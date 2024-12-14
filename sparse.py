import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from PIL import Image
from lithosim_cuda import LithoSim
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel_torch_data_path = './lithosim_kernels/torch_tensor'
dt = 0.1
dx = 1.0

class _MaskRuleCheck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, min_distance):
        ctx.min_distance = min_distance
        ctx.save_for_backward(mask)
        
        # 提取轮廓
        mask_np = mask.detach().cpu().numpy()
        labeled_array, num_features = ndimage.label(mask_np)
        centers = ndimage.center_of_mass(mask_np, labeled_array, range(1, num_features + 1))
        
        # 如果没有找到轮廓，返回0惩罚
        if len(centers) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=mask.device)
        
        # 计算所有轮廓之间的距离
        tree = KDTree(centers)
        distances, indices = tree.query(centers, k=2)
        penalty = 0.0
        for i in range(len(distances)):
            if distances[i][1] < min_distance:
                penalty += min_distance - distances[i][1]
        
        return torch.tensor(penalty, dtype=torch.float32, device=mask.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        min_distance = ctx.min_distance
        
        # 提取轮廓
        mask_np = mask.detach().cpu().numpy()
        labeled_array, num_features = ndimage.label(mask_np)
        centers = ndimage.center_of_mass(mask_np, labeled_array, range(1, num_features + 1))
        
        # 如果没有找到轮廓，返回0梯度
        if len(centers) == 0:
            return torch.zeros_like(mask), None
        
        # 计算梯度
        grad_mask = torch.zeros_like(mask)
        centers_np = np.array(centers)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.linalg.norm(centers_np[i] - centers_np[j])
                if distance < min_distance:
                    direction = (centers_np[i] - centers_np[j]) / distance
                    grad_mask[labeled_array == i + 1] += direction[0] * grad_output.item()
                    grad_mask[labeled_array == j + 1] -= direction[0] * grad_output.item()
    
        return grad_mask, None

def apply_smoothing(mask):
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device) / 9.0
    smoothed_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return smoothed_mask.squeeze(0).squeeze(0)

class _Binarize(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, levelset): 
        ctx.save_for_backward(levelset)
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask
    
    @staticmethod
    def backward(ctx, grad_output): 
        levelset, = ctx.saved_tensors
        gradX, gradY = gradImage(levelset)
        l2norm = torch.sqrt(gradX**2 + gradY**2)
        return -l2norm * grad_output
    
class Binarize(nn.Module): 
    def __init__(self): 
        super(Binarize, self).__init__()
        pass

    def forward(self, levelset): 
        return _Binarize.apply(levelset)

def load_image(path):
    img = Image.open(path).convert('L')
    img_array = np.array(img) / 255.0
    binary = (img_array > 0.5).astype(np.float32)
    return torch.tensor(binary, dtype=torch.float32).to(device)

def extract_contour(binary_image):
    dist_transform = distance_transform_edt(binary_image.cpu().numpy())
    dist_transform_outside = distance_transform_edt(1 - binary_image.cpu().numpy())
    sdf = dist_transform_outside - dist_transform
    return torch.tensor(sdf, dtype=torch.float32).to(device)

def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image, (1, 1, 1, 1), mode='replicate')[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)

def photolithograph(mask):
    lithosim = LithoSim("config/lithosimple.txt")
    wafer_image, wafer_image_min, wafer_image_max = lithosim(mask)
    return wafer_image, wafer_image_min, wafer_image_max

def calculate_levelset_with_mask_check(phi, target, iter=50, min_distance=30):

    phi = phi.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=0.8)
    binarize = Binarize()
    
    lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
    best_phi = phi
    best_mask = None
    
    # 加入进度条
    for _ in tqdm(range(iter), desc="Optimization Iteration"):
        mask = binarize(phi)
        
        MRC_penalty = _MaskRuleCheck.apply(mask, min_distance)
        
        smoothed_mask = apply_smoothing(mask)

        with torch.no_grad():
            printedNom, printedMax, printedMin = photolithograph(smoothed_mask)
        
        l2loss = F.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = F.mse_loss(printedMax, target, reduction="sum") + F.mse_loss(printedMin, target, reduction="sum")
        pvbloss = F.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= 0.225) != (printedMin >= 0.225))

        loss = (l2loss + 0.2 * pvbl2 + 0.8 * pvbloss) + MRC_penalty

        if loss.item() < lossMin:
            lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
            bestParams = phi.detach().clone()
            bestMask = smoothed_mask.detach().clone()
            print_img = printedNom

        opt.zero_grad()
        loss.backward()
        opt.step()

    return l2Min, pvbMin, bestParams, bestMask, print_img

# ————————————————————————————————————————————#

input_path = 'image/CurvILT_target6.png'
save_path = 'image/result.png'
resist_path = 'image/resist_img.png'

binary_image = load_image(input_path)
phi = extract_contour(binary_image)

begin = time.time()
l2Min, pvbMin, phi_opt, mask_opt, resist_result = calculate_levelset_with_mask_check(phi, binary_image)
runtime = time.time() - begin

print(f"Minimum L2 Loss: {l2Min}nm^2")
print(f"Minimum PV Bandwidth: {pvbMin}nm^2")
print(f"Total Runtime: {runtime}s")

plt.figure(figsize=(34.4, 14.4))

plt.subplot(221)
plt.imshow(binary_image.cpu(), cmap="gray")
plt.title("Layout")

plt.subplot(222)
pic1 = plt.imshow(extract_contour(binary_image).cpu().detach().numpy(), cmap="RdYlBu")
plt.contour(extract_contour(binary_image).cpu().detach().numpy(), levels=50, colors="white", linewidths=0.5)
plt.colorbar(pic1)
plt.title("Level Set Function (LSF)")

plt.subplot(223)
pic2 = plt.imshow(phi_opt.cpu().detach().numpy(), cmap="RdYlBu")
plt.contour(phi_opt.cpu().detach().numpy(), levels=50, colors="white", linewidths=0.5)
plt.colorbar(pic2)
plt.title("Optimized LSF")

plt.subplot(224)
plt.imshow(mask_opt.cpu().detach().numpy(), cmap="gray")
plt.title("Optimized Mask")

plt.tight_layout()
plt.savefig(save_path)

plt.figure(figsize=(10.24, 10.24))
plt.imshow(resist_result.cpu().detach().numpy())
plt.savefig(resist_path)