import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale

from lithosim_cuda import LithoSim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel_torch_data_path = './lithosim_kernels/torch_tensor'
dt = 0.1
dx = 1.0

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
    #img = img.resize((512,512),Image.LANCZOS)
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

def pad_replicate_2d(phi):
    
    top = phi[0:1, :]
    bottom = phi[-1:, :]
    phi_padded = torch.cat([top, phi, bottom], dim=0)
    
    left = phi_padded[:, 0:1]
    right = phi_padded[:, -1:]
    phi_padded = torch.cat([left, phi_padded, right], dim=1)
    
    return phi_padded


def reinitialize_phi(phi, iterations=10, delta_tau=0.1, epsilon=1e-6):
    """
    对水平集函数 phi 进行重新初始化，使其回到 SDF 状态。
    参数：
        phi: torch.Tensor，初始水平集函数。
        iterations: int，迭代次数。
        delta_tau: float，时间步长。
        epsilon: float，用于稳定符号函数的分母。
    返回：
        reinitialized_phi: torch.Tensor，重新初始化后的水平集函数。
    """
    phi = phi.clone()
    sign_phi = phi / torch.sqrt(phi**2 + epsilon**2)

    for _ in range(iterations):

        grad_x = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / 2
        grad_y = (phi[1:-1, 2:] - phi[1:-1, :-2]) / 2
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + epsilon)

        correction = sign_phi[1:-1, 1:-1] * (grad_norm - 1)
        phi[1:-1, 1:-1] -= delta_tau * correction

        phi = pad_replicate_2d(phi[1:-1, 1:-1])

    return phi

def photolithograph(mask):
    lithosim = LithoSim("config/lithosimple.txt")
    wafer_image, wafer_image_min, wafer_image_max = lithosim(mask)
    
    return wafer_image, wafer_image_min, wafer_image_max

def calculate_levelset(phi, target, iter=400):

    phi = phi.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([phi], lr=0.8)
    binarize = Binarize()

    lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12

    for iter in range(iter):
        '''
        if iter != 0 and iter % 20 == 0:
            phi = reinitialize_phi(phi)
        '''
        mask = binarize(phi)
        #mask = add_sraf(mask)
        #mask = (phi < 0).float()
        printedNom, printedMax, printedMin = photolithograph(mask)
        l2loss = F.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = F.mse_loss(printedMax, target, reduction="sum") + F.mse_loss(printedMin, target, reduction="sum")
        pvbloss = F.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= 0.225) != (printedMin >= 0.225))

        loss = (l2loss + 0.2 * pvbl2 + 0.8 * pvbloss)

        if loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = phi.detach().clone()
                bestMask = mask.detach().clone()
                print = printedNom
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    return l2Min, pvbMin, bestParams, bestMask, print
#————————————————————————————————————————————————————————————————————————————#
input_path = "image/CurvILT_target10.png"
save_path = 'image/result.png'
resist_path = 'image/resist_img.png'

binary_image = load_image(input_path)
phi = extract_contour(binary_image)
begin = time.time()
l2Min, pvbMin, phi_opt, mask_opt, resist_result = calculate_levelset(phi, binary_image)
runtime = time.time() - begin

print(f"Minimum L2 Loss: {l2Min}nm^2")
print(f"Minimum PV Bandwidth: {pvbMin}nm^2")
print(f"Total Runtime: {runtime}s")

'''
fig, axes = plt.subplots(2, 2)

ax1, ax2, ax3, ax4 = axes.flatten()
ax1.imshow(binary_image.cpu(), cmap="gray")
ax1.set_title("Layout")

ax2.imshow(extract_contour(binary_image).cpu().detach().numpy(), cmap="RdYlBu")
ax2.contour(extract_contour(binary_image).cpu().detach().numpy(),levels=50, colors="white", linewidths=0.5)
fig.colorbar(ax2.imshow(extract_contour(binary_image).cpu().detach().numpy(), cmap="RdYlBu"), ax=ax2)
ax2.set_title("Level Set Function (Phi)")

ax3.imshow(phi_opt.cpu().detach().numpy(), cmap="RdYlBu")
ax3.contour(phi_opt.cpu().detach().numpy(),levels=50, colors="white", linewidths=0.5)
fig.colorbar(ax3.imshow(phi_opt.cpu().detach().numpy(), cmap="RdYlBu"), ax=ax3)
ax3.set_title("Optimized LSF")

ax4.imshow(mask_opt.cpu().detach().numpy(), cmap="gray")
ax4.set_title("Optimized Mask")
'''
plt.figure(figsize=(34.4,14.4))

plt.subplot(221)
plt.imshow(binary_image.cpu(), cmap="gray")
plt.title("Layout")

plt.subplot(222)
pic1 = plt.imshow(extract_contour(binary_image).cpu().detach().numpy(), cmap="RdYlBu")
plt.contour(extract_contour(binary_image).cpu().detach().numpy(),levels=50, colors="white", linewidths=0.5)
plt.colorbar(pic1)
plt.title("Level Set Function (LSF)")

plt.subplot(223)
pic2 = plt.imshow(phi_opt.cpu().detach().numpy(), cmap="RdYlBu")
plt.contour(phi_opt.cpu().detach().numpy(),levels=50, colors="white", linewidths=0.5)
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