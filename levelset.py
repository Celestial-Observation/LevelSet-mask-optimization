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
from utils import Binarize, _MaskRuleCheck, Dilation, Erosion
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale

from lithosim_cuda import LithoSim
from utils import evaluate

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel_torch_data_path = './lithosim_kernels/torch_tensor'
kernel_size = 10

def apply_smoothing(mask):
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device) / 9.0
    smoothed_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return smoothed_mask.squeeze(0).squeeze(0)

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

def photolithograph(mask):
    lithosim = LithoSim("config/lithosimple.txt")
    wafer_image, wafer_image_min, wafer_image_max = lithosim(mask)
    
    return wafer_image, wafer_image_min, wafer_image_max

def calculate_levelset_with_mask_check(phi, target, iter=400, min_distance=30):

    phi = phi.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=0.8)
    binarize = Binarize()
    dilation = Dilation.apply
    erosion = Erosion.apply
    #mask_check = _MaskRuleCheck.apply(min_distance=min_distance)
    
    lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
    best_phi = phi
    best_mask = None
    
    for iter in tqdm(range(iter), desc="Optimization Iteration"):
        mask = binarize(phi)
        
       #MRC_penalty = _MaskRuleCheck.apply(mask, min_distance)
       #print(iter': 'str(MRC_penalty.item())) 
        
        smoothed_mask = apply_smoothing(mask)
        #mask_test = torch.sigmoid(4 * (smoothed_mask))

        printedNom, printedMax, printedMin = photolithograph(smoothed_mask)
        
        l2loss = F.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = F.mse_loss(printedMax, target, reduction="sum") + F.mse_loss(printedMin, target, reduction="sum")
        #pvbloss = F.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= 0.225) != (printedMin >= 0.225))

        #loss = (l2loss + 0.2 * pvbl2 + 0.8 * pvbloss) + MRC_penalty
        loss = (l2loss + 1 * pvbl2)

        if loss.item() < lossMin:
            lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
            bestPhi = phi.detach().clone()
            bestMask = smoothed_mask.detach().clone()
            dilation_mask = dilation(mask, kernel_size)
            bestMask = erosion(dilation_mask, kernel_size)
            print_img = printedNom

        opt.zero_grad()
        loss.backward()
        opt.step()

    return l2Min, pvbMin, bestPhi, bestMask, print_img

# ————————————————————————————————————————————#
def main():
    input_path = 'image/CurvILT_target1.png'
    save_path = 'image/result.png'
    resist_path = 'image/resist_img.png'
    
    lithosim = LithoSim("config/lithosimple.txt")

    binary_image = load_image(input_path)
    phi = extract_contour(binary_image)

    begin = time.time()
    _, _, phi_opt, mask_opt, resist_result = calculate_levelset_with_mask_check(phi, binary_image)
    runtime = time.time() - begin
    
    resist_binary = torch.zeros_like(resist_result)
    resist_binary[resist_result >= 0.5] = 1
    
    l2Min, pvbMin, epe = evaluate(mask_opt, binary_image, lithosim)

    print(f"Minimum L2 Loss: {l2Min}nm^2")
    print(f"Minimum PV Bandwidth: {pvbMin}nm^2")
    print(f'Minimum EPE chount: {epe}')
    print(f"Total Runtime: {runtime}s")

    plt.figure(figsize=(20, 20))

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
    #plt.show()

    plt.figure(figsize=(10.24, 10.24))
    plt.imshow(resist_binary.cpu().detach().numpy(), cmap="gray")
    plt.savefig(resist_path)
    #plt.show()

if __name__ == "__main__":
    
    import cProfile
    cProfile.run("main()", "output.prof")
    