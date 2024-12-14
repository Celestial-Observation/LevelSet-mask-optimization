import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image

from lithosim_cuda import LithoSim

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel_torch_data_path = './lithosim_kernels/torch_tensor'
dt = 0.1
dx = 1.0

def load_image(path):
    img = Image.open(path).convert('L')
    img_array = np.array(img) / 255.0
    binary = (img_array > 0.5).astype(np.float32)
    return torch.tensor(binary, dtype=torch.float32).to(device)

def apply_smoothing(mask, sigma=1.0):
    """
    Apply Gaussian smoothing to the mask using a Gaussian kernel.
    """
    # Create Gaussian kernel
    size = int(2 * (3 * sigma) + 1)  # Kernel size based on sigma
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=mask.device)
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()  # Normalize

    # Create 2D Gaussian kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel = kernel.view(1, 1, size, size)  # Reshape for convolution

    # Apply convolution with Gaussian kernel
    smoothed_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=size // 2)
    return smoothed_mask.squeeze(0).squeeze(0)

def grad_image(mask):
    """
    Compute gradients of the mask for PLIC.
    """
    grad_x_kernel = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=mask.device)
    grad_y_kernel = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32, device=mask.device)
    
    # Add padding to maintain same dimensions
    grad_x = F.conv2d(mask.unsqueeze(0).unsqueeze(0), grad_x_kernel, padding=(0, 1))
    grad_y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), grad_y_kernel, padding=(1, 0))
    
    # Ensure output dimensions match input
    grad_x = grad_x.squeeze(0).squeeze(0)[:, :mask.shape[1]]
    grad_y = grad_y.squeeze(0).squeeze(0)[:mask.shape[0], :]
    return grad_x, grad_y

def reconstruct_interface(mask):
    """
    Perform Piecewise Linear Interface Calculation (PLIC) to accurately capture the interface.
    """
    grad_x, grad_y = grad_image(mask)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # Avoid division by zero
    normalized_grad_x = grad_x / (magnitude + 1e-8)
    normalized_grad_y = grad_y / (magnitude + 1e-8)

    # Reconstruct interface using gradients
    reconstructed_mask = mask.clone()
    interface_mask = (mask > 0.0) & (mask < 1.0)

    # Update interface using gradient information
    reconstructed_mask[interface_mask] += 0.1 * (
        normalized_grad_x[interface_mask] + normalized_grad_y[interface_mask]
    )
    reconstructed_mask = torch.clamp(reconstructed_mask, 0.0, 1.0)

    return reconstructed_mask

def photolithograph(mask):
    lithosim = LithoSim("config/lithosimple.txt")
    wafer_image, wafer_image_min, wafer_image_max = lithosim(mask)
    return wafer_image, wafer_image_min, wafer_image_max

def calculate_vof(mask, target, iter=50, min_distance=30):
    mask = mask.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([mask], lr=0.8)

    lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
    best_mask = None

    for iteration in tqdm(range(iter), desc="Optimization Iteration"):
        # Smooth mask to avoid artifacts
        smoothed_mask = apply_smoothing(mask)

        # Reconstruct interface using PLIC
        #reconstructed_mask = reconstruct_interface(smoothed_mask)

        # Simulate photolithography process
        printedNom, printedMax, printedMin = photolithograph(smoothed_mask)

        # Define losses
        l2loss = F.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = F.mse_loss(printedMax, target, reduction="sum") + F.mse_loss(printedMin, target, reduction="sum")
        pvbloss = F.mse_loss(printedMax, printedMin, reduction="sum")
        pvband = torch.sum((printedMax >= 0.225) != (printedMin >= 0.225))

        # Total loss
        loss = l2loss + 0.2 * pvbl2 + 0.8 * pvbloss

        if loss.item() < lossMin:
            lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
            best_mask = smoothed_mask.detach().clone()
            print_img = printedNom

        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Enforce VOF bounds
        
        with torch.no_grad():
            mask.clamp_(0.0, 1.0)
        
    return l2Min, pvbMin, best_mask, print_img

# Main function
def main():
    input_path = 'image/CurvILT_target1.png'
    save_path = 'image/result_vof.png'

    # Load and preprocess target image
    binary_image = load_image(input_path)

    # Initialize VOF mask
    mask = binary_image.clone().detach() + torch.randn_like(binary_image) * 0.01

    # Run optimization
    begin = time.time()
    l2Min, pvbMin, mask_opt, resist_result = calculate_vof(mask, binary_image)
    runtime = time.time() - begin

    print(f"Minimum L2 Loss: {l2Min}nm^2")
    print(f"Minimum PV Bandwidth: {pvbMin}nm^2")
    print(f"Total Runtime: {runtime}s")

    # Visualization
    plt.figure(figsize=(20, 10))

    plt.subplot(131)
    plt.imshow(binary_image.cpu(), cmap="gray")
    plt.title("Layout")

    plt.subplot(132)
    plt.imshow(mask_opt.cpu().detach().numpy(), cmap="gray")
    plt.title("Optimized Mask")

    plt.subplot(133)
    plt.imshow(resist_result.cpu().detach().numpy(), cmap="gray")
    plt.title("Resist Result")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()
