import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from levelset import load_image, calculate_levelset_with_mask_check, gradImage, photolithograph, apply_smoothing, Binarize, extract_contour

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_patch(patch, scale):

    h, w = patch.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(patch).resize((new_w, new_h), Image.BILINEAR))

def poisson_blend(source, target, mask):

    laplacian = torch.nn.functional.conv2d(
        source.unsqueeze(0).unsqueeze(0),
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
        padding=1
    )[0, 0]

    result = target.clone()
    result[mask] = source[mask] + 0.25 * laplacian[mask]

    return result

def split_image_into_patches(image, patch_size, overlap):

    h, w = image.shape
    ph, pw = patch_size
    oh, ow = overlap
    
    patches = []
    coordinates = []
    
    for y in range(0, h - ph + 1, ph - oh):
        for x in range(0, w - pw + 1, pw - ow):
            patch = image[y:y+ph, x:x+pw]
            patches.append(patch)
            coordinates.append((y, x))
    
    if h % (ph - oh) != 0:
        for x in range(0, w - pw + 1, pw - ow):
            patch = image[h-ph:h, x:x+pw]
            patches.append(patch)
            coordinates.append((h-ph, x))
    
    if w % (pw - ow) != 0:
        for y in range(0, h - ph + 1, ph - oh):
            patch = image[y:y+ph, w-pw:w]
            patches.append(patch)
            coordinates.append((y, w-pw))
    
    if h % (ph - oh) != 0 and w % (pw - ow) != 0:
        patch = image[h-ph:h, w-pw:w]
        patches.append(patch)
        coordinates.append((h-ph, w-pw))
    
    return patches, coordinates

def split_image_into_patches_with_padding(image, patch_size, overlap, padding=250):

    h, w = image.shape
    ph, pw = patch_size
    oh, ow = overlap

    patches = []
    coordinates = []

    for y in range(0, h - ph + 1, ph - oh):
        for x in range(0, w - pw + 1, pw - ow):
            patch = np.zeros((ph + 2 * padding, pw + 2 * padding), dtype=image.dtype)
            patch[padding:padding + ph, padding:padding + pw] = image[y:y + ph, x:x + pw]
            patches.append(patch)
            coordinates.append((y, x))

    if h % (ph - oh) != 0:
        for x in range(0, w - pw + 1, pw - ow):
            patch = np.zeros((ph + 2 * padding, pw + 2 * padding), dtype=image.dtype)
            patch[padding:padding + (h - (h - ph)), padding:padding + pw] = image[h - ph:h, x:x + pw]
            patches.append(patch)
            coordinates.append((h - ph, x))
    
    if w % (pw - ow) != 0:
        for y in range(0, h - ph + 1, ph - oh):
            patch = np.zeros((ph + 2 * padding, pw + 2 * padding), dtype=image.dtype)
            patch[padding:padding + ph, padding:padding + (w - (w - pw))] = image[y:y + ph, w - pw:w]
            patches.append(patch)
            coordinates.append((y, w - pw))
    
    if h % (ph - oh) != 0 and w % (pw - ow) != 0:
        patch = np.zeros((ph + 2 * padding, pw + 2 * padding), dtype=image.dtype)
        patch[padding:padding + (h - (h - ph)), padding:padding + (w - (w - pw))] = image[h - ph:h, w - pw:w]
        patches.append(patch)
        coordinates.append((h - ph, w - pw))
    
    return patches, coordinates

def merge_patches(patches, coordinates, output_shape, patch_size, overlap):

    h, w = output_shape
    ph, pw = patch_size
    oh, ow = overlap
    
    output = torch.zeros((h, w), device=device)
    weight = torch.zeros((h, w), device=device)
    
    for patch, (y, x) in tqdm(zip(patches, coordinates), total=len(patches), desc="Merging Patches"):
        patch_h, patch_w = patch.shape
        output[y:y+patch_h, x:x+patch_w] += patch
        weight[y:y+patch_h, x:x+patch_w] += 1
    
    weight[weight == 0] = 1

    output /= weight

    output_np = output.detach().cpu().numpy()
    smoothed_output_np = output_np.copy()

    for patch, (y, x) in tqdm(zip(patches, coordinates), total=len(patches), desc="Smoothing Overlaps"):

        y_start, y_end = max(y, 0), min(y + ph, h)
        x_start, x_end = max(x, 0), min(x + pw, w)

        if y > 0:
            overlap_y_start, overlap_y_end = y, y + oh
            smoothed_output_np[overlap_y_start:overlap_y_end, x_start:x_end] = gaussian_filter(
                output_np[overlap_y_start:overlap_y_end, x_start:x_end], sigma=1.0
            )

        if x > 0:
            overlap_x_start, overlap_x_end = x, x + ow
            smoothed_output_np[y_start:y_end, overlap_x_start:overlap_x_end] = gaussian_filter(
                output_np[y_start:y_end, overlap_x_start:overlap_x_end], sigma=1.0
            )

    smoothed_output = torch.tensor(smoothed_output_np, device=device, dtype=output.dtype)

    return smoothed_output

def merge_patches_without_padding(patches, coordinates, output_shape, patch_size, overlap, padding=250):

    h, w = output_shape
    ph, pw = patch_size
    oh, ow = overlap

    output = torch.zeros((h, w), device=device)
    weight = torch.zeros((h, w), device=device)

    for patch, (y, x) in tqdm(zip(patches, coordinates), total=len(patches), desc="Merging Patches"):
        patch = patch[padding:padding + ph, padding:padding + pw]  # 去除黑边
        patch_h, patch_w = patch.shape

        output[y:y + patch_h, x:x + patch_w] += patch
        weight[y:y + patch_h, x:x + patch_w] += 1

    weight[weight == 0] = 1
    output /= weight
    return output


'''def merge_patches(patches, coordinates, output_shape, patch_size, overlap):

    h, w = output_shape
    ph, pw = patch_size
    oh, ow = overlap

    output = torch.zeros((h, w), device=device)
    weight = torch.zeros((h, w), device=device)

    for patch, (y, x) in tqdm(zip(patches, coordinates), total=len(patches), desc="Merging Patches"):
        patch_h, patch_w = patch.shape

        # Non-overlapping region
        output[y:y+patch_h, x:x+patch_w] += patch
        weight[y:y+patch_h, x:x+patch_w] += 1

        # Process overlap with previous patches
        if x > 0:  # Left overlap
            overlap_region = output[y:y+patch_h, x:x+ow]
            patch_overlap = patch[:, :ow]
            mask = weight[y:y+patch_h, x:x+ow] > 0
            fused_overlap = poisson_blend(patch_overlap, overlap_region, mask)
            output[y:y+patch_h, x:x+ow] = fused_overlap

        if y > 0:  # Top overlap
            overlap_region = output[y:y+oh, x:x+patch_w]
            patch_overlap = patch[:oh, :]
            mask = weight[y:y+oh, x:x+patch_w] > 0
            fused_overlap = poisson_blend(patch_overlap, overlap_region, mask)
            output[y:y+oh, x:x+patch_w] = fused_overlap

    # Normalize to handle weight overlaps
    output /= weight
    return output'''
    
def main():
    input_path = "image/gcd_45_output.png"
    save_path = 'image/result.png'
    resist_path = 'image/resist_img.png'
    
    binary_image = load_image(input_path)
    h, w = binary_image.shape
    
    patch_size = (3000, 3000)
    overlap = (250, 250)
    scale_factor = 0.5
    
    patches, coordinates = split_image_into_patches_with_padding(binary_image.cpu().numpy(), patch_size, overlap)
    
    best_phis = []
    best_masks = []
    best_print_imgs = []
    
    temp_folder = 'temp_patches'
    os.makedirs(temp_folder, exist_ok=True)
    
    for i, patch in enumerate(tqdm(patches, desc="Processing Patches")):
        
        resized_patch = resize_patch(patch, scale_factor)
        patch_tensor = torch.tensor(resized_patch, dtype=torch.float32, device=device)
        
        phi = extract_contour(patch_tensor)
        l2Min, pvbMin, phi_opt, mask_opt, resist_result = calculate_levelset_with_mask_check(phi, patch_tensor)
        
        phi_opt_resized = resize_patch(phi_opt.detach().cpu().numpy(), 1 / scale_factor)
        mask_opt_resized = resize_patch(mask_opt.detach().cpu().numpy(), 1 / scale_factor)
        resist_result_resized = resize_patch(resist_result.detach().cpu().numpy(), 1 / scale_factor)
        
        phi_opt = torch.tensor(phi_opt_resized, dtype=torch.float32, device=device)
        mask_opt = torch.tensor(mask_opt_resized, dtype=torch.float32, device=device)
        resist_result = torch.tensor(resist_result_resized, dtype=torch.float32, device=device)
        
        torch.save(phi_opt, os.path.join(temp_folder, f'phi_{i}.pt'))
        torch.save(mask_opt, os.path.join(temp_folder, f'mask_{i}.pt'))
        torch.save(resist_result, os.path.join(temp_folder, f'resist_{i}.pt'))
        
        del patch_tensor, phi, l2Min, pvbMin, phi_opt, mask_opt, resist_result
        torch.cuda.empty_cache()
    
    for i in tqdm(range(len(patches)), desc="Loading Patches"):
        phi_opt = torch.load(os.path.join(temp_folder, f'phi_{i}.pt'))
        mask_opt = torch.load(os.path.join(temp_folder, f'mask_{i}.pt'))
        resist_result = torch.load(os.path.join(temp_folder, f'resist_{i}.pt'))
        
        best_phis.append(phi_opt)
        best_masks.append(mask_opt)
        best_print_imgs.append(resist_result)
        
        del phi_opt, mask_opt, resist_result
        torch.cuda.empty_cache()

    binarize = Binarize()
    
    merged_phi = merge_patches_without_padding(best_phis, coordinates, (h, w), patch_size, overlap)
    #merged_mask = merge_patches(best_masks, coordinates, (h, w), patch_size, overlap)
    merged_mask = binarize(merged_phi)
    merged_resist_result = merge_patches_without_padding(best_print_imgs, coordinates, (h, w), patch_size, overlap)
    #merged_mask = torch.where(merged_mask > 0.675, torch.tensor(1), torch.tensor(0))
    
    plt.figure(figsize=(200, 200))
    
    plt.subplot(221)
    plt.imshow(binary_image.cpu(), cmap="gray")
    plt.title("Layout")
    
    plt.subplot(222)
    pic1 = plt.imshow(extract_contour(binary_image).cpu().detach().numpy(), cmap="RdYlBu")
    plt.contour(extract_contour(binary_image).cpu().detach().numpy(), levels=50, colors="white", linewidths=0.5)
    plt.colorbar(pic1)
    plt.title("Level Set Function (LSF)")
    
    plt.subplot(223)
    pic2 = plt.imshow(merged_phi.cpu().detach().numpy(), cmap="RdYlBu")
    plt.contour(merged_phi.cpu().detach().numpy(), levels=50, colors="white", linewidths=0.5)
    plt.colorbar(pic2)
    plt.title("Optimized LSF")
    
    plt.subplot(224)
    plt.imshow(merged_mask.cpu().detach().numpy(), cmap="gray")
    plt.title("Optimized Mask")
    
    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()
    
    plt.figure(figsize=(102.4, 102.4))
    plt.imshow(merged_resist_result.cpu().detach().numpy(), cmap="gray")
    plt.savefig(resist_path)
    #plt.show()
    
    import shutil
    shutil.rmtree(temp_folder)

if __name__ == "__main__":
    
    import cProfile
    cProfile.run("main()", "result")
    
    import pstats
    pstats.Stats('result').sort_stats(-1).print_stats()