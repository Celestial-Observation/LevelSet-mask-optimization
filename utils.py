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
import lithosim_cuda as lithosim
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REALTYPE = torch.float32

EPE_CONSTRAINT = 15
EPE_CHECK_INTERVEL = 40
MIN_EPE_CHECK_LENGTH = 80
EPE_CHECK_START_INTERVEL = 40

#主要代码修改自OpenILT的evaluation.py。

class Basic: 
    def __init__(self, litho=lithosim.LithoSim("./config/lithosimple.txt"), thresh=0.5, device=DEVICE): 
        self._litho  = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = F.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return l2loss.item(), pvband.item()

    def sim(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = F.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return mask, binaryNom

def boundaries(target, dtype=REALTYPE, device=DEVICE):
    boundary   = torch.zeros_like(target)
    corner     = torch.zeros_like(target)
    vertical   = torch.zeros_like(target)
    horizontal = torch.zeros_like(target)

    padded = F.pad(target[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    upperleft  = padded[2:,  :-2] == 1
    upperright = padded[2:,  2:]  == 1
    lowerleft  = padded[:-2, :-2] == 1
    lowerright = padded[:-2, 2:]  == 1
    boundary = (target == 1)
    boundary[upper & lower & left & right & upperleft & upperright & lowerleft & lowerright] = False
    
    padded = F.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper  = padded[2:,   1:-1]  == 1
    lower  = padded[:-2,  1:-1]  == 1
    left   = padded[1:-1, :-2]   == 1
    right  = padded[1:-1, 2:]    == 1
    center = padded[1:-1, 1:-1]  == 1

    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort((vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy()))
    vsites = vsites[vindices]
    vstart = torch.cat((torch.tensor([True], device=vsites.device), vsites[:, 0][1:] != vsites[:, 0][:-1] + 1))
    vend   = torch.cat((vsites[:, 0][1:] != vsites[:, 0][:-1] + 1, torch.tensor([True], device=vsites.device)))
    vstart = vsites[(vstart == True).nonzero()[:, 0], :]
    vend   = vsites[(vend   == True).nonzero()[:, 0], :]
    vposes = torch.stack((vstart, vend), axis=2)
    
    horizontal = center.clone()
    horizontal[upper & lower] = False
    hsites = horizontal.nonzero()
    hindices = np.lexsort((hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy()))
    hsites = hsites[hindices]
    hstart = torch.cat((torch.tensor([True], device=hsites.device), hsites[:, 1][1:] != hsites[:, 1][:-1] + 1))
    hend   = torch.cat((hsites[:, 1][1:] != hsites[:, 1][:-1] + 1, torch.tensor([True], device=hsites.device)))
    hstart = hsites[(hstart == True).nonzero()[:, 0], :]
    hend   = hsites[(hend   == True).nonzero()[:, 0], :]
    hposes = torch.stack((hstart, hend), axis=2)

    return vposes.float(), hposes.float()


def check(image, sample, target, direction):
    if direction == 'v':
        if ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0)): #left ,x small
            inner = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif ((target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0) and (target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1)): #right, x large
            inner = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    if direction == 'h':
        if((target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0)): #up, y small
            inner = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0) and (target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1): #low, y large
            inner = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    return inner, outer


def epecheck(mask, target, vposes, hposes):
    '''
    input: binary image tensor: (b, c, x, y); vertical points pair vposes: (N_v,4,2); horizontal points pair: (N_h, 4, 2), target image (b, c, x, y)
    output the total number of epe violations
    '''
    inner = 0
    outer = 0
    epeMap = torch.zeros_like(target)
    vioMap = torch.zeros_like(target)

    for idx in range(vposes.shape[0]):
        center = (vposes[idx, :, 0] + vposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0) #(1, 2)
        if (vposes[idx, 0, 1] - vposes[idx, 0, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        else:
            sampleY = torch.cat((torch.arange(vposes[idx, 0, 0] + EPE_CHECK_START_INTERVEL, center[0, 0] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(vposes[idx, 0, 1] - EPE_CHECK_START_INTERVEL, center[0, 0],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = vposes[idx, :, 0].repeat(sampleY.shape[0], 1)
            sample[:, 0] = sampleY
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'v')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1

    for idx in range(hposes.shape[0]):
        center = (hposes[idx, :, 0] + hposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if (hposes[idx, 1, 1] - hposes[idx, 1, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        else: 
            sampleX = torch.cat((torch.arange(hposes[idx, 1, 0] + EPE_CHECK_START_INTERVEL, center[0, 1] + 1, step = EPE_CHECK_INTERVEL), 
                                 torch.arange(hposes[idx, 1, 1] - EPE_CHECK_START_INTERVEL, center[0, 1],     step = -EPE_CHECK_INTERVEL))).unique()
            sample = hposes[idx, :, 0].repeat(sampleX.shape[0], 1)
            sample[:, 1] = sampleX
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, 'h')
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    return inner, outer, vioMap


class EPEChecker: 
    def __init__(self, litho=lithosim.LithoSim("./config/lithosimple.txt"), thresh=0.5, device=DEVICE): 
        self._litho  = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1): 
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad(): 
            mask[mask >= self._thresh]  = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1: 
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            vposes, hposes = boundaries(target)
            epeIn, epeOut, _ =  epecheck(binaryNom, target, vposes, hposes)
        return epeIn, epeOut
    
def evaluate(mask, target, litho, scale=1, shots=False, verbose=False): 
    test = Basic(litho, 0.5)
    epeCheck = EPEChecker(litho, 0.5)
    #shotCount = ShotCounter(litho, 0.5)

    l2, pvb = test.run(mask, target, scale=scale)
    epeIn, epeOut = epeCheck.run(mask, target, scale=scale)
    epe = epeIn + epeOut
    if verbose: 
        print(f"[]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f};")

    return l2, pvb, epe

class _MaskRuleCheck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, min_distance):
        ctx.min_distance = min_distance
        ctx.save_for_backward(mask)
        
        labeled_array, num_features = ndimage.label(mask.detach().cpu().numpy())
        centers = ndimage.center_of_mass(mask.detach().cpu().numpy(), labeled_array, range(1, num_features + 1))
        
        if len(centers) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=mask.device)
        
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
        
        mask_np = mask.detach().cpu().numpy()
        labeled_array, num_features = ndimage.label(mask_np)
        centers = ndimage.center_of_mass(mask_np, labeled_array, range(1, num_features + 1))
        
        if len(centers) == 0:
            return torch.zeros_like(mask), None
        
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
    
class Dilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size):

        if input.dim() == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim() != 4:
            raise ValueError("Input tensor must be 2D, 3D, or 4D.")

        ctx.kernel_size = kernel_size
        dilation = F.max_pool2d(input, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        ctx.save_for_backward(input, dilation)
        return dilation.squeeze(0).squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        input, dilation = ctx.saved_tensors
        kernel_size = ctx.kernel_size

        if grad_output.dim() == 2:
            grad_output = grad_output.unsqueeze(0).unsqueeze(0)
        elif grad_output.dim() == 3:
            grad_output = grad_output.unsqueeze(0)
        elif grad_output.dim() != 4:
            raise ValueError("grad_output must be 2D, 3D, or 4D.")

        grad_input = torch.zeros_like(input)
        max_mask = (input == dilation)
        grad_input[max_mask] = grad_output[max_mask]
        return grad_input.squeeze(0).squeeze(0), None


class Erosion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size):

        if input.dim() == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim() != 4:
            raise ValueError("Input tensor must be 2D, 3D, or 4D.")
        
        ctx.kernel_size = kernel_size
        erosion = -F.max_pool2d(-input, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        ctx.save_for_backward(input, erosion)
        return erosion.squeeze(0).squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        input, erosion = ctx.saved_tensors
        kernel_size = ctx.kernel_size

        if grad_output.dim() == 2:
            grad_output = grad_output.unsqueeze(0).unsqueeze(0)
        elif grad_output.dim() == 3:
            grad_output = grad_output.unsqueeze(0)
        elif grad_output.dim() != 4:
            raise ValueError("grad_output must be 2D, 3D, or 4D.")

        grad_input = torch.zeros_like(input)
        min_mask = (input == erosion)
        grad_input[min_mask] = grad_output[min_mask]
        return grad_input.squeeze(0).squeeze(0), None
    
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
    

def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = F.pad(image, (1, 1, 1, 1), mode='replicate')[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)