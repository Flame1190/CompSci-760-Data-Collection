import torch
import math
from kornia.metrics import ssim as compute_ssim
from utils import warp

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.shape == img2.shape

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    
    return 10 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.shape == img2.shape

    return torch.mean(compute_ssim(img1, img2, 9))

# , threshold: float = 0.1
    # """
    # - warp into one frame of reference
    # - calculate difference map
    # l1 loss between diff maps?

    # or mask and check masking ratio?
    # """
def wsdr(l_sr: torch.Tensor, r_sr: torch.Tensor, l_hr: torch.Tensor, r_hr: torch.Tensor, d_hr: torch.Tensor, warping_coeff: float = 0.1845) -> float:
    assert l_sr.shape == r_sr.shape == l_hr.shape == r_hr.shape 
    assert d_hr.shape[1] == 1
    assert d_hr.shape[2:] == l_hr.shape[2:]
    assert d_hr.shape[0] == l_hr.shape[0]

    disparity = d_hr * warping_coeff
    disparity = torch.cat([disparity, torch.zeros_like(disparity)], dim=1) # exclude vertical disparity
    
    r_sr_warped = warp(r_sr, disparity)
    r_hr_warped = warp(r_hr, disparity)

    diff_sr = torch.abs(l_sr - r_sr_warped)
    diff_hr = torch.abs(l_hr - r_hr_warped)

    return diff_sr.mean() / diff_hr.mean()
