import torch
import numpy as np
import math
from pytorch_msssim import ssim as _ssim

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.shape == img2.shape

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    
    return 10 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.shape == img2.shape
    
    return _ssim(img1, img2, data_range=255, size_average=True)