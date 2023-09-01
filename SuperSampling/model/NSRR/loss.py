import torch
from kornia.metrics import ssim as compute_ssim

def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float=1) -> torch.Tensor:
    """
    Computes just the SSIM loss.
    TODO: add vgg16 loss
    
    """
    # # SSIM currently seems broken?
    loss_ssim = 1 - torch.mean(compute_ssim(output, target, 9))
    return loss_ssim