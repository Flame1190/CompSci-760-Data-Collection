import torch
from pytorch_msssim import ssim

def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float=1) -> torch.Tensor:
    """
    Computes just the SSIM loss.
    TODO: add vgg16 loss
    
    """
    # # SSIM currently seems broken?
    # loss_ssim = 1 - ssim(output, target, data_range=255, size_average=True)
    # return loss_ssim
    l1_loss = torch.nn.L1Loss()
    loss_l1 = l1_loss(output, target)
    
    return loss_l1 