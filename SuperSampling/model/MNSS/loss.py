import torch
import torch.nn as nn
import torch.nn.functional as F
from model.perceptual_loss import PerceptualLoss
from kornia.metrics import ssim 

class MNSSLoss(nn.Module):
    def __init__(self, scale_factor: int, k: float, w: float) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.k = k
        self.w = w
        # TODO: check what layers they use
        self.perceptual_loss = PerceptualLoss() 
        self.eval()

    def forward(self,
                img_aa: torch.Tensor,
                img_ss: torch.Tensor,
                img_truth: torch.Tensor,
                jitter: tuple[int, int]):
        structural_loss = 1 - torch.mean(ssim(img_ss, img_truth, 11))
        perceptual_loss = self.perceptual_loss(img_ss, img_truth)
        antialiasing_loss = F.l1_loss(img_aa, img_truth[:, :, jitter[0]::self.scale_factor, jitter[1]::self.scale_factor])
        return structural_loss + self.k * antialiasing_loss + self.w * perceptual_loss

