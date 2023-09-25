import torch
import torch.nn as nn
import torch.nn.functional as F
from model.perceptual_loss import PerceptualLoss
from kornia.metrics import ssim 

class INSSLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.perceptual_loss = PerceptualLoss() 
        self.eval()

    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                ):
        pixel_loss = F.l1_loss(output, target)
        structural_loss = 1 - torch.mean(ssim(output, target, 11))
        perceptual_loss = self.perceptual_loss(output, target)
        return self.alpha * pixel_loss + self.beta * structural_loss + self.gamma * perceptual_loss

