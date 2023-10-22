import torch
import torch.nn as nn
import torch.nn.functional as F
from model.perceptual_loss import PerceptualLoss
from kornia.metrics import ssim 

class SINSSLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.perceptual_loss = PerceptualLoss() 
        self.eval()

    def forward(self,
                left_output: torch.Tensor,
                left_target: torch.Tensor,
                right_output: torch.Tensor,
                right_target: torch.Tensor,
                ):
        pixel_loss = F.l1_loss(left_output, left_target) + F.l1_loss(right_output, right_target)
        structural_loss = 2 - torch.mean(ssim(left_output, left_target, 11)) - torch.mean(ssim(right_output, right_target, 11))
        perceptual_loss = self.perceptual_loss(left_output, left_target) + self.perceptual_loss(right_output, right_target)


        return (self.alpha * pixel_loss + self.beta * structural_loss + self.gamma * perceptual_loss) / 2

