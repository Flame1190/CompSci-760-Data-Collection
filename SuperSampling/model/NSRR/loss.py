import torch
import torch.nn as nn
from model.perceptual_loss import PerceptualLoss
from kornia.metrics import ssim

class NSRRLoss(nn.Module):
    def __init__(self, w: float=1):
        super().__init__()
        self.w = w
        # they use 5 layers in xiao et al
        # it appears they use all 4-style layers in the paper
        # plus the feature layer
        self.vgg_loss = PerceptualLoss(*[
            ("3", "relu1_2"),
            ("8", "relu2_2"),
            ("15", "relu3_3"),
            ("15", "relu3_3"), # feature layer
            ("22", "relu4_3")  
        ])
        self.eval()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert output.shape == target.shape

        loss_ssim = 1 - torch.mean(ssim(output, target, 11))
        loss_perceptual = self.vgg_loss(output, target) 

        return loss_ssim + self.w * loss_perceptual
