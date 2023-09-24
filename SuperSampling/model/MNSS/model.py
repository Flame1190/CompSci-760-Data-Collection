import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import warp

class MNSS(BaseModel):
    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.warping = Warping()
        self.inpainting = Inpainting(scale_factor)
        self.blending = Blending(scale_factor)
        self.upscaling = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self,
                current_color: torch.Tensor,
                current_depth: torch.Tensor,
                previous_color: torch.Tensor,
                previous_depth: torch.Tensor, # low resolution
                motion: torch.Tensor,
                jitter: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param current_color: the color of the current pixel
        :param current_depth: the depth of the current pixel
        :param previous_color: the color of the previous pixel
        :param previous_depth: the depth of the previous pixel
        :param motion: the motion vector of the current pixel
            2, H, W
        :param jitter: the subpixel jitter of the current pixel
            2

        :return: the reconstructed color of the current pixel
        """
        previous_depth = self.upscaling(previous_depth)
        motion = self.upscaling(motion)

        warped_color, warped_depth = self.warping(previous_color, previous_depth, motion)
        previous_color = self.inpainting(current_color, current_depth, warped_color, warped_depth, jitter)
        return self.blending(current_color, previous_color, jitter)

class Warping(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prev_color: torch.Tensor, prev_depth: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """
        :param prev_color: the color of the previous pixel
        :param prev_depth: the depth of the previous pixel
        :param motion: the motion vector of the current pixel
            2, H, W

        :return: the warped color of the current pixel
        """
        return warp(prev_color, motion), warp(prev_depth, motion)

class Inpainting(BaseModel):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        # this breaks autograd ?
        self.difference_net = nn.Sequential(
            nn.Conv2d(4, 3, 3, padding=1, stride=1),
            # nn.Sigmoid() # can use clipped TANH instead
            nn.Hardtanh(min_val=0, max_val=1)
        )
        # TODO: replace with jitter conditioned bilinear upsampling
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, 
                current_color: torch.Tensor, 
                current_depth: torch.Tensor, 
                previous_color: torch.Tensor, 
                previous_depth: torch.Tensor,
                jitter: tuple[int, int]) -> torch.Tensor:
        color_dif = current_color - previous_color[:, :, jitter[0]::self.scale_factor, jitter[1]::self.scale_factor]
        depth_dif = current_depth - previous_depth[:, :, jitter[0]::self.scale_factor, jitter[1]::self.scale_factor]
        difference = torch.cat([color_dif, depth_dif], dim=1)
        difference = self.difference_net(difference)

        difference = self.upsample(difference)
        color = self.upsample(current_color)
        
        # difference has to be detached???
        return (difference * color) + ((1 - difference) * previous_color)
        
        

class Blending(BaseModel):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.pool = nn.AvgPool2d(scale_factor)
        self.antialiasing = nn.Sequential(
            nn.Conv2d(6, 3, 1, 1),
            nn.ReLU()
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, 
                current_color: torch.Tensor, 
                previous_color: torch.Tensor, 
                jitter: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param antialiased_color: the antialiased color of the current pixel
        :param previous_color: the color of the previous pixel
        :param jitter: the subpixel jitter of the current pixel
            H, W

        :return: the reconstructed color of the current pixel
        """
        pooled_previous_color = self.pool(previous_color)
        antialiased_color = self.antialiasing(torch.cat([current_color, pooled_previous_color], dim=1))

        # populate the previous color with the antialiased color according to subpixel jitter
        updated_previous_color = previous_color.clone()
        updated_previous_color[:, :, jitter[0]::self.scale_factor, jitter[1]::self.scale_factor] = antialiased_color

        return self.reconstruction(previous_color), antialiased_color

    