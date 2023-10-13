
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import warp, retrieve_elements_from_indices, flatten

# what if in-painting also returns a feature map?
# just use depth for warping?

class SINSS(BaseModel):
    def __init__(self, scale_factor: int, f: int = 16, m: int = 1, stereo_warping_coefficient: float = 0.1845):
        super().__init__()
        self.scale_factor = scale_factor
        self.shuffled_channel_count = 3 * (scale_factor ** 2)
        self.space_to_depth = nn.PixelUnshuffle(scale_factor)
        self.depth_to_space = nn.PixelShuffle(scale_factor)

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        # self.stereo_warping_coefficient = stereo_warping_coefficient
        self.warping = Warping(scale_factor=scale_factor)
        self.stereo_warping = StereoWarping(stereo_warping_coefficient)

        self.inpainting = Inpainting()
        self.net = Network(self.shuffled_channel_count, f, m)
        

    def forward(self, 
        left_current_color, left_current_depth, left_current_motion, left_prev_color, left_prev_depth,
        right_current_color, right_current_depth, right_current_motion, right_prev_color, right_prev_depth):

        # warp historical high res
        left_prev_color, left_prev_depth = self.warping(left_prev_color, left_prev_depth, left_current_depth, left_current_motion)
        right_prev_color, right_prev_depth = self.warping(right_prev_color, right_prev_depth, right_current_depth, right_current_motion)

        # warp current low res cross view
        left_warped_color, left_warped_depth = self.stereo_warping(right_current_color, right_current_depth, left_current_depth, left_to_right=True)
        right_warped_color, right_warped_depth = self.stereo_warping(left_current_color, left_current_depth, right_current_depth, left_to_right=False)
    
        # Upsample for inpainting
        left_up_current_color, left_up_current_depth = self.upsample(left_current_color), self.upsample(left_current_depth)
        right_up_current_color, right_up_current_depth = self.upsample(right_current_color), self.upsample(right_current_depth)
        left_prev_depth, right_prev_depth = self.upsample(left_prev_depth), self.upsample(right_prev_depth)

        # Inpaint
        # historical
        left_prev_color = self.inpainting(
            left_up_current_color,
            left_up_current_depth,  
            left_prev_color,
            left_prev_depth
        )
        left_prev_color = self.space_to_depth(left_prev_color)

        right_prev_color = self.inpainting(
            right_up_current_color,
            right_up_current_depth,  
            right_prev_color,
            right_prev_depth
        )
        right_prev_color = self.space_to_depth(right_prev_color)

        # stereo
        left_warped_color = self.inpainting(
            left_warped_color,
            left_warped_depth,
            right_current_color,
            right_current_depth
        )

        right_warped_color = self.inpainting(
            right_warped_color,
            right_warped_depth,
            left_current_color,
            left_current_depth
        )

        # reconstruct
        left_residuals, left_mask = self.net(left_current_color, left_current_depth, left_prev_color)
        left_mix = left_mask * left_residuals + (1 - left_mask) * left_prev_color
        left_mix = self.depth_to_space(left_mix)

        right_residuals, right_mask = self.net(right_current_color, right_current_depth, right_prev_color)
        right_mix = right_mask * right_residuals + (1 - right_mask) * right_prev_color
        right_mix = self.depth_to_space(right_mix)

        return left_mix, right_mix
    
class Network(BaseModel):
    def __init__(self, shuffled_channel_count: int, f: int, m: int):
        super().__init__()
        self.shuffled_channel_count = shuffled_channel_count
        in_channels = sum((
            3, # color current
            1, # depth current
            3, # opposite view color
            self.shuffled_channel_count # color prev shuffled
        ))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
            nn.ReLU(),
            *flatten(
                [[nn.Conv2d(f, f, kernel_size=3, padding=1), nn.ReLU()] for _ in range(m)]),
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.residual_conv = nn.Sequential(
            nn.Conv2d(f, shuffled_channel_count, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(f, shuffled_channel_count, kernel_size=3, padding=1),
            nn.Sigmoid(), # maybe hard-tanh?
        )

    def forward(self, current_color, current_depth, warped_color):
        x = torch.cat([current_color, current_depth, warped_color], dim=1)
        x = self.net(x)

        # residuals, mask = x[:, :self.shuffled_channel_count], x[:, self.shuffled_channel_count:]
        residuals = self.residual_conv(x)
        mask = self.mask_conv(x)

        return residuals, mask
        

class Warping(BaseModel):
    def __init__(self, scale_factor: int, depth_dilation_window: int = 3):
        super().__init__()
        assert depth_dilation_window % 2 == 1
        self.scale_factor = scale_factor
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.pool = nn.MaxPool2d(kernel_size=depth_dilation_window, stride=1, padding=depth_dilation_window // 2, return_indices=True)

    def forward(self, prev_color, prev_depth, current_depth, current_motion):
        # Depth informed dilation
        # Get indices of closest pixels and use those motion vectors
        _, indices = self.pool(current_depth)
        
        current_motion = retrieve_elements_from_indices(current_motion, indices)
        high_res_current_motion = self.upsample(current_motion)

        # Warp previous features and color
        prev_color = warp(prev_color, high_res_current_motion)
        prev_depth = warp(prev_depth, current_motion)

        return prev_color, prev_depth
    

class StereoWarping(BaseModel):
    def __init__(self, warping_coeff: float) -> None:
        super().__init__()
        self.warping_coeff = warping_coeff

    def forward(self, color: torch.Tensor, depth: torch.Tensor, opposite_depth: torch.Tensor, left_to_right: bool):
        disparity = opposite_depth * self.warping_coeff * (-1 if left_to_right else 1)
        disparity = torch.cat([disparity, torch.zeros_like(disparity)], dim=1) # exclude vertical disparity
        return warp(color, disparity), warp(depth, disparity)

class Inpainting(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.difference_net = DoubleConv(4, 8, 3, activation=lambda : nn.Hardtanh(min_val=0, max_val=1))

    def forward(self, 
                current_color: torch.Tensor, 
                current_depth: torch.Tensor, 
                previous_color: torch.Tensor, 
                previous_depth: torch.Tensor) -> torch.Tensor:
        color_dif = current_color - previous_color
        depth_dif = current_depth - previous_depth
        
        difference = torch.cat([color_dif, depth_dif], dim=1)
        difference = self.difference_net(difference)
        
        return (difference * current_color) + ((1 - difference) * previous_color)
    
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            activation()
        )
    
    def forward(self, x):
        return self.net(x)