
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import warp, retrieve_elements_from_indices, flatten

class INSS(BaseModel):
    def __init__(self, scale_factor: int, f: int = 16, m: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        self.shuffled_channel_count = 3 * (scale_factor ** 2)
        self.space_to_depth = nn.PixelUnshuffle(scale_factor)
        self.depth_to_space = nn.PixelShuffle(scale_factor)

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)


        self.warping = Warping(scale_factor=scale_factor)
        self.inpainting = Inpainting()
        self.net = Network(self.shuffled_channel_count, f, m)
        self.artifact_reduction = DoubleConv(3, 8, 3, activation=nn.ReLU)
        

    def forward(self, current_color, current_depth, current_motion, prev_color, prev_depth):
        

        # Warp previous depth and color
        prev_color, prev_depth = self.warping(prev_color, prev_depth, current_depth, current_motion)

        # Upsample for inpainting
        up_current_color = self.upsample(current_color)
        up_current_depth = self.upsample(current_depth)
        prev_depth = self.upsample(prev_depth)

        # Inpaint
        prev_color = self.inpainting(
            up_current_color,
            up_current_depth,
            prev_color,
            prev_depth
        )
        prev_color = self.space_to_depth(prev_color)

        residuals, mask = self.net(current_color, current_depth, prev_color)

        mixed_color = mask * residuals + (1 - mask) * prev_color
        mixed_color = self.depth_to_space(mixed_color)
        return self.artifact_reduction(mixed_color)
    

        
    
class Network(BaseModel):
    def __init__(self, shuffled_channel_count: int, f: int, m: int):
        super().__init__()
        self.shuffled_channel_count = shuffled_channel_count
        in_channels = sum((
            3, # color current
            1, # depth current
            self.shuffled_channel_count # color prev shuffled
        ))
        # out_channels = sum((
        #     shuffled_channel_count, # residuals
        #     shuffled_channel_count  # blending mask
        # ))

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