import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import flatten, warp

class ENSSNoJitter(nn.Module):
    def __init__(self, scale_factor: int, f: int, m: int):
        super().__init__()
        self.scale_factor = scale_factor

        self.space_to_depth = nn.PixelUnshuffle(scale_factor)
        self.depth_to_space = nn.PixelShuffle(scale_factor)
        self.warping = Warping(scale_factor, dilation_block_size=8)
        self.network = Network(scale_factor, f, m)

    def forward(self, 
                current_color,
                current_depth,
                motion,
                previous_color,
                previous_features) -> tuple[torch.Tensor, torch.Tensor]:
        pass
        previous_color, previous_features = self.warping(
            current_depth,
            motion,
            previous_color,
            previous_features
        )
        previous_color = self.space_to_depth(previous_color)
        previous_features = self.space_to_depth(previous_features)

        current_color, current_features = self.network(
            current_color,
            current_depth,
            previous_color,
            previous_features
        )
        return self.depth_to_space(current_color), self.depth_to_space(current_features)

class Network(nn.Module):
    def __init__(self, scale_factor: int, f: int, m: int) -> None:
        super().__init__()
        # color, depth, !jitter, prev_features, prev_color
        in_channels = 3 + 1 + (1 + 3) * (scale_factor ** 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f, 3, padding=1),
            nn.ReLU(),
            *flatten([
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU()
            ] * m)
        )
        self.feature_dec = nn.Conv2d(f, (scale_factor ** 2), 3, padding=1)
        self.mask_dec = nn.Sequential(
            nn.Conv2d(f, 3 * (scale_factor ** 2), 3, padding=1),
            nn.Sigmoid()
        )
        self.color_dec = nn.Sequential(
            nn.Conv2d(f, 3 * (scale_factor ** 2), 3, padding=1),
            nn.ReLU()
        )

    def forward(self, 
                current_color,
                current_depth,
                previous_color,
                previous_features) -> tuple[torch.Tensor, torch.Tensor]:

        x = torch.cat([
            current_color,
            current_depth,
            previous_features,
            previous_color
        ], dim=1)
        x = self.net(x)
        features = self.feature_dec(x)
        mask = self.mask_dec(x)
        color = self.color_dec(x)

        color = color * mask + previous_color * (1 - mask)
        return color, features

class Warping(nn.Module):
    def __init__(self, scale_factor: int, dilation_block_size: int) -> None:
        super().__init__()
        assert dilation_block_size % scale_factor == 0

        self.scale_factor = scale_factor
        self.dilation_block_size = dilation_block_size

    def depth_dilate(self, depth: torch.Tensor, motion: torch.Tensor):
        low_res_block_size = self.dilation_block_size // self.scale_factor
        B, C, H, W = motion.shape
        _, indices = F.max_pool2d_with_indices(-depth, low_res_block_size, low_res_block_size)
        indices = indices.view(B, 1, -1).repeat(1, 2, 1)
        
        motion = motion.flatten(start_dim=2)
        motion = motion.gather(dim=2, index=indices)
        motion = motion.view(B, C, H // low_res_block_size, W // low_res_block_size).repeat(1, low_res_block_size ** 2, 1, 1)
        motion = F.pixel_shuffle(motion, low_res_block_size)

        return motion

    def forward(self,
                depth,
                motion,
                prev_color,
                prev_features) -> tuple[torch.Tensor, torch.Tensor]:
        
        dilated_motion = self.depth_dilate(depth, motion)
        dilated_motion = F.interpolate(dilated_motion, scale_factor=self.scale_factor, mode='nearest')
        prev_color = warp(prev_color, dilated_motion)
        prev_features = warp(prev_features, dilated_motion)
        return prev_color, prev_features