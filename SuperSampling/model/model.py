import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import upsample_zero
import torch

from typing import List

class NSRR(BaseModel):
    def __init__(self, scale_factor: int = 2, num_frames: int = 5):
        super().__init__()
        # may not need given not converting to YCbCr
        self.feature_extraction_current = FeatureExtraction() 
        self.feature_extraction_previous = FeatureExtraction()
        
        self.upsampler = ZeroUpsampling(scale_factor=scale_factor)
        self.warper = BackwardsWarping(scale_factor=scale_factor)

        self.feature_reweighting = FeatureReweighting(num_frames=num_frames)
        self.reconstruction = Reconstruction(num_frames=num_frames)

        self.num_frames = num_frames

    def forward(self, color_maps: List[torch.Tensor], depth_maps: List[torch.Tensor], motion_vectors: List[torch.Tensor]) -> torch.Tensor:
        assert len(color_maps) == self.num_frames
        assert len(depth_maps) == self.num_frames
        assert len(motion_vectors) == self.num_frames - 1
        
        current_color_depth, *prev_color_depth_maps = [torch.concat((color_map, depth_map), dim=1) for color_map, depth_map in zip(color_maps, depth_maps)]
        assert len(prev_color_depth_maps) == self.num_frames - 1

        current_features = self.feature_extraction_current(current_color_depth)
        current_features = self.upsampler(current_features)

        current_color_depth = self.upsampler(current_color_depth)

        previous_features = [self.feature_extraction_previous(prev_color_depth) for prev_color_depth in prev_color_depth_maps]
        previous_features = [self.upsampler(prev_features) for prev_features in previous_features]

        # accumulative warping of previous features 
        for i in range(1, self.num_frames):
            for j in range(1, i + 1):
                previous_features[-j] = self.warper(previous_features[-j], motion_vectors[-i])
        
        previous_features = self.feature_reweighting(current_color_depth, previous_features)

        reconstructed_frame = self.reconstruction(current_features, previous_features)

        return reconstructed_frame

class FeatureExtraction(BaseModel):
    def __init__(self, kernel_size = 3, padding = 'same'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, color_depth_map: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        -------
        color_depth_map: 
            (batch_size, 4, height, width)
            Don't use ycbcr, apparently will break autograd?

        Outputs:
        --------
        feature_map: 
            (batch_size, 12, height, width)
            (concatenation of color_map, depth_map, and learned features)
        """
        
        
        x = self.net(color_depth_map)

        return torch.concat((color_depth_map, x), dim=1)
    
class ZeroUpsampling(BaseModel):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        -------
        frame: 
            (batch_size, channels, height, width)

        Outputs:
        --------
        zero_upsampled_frame: 
            (batch_size, channels, scale_factor * height, scale_factor * width)
            frame zero-upsampled by a factor of scale_factor
        """
        return upsample_zero(frame, scale_factor=self.scale_factor)

class BackwardsWarping(BaseModel):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.motion_vector_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, frame: torch.Tensor, motion_vector: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        -------
        frame:
            (batch_size, channels, high_res_height, high_res_width)
            RGB image of current frame  
        motion_vector:
            (batch_size, 2, low_res_height, low_res_width)
            Motion vector of current frame
            Each pixel? points to the pixel in the previous frame that it came from
        
        Outputs:
        --------
        warped_frame:
            (batch_size, channels, height, width)
            RGB image of warped frame

        Notes:
        ------
        Due to how we scale the motion vectors we require square images
        """
        # use upsampled frame to get height and width
        _, _, height, width = frame.shape 

        # un-upsampled dims threw some issues with grid_sample so we scale then upsample
        upsampled_motion_vector = self.motion_vector_upsampler(motion_vector * self.scale_factor) 
        
        # scale y? to [-1, 1] for grid_sample
        upsampled_motion_vector[:,0,:,:] = (upsampled_motion_vector[:,0,:,:] / (width - 1)) * 2.0 - 1.0
        upsampled_motion_vector[:,1,:,:] = (upsampled_motion_vector[:,1,:,:] / (height - 1)) * 2.0 - 1.0        
        
        return F.grid_sample(frame, upsampled_motion_vector.permute(0, 2, 3, 1), align_corners=True, mode='bilinear')
        

class FeatureReweighting(BaseModel):
    def __init__(self, kernel_size = 3, padding = 'same', max_amplification: float = 10.0, num_frames: int = 5):
        super().__init__()

        self.max_amplification = max_amplification

        # input channels are the rgb-d for all frames 
        # output channels are the weights for i-1...=i-4 frames
        self.net = nn.Sequential(
            nn.Conv2d(4 * num_frames, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, num_frames - 1, kernel_size=kernel_size, padding=padding),
            nn.Tanh()
        )


    def forward(self, rgb_d_current: torch.Tensor, previous_frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Inputs:
        -------
        rgb_d_current: 
            (batch_size, 4, height, width)
            RGB + Depth map of current frame

        previous_frames:
            list of (batch_size, 12, height, width) tensors
            RGB + Depth map of previous frames + learned features

        Outputs:
        --------
        weighted_prev_frames: 
            list of (batch_size, 12, height, width) tensors
            RGB + Depth map of previous frames + learned features
            weight-multiplied by the learned weights
        """
        # concatenate all previous frames
        # extract the first 4 channels
        rgb_d_previous_frames = [frame[:, :4, :, :] for frame in previous_frames]
        collated_prev_frames = torch.concat(rgb_d_previous_frames, dim=1)
        
        x = torch.concat((rgb_d_current, collated_prev_frames), dim=1)
        x = self.net(x)

        scaled_weights = self.max_amplification * ((x + 1) / 2)

        weighted_prev_frames = [scaled_weights[:, i, : :] * previous_frames[i] for i in range(len(previous_frames))]
        return weighted_prev_frames

class Reconstruction(BaseModel):
    """
    Modified U-net with skip connections
    """
    def __init__(self, kernel_size = 3, padding = 'same', num_frames: int = 5):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(12 * num_frames, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

        self.center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, current_frame: torch.Tensor, prev_frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
        -------
        current_frame: 
            (batch_size, 12, height, width)
            RGB + Depth map of current frame + learned features

        prev_frames:
            list of (batch_size, 12, height, width) tensors
            RGB + Depth map of previous frames + learned features

        Outputs:
        --------
        reconstructed_frame: 
            (batch_size, 3, height, width)
            RGB image of reconstructed frame
        """

        joined_frames = torch.concat([current_frame] + prev_frames, dim=1)

        out_enc1 = self.enc1(joined_frames)

        out_enc2 = self.enc2(self.down(out_enc1))

        out_center = self.center(self.down(out_enc2))

        in_dec1 = torch.concat((self.up(out_center), out_enc2), dim=1)
        out_dec1 = self.dec1(in_dec1)

        in_dec2 = torch.concat((self.up(out_dec1), out_enc1), dim=1)
        out_dec2 = self.dec2(in_dec2)

        return out_dec2