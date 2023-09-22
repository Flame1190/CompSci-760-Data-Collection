import os

from base import BaseDataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from collections import deque
from typing import Union, Tuple, List


class NSRRDataLoader(BaseDataLoader):
    """
    Generate batch of data
    `for x_batch in data_loader:`
    `x_batch` is a list of 4 tensors, meaning `view, depth, motion, view_truth`
    each size is (batch x channel x height x width)
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 batch_size: int,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 downsample: Union[Tuple[int, int], List[int], int] = (2, 2),
                 num_data: Union[int,None] = None,
                 resize_factor : Union[int, None] = None,
                 resize_dimensions : None | Tuple[int, int] = None,
                 num_frames: int = 5,
                 ):
        assert (resize_factor is not None) != (resize_dimensions is not None)
        dataset = NSRRDataset(data_dir,
                              img_dirname=img_dirname,
                              depth_dirname=depth_dirname,
                              motion_dirname=motion_dirname,
                              downsample=downsample,
                              num_data=num_data,
                              resize_factor = resize_factor,
                              resize_dimensions = resize_dimensions,
                              num_frames = num_frames,
                              )
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         validation_split=validation_split,
                         num_workers=num_workers,
                         )


class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 downsample: int = 2,
                 num_data:Union[int, None] = 5,
                 resize_factor:Union[int, None] = None,
                 resize_dimensions: None | Tuple[int, int] = None,
                 num_frames: int = 5,
                 transform: nn.Module = None,
                 ):
        assert (resize_factor is not None) != (resize_dimensions is not None)
        super().__init__()

        self.data_dir = data_dir
        self.img_dirname = img_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname

        self.resize_factor = resize_factor
        self.resize_dimensions = resize_dimensions

        self.downsample = downsample

        if transform is None:
            self.transform = tf.ToTensor()

        self.img_list = os.listdir(os.path.join(self.data_dir, self.img_dirname))
        self.img_list = sorted(self.img_list, key=lambda keys:[ord(i) for i in keys], reverse=False)
        
        if num_data is None:
            num_data = len(self.img_list)

        self.data_list = []
        # maintain a buffer for the last num_frames frames
        img_name_buffer = deque(maxlen=num_frames) 

        for i, img_name in enumerate(self.img_list):

            if(i>=num_data + num_frames - 1):
                break
                
            # handle scene change
            # TODO: more complex scene names — currently just [a-zA-Z]\d+
            if len(img_name_buffer) and img_name[0] != img_name_buffer[0][0]:
                img_name_buffer.clear()

            img_name_buffer.appendleft(img_name)

            if len(img_name_buffer) == num_frames:
                self.data_list.append(list(img_name_buffer))
                
    def __getitem__(self, index):
        data = self.data_list[index]

        view_list, depth_list, motion_list, truth_list = [], [], [], []
        # elements in the lists following the order: current frame i, pre i-1, pre i-2, pre i-3, pre i-4
        for frame in data:
            frame, _ = frame.rsplit('.', 1)
            img_path = os.path.join(self.data_dir, self.img_dirname, f"{frame}.png")
            depth_path = os.path.join(self.data_dir, self.depth_dirname, f"{frame}.exr")
            motion_path = os.path.join(self.data_dir, self.motion_dirname, f"{frame}.exr")
            
            img_view_truth = cv2.imread(img_path)
            img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            img_motion = cv2.imread(motion_path, cv2.IMREAD_UNCHANGED)
            
            
            width, height, _ = img_view_truth.shape
            if self.resize_factor:
                height, width = height//self.resize_factor, width//self.resize_factor
            else:
                height, width = self.resize_dimensions # TODO: reorder, this should be width, height
                

            img_view_truth = cv2.resize(img_view_truth, (height, width), cv2.INTER_NEAREST)

            img_view_input = cv2.resize(img_view_truth, (height//self.downsample, width//self.downsample), cv2.INTER_NEAREST)
            img_depth = cv2.resize(img_depth,  (height//self.downsample, width//self.downsample), cv2.INTER_NEAREST)
            img_motion = cv2.resize(img_motion, (height//self.downsample, width//self.downsample), cv2.INTER_NEAREST)

            target_image = self.transform(img_view_truth)[[2,1,0],:,:]
            img_view = self.transform(img_view_input)[[2,1,0],:,:]

            # depth data is in the 3rd channel (channels are BGR)
            img_depth = self.transform(img_depth)[2:3, :, :]
            
            # motion data is in the 2nd and 3rd channels (channels are BGR)
            img_motion = self.transform(img_motion)[1:3,:,:] 
            # swap channels to match the order of the motion vectors
            img_motion = img_motion[[1,0],:,:]
            img_motion[1] = -img_motion[1] # flip y axis
            img_motion = img_motion * -1 # point backwards

            view_list.append(img_view)
            depth_list.append(img_depth)
            motion_list.append(img_motion)
            truth_list.append(target_image)
            
        # Ignore the last motion vector as it is not used
        # Only return the most recent frame's truth
        first_motion = torch.zeros_like(motion_list[0]) 
        return view_list, depth_list, [first_motion] + motion_list[:-1], truth_list[0]

    def __len__(self) -> int:
        return len(self.data_list)
