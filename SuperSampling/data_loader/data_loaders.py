"""
Adapted from guanrenyangs code
https://github.com/guanrenyang/NSRR-Reimplementation
"""
import os

from torchvision.transforms.transforms import Resize
from base import BaseDataLoader

import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf

from PIL import Image

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
                 num_frames: int = 5,
                 ):
        dataset = NSRRDataset(data_dir,
                              img_dirname=img_dirname,
                              depth_dirname=depth_dirname,
                              motion_dirname=motion_dirname,
                              downsample=downsample,
                              num_data=num_data,
                              resize_factor = resize_factor,
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
                 transform: nn.Module = None,
                 num_data:Union[int, None] = None,
                 resize_factor:Union[int, None] = None,
                 num_frames: int = 5
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.img_dirname = img_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname
        self.resize_factor = resize_factor
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
            img_path = os.path.join(self.data_dir, self.img_dirname, frame)
            depth_path = os.path.join(self.data_dir, self.depth_dirname, frame)
            motion_path = os.path.join(self.data_dir, self.motion_dirname, frame)
            
            img_view_truth = Image.open(img_path)
            img_motion = Image.open(motion_path)
            img_depth = Image.open(depth_path).convert(mode="L") # TODO: check this

            # TODO: fix - this is actually width, height lmao
            height, width = img_view_truth.size
            height, width = height//self.resize_factor, width//self.resize_factor

            img_view_truth = img_view_truth.resize(
                (height, width), Image.ANTIALIAS)
            img_motion = img_motion.resize(
                (height, width), Image.ANTIALIAS)
            img_depth = img_depth.resize(
                (height, width), Image.ANTIALIAS)

            # not sure why the dim swap is needed
            transform_downscale = tf.Resize((width//self.downsample, height//self.downsample))
            comp_transform = tf.Compose([transform_downscale, self.transform])

            target_image = self.transform(img_view_truth)
            img_view = comp_transform(img_view_truth)

            # depth data is in a single-channel image.
            img_depth = comp_transform(img_depth)
            
            # guanrenyang used full-res motion vecs (flow in their case?)
            # Paper states low res sub-pixel motion vecs are used then upsampled bilinearly
            img_motion = comp_transform(img_motion) 
            # ugh
            img_motion = img_motion[:2, :, :] 
            # TODO: realign motion vectors for down-sampled coords

            view_list.append(img_view)
            depth_list.append(img_depth)
            motion_list.append(img_motion)
            truth_list.append(target_image)
            
        # Ignore the last motion vectors
        # Only return the most recent frame's truth
        return view_list, depth_list, motion_list[:-1], truth_list[0]

    def __len__(self) -> int:
        return len(self.data_list)