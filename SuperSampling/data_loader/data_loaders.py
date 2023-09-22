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


class SupersamplingDataLoader(BaseDataLoader):
    """
    Generate batch of data
    `for x_batch in data_loader:`
    `x_batch` is a list of 4 tensors, meaning `view, depth, motion, view_truth`
    each size is (batch x channel x height x width)

    NOTE: validation should always be done by an independent data loader
    
    """
    def __init__(self,
                 data_dirs: str,
                 color_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 batch_size: int,
                 scale_factor: int,
                 num_frames: int,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 num_data: Union[int, None] = None,
                 output_dimensions: Union[int, None] = None,
                 ):
        self.dataset = SupersamplingDataset(data_dirs=data_dirs,
                              color_dirname=color_dirname,
                              depth_dirname=depth_dirname,
                              motion_dirname=motion_dirname,
                              scale_factor=scale_factor,
                              num_data=num_data,
                              output_dimensions=output_dimensions,
                              num_frames=num_frames,
                              )
        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         validation_split=0, # read doc string note
                         num_workers=num_workers,
                         )

class SupersamplingDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dirs: str | List[str],
                 color_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 scale_factor: int,
                 num_frames: int,
                 output_dimensions: None | Tuple[int, int] = None,
                 num_data: Union[int, None] = None,
                 ):
        super().__init__()

        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.color_dirname = color_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname

        self.scale_factor = scale_factor
        self.output_dimensions = output_dimensions

        self.transform = tf.transforms.Compose([
            tf.Lambda(lambda x: x[[2,1,0],:,:]), # BGR to RGB (for cv2)
            tf.ToTensor(),
        ])

        self.clips = {
            data_dir: sorted(os.list(os.path.join(data_dir, self.color_dirname)), reverse=False) for data_dir in self.data_dirs
        }

        if num_data is None:
            num_data = sum([len(clip) for clip in self.clips.values()])

        self.data_list = []
        # maintain a buffer for the last num_frames frames
        img_name_buffer = deque(maxlen=num_frames) 

        for i, clip in enumerate(self.clips.values()):
            for img_name in clip:
                if(i>=num_data + num_frames - 1):
                    break
                    
                img_name_buffer.appendleft(img_name)
                if len(img_name_buffer) == num_frames:
                    self.data_list.append(list(img_name_buffer))

            # handle scene change
            img_name_buffer.clear()

    
    def __getitem__(self, index):
        data = self.data_list[index]

        view_list, depth_list, motion_list, truth_list = [], [], [], []
        # elements in the lists following the order: current frame i, pre i-1, pre i-2, pre i-3, pre i-4
        for frame in data:
            frame, _ = frame.rsplit('.', 1)
            img_path = os.path.join(self.data_dir, self.color_dirname, f"{frame}.png")
            depth_path = os.path.join(self.data_dir, self.depth_dirname, f"{frame}.exr")
            motion_path = os.path.join(self.data_dir, self.motion_dirname, f"{frame}.exr")
            
            img_view_truth = cv2.imread(img_path)
            img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            img_motion = cv2.imread(motion_path, cv2.IMREAD_UNCHANGED)
            
            
            width, height, _ = img_view_truth.shape
            if self.resize_dimensions is not None:
                height, width = self.resize_dimensions # TODO: reorder, this should be width, height
                
            # TODO: check interpolation method
            img_view_truth = cv2.resize(img_view_truth, (height, width), cv2.INTER_NEAREST)

            low_res_height, low_res_width = height // self.scale_factor, width // self.scale_factor
            img_view_input = cv2.resize(img_view_truth, (low_res_height, low_res_width), cv2.INTER_NEAREST)
            img_depth = cv2.resize(img_depth,  (low_res_height, low_res_width), cv2.INTER_NEAREST)
            img_motion = cv2.resize(img_motion, (low_res_height, low_res_width), cv2.INTER_NEAREST)

            target_image = self.transform(img_view_truth)
            img_view = self.transform(img_view_input)
            img_depth = self.transform(img_depth)[0:1, :, :]
            img_motion = self.transform(img_motion)[0:2,:,:] 
            img_motion[1] = -img_motion[1] # flip y axis 
            img_motion = img_motion * -1 # point backwards

            view_list.append(img_view)
            depth_list.append(img_depth)
            motion_list.append(img_motion)
            truth_list.append(target_image)
            
        return view_list, depth_list, motion_list, truth_list

    def __len__(self) -> int:
        return len(self.data_list)


