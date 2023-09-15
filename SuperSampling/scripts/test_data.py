# script to check data validity
import argparse

import sys
if __name__=='__main__':
    sys.path.insert(0,sys.path[0]+'/..')
    
import torch
from data_loader.data_loaders import NSRRDataLoader
from utils import warp, upsample_zero
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Test data loading and analytical transforms used.')

    parser.add_argument('src', type=str,
                        help='Parent folder for data to test')
    parser.add_argument('--color_dirname', default='color')
    parser.add_argument('--depth_dirname', default='depth')
    parser.add_argument('--motion_dirname', default='motion')
    parser.add_argument('--downsample', default=2, type=int)

    args = parser.parse_args()
    return args 

def test_warping(view_list: list[torch.Tensor], motion_list: list[torch.Tensor]):
    # motion is ordered current to earliest
    
    # accumulate warping
    for k, motion in enumerate(motion_list):
        if k == 0:
            continue
        # sample from the motion of this frame according
        # to the accumulated motion of the future frames
        motion_list[k] = warp(motion, motion_list[k-1])

    # warp views
    for k, view in enumerate(view_list):
        view_list[k] = warp(view, motion_list[k])
        cv2.imshow(f"view {k} warped", view_list[k][0].permute(1,2,0).numpy())

    cv2.waitKey(0)

def main():
    args = parse_args()
    
    loader = NSRRDataLoader(
        args.src,
        args.color_dirname,
        args.depth_dirname,
        args.motion_dirname,
        1,
        resize_factor=3,
        downsample=args.downsample,
        shuffle=False,
    )
    
    upsample_bilinear = torch.nn.Upsample(scale_factor=args.downsample, mode='bilinear', align_corners=True)
    for i, x in enumerate(loader):
        view_list, _, motion_list, _ = x
        view_list = [upsample_zero(view, scale_factor=args.downsample) for view in view_list]
        motion_list = [upsample_bilinear(motion) for motion in motion_list]

        test_warping(view_list, motion_list)
        break

if __name__ == "__main__":
    main()