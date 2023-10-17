import os
from base import BaseDataLoader

import torch
from torch.utils.data import Dataset

from data_loader import SupersamplingDataset, StereoSupersamplingDataset

def split_image(image: torch.Tensor, patch_size: int = 264, min_overlap: int = 12, get_indices: bool = False):
    B, C, H, W = image.shape
    assert B == 1, "Only batch size 1 is supported"

    num_patches_h = H // patch_size + 1
    num_patches_w = W // patch_size + 1 

    overlap_h = (num_patches_h * patch_size - H) // (num_patches_h - 1)
    while overlap_h < min_overlap:
        num_patches_h += 1
        overlap_h = (num_patches_h * patch_size - H) // (num_patches_h - 1)

    overlap_w = (num_patches_w * patch_size - W) // (num_patches_w - 1)
    while overlap_w < min_overlap:
        num_patches_w += 1
        overlap_w = (num_patches_w * patch_size - W) // (num_patches_w - 1)

    patches = torch.Tensor(num_patches_h * num_patches_w, C, patch_size, patch_size).to(image.device)
    patch_indices = []
    patch_starts_h = [i * (patch_size - overlap_h) for i in range(num_patches_h)]
    patch_starts_w = [i * (patch_size - overlap_w) for i in range(num_patches_w)]

    for i, p_h in enumerate(patch_starts_h):
        for j, p_w in enumerate(patch_starts_w):
            patch = image[:, :, p_h:p_h + patch_size, p_w:p_w + patch_size]
            patches[i * num_patches_w + j] = patch
            get_indices and patch_indices.append((p_h, p_w))

    return patches, patch_indices

def merge_image(patches: torch.Tensor, patch_indices: list[tuple[int,int]], image_size: tuple, patch_size: int, overlap: int):
    image = torch.zeros(image_size)
    
    for i, (p_h, p_w) in enumerate(patch_indices):
        image[0,:,p_h + overlap // 2 :p_h + patch_size, p_w + overlap // 2:p_w + patch_size] = patches[i, :, overlap // 2:, overlap // 2:]

    return image


class StereoRecurrentTestingDataLoader(BaseDataLoader):
    def __init__(self,
                 data_dirs: str,
                 left_dirname: str,
                 right_dirname: str, 
                 color_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 scale_factor: int,
                 patch_size_hr: int,
                 overlap_hr: int,
                 num_workers: int = 1,
                 output_dimensions: int | None = None,
                 
                 num_data: int | None = None
                 ):
        reverse: bool = True,
        num_frames = 2
        batch_size = 1

        left_data_dirs = [os.path.join(data_dir, left_dirname) for data_dir in data_dirs]
        right_data_dirs = [os.path.join(data_dir, right_dirname) for data_dir in data_dirs]

        self.left_dataset = SupersamplingDataset(
            data_dirs=left_data_dirs,
            color_dirname=color_dirname,
            depth_dirname=depth_dirname,
            motion_dirname=motion_dirname,
            scale_factor=scale_factor,
            num_data=num_data,
            output_dimensions=output_dimensions,
            num_frames=num_frames,
            shuffle=False,
            reverse=reverse
            )
        self.left_dataset = ChunkedDataset(
            dataset=self.left_dataset, 
            patch_size_hr=patch_size_hr, 
            overlap_hr=overlap_hr, 
            scale_factor=scale_factor
            )
        
        self.right_dataset = SupersamplingDataset(
            data_dirs=right_data_dirs,
            color_dirname=color_dirname,
            depth_dirname=depth_dirname,
            motion_dirname=motion_dirname,
            scale_factor=scale_factor,
            num_data=num_data,
            output_dimensions=output_dimensions,
            num_frames=num_frames,
            shuffle=False,
            reverse=reverse
            )
        self.right_dataset = ChunkedDataset(
            dataset=self.right_dataset, 
            patch_size_hr=patch_size_hr, 
            overlap_hr=overlap_hr, 
            scale_factor=scale_factor
        )
        self.dataset = ChunkedStereoDataset(
            left_dataset=self.left_dataset,
            right_dataset=self.right_dataset,
        )

        super().__init__(dataset=self.dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         validation_split=0, # read doc string note
                         num_workers=num_workers,
                         )



class ChunkedDataset(Dataset):
    def __init__(self, 
                 dataset: Dataset,
                 patch_size_hr: int,
                 overlap_hr: int,
                 scale_factor: int):
        assert not patch_size_hr % scale_factor, "Patch size must be divisible by scale factor"
        assert not overlap_hr % scale_factor, "Overlap must be divisible by scale factor"

        super().__init__()

        self.dataset = dataset

        self.patch_size_hr = patch_size_hr
        self.overlap_hr = overlap_hr

        self.patch_size_lr = patch_size_hr // scale_factor
        self.overlap_lr = overlap_hr // scale_factor


    def __getitem__(self, index):
        add_batch_dim = lambda x: torch.unsqueeze(x, 0)
        remove_batch_dim = lambda x: x[0]

        views, depths, motions, truths = self.dataset[index]
        assert len(views) == 2

        unchunked_view = views[1]
        unchunked_truth = truths[1]

        views = map(add_batch_dim, views)
        depths = map(add_batch_dim, depths)
        motions = map(add_batch_dim, motions)
        truths = map(add_batch_dim, truths)


        views = [split_image(view, self.patch_size_lr, self.overlap_lr)[0] for view in views]
        depths = [split_image(depth, self.patch_size_lr, self.overlap_lr)[0] for depth in depths]
        motions = [split_image(motion, self.patch_size_lr, self.overlap_lr)[0] for motion in motions]
        truths = [split_image(truth, self.patch_size_hr, self.overlap_hr, True) for truth in truths]
        indices = truths[1][1]     
        truths = [truth for truth, _ in truths]

        # views = list(map(remove_batch_dim, views))
        # depths = list(map(remove_batch_dim, depths))
        # motions = list(map(remove_batch_dim, motions))
        # truths = list(map(remove_batch_dim, truths))

        return views, depths, motions, truths, indices, unchunked_view, unchunked_truth

    def __len__(self) -> int:
        return len(self.dataset)
    
class ChunkedStereoDataset:
    def __init__(self, 
                 left_dataset: ChunkedDataset,
                 right_dataset: ChunkedDataset) -> None:
        super().__init__()
        self.left_dataset = left_dataset
        self.right_dataset = right_dataset

    def __getitem__(self, index):
        left_views, left_depths, left_motions, left_truths, indices, left_unchunked_view, left_unchunked_truth = self.left_dataset[index]
        right_views, right_depths, right_motions, right_truths, _, right_unchunked_view, right_unchunked_truth = self.right_dataset[index]

        assert len(left_views) == len(right_views)
        n = len(left_views)

        mixed_views = [torch.cat((left_views[i].unsqueeze(0), right_views[i].unsqueeze(0)), dim=0) for i in range(n)]
        mixed_depths = [torch.cat((left_depths[i].unsqueeze(0), right_depths[i].unsqueeze(0)), dim=0) for i in range(n)]
        mixed_motions = [torch.cat((left_motions[i].unsqueeze(0), right_motions[i].unsqueeze(0)), dim=0) for i in range(n)]
        mixed_truths = [torch.cat((left_truths[i].unsqueeze(0), right_truths[i].unsqueeze(0)), dim=0) for i in range(n)]

        return (
            mixed_views, 
            mixed_depths, 
            mixed_motions, 
            mixed_truths, 
            indices, # deterministic, matches for left and right
            (left_unchunked_view, right_unchunked_view), 
            (left_unchunked_truth, right_unchunked_truth)
        )

    def __len__(self) -> int:
        return min(len(self.left_dataset), len(self.right_dataset))