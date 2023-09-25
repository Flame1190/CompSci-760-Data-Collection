import json
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torchvision
import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def no_op(x):
    return x

def upsample_zero(image: torch.Tensor, *, scale_factor: int = 2) -> torch.Tensor:
    batch_size, channels, height, width = image.shape

    zero_upsampled_image = torch.zeros((batch_size, channels, height * scale_factor, width * scale_factor), dtype=torch.float32, device=image.device)
    zero_upsampled_image[:,:,::scale_factor,::scale_factor] = image

    return zero_upsampled_image

def warp(image: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    B, _, H, W = image.size()
    # Construct a grid of relative pixel positions [-1,1] = [left, right] = [top, bottom]
    xx = torch.linspace(-1, 1, W, device=image.device).view(1,-1).repeat(H,1)
    yy = torch.linspace(-1, 1, H, device=image.device).view(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx, yy),1)
    # We use HDRP motion vectors which measure the forward movement of pixels
    # measured relative to the frame, i.e. -1 means the pixel moved to the left 
    # of the frame from the right. However, we invert these in the data loader.
    #
    # To get the pixel in the previous frame that we are sampling from we add 
    # motion vector from the grid of relative pixel positions
    vgrid = grid + motion

    return F.grid_sample(image, vgrid.permute(0, 2, 3, 1).to(image.dtype), mode="bilinear", align_corners=True)

def retrieve_elements_from_indices(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _, iC, _, _ = indices.shape
    assert iC == 1
    B, C, H, W = tensor.shape

    indices = indices.flatten(start_dim=1).view(B, 1, -1)
    tensor = tensor.flatten(start_dim=2)
    
    # duplicate for each channel for gather to work
    indices = indices.repeat(1, C, 1) 

    # if not duplicating indices for each channel
    # we would have to loop over each channel and gather
    tensor = tensor.gather(dim=2, index=indices).view(B, C, H, W)
    return tensor

def flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def save_image(image: torch.Tensor, path: str, format: str | None = None) -> None:
    image = image.detach().cpu()
    torchvision.utils.save_image(image, path, format=format)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def get_least_utilized_gpu():
    # Special thanks https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/2
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
        
    device = None
    list_ids = None
    if n_gpu_use == 1 and n_gpu > 1:
        free_gpu_id = get_least_utilized_gpu()
        torch.cuda.set_device(free_gpu_id)
        list_ids = [free_gpu_id]
    else: 
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
