import argparse
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import ensure_dir

from parse_config import ConfigParser
from PIL import Image
import torchvision
import time
import os

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

    patches = torch.Tensor(num_patches_h * num_patches_w, C, patch_size, patch_size)
    patch_indices = []
    patch_starts_h = [i * (patch_size - overlap_h) for i in range(num_patches_h)]
    patch_starts_w = [i * (patch_size - overlap_w) for i in range(num_patches_w)]

    for i, p_h in enumerate(patch_starts_h):
        for j, p_w in enumerate(patch_starts_w):
            patch = image[:, :, p_h:p_h + patch_size, p_w:p_w + patch_size]
            patches[i * num_patches_w + j] = patch
            get_indices and patch_indices.append((p_h, p_w))

    return patches, patch_indices

def merge_image(patches: torch.Tensor, patch_indices: list[tuple[int,int]], image_size: tuple, patch_size: int):
    image = torch.zeros(image_size)

    for i, (p_h, p_w) in enumerate(patch_indices):
        image[0,:,p_h:p_h + patch_size, p_w:p_w + patch_size] = patches[i, :, :, :]

    return image

def test_inss(
        data_loader,
        model,
        loss_fn,
        metric_fns,
        device,
        config
        ):
    assert config['data_loader']['args']['num_frames'] == 2, "Only 2 frames for efficiency"
    assert torch.is_grad_enabled() == False, "No gradients should be computed"
    assert model.__class__.__name__ == "INSS", "Only INSS is supported"

    toImage = torchvision.transforms.ToPILImage()

    input_path = os.path.join("test_results", "input")
    output_path = os.path.join("test_results", "output")
    truth_path = os.path.join("test_results", "truth")

    ensure_dir(input_path)
    ensure_dir(output_path)
    ensure_dir(truth_path)



    scale_factor = config['data_loader']['args']['scale_factor']

    patch_size_hr = 264
    overlap_hr = 12
    patch_size_lr = patch_size_hr // scale_factor
    overlap_lr = overlap_hr // scale_factor

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_time = 0.0

    indices = None
    prev_color = None
    for i, [low_res_list, depth_list, motion_vector_list, truth] in enumerate(tqdm(data_loader)):
        current_low_res, _ = split_image(low_res_list[1], patch_size_lr, overlap_lr)
        current_depth, _ = split_image(depth_list[1], patch_size_lr, overlap_lr)
        prev_depth, _ = split_image(depth_list[0], patch_size_lr, overlap_lr)
        current_motion, _ = split_image(motion_vector_list[1], patch_size_lr, overlap_lr)

        # dimensions are static so indices are the same for all frames
        if i == 0:
            prev_color, indices = split_image(low_res_list[0], patch_size_hr, overlap_hr, get_indices=True)
        target, _ = split_image(truth, patch_size_hr, overlap_hr)

        current_low_res = current_low_res.to(device)
        current_depth = current_depth.to(device)
        prev_depth = prev_depth.to(device)
        current_motion = current_motion.to(device)
        prev_color = prev_color.to(device)
        target = target.to(device)

        start = time.time()
        output = model(current_low_res, current_depth, current_motion, prev_color, prev_depth)
        total_time += time.time() - start

        # computing loss, metrics on test set
        loss = loss_fn(output, target)
        total_loss += loss.item() 
        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(output, target)

        # merge patches
        merged_output = merge_image(output, indices, truth.shape, patch_size_hr)

        # save result images
        toImage(low_res_list[1][0]).save(os.path.join(input_path, f"{i}.png"))
        toImage(merged_output[0]).save(os.path.join(output_path, f"{i}.png"))
        toImage(truth[0]).save(os.path.join(truth_path, f"{i}.png"))

    return (total_loss, total_metrics, total_time)

def main(config):
    toImage = torchvision.transforms.ToPILImage()

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        img_dirname="color/",
        depth_dirname="depth/",
        motion_dirname="motion/",
        batch_size=1 ,
        shuffle=False,
        num_workers=4,
        num_frames=config['data_loader']['args']['num_frames'],
        scale_factor=config['data_loader']['args']['downsample'],
        output_dimensions=config['data_loader']['args']['output_dimensions'],
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_time = 0.0

    with torch.no_grad():
        total_loss, total_metrics, total_time = test_inss(data_loader, model, loss_fn, metric_fns, device, config)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    print("total runtime: ", total_time)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
