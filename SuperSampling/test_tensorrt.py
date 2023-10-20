import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

import model.model as module_arch
import model.metric as metrics
from utils import ensure_dir

from parse_config import ConfigParser
from PIL import Image
import torchvision
import time
import os

import torch_tensorrt


from test_utils import merge_image, StereoRecurrentTestingDataLoader


def test_mnss(config):
    assert config['data_loader']['args']['num_frames'] == 2, "Only 2 frames for efficiency"
    assert config["arch"]["type"] == "MNSS", "MNSS only punk"

    logger = config.get_logger('test')

    scale_factor = config["globals"]["scale_factor"]
    # patch_size_hr = config["globals"]["patch_size_hr"]
    patch_size_hr = 264
    # overlap_hr = config["globals"]["overlap_hr"]
    overlap_hr = 12
    patch_size_lr = patch_size_hr // scale_factor
    overlap_lr = overlap_hr // scale_factor

    output_dimensions = config["data_loader"]["args"]["output_dimensions"]
    assert output_dimensions is not None, "Sorry mate, you need to specify output dimensions since I won't fix my code"

    # ensure results directories exist
    input_path = os.path.join("test_results", "input")
    output_path = os.path.join("test_results", "output")
    truth_path = os.path.join("test_results", "truth")

    ensure_dir(input_path)
    ensure_dir(output_path)
    ensure_dir(truth_path)

    # setup data_loader instances
    data_loader = StereoRecurrentTestingDataLoader(
        config['data_loader']['args']['data_dirs'],
        left_dirname=config['data_loader']['args']['left_dirname'],
        right_dirname=config['data_loader']['args']['right_dirname'],
        color_dirname=config['data_loader']['args']['color_dirname'],
        depth_dirname=config['data_loader']['args']['depth_dirname'],
        motion_dirname=config['data_loader']['args']['motion_dirname'],

        num_workers=4,

        scale_factor=scale_factor,
        patch_size_hr=patch_size_hr,
        overlap_hr=overlap_hr,

        output_dimensions=output_dimensions,
        num_data=config["data_loader"]["args"]["num_data"]
    )

    # build model architecture
    model = config.init_obj('arch', module_arch, scale_factor=scale_factor)
    logger.info(model)

    # init model weights from checkpoint
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

    # convert to tensorrt
    mnss_input_signature = (
        torch_tensorrt.Input(shape=[patch_size_lr, patch_size_lr], dtype=torch.half), # cur_color (LR)
        torch_tensorrt.Input(shape=[patch_size_lr, patch_size_lr], dtype=torch.half), # cur_depth (LR)
        torch_tensorrt.Input(shape=[patch_size_hr, patch_size_hr], dtype=torch.half), # prev_color (HR)
        torch_tensorrt.Input(shape=[patch_size_lr, patch_size_lr], dtype=torch.half), # prev_depth (LR)
        torch_tensorrt.Input(shape=[patch_size_lr, patch_size_lr], dtype=torch.half), # motion (LR)
    )
    enabled_precisions = {torch.float, torch.half}
    start_compile = time.time()
    trt_ts_module = torch_tensorrt.compile(
        model, input_signature=mnss_input_signature, enabled_precisions=enabled_precisions
    )
    print("Compile time: ", time.time() - start_compile)

    # test
    total_time = 0.0
    data_time = 0.0

    indices = None
    left_prev_high_res, right_prev_high_res = None, None

    perm_for_stereo = lambda res_list: [tensor.squeeze(0) for tensor in res_list]
    with torch.no_grad(): 
        assert torch.is_grad_enabled() == False, "No gradients should be computed"
        t_start_load = time.time()

        for frame_idx, [
            low_res_list, 
            depth_list, 
            motion_vector_list, 
            target_list, 
            _,
            _,
            _
        ] in enumerate(tqdm(data_loader)):
            low_res_list = perm_for_stereo(low_res_list)
            depth_list = perm_for_stereo(depth_list)
            motion_vector_list = perm_for_stereo(motion_vector_list)
            target_list = perm_for_stereo(target_list)

            data_time += time.time() - t_start_load

            left_low_res, right_low_res = low_res_list[1].to(device).half()
            left_depth, right_depth = depth_list[1].to(device).half()
            left_prev_depth, right_prev_depth = depth_list[0].to(device).half()
            left_motion, right_motion = motion_vector_list[1].to(device).half()

            # dimensions are static so indices are the same for all frames
            if frame_idx == 0:
                left_prev_high_res, right_prev_high_res = target_list[0].to(device).half()
                
            start = time.time()
            left_output, _ = trt_ts_module(left_low_res, left_depth, left_prev_high_res, left_prev_depth, left_motion, (0,0))
            right_output, _ = trt_ts_module(right_low_res, right_depth, right_prev_high_res, right_prev_depth, right_motion, (0,0))
            # first one takes much longer due to lazy init i assume
            if frame_idx >= 1:
                total_time += (time.time() - start)

            t_start_load = time.time()

    n_samples = len(data_loader.sampler)
    print(f"runtime: {1000 * total_time / (n_samples - 1)} ms per frame")




def main(config):
    if config["run"] == "mnss":
        test_mnss(config)
    else:
        raise NotImplementedError("Only sinss is supported")


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
