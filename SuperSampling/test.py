import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
import model.metric as metrics
from utils import ensure_dir

from parse_config import ConfigParser
from PIL import Image
import torchvision
import time
import os


from test_utils import merge_image, split_image, StereoRecurrentTestingDataLoader

def test_sinss(config):
    assert config['data_loader']['args']['num_frames'] == 2, "Only 2 frames for efficiency"
    assert config["arch"]["type"] == "SINSS", "SINSS only punk"

    toImage = torchvision.transforms.ToPILImage()
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

    # test
    compute_metrics = True
    total_metrics = torch.zeros(2 + 2 + 1)
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
            indices,
            unchunked_low_res,
            unchunked_high_res
        ] in enumerate(tqdm(data_loader)):
            low_res_list = perm_for_stereo(low_res_list)
            depth_list = perm_for_stereo(depth_list)
            motion_vector_list = perm_for_stereo(motion_vector_list)
            target_list = perm_for_stereo(target_list)

            data_time += time.time() - t_start_load

            left_low_res, right_low_res = low_res_list[1].to(device)
            left_depth, right_depth = depth_list[1].to(device)
            left_prev_depth, right_prev_depth = depth_list[0].to(device)
            left_motion, right_motion = motion_vector_list[1].to(device)
            left_target, right_target = target_list[1].to(device)

            # dimensions are static so indices are the same for all frames
            if frame_idx == 0:
                left_prev_high_res, right_prev_high_res = target_list[0].to(device)
                
            start = time.time()
            left_output, right_output = model(
                left_low_res, left_depth, left_motion, left_prev_high_res, left_prev_depth,
                right_low_res, right_depth, right_motion, right_prev_high_res, right_prev_depth
            )
            # first one takes much longer due to lazy init i assume
            if frame_idx >= 1:
                total_time += (time.time() - start)

            # metrics
            if compute_metrics:
                # computing loss, metrics on test set
                left_output = left_output.cpu().detach()
                right_output = right_output.cpu().detach()
                
                left_target = left_target.cpu().detach()
                right_target = right_target.cpu().detach()

                
                total_metrics[0] += metrics.psnr(left_output, left_target) 
                total_metrics[1] += metrics.psnr(right_output, right_target)

                total_metrics[2] += metrics.ssim(left_output, left_target) 
                total_metrics[3] += metrics.ssim(right_output, right_target)

                total_metrics[4] += metrics.wsdr(
                    left_output, 
                    right_output, 
                    left_target, 
                    right_target, 
                    F.upsample(left_depth, scale_factor=scale_factor).cpu(), 
                    warping_coeff=0.1845
                    )

            # merge patches
            merged_output = merge_image(left_output, indices, (1, 3, *output_dimensions), patch_size_hr, overlap_hr)
            left_input = unchunked_low_res[0]
            left_truth = unchunked_high_res[0]

            # save result images
            toImage(left_input[0]).save(os.path.join(input_path, f"{frame_idx}.png"))
            toImage(merged_output[0]).save(os.path.join(output_path, f"{frame_idx}.png"))
            toImage(left_truth[0]).save(os.path.join(truth_path, f"{frame_idx}.png"))
            t_start_load = time.time()

    n_samples = len(data_loader.sampler)
    log = {
        "left_psnr": total_metrics[0].item() / n_samples,
        "right_psnr": total_metrics[1].item() / n_samples,
        "left_ssim": total_metrics[2].item() / n_samples,
        "right_ssim": total_metrics[3].item() / n_samples,
        "wsdr": total_metrics[4].item() / n_samples,
    }
    logger.info(log)
    print(f"runtime: {1000 * total_time / (n_samples - 1)} ms per frame")

def test_mnss(config):
    assert config['data_loader']['args']['num_frames'] == 2, "Only 2 frames for efficiency"
    assert config["arch"]["type"] == "MNSS", "MNSS only punk"

    toImage = torchvision.transforms.ToPILImage()
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

    # test
    compute_metrics = True
    total_metrics = torch.zeros(2 + 2 + 1)
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
            indices,
            unchunked_low_res,
            _
        ] in enumerate(tqdm(data_loader)):
            low_res_list = perm_for_stereo(low_res_list)
            depth_list = perm_for_stereo(depth_list)
            motion_vector_list = perm_for_stereo(motion_vector_list)
            target_list = perm_for_stereo(target_list)

            data_time += time.time() - t_start_load

            left_low_res, right_low_res = low_res_list[1].to(device)
            left_depth, right_depth = depth_list[1].to(device)
            left_prev_depth, right_prev_depth = depth_list[0].to(device)
            left_motion, right_motion = motion_vector_list[1].to(device)
            left_target, right_target = target_list[1].to(device)

            # dimensions are static so indices are the same for all frames
            if frame_idx == 0:
                left_prev_high_res, right_prev_high_res = target_list[0].to(device)
                
            start = time.time()
            left_output, _ = model(left_low_res, left_depth, left_prev_high_res, left_prev_depth, left_motion, (0,0))
            right_output, _ = model(right_low_res, right_depth, right_prev_high_res, right_prev_depth, right_motion, (0,0))
            # first one takes much longer due to lazy init i assume
            if frame_idx >= 1:
                total_time += (time.time() - start)

            # metrics
            if compute_metrics:
                # computing loss, metrics on test set
                left_output = left_output.cpu().detach()
                right_output = right_output.cpu().detach()
                
                left_target = left_target.cpu().detach()
                right_target = right_target.cpu().detach()

                
                total_metrics[0] += metrics.psnr(left_output, left_target) 
                total_metrics[1] += metrics.psnr(right_output, right_target)

                total_metrics[2] += metrics.ssim(left_output, left_target) 
                total_metrics[3] += metrics.ssim(right_output, right_target)

                total_metrics[4] += metrics.wsdr(
                    left_output, 
                    right_output, 
                    left_target, 
                    right_target, 
                    F.upsample(left_depth, scale_factor=scale_factor).cpu(), 
                    warping_coeff=0.1845
                    )

            # merge patches
            merged_output = merge_image(left_output, indices, (1, 3, *output_dimensions), patch_size_hr, overlap_hr)
            left_input = unchunked_low_res[0]

            # save result images
            toImage(left_input[0]).save(os.path.join(input_path, f"{frame_idx}.png"))
            toImage(merged_output[0]).save(os.path.join(output_path, f"{frame_idx}.png"))

            t_start_load = time.time()

    n_samples = len(data_loader.sampler)
    log = {
        "left_psnr": total_metrics[0].item() / n_samples,
        "right_psnr": total_metrics[1].item() / n_samples,
        "left_ssim": total_metrics[2].item() / n_samples,
        "right_ssim": total_metrics[3].item() / n_samples,
        "wsdr": total_metrics[4].item() / n_samples,
    }
    logger.info(log)
    print(f"runtime: {1000 * total_time / (n_samples - 1)} ms per frame")

def test_nsrr(config):
    
    assert config["arch"]["type"] == "NSRR", "NSRR only punk"

    toImage = torchvision.transforms.ToPILImage()
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
        num_data=config["data_loader"]["args"]["num_data"],
        num_frames=5
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

    # test
    compute_metrics = True
    total_metrics = torch.zeros(2 + 2 + 1)
    total_time = 0.0
    data_time = 0.0

    indices = None

    perm_for_stereo = lambda res_list: [tensor.squeeze(0) for tensor in res_list]
    with torch.no_grad(): 
        assert torch.is_grad_enabled() == False, "No gradients should be computed"
        t_start_load = time.time()

        for frame_idx, [
            low_res_list, 
            depth_list, 
            motion_vector_list, 
            target_list, 
            indices,
            unchunked_low_res,
            _
        ] in enumerate(tqdm(data_loader)):
            low_res_list = perm_for_stereo(low_res_list[::-1])
            depth_list = perm_for_stereo(depth_list[::-1])
            motion_vector_list = perm_for_stereo(motion_vector_list[::-1])
            target_list = perm_for_stereo(target_list[::-1][:1])

            data_time += time.time() - t_start_load

            
            left_depth_list = [depth[0].to(device) for depth in depth_list]
            right_depth_list = [depth[1].to(device) for depth in depth_list]
            left_motion_list = [motion_vector[0].to(device) for motion_vector in motion_vector_list]
            right_motion_list = [motion_vector[1].to(device) for motion_vector in motion_vector_list]
            left_low_res_list = [low_res[0].to(device) for low_res in low_res_list]
            right_low_res_list = [low_res[1].to(device) for low_res in low_res_list]

            start = time.time()
            left_output = model(left_low_res_list, left_depth_list, left_motion_list)
            right_output = model(right_low_res_list, right_depth_list, right_motion_list)
            # first one takes much longer due to lazy init i assume
            if frame_idx >= 1:
                total_time += (time.time() - start)

            # metrics
            left_target, right_target = target_list[0]
            if compute_metrics:
                # computing loss, metrics on test set
                left_output = left_output.cpu().detach()
                right_output = right_output.cpu().detach()
                
                left_target = left_target.cpu().detach()
                right_target = right_target.cpu().detach()

                
                total_metrics[0] += metrics.psnr(left_output, left_target) 
                total_metrics[1] += metrics.psnr(right_output, right_target)

                total_metrics[2] += metrics.ssim(left_output, left_target) 
                total_metrics[3] += metrics.ssim(right_output, right_target)

                total_metrics[4] += metrics.wsdr(
                    left_output, 
                    right_output, 
                    left_target, 
                    right_target, 
                    F.upsample(left_depth_list[0], scale_factor=scale_factor).cpu(), 
                    warping_coeff=0.1845
                    )

            # merge patches
            merged_output = merge_image(left_output, indices, (1, 3, *output_dimensions), patch_size_hr, overlap_hr)
            left_input = unchunked_low_res[0]

            # save result images
            toImage(left_input[0]).save(os.path.join(input_path, f"{frame_idx}.png"))
            toImage(merged_output[0]).save(os.path.join(output_path, f"{frame_idx}.png"))

            t_start_load = time.time()

    n_samples = len(data_loader.sampler)
    log = {
        "left_psnr": total_metrics[0].item() / n_samples,
        "right_psnr": total_metrics[1].item() / n_samples,
        "left_ssim": total_metrics[2].item() / n_samples,
        "right_ssim": total_metrics[3].item() / n_samples,
        "wsdr": total_metrics[4].item() / n_samples,
    }
    logger.info(log)
    print(f"runtime: {1000 * total_time / (n_samples - 1)} ms per frame")



def test_inss(
        data_loader,
        model,
        loss_fn,
        metric_fns,
        device,
        config,
        compute_metrics = False
        ):
    assert config['data_loader']['args']['num_frames'] == 2, "Only 2 frames for efficiency"
    assert torch.is_grad_enabled() == False, "No gradients should be computed"
    # assert model.__class__.__name__ == "INSS", "Only INSS is supported"

    toImage = torchvision.transforms.ToPILImage()

    input_path = os.path.join("test_results", "input")
    output_path = os.path.join("test_results", "output")
    truth_path = os.path.join("test_results", "truth")

    ensure_dir(input_path)
    ensure_dir(output_path)
    ensure_dir(truth_path)

    scale_factor = config['globals']['scale_factor']

    patch_size_hr = 264
    overlap_hr = 12
    patch_size_lr = patch_size_hr // scale_factor
    overlap_lr = overlap_hr // scale_factor

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_time = 0.0
    data_time = 0.0

    indices = None
    prev_color = None
    with torch.no_grad():
        t_start_load = time()
        for frame_idx, [low_res_list, depth_list, motion_vector_list, truth] in enumerate(tqdm(data_loader)):
            data_time += time() - t_start_load
            current_low_res, _ = split_image(low_res_list[1].to(device), patch_size_lr, overlap_lr)
            current_depth, _ = split_image(depth_list[1].to(device), patch_size_lr, overlap_lr)
            prev_depth, _ = split_image(depth_list[0].to(device), patch_size_lr, overlap_lr)
            current_motion, _ = split_image(motion_vector_list[1].to(device), patch_size_lr, overlap_lr)

            # dimensions are static so indices are the same for all frames
            if frame_idx == 0:
                prev_color, indices = split_image(truth[0].to(device), patch_size_hr, overlap_hr, get_indices=True)
            target, _ = split_image(truth[1].to(device), patch_size_hr, overlap_hr)


            # print("target", target.device)

            start = time.time()
            output = model(current_low_res, current_depth, current_motion, prev_color, prev_depth)
            total_time += time.time() - start

            if compute_metrics:
                # computing loss, metrics on test set
                output = output.cpu().detach()
                target = target.cpu().detach()

                loss = loss_fn(output, target)
                total_loss += loss#.item() 
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target)

            # merge patches
            merged_output = merge_image(output, indices, truth[0].shape, patch_size_hr, overlap_hr)

            # save result images
            toImage(low_res_list[1][0]).save(os.path.join(input_path, f"{frame_idx}.png"))
            toImage(merged_output[0]).save(os.path.join(output_path, f"{frame_idx}.png"))
            toImage(truth[1][0]).save(os.path.join(truth_path, f"{frame_idx}.png"))

            t_start_load = time()

    return (total_loss, total_metrics, total_time)



def main(config):
    if config["run"] == "sinss":
        test_sinss(config)
    elif config["run"] == "mnss":
        test_mnss(config)
    elif config["run"] == "nsrr":
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
