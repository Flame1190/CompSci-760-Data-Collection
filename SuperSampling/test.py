import argparse
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from parse_config import ConfigParser
from PIL import Image
import torchvision
import time
import os

def main(config):
    toRGBFromBGR = lambda x: x[[2,1,0],:,:]
    toPILImage = torchvision.transforms.ToPILImage()
    toImage = lambda tensor: toPILImage(toRGBFromBGR(tensor))

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        img_dirname="color/",
        depth_dirname="depth/",
        motion_dirname="motion/",
        batch_size=1 ,
        shuffle=False,
        validation_split=0.0,
        num_workers=4,
        downsample=config['data_loader']['args']['downsample'],
        resize_factor=config['data_loader']['args']['resize_factor'],
        num_frames=config['data_loader']['args']['num_frames']
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
        for i, [low_res_list, depth_list, motion_vector_list, truth] in enumerate(tqdm(data_loader)):
            low_res_list = [low_res.to(device) for low_res in low_res_list]
            depth_list = [depth.to(device) for depth in depth_list]
            motion_vector_list = [motion_vector.to(device) for motion_vector in motion_vector_list]
            target = truth.to(device)

            start = time.time()
            output = model(low_res_list, depth_list, motion_vector_list)
            total_time += time.time() - start

            #
            # save sample images, or do something with output here
            # 
            output_pic_test_path = os.path.join(os.getcwd(), "output_pic_test")
            input_path = os.path.join(output_pic_test_path, "input")
            output_path = os.path.join(output_pic_test_path, "output")
            truth_path = os.path.join(output_pic_test_path, "truth")

            for path in [input_path, output_path, truth_path]:
                if not os.path.exists(path):
                    os.makedirs(path)

            toImage(low_res_list[0][0]).save(os.path.join(input_path, f"{i}.png"))
            toImage(output[0]).save(os.path.join(output_path, f"{i}.png"))
            toImage(truth[0]).save(os.path.join(truth_path, f"{i}.png"))

            # computing loss, metrics on test set
            output = output.cpu()
            target = target.cpu()
            loss = loss_fn(output, target)
            batch_size = output.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

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
