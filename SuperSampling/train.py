import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, **config["globals"])
    valid_data_loader = config.init_obj('valid_data_loader', module_data, **config["globals"])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **config["globals"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        # TODO: switch to DistributedDataParallel
        # will need some work around
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # initialize parameters
    for item in model.modules():
        if isinstance(item, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(item.weight)

    # get function handles of loss and metrics
    # TODO: abstract loss into model config and initialize object dependant on
    # - coupled to training loop abstraction (trainer.py)
    # criterion = getattr(module_loss, config['loss']) 
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # default is to process the whole batch
    if "batch_split_size" not in config["trainer"]["args"]:
        config["trainer"]["args"]["batch_split_size"] = config["data_loader"]["args"]["batch_size"] 
    trainer = Trainer(model, metrics, optimizer,
                      config=config,
                      device=device,
                      device_ids=device_ids,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      **config["trainer"]["args"]
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
