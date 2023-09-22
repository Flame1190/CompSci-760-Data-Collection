import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, no_op
import torchvision
from PIL import Image

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, batch_split_size=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.batch_split_size = batch_split_size

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Modifications copied from https://github.com/guanrenyang/NSRR-Reimplementation : trainer/trainer.py
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, [low_res_list, depth_list, motion_vector_list, target] in enumerate(self.data_loader):
            batch_split_idx = 0
            batch_size = low_res_list[0].shape[0]

            self.optimizer.zero_grad()
            total_loss = 0
            output = torch.zeros_like(target)
            while batch_split_idx < batch_size:
                sub_batch_size = min(batch_size - batch_split_idx, self.batch_split_size)
                batch_split_idx_end = min(batch_size, batch_split_idx + self.batch_split_size)

                sub_batch_low_res_list = [low_res[batch_split_idx:batch_split_idx_end].to(self.device) for low_res in low_res_list]
                sub_batch_depth_list = [depth[batch_split_idx:batch_split_idx_end].to(self.device) for depth in depth_list]
                sub_batch_motion_vector_list = [motion_vector[batch_split_idx:batch_split_idx_end].to(self.device) for motion_vector in motion_vector_list]
                sub_batch_target = target[batch_split_idx:batch_split_idx_end].to(self.device)

                sub_batch_output = self.model(sub_batch_low_res_list, sub_batch_depth_list, sub_batch_motion_vector_list)
                loss = self.criterion(sub_batch_output, sub_batch_target) * (sub_batch_size / batch_size)
                loss.backward()
                total_loss += loss.item()
                output[batch_split_idx:batch_split_idx_end] = sub_batch_output[:]
                batch_split_idx += self.batch_split_size

            self.optimizer.step()

            # Save batch result
            # toRGBFromBGR = lambda x: x[[2,1,0],:,:]
            toRGBFromBGR = no_op
            toPILImage = torchvision.transforms.ToPILImage()
            toImage = lambda tensor: toPILImage(toRGBFromBGR(tensor))
            output_dir = './output_pic'
            image_name = f'epoch_{epoch}_batch_{batch_idx}.png'
            

            if batch_idx % 4 == 0:
                toImage(output[0]).save(f'{output_dir}/output/{image_name}', format='PNG')
                toImage(target[0]).save(f'{output_dir}/ground_truth/{image_name}', format='PNG')
                toImage(low_res_list[0][0]).save(f'{output_dir}/input/{image_name}', format='PNG')

            # backprop
            # loss = self.criterion(output, target)
            # loss.backward()
            # self.optimizer.step()
            output = output.cpu().detach()
            target = target.cpu().detach()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss))
                self.writer.add_image('input', make_grid(low_res_list[0].cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, [low_res_list, depth_list, motion_vector_list, target] in enumerate(self.valid_data_loader):
                batch_split_idx = 0
                batch_size = low_res_list[0].shape[0]

                total_loss = 0
                total_metrics = np.zeros(len(self.metric_ftns))
                output = torch.zeros_like(target)
                while batch_split_idx < batch_size:
                    sub_batch_size = min(batch_size - batch_split_idx, self.batch_split_size)
                    batch_split_idx_end = min(batch_size, batch_split_idx + self.batch_split_size)

                    sub_batch_low_res_list = [low_res[batch_split_idx:batch_split_idx_end].to(self.device) for low_res in low_res_list]
                    sub_batch_depth_list = [depth[batch_split_idx:batch_split_idx_end].to(self.device) for depth in depth_list]
                    sub_batch_motion_vector_list = [motion_vector[batch_split_idx:batch_split_idx_end].to(self.device) for motion_vector in motion_vector_list]
                    sub_batch_target = target[batch_split_idx:batch_split_idx_end].to(self.device)

                    sub_batch_output = self.model(sub_batch_low_res_list, sub_batch_depth_list, sub_batch_motion_vector_list)
                    total_loss += (self.criterion(sub_batch_output, sub_batch_target) * (sub_batch_size / batch_size)).item()
                    
                    for i, met in enumerate(self.metric_ftns):
                        total_metrics[i] += (met(sub_batch_output, sub_batch_target) * (sub_batch_size / batch_size))

                    output[batch_split_idx:batch_split_idx_end] = sub_batch_output[:]
                    batch_split_idx += self.batch_split_size

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', total_loss)
                for i, met in enumerate(self.metric_ftns):
                    self.valid_metrics.update(met.__name__, total_metrics[i])
                self.writer.add_image('input', make_grid(low_res_list[0].cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
