import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, no_op
import torchvision
from model.loss import NSRRLoss, MNSSLoss

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, 
                 batch_split_size=None, method=None, use_prev_high_res=False):
        assert method is not None
        assert f"init_{method}" in Trainer.__dict__
        assert f"train_{method}" in Trainer.__dict__

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
        self.use_prev_high_res = use_prev_high_res
        
        # initilize training method
        getattr(self, f"init_{method}")()
        self.training_method = getattr(self, f"train_{method}")

        super().__init__(model, self.criterion, metric_ftns, optimizer, config)
        

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        toImage = torchvision.transforms.ToPILImage()
        output_dir = './results'

        # Modifications copied from https://github.com/guanrenyang/NSRR-Reimplementation : trainer/trainer.py
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, [low_res_list, depth_list, motion_vector_list, target_list] in enumerate(self.data_loader):
            batch_split_idx = 0
            batch_size = low_res_list[0].shape[0]

            total_loss = 0
            total_metrics = np.zeros(len(self.metric_ftns))

            output = torch.zeros_like(target_list[0])

            self.optimizer.zero_grad()
            while batch_split_idx < batch_size:
                sub_batch_size = min(batch_size - batch_split_idx, self.batch_split_size)
                batch_split_idx_end = min(batch_size, batch_split_idx + self.batch_split_size)

                sub_batch_low_res_list = [low_res[batch_split_idx:batch_split_idx_end] for low_res in low_res_list]
                sub_batch_depth_list = [depth[batch_split_idx:batch_split_idx_end] for depth in depth_list]
                sub_batch_motion_vector_list = [motion_vector[batch_split_idx:batch_split_idx_end] for motion_vector in motion_vector_list]
                sub_batch_target_list = [target[batch_split_idx:batch_split_idx_end] for target in target_list]

                sub_batch_output, loss, metrics = self.training_method(
                    sub_batch_low_res_list, sub_batch_depth_list, sub_batch_motion_vector_list, sub_batch_target_list, 
                    weight = (sub_batch_size / batch_size))

                # loss & metrics
                total_loss += loss
                total_metrics += metrics

                # prepare for next sub-batch
                output[batch_split_idx:batch_split_idx_end] = sub_batch_output[:]
                batch_split_idx += self.batch_split_size


            self.optimizer.step()

            # Save batch result
            image_name = f'epoch_{epoch}_batch_{batch_idx}.png'
            # TODO: use verbosity
            if batch_idx % 4 == 0:
                toImage(output[0]).save(f'{output_dir}/output/{image_name}', format='PNG')
                

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)
            for i, met in enumerate(self.metric_ftns):
                self.train_metrics.update(met.__name__, total_metrics[i])

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
    
    def init_nsrr(self):
        """
        Initialize trainer for Neural Supersampling for Real-time Rendering
        """
        self.criterion = NSRRLoss(0.1).to(self.device)

    def train_nsrr(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        """
        Training logic for one batch / sub batch  Neural Supersampling for Real-time Rendering

        """

        target = target_list[0].to(self.device)
        depth_list = [depth.to(self.device) for depth in depth_list]
        motion_vector_list = [motion_vector.to(self.device) for motion_vector in motion_vector_list]
        low_res_list = [low_res.to(self.device) for low_res in low_res_list]

        output = self.model(low_res_list, depth_list, motion_vector_list)

        loss = self.criterion(output, target) * weight
        if accumulate_gradients:
            loss.backward()

        metrics = np.zeros(len(self.metric_ftns))
        for i, met in enumerate(self.metric_ftns):
            metrics[i] = met(output, target) * weight

        return output, loss.item(), metrics
    
    def init_mnss(self):
        
        self.criterion = MNSSLoss(
            self.config["globals"]["scale_factor"],
            k=5,
            w=0.1
        ).to(self.device)

    def train_mnss(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        n = len(low_res_list)

        img_ss = None
        prev_high_res = None
        avg_loss = 0
        avg_metrics = np.zeros(len(self.metric_ftns))

        for i in range(1,n):
            low_res = low_res_list[i].to(self.device)
            depth = depth_list[i].to(self.device)
            prev_depth = depth_list[i-1].to(self.device)
            motion = motion_vector_list[i].to(self.device)
            target = target_list[i].to(self.device)

            if i == 1 or self.use_prev_high_res:
                prev_high_res = target_list[i-1].to(self.device)
            else:
                prev_high_res = img_ss.detach()

            jitter = (0,0)
            img_ss, img_aa = self.model(low_res, depth, prev_high_res, prev_depth, motion, jitter)
            
            loss = self.criterion(img_aa, img_ss, target, jitter) * weight / (n - 1) # average over clip
            if accumulate_gradients:
                loss.backward()

            avg_loss += loss.item()  

            for i, met in enumerate(self.metric_ftns):
                avg_metrics[i] = met(img_ss, target) * weight / (n - 1) # average over clip

        return img_ss, avg_loss, avg_metrics

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, [low_res_list, depth_list, motion_vector_list, target_list] in enumerate(self.valid_data_loader):
                batch_split_idx = 0
                batch_size = low_res_list[0].shape[0]

                total_loss = 0
                total_metrics = np.zeros(len(self.metric_ftns))
                output = torch.zeros_like(target_list[0])

                while batch_split_idx < batch_size:
                    # split batch into sub-batches
                    sub_batch_size = min(batch_size - batch_split_idx, self.batch_split_size)
                    batch_split_idx_end = min(batch_size, batch_split_idx + self.batch_split_size)

                    sub_batch_low_res_list = [low_res[batch_split_idx:batch_split_idx_end] for low_res in low_res_list]
                    sub_batch_depth_list = [depth[batch_split_idx:batch_split_idx_end] for depth in depth_list]
                    sub_batch_motion_vector_list = [motion_vector[batch_split_idx:batch_split_idx_end] for motion_vector in motion_vector_list]
                    sub_batch_target_list = [target[batch_split_idx:batch_split_idx_end] for target in target_list]

                    # forward pass
                    sub_batch_output, loss, metrics = self.training_method(
                        sub_batch_low_res_list, sub_batch_depth_list, sub_batch_motion_vector_list, sub_batch_target_list,
                        weight = (sub_batch_size / batch_size),
                        accumulate_gradients = False
                        )

                    # loss & metrics
                    total_loss += loss
                    total_metrics += metrics

                    # prepare for next sub-batch
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
