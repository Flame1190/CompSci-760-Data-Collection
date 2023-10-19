import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, ensure_dir
import torchvision
from model.loss import NSRRLoss, MNSSLoss, INSSLoss, SINSSLoss
from time import time

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, 
                 batch_split_size=None, method=None, use_prev_high_res=False, device_ids=None,
                 image_verbosity=None):
        assert method is not None
        assert f"init_{method}" in Trainer.__dict__
        assert f"train_{method}" in Trainer.__dict__

        self.config = config
        self.scale_factor = config["globals"]["scale_factor"]
        self.device = device
        self.device_ids = device_ids
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
        self.image_verbosity = image_verbosity
        
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
        ensure_dir(F"{output_dir}/output")
        # Modifications copied from https://github.com/guanrenyang/NSRR-Reimplementation : trainer/trainer.py
        self.model.train()
        self.train_metrics.reset()
        t0 = time()

        is_stereo = False
        for batch_idx, [low_res_list, depth_list, motion_vector_list, target_list] in enumerate(self.data_loader):
            is_stereo = len(low_res_list[0].shape) == 5 # dodgy

            batch_split_idx = 0
            batch_size = low_res_list[0].shape[0]

            total_loss = 0
            total_metrics = np.zeros(len(self.metric_ftns))

            # output = torch.zeros_like(target_list[0])

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

                output = sub_batch_output
                # prepare for next sub-batch
                batch_split_idx += self.batch_split_size


            self.optimizer.step()

            # Save batch result
            image_name = f'epoch_{epoch}_batch_{batch_idx}.png'

            if batch_idx % self.image_verbosity == 0:
                toImage(output[0]).save(f'{output_dir}/output/{image_name}', format='PNG')
                

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)
            for i, met in enumerate(self.metric_ftns):
                self.train_metrics.update(met.__name__, total_metrics[i])

            if batch_idx % self.log_step == 0:
                img = low_res_list[0]
                if is_stereo:
                    img = img.permute(1,0,2,3,4)[0]
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss))
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self.logger.debug(f"Train Epoch: {epoch} Duration: {time() - t0:.3f} seconds")
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def get_patching_transforms(self, cur_size_hr: int, new_size_hr: int):
        # create a random 264 x 264 patch at the target resolution
        # patch_size_HR = 528
        new_size_lr = new_size_hr // self.scale_factor
        
        idx_h = np.random.randint(0, cur_size_hr - new_size_hr)
        idx_w = np.random.randint(0, cur_size_hr - new_size_hr)
        get_target_patch_high_res = lambda x : x[:, :, idx_h:idx_h+new_size_hr, idx_w:idx_w+new_size_hr]
        
        idx_h = idx_h // self.scale_factor
        idx_w = idx_w // self.scale_factor
        get_target_patch_low_res = lambda x : x[:, :, idx_h:idx_h+new_size_lr, idx_w:idx_w+new_size_lr]
        return get_target_patch_high_res, get_target_patch_low_res

    def init_nsrr(self):
        """
        Initialize trainer for Neural Supersampling for Real-time Rendering
        """
        self.criterion = NSRRLoss(0.1).to(self.device)
        # if len(self.device_ids) > 1:
        #     self.criterion = torch.nn.DataParallel(self.criterion, device_ids=self.device_ids)

    def train_nsrr(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        """
        Training logic for one batch / sub batch  Neural Supersampling for Real-time Rendering

        """
        _, _, H, W = target_list[0].shape
        assert H == W
        transform_hr, transform_lr = self.get_patching_transforms(H, 900)
        target_list = [transform_hr(target) for target in target_list[:1]]
        low_res_list = [transform_lr(low_res) for low_res in low_res_list]
        depth_list = [transform_lr(depth) for depth in depth_list]
        motion_vector_list = [transform_lr(motion_vector) for motion_vector in motion_vector_list]

        target = target_list[0].to(self.device)
        depth_list = [depth.to(self.device) for depth in depth_list]
        motion_vector_list = [motion_vector.to(self.device) for motion_vector in motion_vector_list]
        low_res_list = [low_res.to(self.device) for low_res in low_res_list]

        output = self.model(low_res_list, depth_list, motion_vector_list)

        loss = self.criterion(output, target) * weight
        # Pytorch can't gather loss
        # if hasattr(loss, "shape"):
        #     self.logger.debug("Averaging")
        #     loss = torch.mean(loss)

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
        # if len(self.device_ids) > 1:
        #     self.criterion = torch.nn.DataParallel(self.criterion, device_ids=self.device_ids)

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
                avg_metrics[i] += met(img_ss, target) * weight / (n - 1) # average over clip

        return img_ss, avg_loss, avg_metrics
    
    def init_inss(self):
        self.criterion = INSSLoss(
            alpha=1.5,
            beta=1,
            gamma=0.1
        ).to(self.device)
        # if len(self.device_ids) > 1:
        #     self.criterion = torch.nn.DataParallel(self.criterion, device_ids=self.device_ids)

    def train_inss(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        n = len(low_res_list)

        output = None
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
                prev_high_res = output.detach()

            output = self.model(low_res, depth, motion, prev_high_res, prev_depth)
            
            loss = self.criterion(output, target) * weight / (n - 1) # average over clip
            if accumulate_gradients:
                loss.backward()

            avg_loss += loss.item()  

            for i, met in enumerate(self.metric_ftns):
                avg_metrics[i] += met(output, target) * weight / (n - 1) # average over clip

        return output, avg_loss, avg_metrics

    def init_sinss(self):
        self.criterion = SINSSLoss(
            alpha=1.5,
            beta=1,
            gamma=0.1
        ).to(self.device)
        # if len(self.device_ids) > 1:
        #     self.criterion = torch.nn.DataParallel(self.criterion, device_ids=self.device_ids)

    def train_sinss(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        B, test, C, H, W = target_list[0].shape
        assert test == 2, "SINSS requires a stereo pair"
        low_res_list = [tensor.permute(1, 0, 2, 3, 4) for tensor in low_res_list]
        depth_list = [tensor.permute(1, 0, 2, 3, 4) for tensor in depth_list]
        motion_vector_list = [tensor.permute(1, 0, 2, 3, 4) for tensor in motion_vector_list]
        target_list = [tensor.permute(1, 0, 2, 3, 4) for tensor in target_list]

        n = len(low_res_list)
         
        output = None
        left_prev_high_res, right_prev_high_res = None, None
        avg_loss = 0
        avg_metrics = np.zeros(len(self.metric_ftns))

        for i in range(1,n):
            left_low_res, right_low_res = low_res_list[i].to(self.device)
            left_depth, right_depth = depth_list[i].to(self.device)
            left_prev_depth, right_prev_depth = depth_list[i-1].to(self.device)
            left_motion, right_motion = motion_vector_list[i].to(self.device)
            left_target, right_target = target_list[i].to(self.device)

            if i == 1 or self.use_prev_high_res:
                left_prev_high_res, right_prev_high_res = target_list[i-1].to(self.device)
            else:
                left_prev_high_res, right_prev_high_res = output

            left_output, right_output = self.model(
                left_low_res, left_depth, left_motion, left_prev_high_res, left_prev_depth,
                right_low_res, right_depth, right_motion, right_prev_high_res, right_prev_depth
            )
            
            loss = self.criterion(left_output, left_target, right_output, right_target) * weight / (n - 1) # average over clip
            
            if accumulate_gradients:
                loss.backward()

            avg_loss += loss.item()  

            for i, met in enumerate(self.metric_ftns):
                avg_metrics[i] += met(left_output, left_target) * 0.5 * weight / (n - 1) # average over clip
                avg_metrics[i] += met(right_output, right_target) * 0.5 * weight / (n - 1) # average over clip

            output = (left_output.detach(), right_output.detach())
        
        output = torch.cat((output[0], output[1]), dim=3)
        return output, avg_loss, avg_metrics

    def init_enss(self):
        self.criterion = torch.nn.L1Loss().to(self.device)

    def train_enss(self, low_res_list, depth_list, motion_vector_list, target_list, weight=1, accumulate_gradients=True):
        # create a random 264 x 264 patch at the target resolution
        patch_size_HR = 528
        patch_size_LR = patch_size_HR // self.scale_factor
        B, _, H_HR, W_HR = target_list[0].shape
        
        idx_h = np.random.randint(0, H_HR - patch_size_HR)
        idx_w = np.random.randint(0, W_HR - patch_size_HR)
        get_target_patch_high_res = lambda x : x[:, :, idx_h:idx_h+patch_size_HR, idx_w:idx_w+patch_size_HR]
        
        idx_h = idx_h // self.scale_factor
        idx_w = idx_w // self.scale_factor
        get_target_patch_low_res = lambda x : x[:, :, idx_h:idx_h+patch_size_LR, idx_w:idx_w+patch_size_LR]

        # get target patches
        target_list = [get_target_patch_high_res(target) for target in target_list]
        low_res_list = [get_target_patch_low_res(low_res) for low_res in low_res_list]
        depth_list = [get_target_patch_low_res(depth) for depth in depth_list]
        motion_vector_list = [get_target_patch_low_res(motion_vector) for motion_vector in motion_vector_list]
        
        # feature and color reccurent training
        n = len(low_res_list)
        
        loss = 0
        metrics = np.zeros(len(self.metric_ftns))

        prev_color = target_list[0].clone().to(self.device)
        prev_features = torch.zeros(B, 1, patch_size_HR, patch_size_HR).to(self.device)

        for i in range(1,n):
            low_res = low_res_list[i].to(self.device)
            depth = depth_list[i].to(self.device)
            motion = motion_vector_list[i].to(self.device)
            target = target_list[i].to(self.device)

            if self.use_prev_high_res:
                prev_color = target_list[i-1].to(self.device)

            output, new_features = self.model(low_res, depth, motion, prev_color, prev_features)
            
            loss += self.criterion(output, target) * weight / (n - 1) # average over clip

            for i, met in enumerate(self.metric_ftns):
                metrics[i] += met(output, target) * weight / (n - 1) # average over clip

            prev_color = output
            prev_features = new_features

        if accumulate_gradients:
            loss.backward()

        return prev_color, loss, metrics



    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        t0 = time()
        is_stereo = False
        with torch.no_grad():
            for batch_idx, [low_res_list, depth_list, motion_vector_list, target_list] in enumerate(self.valid_data_loader):
                is_stereo = len(low_res_list[0].shape) == 5 # dodgy
                batch_split_idx = 0
                batch_size = low_res_list[0].shape[0]

                total_loss = 0
                total_metrics = np.zeros(len(self.metric_ftns))
                # output = None

                while batch_split_idx < batch_size:
                    # split batch into sub-batches
                    sub_batch_size = min(batch_size - batch_split_idx, self.batch_split_size)
                    batch_split_idx_end = min(batch_size, batch_split_idx + self.batch_split_size)

                    sub_batch_low_res_list = [low_res[batch_split_idx:batch_split_idx_end] for low_res in low_res_list]
                    sub_batch_depth_list = [depth[batch_split_idx:batch_split_idx_end] for depth in depth_list]
                    sub_batch_motion_vector_list = [motion_vector[batch_split_idx:batch_split_idx_end] for motion_vector in motion_vector_list]
                    sub_batch_target_list = [target[batch_split_idx:batch_split_idx_end] for target in target_list]

                    # forward pass
                    _, loss, metrics = self.training_method(
                        sub_batch_low_res_list, sub_batch_depth_list, sub_batch_motion_vector_list, sub_batch_target_list,
                        weight = (sub_batch_size / batch_size),
                        accumulate_gradients = False
                        )

                    # loss & metrics
                    total_loss += loss
                    total_metrics += metrics

                    # output = sub_batch_output
                    # prepare for next sub-batch
                    batch_split_idx += self.batch_split_size

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', total_loss)
                for i, met in enumerate(self.metric_ftns):
                    self.valid_metrics.update(met.__name__, total_metrics[i])
                img = low_res_list[0]
                if is_stereo:
                    img = img.permute(1,0,2,3,4)[0]
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

        self.logger.debug(f"Validate Epoch: {epoch} Duration: {time() - t0:.3f} seconds")
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
