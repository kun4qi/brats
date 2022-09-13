import argparse
import os
import json
import datetime
import time
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning
from pytorch_lightning import LightningModule

from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from utils import norm
from utils import denorm
from dataio.settings import TRAIN_PATIENT_IDS
from dataio.settings import TEST_PATIENT_IDS
import functions.pytorch_ssim as pytorch_ssim



def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )

class CKBrainMet(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.alpha = self.config.training.alpha
        self.beta = self.config.training.beta
        self.margin = self.config.training.margin
        self.batch_size = self.config.dataset.batch_size
        self.fixed_z = torch.randn(calc_latent_dim(self.config))
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.

        # networks
        self.E = Encoder(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.enc_filters, activation=self.config.model.enc_activation).float()
        self.D = Decoder(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.dec_filters, activation=self.config.model.dec_activation, final_activation=self.config.model.dec_final_activation).float()
        if config.model.enc_spectral_norm:
            apply_spectral_norm(self.E)
        if config.model.dec_spectral_norm:
            apply_spectral_norm(self.D)

        if config.training.use_cuda:
            self.E = nn.DataParallel(self.E)
            self.D = nn.DataParallel(self.D)

    def l_recon(self, recon: torch.Tensor, target: torch.Tensor):
        if 'ssim' in self.config.training.loss:
            ssim_loss = pytorch_ssim.SSIM(window_size=11)

        if self.config.training.loss == 'l2':
            loss = F.mse_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'l1':
            loss = F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

        elif self.config.training.loss == 'ssim+l1':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.l1_loss(recon, target, reduction='sum')

        elif self.config.training.loss == 'ssim+l2':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.mse_loss(recon, target, reduction='sum')

        else:
            raise NotImplementedError

        return self.beta * loss / self.batch_size

    def l_reg(self, mu: torch.Tensor, log_var: torch.Tensor):
        loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
        return loss / self.batch_size

    def validation_step(self, batch, batch_idx):
        #save image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_epoch_interval == 0:
                    image = batch['image']
                    z, _, _ = self.E(image)
                    x_r = self.D(z)
                    x_p = self.D(self.fixed_z)

                    image = image.detach().cpu()
                    x_r = x_r.detach().cpu()
                    x_p = x_p.detach().cpu()

                    image = image[:self.config.save.n_save_images, ...]
                    x_r = x_r[:self.config.save.n_save_images, ...]
                    x_p = x_p[:self.config.save.n_save_images, ...]
                    self.logger.log_images(torch.cat([image, x_r, x_p]), self.current_epoch)


        
    def configure_optimizers(self):
        e_optim = optim.Adam(filter(lambda p: p.requires_grad, self.E.parameters()), self.config.optimizer.enc_lr, [0.9, 0.9999])
        d_optim = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.config.optimizer.dec_lr, [0.9, 0.9999])
        return [e_optim, d_optim]
