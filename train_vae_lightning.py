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
from pytorch_lightning import LightningModule, Trainer

from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from utils import load_json
from utils import check_manual_seed
from utils import norm
from utils import denorm
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio.settings import TRAIN_PATIENT_IDS
from dataio.settings import TEST_PATIENT_IDS
from dataio import MNISTDataModule
import functions.pytorch_ssim as pytorch_ssim
from common_part import CKBrainMet


class vae(CKBrainMet):
    def __init__(self, config, needs_save): 
        super().__init__(config, needs_save)
        self.config = config
        self.needs_save = needs_save

    def training_step(self, batch, batch_idx):
        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)

        e_optim, d_optim = self.optimizers()
        e_optim.zero_grad()
        d_optim.zero_grad()


        # image = norm(batch['image'])
        image = batch['image']
        z, z_mu, z_logvar = self.E(image)
        #print(z.shape)
        x_r = self.D(z)

        l_vae_reg = self.l_reg(z_mu, z_logvar)

        l_vae_recon = self.l_recon(x_r, image)

        l_vae_total = self.alpha * l_vae_reg + self.beta * l_vae_recon

        self.manual_backward(l_vae_total)

        e_optim.step()
        d_optim.step()

        if self.needs_save:
            self.log('l_vae_reg', l_vae_reg, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('l_vae_recon', l_vae_recon, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('l_vae_total', l_vae_total, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        
        return {'loss': l_vae_total}


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'l_vae_reg', 'l_vae_recon', 'l_vae_total']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)
    
    #save config
    logger.log_hyperparams(config, needs_save)

    #set callbacks
    #save_interval??????????????????????????????limit_num????????????
    checkpoint_callback = ModelSaver(
        limit_num=config.save.n_saved,
        save_interval=config.save.save_epoch_interval,
        monitor=None,
        dirpath=logger.log_dir,
        filename='ckpt-{epoch:04d}',
        save_top_k=-1,
        save_last=False
    )

    #time per epoch
    timer = Time(config)

    dm = MNISTDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        gpus=1,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = vae(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      model = vae.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
      trainer.fit(model, dm)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)
