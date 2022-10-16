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
from torchvision.utils import save_image

import pytorch_lightning
from pytorch_lightning import LightningModule

from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from utils import norm
from utils import denorm
import argparse
import os
import json
import datetime
import time
from functools import partial
from collections import defaultdict
import cv2

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
from utils import load_json
from utils import check_manual_seed
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


class Ramdomsampling(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dir = config.save.output_root_dir
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

    def save_image(self, seed):
        #save image
        seed = check_manual_seed(seed)
        print('Using manual seed: {}'.format(seed))
        fixed_z = torch.randn(calc_latent_dim(self.config))
        self.D.eval()
        with torch.no_grad():
            x_p = self.D(fixed_z)
            x_p = x_p.detach().cpu()
            x_p = x_p[:self.config.save.n_save_images, ...]
            os.makedirs(os.path.join(config.save.output_root_dir, 'sampling_images'), exist_ok=True)
            save_image(x_p.data, os.path.join(config.save.output_root_dir, 'sampling_images', f'seed_{seed}_img.jpg'))


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    rs = Ramdomsampling.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config = config)
    value = torch.randint(low=0, high=100000, size=(config.save.num_seed,))
    for seed in value:
        rs.save_image(seed)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    
    main(config)