"""
Example template for defining a system
"""
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_msssim import SSIM
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import ProbaDataset
from models.SRGAN import Discriminator
from models.SRResNet import MSRResNet
from utils import max_cPSNR


class ProbaModel(LightningModule):

    def __init__(self, hparams):
        super(ProbaModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        self.gen = MSRResNet(in_nc=hparams.nc + 2, out_nc=1, nf=48, nb=16, upscale=3)
        self.dis = Discriminator()

        self.loss_MSE = torch.nn.MSELoss()
        self.loss_ssim = SSIM(win_size=12, win_sigma=1.5, data_range=1., size_average=True, channel=1)

        path = os.path.join('/local/pajot/data/proba_v', "norm.csv")
        self.baseline_cpsnrs = pd.read_csv(path, ' ', names=['name', 'score']).set_index('name').to_dict()['score']

        # self.gan_loss = GANLoss(gan_mode='hinge')

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        lr, hr, mask_hr, name, sr = batch

        srs = self.gen(lr, sr)

        # if optimizer_i == 1:
        #     # set_requires_grad(self.dis, True)
        #     # set_requires_grad(self.gen, False)
        #
        #     # pred_fake = self.dis(srs.detach())
        #     # pred_real = self.dis(hr.unsqueeze(1))
        #
        #     # loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
        #     # loss_dis = (loss_dis_real + loss_dis_fake) / 2
        #
        #     # tqdm_dict = {'g_loss': loss_dis}
        #
        #     output = OrderedDict({
        #             'loss':         loss_dis,
        #             'progress_bar': tqdm_dict,
        #             'log':          tqdm_dict,
        #             })
        #
        #     return output

        # if optimizer_i == 0:
        # pred_fake = self.dis(srs)

        # loss_gen = self.gan_loss.G_loss(pred_fake)
        # print(hr.min(), hr.max())
        loss_ssim = 1 - self.loss_ssim(srs, hr.unsqueeze(1))
        loss_MSE = self.loss_MSE(srs, hr.unsqueeze(1))
        loss = loss_ssim * self.hparams.alpha + loss_MSE
        # loss =  self.loss_fn(srs.squeeze(), hr)

        # loss_gen = loss_gen + self.hparams.alpha * loss

        # tqdm_dict = {'g_loss': loss_gen, 'loss_mse': loss}
        tqdm_dict = {'loss_ssim': loss} # For logging purposes

        output = OrderedDict({
                'loss':         loss,
                'progress_bar': tqdm_dict,
                'log':          tqdm_dict,
                })

        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        lr, hr, mask_hr, name, sr = batch
        hrs = hr.cpu().numpy()
        hr_maps = mask_hr.cpu().numpy()
        srs = self.gen(lr, sr)[:, 0]
        srs = srs.detach().cpu().numpy()
        baseline_score = self.baseline_cpsnrs[name[0]]
        val_score = baseline_score / max_cPSNR(srs[0], hrs[0], hr_maps[0])

        output = OrderedDict({
                'val_loss': torch.Tensor([val_score]),
                'srs':      srs # for visualisation
                })
        return output

    def validation_end(self, outputs):

        val_loss_mean = 0

        for output in outputs:
            val_loss = output['val_loss']
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        srs = output['srs']
        self.logger.experiment.add_image('SR Image', (srs[0] - np.min(srs[0])) / np.max(srs[0]), dataformats='HW') # scaling for visualisation

        tqdm_dict = {'val_loss': val_loss_mean}

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        lrg = self.hparams.lrg
        lrd = self.hparams.lrd
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lrg, betas=(b1, b2))
        # opt_d = torch.optim.Adam(self.dis.parameters(), lr=lrd, betas=(b1, b2))

        scheduler = lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.98,
                                                   verbose=True, patience=3)

        return [opt_g], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        logging.info('training data loader called')
        train_dataset = ProbaDataset(root=self.hparams.data_root, train_test_val='train', top_k=self.hparams.topk, rand=self.hparams.rand,
                                     stat=self.hparams.stat)

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                      shuffle=True, num_workers=8,
                                      pin_memory=True)

        return train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        logging.info('val data loader called')

        val_dataset = ProbaDataset(root=self.hparams.data_root, train_test_val='val', top_k=self.hparams.topk, rand=self.hparams.rand,
                                   stat=self.hparams.stat)
        val_dataloader = DataLoader(val_dataset, batch_size=1,
                                    shuffle=False, num_workers=8,
                                    pin_memory=True)
        return val_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--lrd', default=0.0001, type=float)
        parser.add_argument('--lrg', default=0.0001, type=float)

        parser.add_argument('--topk', default=5, type=int)
        parser.add_argument('--rand', default=False, type=bool)
        parser.add_argument('--stat', default=False, type=bool)
        parser.add_argument('--alpha', default=500, type=float)

        parser.add_argument('--b1', default=0.5, type=float)
        parser.add_argument('--b2', default=0.99, type=float)

        parser.add_argument('--data_root', type=str)
        parser.add_argument('--nc', default=3, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser
