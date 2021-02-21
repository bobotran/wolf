import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
import random
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from wolf.data.dataset import get_first_batch, ctscansModule, xrayModule, m2sDataModule, CTDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

import comet_ml
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from wolf.modules.classifier.classifier import CovidNet
from wolf import WolfModel
import json

class Flow2Flow(pl.LightningModule):
  def __init__(self, params, init_batch, lr, betas, lr_decay, weight_decay, warmup_steps, temp,
               src_classifier_path, init_lr=1e-7, eps=1e-8, amsgrad=False, n_bits=8):
    super(Flow2Flow, self).__init__()
    self.save_hyperparameters()
    
    self.wolf = WolfModel.from_params(params)
    self.classifier = CovidNet.load_from_checkpoint(src_classifier_path, unit_embed=True)

    self.classifier.eval()
    self.classifier.freeze()

  def generate(self, img):
    z, epsilon = self.wolf.encode(img, n_bits=self.hparams.n_bits, random=False)
    epsilon = epsilon.squeeze(1)
    new_eps = self.hparams.temp * torch.randn_like(epsilon)
    z = z.squeeze(1) if z is not None else z
    img_recon = self.wolf.decode(new_eps, z=z, n_bits=self.hparams.n_bits)
    return img_recon

  def generate_and_save(self, batch, batch_idx, idx):
    img, label = batch
    with torch.no_grad():
      samples = self.generate(img)
    for ctr, i in enumerate(idx):
      save_image_tensor(samples, label, i, batch_idx, 'synth{}'.format(ctr))

  def test_step(self, batch, batch_idx):
    self.classifier.eval()
    self.wolf.eval()

    img, label = batch
    # for i in range(img.size(0)):
    #   save_image_tensor(img, label, i, batch_idx, 'real')
    self.generate_and_save(batch, batch_idx, range(len(label)))

    # 30% COVID = 0.6862848015
    # 40% COVID = 1.623109691
    # 50% COVID = 2.934664537
    oversample_ratio = 0.0
    # Only augment positive class
    pos_idx = (label==1).nonzero(as_tuple=True)[0]
    oversample_num = int(len(pos_idx) * oversample_ratio)
    if oversample_num > 0:
      aug_idx = np.random.randint(low=0, high=len(pos_idx), size=oversample_num)
      self.generate_and_save(batch, batch_idx, pos_idx[aug_idx])

    return {
      'oversample_num': oversample_num,
      'total_positives': oversample_num + len(pos_idx),
      'negatives': len((label==0).nonzero())
    }
  
  def test_epoch_end(self, test_step_outputs):
    oversample_total = sum([l['oversample_num'] for l in test_step_outputs])
    positives_total = sum([l['total_positives'] for l in test_step_outputs])
    negatives_total = sum([l['negatives'] for l in test_step_outputs])
    print('Synthetic Examples Generated: {}'.format(oversample_total))
    print('Original Positive + Synthetic Positive: {}'.format(positives_total))
    print('Negative: {}'.format(negatives_total))
    print('Total: {}'.format(negatives_total + positives_total))

def save_image_tensor(img, label, i, batch_idx, suffix=''):
  filename = '/tmp/hsperfdata_hpdas/wolf_synthset/{}/{}_{}_{}.png'.format(label[i], batch_idx, i, suffix)
  torchvision.utils.save_image(img[i], fp=filename)
  # ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).astype(np.uint8)
  # im = Image.fromarray(ndarr)
  # im.save(fp, format=format)

if __name__ == '__main__':
  # torch.autograd.set_detect_anomaly(True)

  seed = random.randint(1, 1000)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  source = 'ct'
  resize_shape = (64,64)
  num_channels = 1

  if source == 'xray':
    print("using source x-ray")
    dm = xrayModule(
          resize_shape=args.resize_shape,
          num_channels=num_channels,
          batch_size=args.batch_size,
          num_workers=args.num_workers,
    )

  elif source == 'ct':
    dm = ctscansModule(
          resize_shape=resize_shape,
          num_channels=num_channels,
          batch_size=1024,
          num_workers=8,
    )

  elif source == 'mnist':
    dm = m2sDataModule(
          resize_shape=args.resize_shape,
          num_channels=num_channels,
          batch_size=args.batch_size,
          num_workers=args.num_workers,
    )
  params = json.load(open('configs/4level.json', 'r'))
  model = Flow2Flow.load_from_checkpoint('checkpoints/wolf_4level1024/Train_MLE_Loss=-9751.2558.ckpt', params=params)

  dm.setup()
  train_loader = dm.train_dataloader()

  trainer = pl.Trainer(max_epochs=2000,
      gpus=1,
      logger=None,
      accelerator='ddp',
      sync_batchnorm=True,
      profiler=True,
      fast_dev_run=False)
  
  # Generate synthetic set from train set
  trainer.test(model=model, test_dataloaders=train_loader)
