import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import sys
import glob
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from wolf.data.dataset import get_first_batch, ctscansModule, xrayModule, m2sDataModule, CTDataset
from flow2flow import Flow2Flow

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

import comet_ml
from comet_ml import Experiment

from experiments.options import parse_args

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

if __name__ == '__main__':
  # torch.autograd.set_detect_anomaly(True)

  args = parse_args()

  seed = random.randint(1, 1000)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  source = 'ct'
  args.seed = seed # Log for reproducibility

  if source == 'xray':
    print("using source x-ray")
    dm = xrayModule(
          resize_shape=args.resize_shape,
          num_channels=args.num_channels,
          batch_size=args.batch_size,
          num_workers=args.num_workers,
    )

  elif source == 'ct':
    dm = ctscansModule(
          resize_shape=(args.image_size, args.image_size),
          num_channels=args.num_channels,
          batch_size=args.batch_size,
          num_workers=args.workers,
    )

  elif source == 'mnist':
    dm = m2sDataModule(
          resize_shape=args.resize_shape,
          num_channels=args.num_channels,
          batch_size=args.batch_size,
          num_workers=args.num_workers,
    )
  params = json.load(open(args.config, 'r'))
  model = Flow2Flow(params, get_first_batch(dm, init_batch_size=args.init_batch_size), 
    lr=args.lr, betas=(args.beta1, args.beta2), 
    lr_decay=args.lr_decay, weight_decay=args.weight_decay, 
    warmup_steps=args.warmup_steps, temp=args.tau,
    src_classifier_path=args.src_classifier_path)
  
  exp_name = args.name

  comet_logger = CometLogger(
    api_key='bX9IlSNN49zxWDmnoSB5GwX4i',
    project_name='benchmark_wolf',
    experiment_name=args.name
  )
  comet_logger.log_hyperparams(args)

  ckpt_callback = ModelCheckpoint(
      dirpath='./checkpoints/{}'.format(exp_name),
      save_last=True
  )
  best_bpd = ModelCheckpoint(
      dirpath='./checkpoints/{}'.format(args.name),
      filename='{Train MLE Loss:.4f}',
      monitor='Train MLE Loss',
      mode='min'
  )
  best_f1 = ModelCheckpoint(
      dirpath='./checkpoints/{}'.format(args.name),
      filename='{val_f1:.4f}',
      monitor='val_f1',
      mode='max'
  )
  lr_monitor = LearningRateMonitor(logging_interval='epoch')

  # # * means all if need specific format then *.csv
  # list_of_files = glob.glob('./checkpoints/{}/*'.format(exp_name))
  # vals = [x for x in list_of_files if 'val' in x]
  # # sort by highest validation
  # vals = sorted(vals, key=lambda x: int(x[-9:-5]), reverse=True)

  # resume_path = None
  # if vals:
  #     resume_path = vals[0]
  #     print("====== resuming training from {} ======".format(resume_path))
  # else:
  #     print("======\nno feasible model weights found from\n{}\n======".format(
  #         list_of_files))

  trainer = pl.Trainer(max_epochs=args.epochs,
      resume_from_checkpoint='./checkpoints/{}/{}.ckpt'.format(exp_name, args.checkpoint) if args.checkpoint else None, 
      #resume_from_checkpoint=resume_path,
      callbacks=[ckpt_callback, best_f1, best_bpd, lr_monitor],
      accumulate_grad_batches=args.batch_steps,
      # progress_bar_refresh_rate=67,
      gpus=args.gpus,
      gradient_clip_val=args.grad_clip,
      accelerator='ddp',
      sync_batchnorm=True,
      #plugins='ddp_sharded',
      profiler=True,
      logger=comet_logger,
      # log_every_n_steps=1000,
      # flush_logs_every_n_steps=1000,
      fast_dev_run=False)
  
  trainer.fit(model, dm)
