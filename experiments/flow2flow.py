  
import os
import sys
import gc

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import random
import math
import numpy as np

import torch
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image


from wolf.data import load_datasets, get_batch, preprocess, postprocess
from wolf import WolfModel
from wolf.utils import total_grad_norm
from wolf.optim import ExponentialScheduler

from wolf.modules.classifier.classifier import CovidNet
import pytorch_lightning as pl
from timebudget import timebudget

# TODO: self.g.flow.blocks[0].steps[0].conv1x1.weight.grad

class Flow2Flow(pl.LightningModule):
  def __init__(self, params, init_batch, lr, betas, lr_decay, weight_decay, warmup_steps, temp,
               src_classifier_path, init_lr=1e-7, eps=1e-8, amsgrad=False, n_bits=8):
    super(Flow2Flow, self).__init__()

    self.save_hyperparameters()

    self.wolf = WolfModel.from_params(params)
    self.classifier = CovidNet.load_from_checkpoint(src_classifier_path, unit_embed=True)

    self.classifier.eval()
    for param in self.classifier.parameters():
      param.requires_grad = False

    self.init_batch = init_batch

    self.val_acc = pl.metrics.Accuracy(compute_on_step=False)
    self.val_f1 = pl.metrics.F1(num_classes=2, average='macro', compute_on_step=False)
    self.val_precision = pl.metrics.Precision(num_classes=2, average='macro', compute_on_step=False)
    self.val_recall = pl.metrics.Recall(num_classes=2, average='macro', compute_on_step=False)

    # Used during testing
    self.inception = None
    # Save validation batch for visualization
    self.val_batch = None

  def on_train_start(self):
    #self.logger.log_hyperparams(self.hparams)
    # Verify correct classifier weights were loaded
    img, label = self.init_batch
    img, label = img.to(self.device), label.to(self.device)
    with torch.no_grad():
      pred = self.classifier.predict(img)
      self.val_f1(pred, label)
    print('Classifier F1 on initialization batch: {}'.format(self.val_f1.compute()))
    self.wolf.sync()

  def setup(self, stage=None):
    with timebudget("\n===== setup on gpu {}".format(self.device)):
      self.wolf.eval()
      img = self.init_batch[0].to(self.device)
      with torch.no_grad():
        self.wolf.init(preprocess(img, self.hparams.n_bits))
      if stage == 'test':
        self.inception = InceptionV3()

  def configure_optimizers(self):
    optimizer = AdamW(self.wolf.parameters(), lr=self.hparams.lr, betas=self.hparams.betas,
                      eps=self.hparams.eps, amsgrad=self.hparams.amsgrad, weight_decay=self.hparams.weight_decay)
    optimizers = [optimizer]
    schedulers = [{'scheduler': ExponentialScheduler(
      optimizer, self.hparams.lr_decay, self.hparams.warmup_steps, self.hparams.init_lr), 'name': 'lr_g'}]

    return optimizers, schedulers

  def on_after_backward(self):
    self.log('g grad norm', total_grad_norm(self.wolf.parameters()), prog_bar=True, logger=True)

  def training_step(self, batch, batch_idx):
    img = batch[0]

    loss_gen, loss_kl, loss_dequant = self.wolf.loss(img, n_bits=self.hparams.n_bits, nsamples=1)
    loss_gen = loss_gen.sum()
    loss_kl = loss_kl.sum()
    loss_dequant = loss_dequant.sum()
    loss = (loss_gen + loss_kl + loss_dequant) / img.size(0)

    self.log('Train MLE Loss', loss_gen.item() / img.size(0), prog_bar=True, logger=True)
    self.log('Train KL Loss', loss_kl.item() / img.size(0), prog_bar=True, logger=True)

    return loss

  def training_epoch_end(self, training_step_outputs):
    self.classifier.eval()
    self.wolf.eval()
    self.wolf.sync()

  def validation_step(self, batch, batch_idx):
    img, label = batch

    sample = self.generate(img)
    pred = self.classifier.predict(sample)

    self.val_acc(pred, label)
    self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
             prog_bar=False, logger=True)

    self.val_f1(pred, label)
    self.log('val_f1', self.val_f1, on_step=False, on_epoch=True,
             prog_bar=True, logger=True)

    self.val_precision(pred, label)
    self.log('val_precision', self.val_precision, on_step=False, on_epoch=True,
             prog_bar=False, logger=True)

    self.val_recall(pred, label)
    self.log('val_recall', self.val_recall, on_step=False, on_epoch=True,
             prog_bar=False, logger=True)

    # Reservoir Sampling to save validation batch
    if np.random.uniform() <= 1 / (batch_idx + 1):
        self.val_batch = batch

  def validation_epoch_end(self, validation_step_outputs):
    self.log('curr_epoch', self.current_epoch)

    img, label = self.val_batch
    sample = self.generate(img)
    self.log_reconstruct(img, sample, label)

  def generate(self, img):
    z, epsilon = self.wolf.encode(img, n_bits=self.hparams.n_bits, random=False)
    epsilon = epsilon.squeeze(1)
    new_eps = self.hparams.temp * torch.randn_like(epsilon)
    z = z.squeeze(1) if z is not None else z
    img_recon = self.wolf.decode(new_eps, z=z, n_bits=self.hparams.n_bits)
    return img_recon

  def rescale(self, tensor):
    if tensor.size(1) == 2:
        tensor = tensor.narrow(dim=1, start=1, length=1)

    tensor = tensor.cpu().float()
    tensor *= 255.
    tensor = tensor.type(torch.uint8)

    return tensor
  
  def make_grid(self, tensor, nrow=8, padding=2, normalize=False, range_=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/utils.py
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range_ (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float or tuple, optional): Value for the padded pixels.
            If tuple, one per channel.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range_ is not None:
            assert isinstance(range_, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range_)
        else:
            norm_range(tensor, range_)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding)

    # fill with the pad value
    if isinstance(pad_value, float) or isinstance(pad_value, int):
        grid.fill_(pad_value)
    else:
        if len(pad_value) != 3:
            raise ValueError('Specified tuple pad_value per channel, \
                              but has {} != 3 elements'.format(len(pad_value)))
        # Pad per channel
        for i, v in enumerate(pad_value):
            grid[i, :, :] = v

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid

  def log_reconstruct(self, img, sample, labels):
    image_tensor_dict = {'sample': sample,
        'img': postprocess(preprocess(img, self.hparams.n_bits), self.hparams.n_bits)}

    image_dict = {k: self.rescale(v) for k, v in image_tensor_dict.items()}

    nrow=2
    plot_groups = ('img',)
    image_keys = {'img': ('img', 'sample')}
    
    for plot_group in plot_groups:
      image_batches=[image_dict[k] for k in image_keys[plot_group]]
      for i in range(16): # number of reconstructions to generate
          images=[]
          label=[]
          for image_batch in image_batches:
              if len(image_batch) > i:
                  idx = torch.randint(0, len(image_batch), (1,)).item()
                  images.append(image_batch[idx])
                  label.append(labels[idx])
          if len(images) == 0:
              continue
          images_title='{}/{}_{}'.format(plot_group,
                                          '_'.join(image_keys[plot_group]), i+1)

          # need access to label at this point to distinguish
          covid_negative = (128, 128, 128)
          covid_positive = (212, 176, 190)
          padding = covid_positive if label[0] else covid_negative # 0 - Non-COVID19, 1 - COVID19

          images_concat=self.make_grid(
              images, nrow=nrow, padding=4, pad_value=padding)
          self.logger.experiment.log_image(
              name=images_title, image_data=images_concat.permute(1, 2, 0), step=self.current_epoch)

  def test_step(self, batch, batch_idx, output_dir='evaluation/'):
      img, label = batch
      
      embed = self.e(img)
      v = self.args.temp * torch.randn((img.size(0),) + self.g.latent_shape, device=self.device)
      sample = self.g(v, embed, reverse=True)

      pred = torch.argmax(self.e.predict(sample), dim=1).cpu().numpy()
      label = label.cpu().numpy()
      
      acc = accuracy_score(label, pred)
      self.log('accuracy', torch.tensor(acc, device=self.device), prog_bar=True, logger=True, sync_dist=True)

      f1 = f1_score(label, pred, pos_label=0) # TODO
      self.log('f1', torch.tensor(f1, device=self.device), prog_bar=True, logger=True, sync_dist=True)

      # # Standard FID
      # inception_real_act = self.inception(img)
      # inception_synthetic_act = self.inception(sample)

      return {
          'real_act': embed.cpu().numpy(),
          'synthetic_act': self.e(sample).cpu().numpy(),
          # 'inception_real_act': inception_real_act.cpu().numpy(),
          # 'inception_synthetic_act': inception_synthetic_act.cpu().numpy()
      }

      # for idx in len(sample):
      #     image = sample[idx]
      #     torchvision.utils.save_image(image, 
      #         fp=os.path.join(output_dir, '{}_{}'.format(batch_idx, idx)),
      #         format='png')

  def test_epoch_end(self, outputs):
      def log_fid(real_act, synthetic_act):
          mu_real = np.mean(real_act, axis=0)
          sigma_real = np.cov(real_act, rowvar=False)

          mu_synthetic = np.mean(synthetic_act, axis=0)
          sigma_synthetic = np.cov(synthetic_act, rowvar=False)

          fid = calculate_frechet_distance(mu_synthetic, sigma_synthetic, mu_real, sigma_real)
          self.log('fid', fid, prog_bar=True, logger=True)

      real_act = np.concatenate([out['real_act'] for out in outputs], axis=0)
      synthetic_act = np.concatenate([out['synthetic_act'] for out in outputs], axis=0)
      log_fid(real_act, synthetic_act)

      # # Standard FID
      # inception_real_act = np.concatenate([out[2] for out in outputs], axis=0)
      # inception_synthetic_act = np.concatenate([out[3] for out in outputs], axis=0)
      # log_fid(inception_real_act, inception_synthetic_act)
