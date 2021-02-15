import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
# import augmentations
import torch
from torchvision import datasets
from torchvision.utils import save_image

import pytorch_lightning as pl
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class eight_percent(object):
  def __call__(self, img):
    """
    :param img: (PIL): Image 
    :return: ycbr color space image (PIL)
    """
    # ! assuming its stored (H x W)
    h, w = img.size

    top = int(h * 0.08)
    left = 0

    height = h - top
    width = w - left
    return transforms.functional.crop(img, top, left, height, width)

class m2sDataModule(pl.LightningDataModule):
  def __init__(self,
               resize_shape = (32, 32),
               input_normalize = (0.5, 0.5),
               num_channels = 1,
               data_dir: str = './',
               batch_size: int = 32,
               num_workers: int = 8):

    super().__init__()

    self.resize_shape = resize_shape
    self.input_normalize = input_normalize
    self.num_channels = num_channels

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

  def prepare_data(self):
    # download
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage=None):

    # Assign train/val datasets for use in dataloaders
    if stage == 'fit' or stage is None:
      mnist_full = MNIST(self.data_dir, train=True, transform=get_transform_fn(resize_shape=self.resize_shape))
      # mnist_full, _ = random_split(mnist_full, [10000, len(mnist_full) - 10000])

      split = 0.80 # TODO temporarily here to test end of training stuff

      m_length = len(mnist_full)
      m_l_train = int(m_length * split)
      m_l_val = m_length - m_l_train

      self.mnist_train, self.mnist_val = random_split(mnist_full, [m_l_train, m_l_val])

      self.dims = self.mnist_train[0][0].shape

    # Assign test dataset for use in dataloader(s)
    if stage == 'test' or stage is None:
      self.mnist_test = MNIST(self.data_dir, train=False, transform=get_transform_fn(resize_shape=self.resize_shape))

      self.dims = self.mnist_test[0][0].shape
      print("self.dims {}".format(self.dims))

  def train_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.mnist_train,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

  def val_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.mnist_val,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

# class xrayModuleNew(pl.LightningDataModule):

class xrayModule(pl.LightningDataModule):
  #todo: augment this with the pneumonia rsna dataset
  def __init__(self,
               resize_shape = (128, 128), 
               num_channels = 1,
               batch_size: int = 4, 
               num_workers: int = 0):
    super().__init__()

    self.resize_shape = resize_shape
    self.num_channels = num_channels
    self.batch_size = batch_size
    self.num_workers = num_workers

  def setup(self, stage=None):

    if self.num_channels == 3:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif self.num_channels == 1:
      normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
      
    train_transformer = transforms.Compose([
        eight_percent(),
        transforms.Resize(self.resize_shape),
        # transforms.RandomAffine(10, (0, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((self.resize_shape), scale=(0.5, 1.0)),
        transforms.Grayscale(num_output_channels=self.num_channels),
        transforms.ToTensor(),
    ])

    val_transformer = transforms.Compose([
        eight_percent(),
        transforms.Resize(self.resize_shape),
        transforms.CenterCrop(self.resize_shape),
        transforms.Grayscale(num_output_channels=self.num_channels),
        transforms.ToTensor(),
    ])

    self.xray_train = datasets.ImageFolder(root='../ct_x_data_full/xray_split/train', transform=train_transformer)
    self.xray_val = datasets.ImageFolder(root='../ct_x_data_full/xray_split/val', transform=val_transformer)
    print("============-------{}------------==========".format(self.xray_train.class_to_idx))

    # if stage == 'fit' or stage is None:
    #   self.xray_train, self.xray_val = random_split(self.xray, self.split(self.xray, 0.85))

  def split(self, n, p):
    n = len(n)
    return [int(n*p), n - int(n*p)]

  def train_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.xray_train,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

  def val_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.xray_val,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

class CTDataset(Dataset):
  def __init__(self, stage, num_channels, resize_shape, augment=False, feature_extractor=None):
    self.stage = stage
    self.augment = augment
    assert stage in ['train', 'val', 'test']
    
    #annotations_file = '../covidxct/{}_COVIDx_CT-2A.txt'.format(stage)
    #annotations_file = '../covidxct_v1/{}_COVIDx-CT.txt'.format(stage)
    annotations_file = '/tmp/hsperfdata_bobotran/data/{}_COVIDx-CT.txt'.format(
        stage)
    # filename class xmin ymin xmax ymax
    # Normal=0, Pneumonia=1, and COVID-19=2
    lines = []
    with open(annotations_file) as file_in:
      for line in file_in:
        line_list = line.split()
        # if int(line_list[1]) == 0:
        lines.append(line_list)
    self.annotations = np.array(lines)
    #self.images_dir = '../covidxct/2A_images/2A_images'
    #self.images_dir = '../covidxct_v1/data/COVIDx-CT'
    self.images_dir = '/tmp/hsperfdata_bobotran/data/COVIDx-CT'

    self.grayscale = transforms.Grayscale()
    self.resize = transforms.Resize(resize_shape)
    self.to_tensor = transforms.ToTensor()
    if num_channels == 2:
      self.to_tensor = transforms.Compose([self.to_tensor, CopyChannel()])
    # if self.stage == 'train':
    #   self.to_image = transforms.ToPILImage()
    #   self.random_exterior_exclusion = augmentations.Random_Exterior_Exclusion()
    #   self.bbox_jitter = augmentations.Random_BBOX_Jitter()
    #   self.rotation = augmentations.Random_Rotation()
    #   self.shear = augmentations.Random_Shear()
    #   self.flip = transforms.RandomHorizontalFlip()
    #   self.shift_scale = augmentations.Random_Shift_Scale()
    self.embeddings = None
    if feature_extractor:
      feature_extractor = feature_extractor.to('cuda:1')
      embeddings = []
      for idx in tqdm(range(self.__len__())):
        img = self._getimage(idx).to('cuda:1')
        embeddings.append(feature_extractor(img).unsqueeze(0))
      self.embeddings = torch.cat(embeddings, 0).cpu()

  def _getimage(self, idx):
    img_path = os.path.join(self.images_dir, self.annotations[idx][0])

    image = Image.open(img_path).convert('RGB')
    image_width, image_height = image.size
    xmin = int(self.annotations[idx][2])
    ymin = int(self.annotations[idx][3])
    xmax = int(self.annotations[idx][4])
    ymax = int(self.annotations[idx][5])
    bbox = np.array([xmin, ymin, xmax, ymax])

    image = self.grayscale(image)

    if self.stage == 'train' and self.augment:
      arr = self.random_exterior_exclusion(np.array(image))
      image = self.to_image(arr)
      bbox = self.bbox_jitter(bbox, image_height, image_width)
      image, bbox = self.rotation(image, bbox)
      image, bbox = self.shear(image, bbox)
      image = TF.crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
      image = self.flip(image)
      image = self.resize(image)
      tensor = self.to_tensor(image)
      tensor = self.shift_scale(tensor)
    else:
      image = TF.crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
      image = self.resize(image)
      tensor = self.to_tensor(image)

    return tensor

  def __len__(self):
    return self.annotations.shape[0]

  def __getitem__(self, idx):
    label = int(self.annotations[idx][1])
    # Remap Pneumonia to 0, COVID to 1
    if label == 1 or label == 2:
      label -= 1
    label = torch.tensor(label, dtype=torch.long)
    tensor = self._getimage(idx)
    if self.embeddings:
      embedding = self.embeddings[idx]
      return (tensor, label, embedding)
    else:
      return (tensor, label)

class ctscansModule(pl.LightningDataModule):
  
  def __init__(self,
               resize_shape = (64, 64), 
               num_channels = 1,
               batch_size: int = 4, 
               num_workers: int = 0,
               augment: bool = False,
               synthetic: bool = False):
    super().__init__()

    self.resize_shape = resize_shape
    self.num_channels = num_channels
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.augment = augment
    self.synthetic = synthetic

  def setup(self, stage=None):
    if self.synthetic:
      transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
      ])
      self.ct_train = datasets.ImageFolder(root='/tmp/hsperfdata_bobotran/synthetic', transform=transform)
    else:
      self.ct_train = CTDataset('train', self.num_channels, self.resize_shape, self.augment)

    self.ct_val = CTDataset('val', self.num_channels, self.resize_shape, self.augment)
    self.ct_test = CTDataset('test', self.num_channels, self.resize_shape, self.augment)
  
  # def get_dataset(self):
  #   return self.ct_train

  # def split(self, n, p):
  #   n = len(n)
  #   return [int(n*p), n - int(n*p)]

  def train_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.ct_train,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

  def val_dataloader(self):
    loader = torch.utils.data.DataLoader(
      self.ct_val,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

  def test_dataloader(self):
    loader = torch.utils.data.DataLoader(
        self.ct_test,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        pin_memory=True
    )
    return loader


class xray2ctscanModule(pl.LightningDataModule):

  def __init__(self,
               resize_shape = (64, 64),
               input_normalize = (0.5, 0.5),
               num_channels = 1,
               data_dir: str = './x2c_small',
               batch_size: int = 64,
               num_workers: int = 0):
    super().__init__()

    self.resize_shape = resize_shape
    self.input_normalize = input_normalize
    self.num_channels = num_channels

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        # transforms.Grayscale(num_output_channels=1), # todo: remove this if in_channels is 3
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.Grayscale(num_output_channels=1), # todo: remove this if in_channels is 3
            transforms.ToTensor(),
            normalize
    ])

    # self.ct_cp = datasets.ImageFolder(root='./ct_x_data_full/ct/CP', transform=self._get_transform_fn())
    # self.ct_ncp = datasets.ImageFolder(root='./ct_x_data_full/ct/NCP', transform=self._get_transform_fn())
    # self.ct_n = datasets.ImageFolder(root='./ct_x_data_full/ct/Normal', transform=self._get_transform_fn())

    self.ct = datasets.ImageFolder(root='../../ct_x_data_full/ct_class', transform=train_transformer)
    self.xray_train = datasets.ImageFolder(root='../../ct_x_data_full/COVID-Net/data/train', transform=train_transformer)
    self.xray_val = datasets.ImageFolder(root='../../ct_x_data_full/COVID-Net/data/test', transform=train_transformer)

  def _get_transform_fn(self):
    transforms_list = []
    if self.resize_shape is not None: # allowing variable to be 0
      transforms_list.append(transforms.Resize(self.resize_shape))
    transforms_list.append(transforms.Grayscale(num_output_channels=1))
    transforms_list += [transforms.ToTensor()]
    if self.input_normalize is not None:
      mean, std = self.input_normalize
      transforms_list.append(transforms.Normalize((mean,)*self.num_channels, (std,)*self.num_channels))
    return transforms.Compose(transforms_list)
  
  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      self.ct_train, self.ct_val = random_split(self.ct, self.split(self.ct, 0.85))

  def train_dataloader(self):
    paired_dataset = PairedDataset(
      self.ct_train,
      self.xray_train
    )
    loader = torch.utils.data.DataLoader(
      paired_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader

  def val_dataloader(self):
    paired_dataset = PairedDataset(
      self.ct_val,
      self.xray_val
    )
    loader = torch.utils.data.DataLoader(
      paired_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    return loader


class CopyChannel(object):
    """Converts one channel tensor to two channel by copying first channel.
    """

    def __call__(self, tensor):
        return torch.cat([tensor, tensor], 0)

def get_transform_fn(resize_shape=(32,32)):
  transforms_list = []
  if resize_shape is not None:  # allowing variable to be 0
    transforms_list.append(transforms.Resize(resize_shape))
  #transforms_list += [transforms.Grayscale(num_output_channels=3)]
  transforms_list += [transforms.ToTensor()]
  #transforms_list += [CopyChannel()]
  
  # if self.input_normalize is not None:
  #   mean, std = self.input_normalize
  #   transforms_list.append(transforms.Normalize((mean,)*self.num_channels, (std,)*self.num_channels))
  return transforms.Compose(transforms_list)

def get_first_batch(data_module, init_batch_size=8):
  """
  Returns random datapoint for initialization
  """
  data_module.prepare_data()
  data_module.setup()
  dataset = data_module.train_dataloader().dataset
  rand_idx = np.random.random_integers(0, len(dataset)-1, size=init_batch_size)

  imgs, labels = [], []
  for idx in rand_idx:
    datum = dataset[idx]
    imgs.append(datum[0].unsqueeze(0))
    labels.append(datum[1].unsqueeze(0))
  batch = [torch.cat(imgs, 0), torch.cat(labels, 0)]

  return batch
