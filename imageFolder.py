import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

VAL_END_IDX = 1000

def create_datasets(data_dir, split, patch_size, size=0):
  transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(patch_size),
                                        transforms.ToTensor(),])

  full_dset = dset.ImageFolder(root=data_dir, transform=transform)
  indices = range(VAL_END_IDX, len(full_dset)) if split == 'train' else range(VAL_END_IDX)
  
  if (size != 0):
    return torch.utils.data.Subset(full_dset, indices[:size])

  return torch.utils.data.Subset(full_dset, indices)