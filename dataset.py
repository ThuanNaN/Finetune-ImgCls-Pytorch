from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import  transforms, datasets

import numpy as np

class CustomDataset():
  def __init__ (self,dir_path, batch_size, IMG_SIZE = 224):
    self.dir_path = dir_path
    self.batch_size = batch_size
    self.IMG_SIZE = IMG_SIZE


  def load_dataset(self):
    phases = ['train', 'valid', 'test'] 
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_SDEV = [0.229, 0.224, 0.225]
    data_transforms = {
      'train':
          transforms.Compose([
              transforms.RandomRotation(30),
              transforms.RandomResizedCrop(self.IMG_SIZE),
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.ToTensor(),
              transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
      'valid':
          transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(self.IMG_SIZE),
              transforms.ToTensor(),
              transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
      'test':
          transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(self.IMG_SIZE),
              transforms.ToTensor(),
              transforms.Normalize(IMG_MEAN, IMG_SDEV)])
    }
    
    image_datasets = {n: datasets.ImageFolder(
                            self.dir_path, transform=data_transforms[n])
                      for n in phases}
                  
    valid_size = 0.2
    test_size = 0.1
    num_data = len(image_datasets['train'])
    indices = list(range(num_data))

    split_valid = int(np.floor((1.0 - (valid_size+test_size)) * num_data))
    split_test = int(np.floor((1.0 - test_size) * num_data))

    np.random.seed(42)
    np.random.shuffle(indices)

    data_idx = {"train" : indices[:split_valid],
                "valid":indices[split_valid:split_test],
                "test":indices[split_test:]}
  
    data_sampler = {n: SubsetRandomSampler(data_idx[n]) for n in phases}

    dataloaders = {n: DataLoader(
                        image_datasets[n], batch_size=self.batch_size, sampler = data_sampler[n])
                    for n in phases}

    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx

