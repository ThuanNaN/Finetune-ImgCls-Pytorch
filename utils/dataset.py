import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset


class CashewDataset(Dataset):
    def __init__(self, dir_path,  data_transforms):
        self.dir_path = dir_path
        self.images = datasets.ImageFolder(
                        dir_path, transform=data_transforms
                    )
        self.class_dict = self.images.class_to_idx

    
    def __getitem__(self, index):
        image, class_id = self.images[index] 
        return image, class_id
    
    def __len__(self):
        return len(self.images)

