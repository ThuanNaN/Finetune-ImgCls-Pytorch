
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class CashewDataset(Dataset):
    def __init__(self, dir_path,  data_transforms):
        self.dir_path = dir_path
        self.images = datasets.ImageFolder(
                        dir_path, transform=data_transforms
                    )
        self.label2id = self.images.class_to_idx

    
    def __getitem__(self, index):
        image, class_id = self.images[index] 
        return image, class_id
    
    def __len__(self):
        return len(self.images)


def get_data_transforms(**image_norm):
    image_size = image_norm["img_size"]
    image_mean = image_norm["img_mean"]
    image_std = image_norm["img_std"]
    data_transforms = {
        "train": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(image_mean, image_std)]),

        "val": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(image_mean, image_std)]),
        
        "test": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(image_mean, image_std)]),
    }

    return data_transforms


IMAGE_NORM = {
    "resnet101-imagenetV1": 
        {
            "img_size": 224,
            "img_mean": [0.485, 0.456, 0.406],
            "img_std": [0.229, 0.224, 0.225]
        },

    "resnet101-imagenetV2": 
        {
            "img_size": 224,
            "img_mean": [0.485, 0.456, 0.406],
            "img_std": [0.229, 0.224, 0.225]
        },
}

