
from torchvision import transforms


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_SDEV = [0.229, 0.224, 0.225]



def get_data_transforms(IMG_SIZE = 232):
    data_transforms = {
        "train": transforms.Compose([
                    transforms.RandomRotation(30),
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(IMG_MEAN, IMG_SDEV)]),

        "val": transforms.Compose([
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
        
        "test": transforms.Compose([
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMG_MEAN, IMG_SDEV)]),
    }

    return data_transforms