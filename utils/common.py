import os
import torch
from torchvision import transforms

IMG_NORM = {
    "General": {
        "IMG_MEAN": [0.485, 0.456, 0.406],
        "IMG_STD": [0.229, 0.224, 0.225]
    }
}


IMG_SIZE = {
    "Resnet101_ImgNetV2": 232,
}


def get_data_transforms(image_size, image_norm):
    data_transforms = {
        "train": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(image_norm["IMG_MEAN"], image_norm["IMG_STD"])]),

        "val": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(image_norm["IMG_MEAN"], image_norm["IMG_STD"])]),
        
        "test": transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(image_norm["IMG_MEAN"], image_norm["IMG_STD"])]),
    }

    return data_transforms

def save_ckpt(path_save, name_ckpt, model, optimizer):
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(path_save, name_ckpt)
    )


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

