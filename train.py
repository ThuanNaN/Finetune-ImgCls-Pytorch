import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.ResNet import ResNetModel
from models.AlexNet import AlexNetModel
from models.DenseNet import DenseNetModel
from models.InceptionV3Net import InceptionV3NetModel
from models.VGGNet import VGGNetModel

from torch.utils.data import DataLoader
from utils.dataset import CashewDataset
from utils.dataset import get_data_transforms, ModelAndWeights
import yaml
from yaml.loader import SafeLoader
from utils.trainer import train_model



def load_model(opt):
    if opt.model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return ResNetModel(opt)
    elif opt.model_name in ["alexnet"]:
        return AlexNetModel(opt)
    elif opt.model_name in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        return DenseNetModel(opt)
    elif opt.model_name in ["inception_v3"]:
        return InceptionV3NetModel(opt) 
    elif opt.model_name in ["vgg11", "vgg11bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]:
        return VGGNetModel(opt) 


if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)


    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))

    opt.n_classes = DATA_CONFIG["n_classes"]
     
    
    if torch.cuda.is_available():
        opt.device = "cuda"
    elif torch.backends.mps.is_available():
        opt.device = "mps"
    else:
        opt.device = "cpu"
    
    print("DEVICE: ", opt.device)

    if opt.default_data_transform:
        data_transforms = get_data_transforms(ModelAndWeights, opt.model_name, opt.weight_name)
    else:
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])

    print("Data transforms: ")
    print(data_transforms)

    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms)
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms)

    label_dict = train_dataset.label2id

    dataloaders = {
        "train": DataLoader(train_dataset, opt.batch_size, shuffle=True, num_workers = 8), 
        "val": DataLoader(val_dataset, opt.batch_size)
    }


    model = load_model(opt)
    optimizer = optim.Adam(model.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    model, hist, f_maxtrix = train_model(model, dataloaders, criterion, optimizer, opt)

    print(hist)
    print(f_maxtrix)


