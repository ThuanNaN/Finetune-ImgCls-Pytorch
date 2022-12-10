import argparse
import torch
import torch.nn as nn
import torch.optim as optim

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
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = get_data_transforms(ModelAndWeights, opt.model_name, opt.weight_name)


    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms)
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms)

    label_dict = train_dataset.label2id

    dataloaders = {
        "train": DataLoader(train_dataset, opt.batch_size, shuffle=True), 
        "val": DataLoader(val_dataset, opt.batch_size)
    }


    model = load_model(opt)
    optimizer = optim.Adam(model.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # if opt.fp16:
    #     if not is_apex_available():
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)

    model, hist, f_maxtrix = train_model(model, dataloaders, criterion, optimizer, opt)

    print(hist)
    print(f_maxtrix)


