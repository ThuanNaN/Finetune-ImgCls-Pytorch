import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.finetune.ResNet import CustomResNet101
from utils.dataset import CashewDataset
from utils.common import get_data_transforms, IMG_NORM, IMG_SIZE
import yaml
from yaml.loader import SafeLoader

from utils.trainer import train_model



if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)


    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))


    data_transforms = get_data_transforms(image_size=IMG_SIZE["Resnet101_ImgNetV2"], image_norm=IMG_NORM["General"])

    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms["train"])
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms["val"])

    dataloaders = {
        "train": DataLoader(train_dataset, opt.batch_size, shuffle=True), 
        "val": DataLoader(val_dataset, opt.batch_size)
    }

    model = CustomResNet101(DATA_CONFIG["n_classes"])

    params_to_update = []
    print("Parameters will update !")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    optimizer = optim.Adam(params_to_update, lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=opt.n_epochs, device=opt.device)