import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ResNet import ResNetModel
from utils.dataset import CashewDataset
from utils.dataset import get_data_transforms, IMAGE_NORM
import yaml
from yaml.loader import SafeLoader
from utils.trainer import train_model




if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)


    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))

    num_classes = DATA_CONFIG["n_classes"]

    image_norm = IMAGE_NORM["IMAGENET1K_V2"]
    data_transforms = get_data_transforms(**image_norm)

    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms["train"])
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms["val"])

    if opt.is_ViT:
        label_dict = train_dataset.label2id
        label2id = { l: str(id) for l, id in zip(label_dict.keys(), label_dict.values())}
        id2label = { str(id): l for l, id in zip(label_dict.keys(), label_dict.values())}


    dataloaders = {
        "train": DataLoader(train_dataset, opt.batch_size, shuffle=True), 
        "val": DataLoader(val_dataset, opt.batch_size)
    }

    model = ResNetModel(num_classes, resnet_version=opt.model_name, weight_pretrained=opt.weight_pretrained, freeze_backbone = opt.freeze_backbone)
    optimizer = optim.Adam(model.params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # if opt.fp16:
    #     if not is_apex_available():
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)

    model, hist, f_maxtrix = train_model(model, dataloaders, criterion, optimizer, opt)

    print(hist)
    print(f_maxtrix)