import argparse
import yaml
from yaml import SafeLoader
from utils.dataset import IMAGE_NORM, get_data_transforms, CashewDataset
from torch.utils.data import DataLoader
from models.ResNet import ResNetModel
import torch
from PIL import Image
# import cv2
import glob

if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)

    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))

    image_norm = IMAGE_NORM["resnet101-IMAGENET1KV2"]
    data_transforms = get_data_transforms(**image_norm)["test"]

    model = ResNetModel(DATA_CONFIG["n_classes"], weight_pretrain=False, freeze_backbone=False)

    ckpt_path = "./ckpt/epoch_4_resnet101/best.pt"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    model.eval()

    for img_path in glob.glob("./test_case/*.jpg"):
        img = Image.open(img_path)
        img_transformed = data_transforms(img).unsqueeze(0)
        output = model(img_transformed)
        _, pred = torch.max(output, 1)

        print(pred)


    