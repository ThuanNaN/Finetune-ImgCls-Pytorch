import argparse
import yaml
from yaml import SafeLoader
from utils.dataset import IMAGE_NORM, get_data_transforms, CashewDataset
from torch.utils.data import DataLoader
from models.ResNet import ResNetModel
import torch


def test_model(model, test_loader, device):
    model.to(device)
    totals = 0
    corrects = 0
    confusion_matrix = torch.zeros(model.num_classes,model.num_classes)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            totals += inputs.size(0)
            corrects += torch.sum(preds == labels.data)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long, p.long] +=1
    acc = corrects / totals
    return acc, confusion_matrix


if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)

    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))

    image_norm = IMAGE_NORM["resnet101-imagenetV2"]
    data_transforms = get_data_transforms(**image_norm)

    test_dataset = CashewDataset(DATA_CONFIG["test"], data_transforms["test"])
    test_loader = DataLoader(test_dataset, opt.batch_size)

    model = ResNetModel(DATA_CONFIG["n_classes"], weight_pretrain=False, freeze_backbone=False)

    ckpt_path = "./ckpt/best.pt"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    model.eval()
        
    accuracy, confusion_matrix = test_model(model, test_loader, opt.device)

    print(accuracy)
    print(confusion_matrix)