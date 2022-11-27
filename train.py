import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.finetune.ResNet import CustomResNet101
from utils.dataset import CashewDataset
from utils.common import get_data_transforms


import yaml
from yaml.loader import SafeLoader

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)

    
    with open("./config/model_config.yaml", "r") as f:
        MODEL_CONFIG  = yaml.load(f, Loader=SafeLoader)


    data_transforms = get_data_transforms()
    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms["train"])
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms["val"])

    dataloaders = {
        "train": DataLoader(train_dataset,DATA_CONFIG["batch_size"], shuffle=True), 
        "val": DataLoader(val_dataset, DATA_CONFIG["batch_size"])
    }

    model = CustomResNet101(MODEL_CONFIG["n_classes"])

    params_to_update = []
    print("Parameters will update !")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    optimizer = optim.Adam(params_to_update, lr=MODEL_CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=MODEL_CONFIG["n_epochs"], device=MODEL_CONFIG["device"])