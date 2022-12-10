import torch
import torch.nn as nn
from torchvision import models
from models.utils import set_parameter_requires_grad


def load_resnet(opt):
    if opt.model_name == "resnet18":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.resnet18(weights = opt.weight_name)
        return models.resnet18(weights = None)

    elif opt.model_name == "resnet34":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.resnet34(weights = opt.weight_name)
        return models.resnet34(weights = None)

    elif opt.model_name == "resnet50":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.resnet50(weights = opt.weight_name)
        return models.resnet50(weights = None)

    elif opt.model_name == "resnet101":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.resnet101(weights = opt.weight_name)
        return models.resnet101(weights = None)

    elif opt.model_name == "resnet152":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.resnet152(weights = opt.weight_name)
        return models.resnet152(weights = None)

class ResNetModel(nn.Module):
    def __init__(self, opt):
        super(ResNetModel, self).__init__()
        self.num_classes = opt.n_classes
        
        self.model =  load_resnet(opt)
        
        if opt.freeze_backbone:
            set_parameter_requires_grad(self.model)


        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.num_classes)
        )

        params_to_update = []
        name_params_to_update = []

        for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                name_params_to_update.append(name)

        self.params_to_update = params_to_update
        self.name_params_to_update = name_params_to_update

    def forward(self, x):
        x = self.model(x)
        return x

