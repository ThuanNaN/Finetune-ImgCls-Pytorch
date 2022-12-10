import torch
import torch.nn as nn
from torchvision import models
from models.utils import set_parameter_requires_grad


def load_vgg(opt):
    if opt.model_name == "vgg11":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg11(weights = opt.weight_name)
        return models.vgg11(weights = None)
    if opt.model_name == "vgg11_bn":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg11_bn(weights = opt.weight_name)
        return models.vgg11_bn(weights = None)
    if opt.model_name == "vgg13":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg13(weights = opt.weight_name)
        return models.vgg13(weights = None)
    if opt.model_name == "vgg13_bn":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg13_bn(weights = opt.weight_name)
        return models.vgg13_bn(weights = None)
    if opt.model_name == "vgg16":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg16(weights = opt.weight_name)
        return models.vgg16(weights = None)
    if opt.model_name == "vgg16_bn":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg16_bn(weights = opt.weight_name)
        return models.vgg16_bn(weights = None)
    if opt.model_name == "vgg19":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg19(weights = opt.weight_name)
        return models.vgg19(weights = None)
    if opt.model_name == "vgg19_bn":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.vgg19_bn(weights = opt.weight_name)
        return models.vgg19_bn(weights = None)
    

class VGGNetModel(nn.Module):
    def __init__(self, opt):
        super(VGGNetModel, self).__init__()
        self.num_classes = opt.n_classes

        self.model = load_vgg(opt)

        if opt.freeze_backbone:
            set_parameter_requires_grad(self.model)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

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

