import torch
import torch.nn as nn
from torchvision import models
from models.utils import set_parameter_requires_grad


def load_inception_v3(opt):
    if opt.model_name == "inception_v3":
        if opt.load_weight:
            print("Load pretrained model weight !")
            return models.inception_v3(weights = opt.weight_name)
        return models.inception_v3(weights = None)

class InceptionV3NetModel(nn.Module):
    def __init__(self, opt):
        super(InceptionV3NetModel, self).__init__()
        self.num_classes = opt.n_classes

        self.model = load_inception_v3(opt)

        if opt.freeze_backbone:
            set_parameter_requires_grad(self.model)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

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

        return x.logits

