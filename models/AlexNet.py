import torch
import torch.nn as nn
from models.utils import set_parameter_requires_grad


class AlexNetModel(nn.Module):
    def __init__(self, num_class, model_name = "alexnet", weight_pretrained = "IMAGENET1K_V1", freeze_backbone = False):
        super(AlexNetModel, self).__init__()
        self.num_classes = num_class

        self.model = torch.hub.load("pytorch/vision", model_name, weights=weight_pretrained)

        if freeze_backbone:
            set_parameter_requires_grad(self.model)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_class)

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


