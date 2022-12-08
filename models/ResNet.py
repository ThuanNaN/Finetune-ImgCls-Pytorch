import torch.nn as nn
from torchvision import models
from models.utils import set_parameter_requires_grad


def load_resnet_model(resnet_version, weight_pretrained = None):
    # Resnet 18
    if resnet_version == "resnet18":
        if weight_pretrained:
            return models.resnet18(weights = weight_pretrained)
        return models.resnet18(weights = None)
    
    # Resnet 34
    if resnet_version == "resnet34":
        if weight_pretrained:
            return models.resnet34(weights = weight_pretrained)
        return models.resnet34(weights = None)

    # Resnet 50
    elif resnet_version == "resnet50":
        if weight_pretrained:
            return models.resnet50(weights = weight_pretrained)
        return models.resnet50(weights = None)

    # Resnet 101
    elif resnet_version == "resnet101":
        if weight_pretrained:
            return models.resnet101(weights = weight_pretrained)
        return models.resnet101(weights = None)

    # Resnet 152
    elif resnet_version == "resnet152":
        if weight_pretrained:
            return models.resnet152(weights = weight_pretrained)
        return models.resnet152(weights = None)

class ResNetModel(nn.Module):
    def __init__(self, num_class, resnet_version = "resnet101",  weight_pretrained = "IMAGENET1K_V2", freeze_backbone = False):
        super(ResNetModel, self).__init__()
        self.num_classes = num_class
        
        self.resnet =  load_resnet_model(resnet_version, weight_pretrained)
        
        if freeze_backbone:
            set_parameter_requires_grad(self.resnet, feature_extracting=True)


        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_class)
        )

        params_to_update = []
        name_params_to_update = []

        for name,param in self.resnet.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                name_params_to_update.append(name)

        self.params_to_update = params_to_update
        self.name_params_to_update = name_params_to_update

    def forward(self, x):
        x = self.resnet(x)
        return x

