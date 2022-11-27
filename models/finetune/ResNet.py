import torch
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CustomResNet101(nn.Module):
    def __init__(self, num_class):
        super(CustomResNet101, self).__init__()
        self.resnet = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V2)
        # Freeze 
        set_parameter_requires_grad(self.resnet, feature_extracting=True)

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

