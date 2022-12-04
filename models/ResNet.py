import torch.nn as nn
from torchvision import models
from models.utils import set_parameter_requires_grad


class ResNetModel(nn.Module):
    def __init__(self, num_class, weight_pretrain = models.ResNet101_Weights.IMAGENET1K_V2, freeze_backbone = True):
        super(ResNetModel, self).__init__()
        self.num_classes = num_class
        if weight_pretrain:
            self.resnet = models.resnet101(weights = weight_pretrain)
        else:
            self.resnet = models.resnet101()
        
        if freeze_backbone:
            set_parameter_requires_grad(self.resnet, feature_extracting=True)


        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_class)
        )

        params_to_update = []
        print("Parameters will update !")
        for name,param in self.resnet.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)

        self.params_to_update = params_to_update

    def forward(self, x):
        x = self.resnet(x)
        return x

