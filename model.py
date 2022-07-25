import torch
import torch.nn as nn
from torchvision import models


class Net():
  def __init__(self, num_class, model_name = 'vgg16'):
    self.num_class = num_class
    self.model_name = model_name
    
  
  def create_model(self):
    if self.model_name == "efficientnet-b4":
      model = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 )
      for params in model.parameters():
        params.requires_grad = False
      model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_class)

    elif self.model_name == "mobilenet":
      model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.IMAGENET1K_V1)
      for params in model.parameters():
        params.requires_grad = False
      model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_class)

    else:
      model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
      for param in model.parameters():
          param.requires_grad = False
      model.classifier[6] = nn.Linear(4096,self.num_class)
      
    return model
