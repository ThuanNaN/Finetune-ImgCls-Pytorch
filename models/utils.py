
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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