from torchvision import datasets, models
from torch.utils.data import Dataset

class CashewDataset(Dataset):
    def __init__(self, dir_path,  data_transforms):
        self.dir_path = dir_path
        self.images = datasets.ImageFolder(
                        dir_path, transform=data_transforms
                    )
        self.label2id = self.images.class_to_idx

    
    def __getitem__(self, index):
        image, class_id = self.images[index] 
        return image, class_id
    
    def __len__(self):
        return len(self.images)


def get_data_transforms(ModelAndWeights, model_name, weight_name):
    try:
        return ModelAndWeights[model_name][weight_name].transforms()
    except:
        raise Exception("Can not find pretrained weight !!!")



ModelAndWeights = {

    # AlexNet
    "alexnet": {
        "IMAGENET1K_V1" : models.AlexNet_Weights.IMAGENET1K_V1
    },
    
    # Resnet
    "resnet18": {
        "IMAGENET1K_V1": models.ResNet18_Weights.IMAGENET1K_V1
    },
    "resnet34": {
        "IMAGENET1K_V1": models.ResNet34_Weights.IMAGENET1K_V1
    },
    "resnet50": {
        "IMAGENET1K_V1": models.ResNet50_Weights.IMAGENET1K_V1,
        "IMAGENET1K_V2": models.ResNet50_Weights.IMAGENET1K_V2
    },
    "resnet101": {
        "IMAGENET1K_V1": models.ResNet101_Weights.IMAGENET1K_V1,
        "IMAGENET1K_V2": models.ResNet101_Weights.IMAGENET1K_V2
    },
    "resnet152": {
        "IMAGENET1K_V1": models.ResNet152_Weights.IMAGENET1K_V1,
        "IMAGENET1K_V2": models.ResNet152_Weights.IMAGENET1K_V2
    },

    # DenseNet
    "densenet121": {
        "IMAGENET1K_V1": models.DenseNet121_Weights.IMAGENET1K_V1
    }, 
    "densenet161": {
        "IMAGENET1K_V1": models.DenseNet161_Weights.IMAGENET1K_V1
    },
    "densenet169": {
        "IMAGENET1K_V1": models.DenseNet169_Weights.IMAGENET1K_V1
    },
    "densenet201": {
        "IMAGENET1K_V1": models.DenseNet201_Weights.IMAGENET1K_V1
    },

    #InceptionV3Net
    "inception_v3": {
        "IMAGENET1K_V1": models.Inception_V3_Weights.IMAGENET1K_V1
    },

    #VGGNet
    "vgg11":{
        "IMAGENET1K_V1": models.VGG11_Weights.IMAGENET1K_V1
    },
    "vgg11_bn":{
        "IMAGENET1K_V1": models.VGG11_BN_Weights.IMAGENET1K_V1
    },
    "vgg13":{
        "IMAGENET1K_V1": models.VGG13_Weights.IMAGENET1K_V1
    },
    "vgg13_bn":{
        "IMAGENET1K_V1": models.VGG13_BN_Weights.IMAGENET1K_V1
    },
    "vgg16":{
        "IMAGENET1K_V1": models.VGG16_Weights.IMAGENET1K_V1
    },
    "vgg16_bn":{
        "IMAGENET1K_V1": models.VGG16_BN_Weights.IMAGENET1K_V1
    },
    "vgg19":{
        "IMAGENET1K_V1": models.VGG19_Weights.IMAGENET1K_V1
    },
    "vgg19_bn":{
        "IMAGENET1K_V1": models.VGG19_BN_Weights.IMAGENET1K_V1
    },
}


