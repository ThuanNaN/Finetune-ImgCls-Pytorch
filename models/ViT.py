import torch.nn as nn
from transformers import ViTForImageClassification

class ViT_model(nn.Module):
    def __init__(self, model_name, id2label, label2id):
        super(ViT_model, self).__init__()
        self.model_name = model_name
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            id2label = id2label,
            label2id = label2id
        )
    
    def forward(self, x):
        x = self.model(x)
        return x