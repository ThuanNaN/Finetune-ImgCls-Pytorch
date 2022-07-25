import torch

from utils import load_checkpoint

dir_path = "./checkpoint/checkpoint-1.pth"
num_class = 30
device = "cuda"
epoch, model, optimizer = load_checkpoint(dir_path, num_class, device)

print(model)