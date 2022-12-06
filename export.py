import onnx
import torch
from models.ResNet import ResNetModel




def main():
    n_classes = 3
    model = ResNetModel(n_classes, weight_pretrain=False, freeze_backbone=False)
    ckpt_path = "./ckpt/epoch_4_resnet101/best.pt"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.eval()
    input = torch.randn(1, 3, 232, 232, requires_grad=True)
    ONNX_FILE_PATH = './ckpt/resnet101.onnx'

    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True)

    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    main()