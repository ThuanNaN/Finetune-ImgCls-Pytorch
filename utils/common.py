import os
import torch
import torchmetrics

def save_ckpt(model, optimizer, PATH, name_ckpt):
    path_save = os.path.join(PATH, "weights")
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    
    models_ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(models_ckpt,os.path.join(path_save, name_ckpt))


def load_ckpt(ckpt_path, model, optimizer):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer

def get_metrics(num_cls, device, average = None):

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_cls, average=average).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=num_cls, average=average).to(device)
    recall= torchmetrics.Recall(task="multiclass", num_classes=num_cls, average=average).to(device)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_cls, average=average).to(device)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def save_result(epoch, predicts, targets, metrics, result_file):
    
    precision = metrics['precision'](predicts, targets).detach().cpu().numpy()
    recall = metrics['recall'](predicts, targets).detach().cpu().numpy()
    accuracy = metrics['accuracy'](predicts, targets).detach().cpu().numpy()
    f1_score = metrics['f1_score'](predicts, targets).detach().cpu().numpy()

    with open(result_file, "a+") as f:
        f.write(f"Epoch: {epoch}\t{accuracy:.6f}\t{precision:.6f}\t{recall:.6f}\t{f1_score:.6f}\n\n")





def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

