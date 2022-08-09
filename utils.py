
import os 
import errno
import torch
from model import Net


def save_checkpoint(PATH, epoch, model, optimizer, loss, acc, device):

    if not os.path.exists(os.path.dirname(PATH)):
        try:
            os.makedirs(os.path.dirname(PATH))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    path_save = PATH + "/checkpoint-epoch"+str(epoch)+".pth"

    model.to('cpu') 

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        "acc": acc
    },
    path_save)

    model.to(device)



def load_checkpoint(PATH, num_class, device):


    if device == "gpu":
        checkpoint = torch.load(PATH, map_location='cuda:0')
    else:
        checkpoint = torch.load(PATH, map_location='cpu')

    epoch = checkpoint['epoch']  

    model = Net(num_class).create_model()
    model.load_state_dict = checkpoint['model_state_dict'] 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict = checkpoint['optimizer_state_dict'] 

    return epoch, model, optimizer


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