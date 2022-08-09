import argparse
from dataset import CustomDataset
from model import Net
import torch
from utils import save_checkpoint, load_checkpoint, colorstr

from tqdm import tqdm

import logging
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger('CS-PyTorch')



def train_model(model, dataloaders, optimizer, start_epoch, N_EPOCHS, device, checkpoint_PATH):

    LOGGER.info(f"{colorstr('Optimizer:')} {model}")
    LOGGER.info(f"{colorstr('Optimizer:')} {optimizer}")

    criterion = torch.nn.CrossEntropyLoss()
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    

    for epoch in range(N_EPOCHS):
        LOGGER.info(colorstr(f'\nEpoch {start_epoch+epoch + 1}/{start_epoch + N_EPOCHS }:'))

        for phase in ["train", "valid"]:

            running_loss = 0
            running_accuracy = 0
            correct = 0
            total = 0

            if phase == "train":
                LOGGER.info(colorstr('black', 'bold', '%20s' + '%15s' * 3) % 
                                ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('black', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Validation:','gpu_mem', 'loss', 'acc'))
                model.eval()
            
            with tqdm(dataloaders[phase],
                total=len(dataloaders[phase]),
                bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                unit='batch') as _phase:

                for inputs, labels in _phase:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase =='train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                    running_loss += loss.item() / len(dataloaders[phase])
                    running_accuracy = 100.*correct/total

                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                    desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss, running_accuracy)
                    _phase.set_description_str(desc)

        #save_checkpoint
        save_epoch = start_epoch+epoch + 1
        save_checkpoint(checkpoint_PATH, epoch=save_epoch, model=model, optimizer=optimizer, loss=running_loss, acc= running_accuracy, device=device)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image classification with pytorch')
    parser.add_argument(
        "--load_checkpoint",  
        type=bool,
        nargs="?", 
        help="Load check point", 
        default= False
    )
    parser.add_argument(
        "--N_EPOCH",  
        type=int,
        nargs="?", 
        help="Num of epoch", 
        default= "2"
    )
    parser.add_argument(
        "--model_name", 
        type=str,  
        nargs="?",
        help="Choose model", 
        default="vgg16"
    )

    parser.add_argument(
        "--n_class", 
        type=int,  
        nargs="?",
        help="Number of classes", 
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    dir_path = "./data_dir"
    dataloaders, _ = CustomDataset(dir_path, batch_size = 64).load_dataset()

    checkpoint_dir = "./checkpoint"
    num_class = args.n_class
    if args.load_checkpoint:
        epoch, model, optimizer = load_checkpoint(dir_path, num_class, device)
        start_epoch = epoch

    else:
        start_epoch = 0
        model = Net(num_class = num_class, model_name=args.model_name).create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    N_EPOCHS = args.N_EPOCH
    model.to(device)
    train_model(model, dataloaders, optimizer, start_epoch, N_EPOCHS, device = device, checkpoint_PATH = checkpoint_dir)   
