import time
import copy
import os
import torch 
import logging
import numpy as np
from tqdm import tqdm
from utils.common import colorstr, save_ckpt, save_result, get_metrics

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")


def train_model(model, dataloaders, criterion, optimizer, opt):
    device, num_epochs, num_cls = opt.device, opt.n_epochs, opt.n_classes

    metrics = get_metrics(num_cls=num_cls, device=device, average = None)

    PATH_SAVE = os.path.join(opt.check_points, opt.ckpt_name)
    if not os.path.exists(PATH_SAVE):
        os.mkdir(PATH_SAVE)


    since = time.time()
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    result_file = os.path.join(PATH_SAVE, "result.txt")
    with open(result_file, "w") as f:
        f.write("\t\t\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-Score\n")

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Validation:','gpu_mem', 'loss', 'acc'))
                model.eval()

            running_items = 0
            running_loss = 0.0
            running_corrects = 0
            list_preds = []
            list_targets = []

            with tqdm(dataloaders[phase],
                total=len(dataloaders[phase]),
                bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                unit='batch') as _phase:

                for inputs, labels in _phase:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    list_targets.append(labels.data)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        list_preds.append(preds)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_items += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / running_items
                    epoch_acc = running_corrects.float() / running_items

                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                    desc = ('%35s' + '%15.6g' * 2) % (mem, epoch_loss, epoch_acc)
                    _phase.set_description_str(desc)

            epoch_preds = torch.cat([x for x in list_preds], dim=0)
            epoch_targets = torch.cat([x for x in list_targets], dim=0)

            save_result(epoch, epoch_preds, epoch_targets, metrics, result_file)

            save_ckpt(model, optimizer, PATH_SAVE, "epoch_{}.pt".format(epoch))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_ckpt(model, optimizer, PATH_SAVE, "best.pt")
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s with {} epochs'.format(time_elapsed // 60, time_elapsed % 60, num_epochs))
    print('Best val Acc: {:4f}'.format(best_acc))
    save_ckpt(model, optimizer, PATH_SAVE, "last.pt")
    model.load_state_dict(best_model_wts)
    
    #plot and save

    return model

