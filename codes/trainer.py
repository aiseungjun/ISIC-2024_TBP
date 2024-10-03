import time
import copy
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from config import CONFIG
from utility import binary_auroc

def criterion(outputs, targets):
    pos_weight = torch.tensor([CONFIG["p:n_ratio"]], device=outputs.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, targets)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    running_loss = 0.0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device)
        metadata = data['metadata'].to(device)
        labels = data['label'].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images, metadata).squeeze()
        loss = criterion(outputs, labels)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
        
        running_loss += (loss.item() * batch_size)
        bar.set_postfix(Epoch=epoch, Train_Loss=loss.item() * batch_size, LR=optimizer.param_groups[0]['lr'])

    if scheduler is not None: # this is when T_max=n_epochs
        scheduler.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        metadata = data['metadata'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.float)
        batch_size = images.size(0)

        outputs = model(images, metadata).squeeze()
        loss = criterion(outputs, labels)
        running_loss += (loss.item() * batch_size)

        all_outputs.append(outputs)
        all_targets.append(labels)
        bar.set_postfix(Epoch=epoch, Valid_Loss=loss.item() * batch_size)

    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    epoch_auroc = binary_auroc(input=all_outputs, target=all_targets).item()
    return epoch_loss, epoch_auroc


def trainer(train_loader, valid_loader, model, optimizer, scheduler, num_epochs=CONFIG['n_epochs'], device=CONFIG['device']):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        # gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, dataloader=train_loader, device=device, epoch=epoch)
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, valid_loader, device=device, epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['lr'].append(scheduler.get_lr()[0])
        
        # deep copy the model
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'test_model.pth')
            #PATH = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            #torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved!")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)    
    return model, history