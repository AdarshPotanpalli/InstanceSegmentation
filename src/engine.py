# imports
import torch
from torch.amp import autocast, GradScaler
from torch import nn
import gc

# set seed
torch.manual_seed(42)
scaler = GradScaler('cuda')

def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler,
               epoch: int,
               device = 'cuda'):
  
  """
  Performs train step (1 epoch of training)

  Inputs:
    model: model to be trained
    train_dataloader: Dataloader for train dataset
    optimizer: optmizer for param update
    lr_scheduler: learning rate scheduler used
    epoch: Current epoch count
    device: device to train the model on (preferred 'cuda')

  Outputs:
    average loss per batch
  """
  model.train()
  total_loss = 0

  for batch_number, (images, target) in enumerate(train_dataloader):
    # images and targets to device
    images = list(image.to(device) for image in images)
    target = [{k: v.to(device) for k,v in t.items()} for t in target]

    # calculate the loss
    with autocast('cuda'): # automatic mixed precision
      loss_dict = model(images, target) # gives a dictionary of losses per key
      loss = sum(loss for loss in loss_dict.values())

    total_loss += loss.item()

    # optimzer zero grad
    optimizer.zero_grad()

    # loss backward
    scaler.scale(loss).backward() # scaling the losses

    # optimizer step
    scaler.step(optimizer) # scaling the optimizer
    scaler.update()

    if batch_number % 50 == 0:
      print(f'Batch No: {batch_number} | Loss in Batch : {loss:.3f}')

  print(f'\n[INFO] Epoch: {epoch+1} | Train Loss per batch: {total_loss/len(train_dataloader):.3f}\n')
  print('-'*50)
  lr_scheduler.step() # lr scheduler

  del images, target, loss_dict, loss  # delete unnecessary variables
  torch.cuda.empty_cache()  # free up GPU memory
  gc.collect()
  return total_loss/len(train_dataloader)


def val_step(model: nn.Module,
              val_dataloader: torch.utils.data.DataLoader,
              device = 'cuda'):

  """
  Performs validation step (1 epoch of validation)

  Inputs:
    model: model to be evaluate
    val_dataloader: Dataloader for val dataset
    device: device to evaluate val losses (preferred 'cuda')

  Outputs:
    average loss per batch
  
  """
  
  total_val_loss = 0
  for batch_number, (images, target) in enumerate(val_dataloader):
    # putting the images and targets on device
    images = list(image.to(device) for image in images)
    target = [{k: v.to(device) for k,v in t.items()} for t in target]

    with autocast('cuda'): # automatic mixed precision
      loss_dict = model(images, target)
      loss = sum(loss for loss in loss_dict.values())

    total_val_loss += loss.item()

    print(f'[INFO] Val Loss per batch: {total_val_loss/len(val_dataloader):.3f}')
    print('-'*50)


  del images, targets, loss_dict, loss  # delete unnecessary variables
  torch.cuda.empty_cache()  # free up GPU memory
  gc.collect()
  return total_val_loss/len(val_dataloader) # returning val loss
