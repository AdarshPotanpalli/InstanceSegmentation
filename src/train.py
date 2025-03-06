# imports
from engine import train_step, val_step
from model_builder import create_model
from data_setup import create_dataloader, CocoCustomDataset, custom_augment
from timeit import default_timer as timer
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from pathlib import Path
from pycocotools.coco import COCO
import gc
import torch

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Hyper params
NUM_EPOCHS = 10

# train, val, annotations directory paths
dataset_path = Path('../../data/COCO') # <--- make changes here ------
train_dir = dataset_path / 'train2017'
val_dir = dataset_path / 'val2017'
train_annotation_file = dataset_path/ 'annotations' / 'instances_train2017.json'
val_annotation_file = dataset_path/ 'annotations' / 'instances_val2017.json'

# Coco annotations
train_annotation_coco = COCO(train_annotation_file)
val_annotation_coco = COCO(val_annotation_file)

cat_nms = [
    # People & Accessories
    'person', 'backpack', 'handbag', 'suitcase',

    # Vehicles
    'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',

    # Furniture
    'bench', 'chair'
]

# Datasets
train_dataset = CocoCustomDataset(root_dir= train_dir,
                                  categories= cat_nms,
                                  coco_annotations= train_annotation_coco,
                                  transforms= custom_augment)

val_dataset = CocoCustomDataset(root_dir= val_dir,
                                categories= cat_nms,
                                coco_annotations= val_annotation_coco,
                                transforms= None)

# Dataloaders
train_dataloader = create_dataloader(dataset= train_dataset,
                                     shuffle= True)

val_dataloader = create_dataloader(dataset= val_dataset,
                                   shuffle = False)

# Model
model = create_model(cat_nms= cat_nms)

# optimizer
optimizer = torch.optim.SGD(params = model.parameters(),
                            lr = 0.005,
                            momentum= 0.9,
                            weight_decay= 0.0005)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=5,
                                               gamma = 0.1)

# results dict
results_dict = {
    'train_loss' : []
}

# set seed
torch.manual_seed(42)


start_time = timer()
model.to(device)
for epoch in tqdm(range(NUM_EPOCHS)):
  
  train_loss = train_step(model = model,
                          train_dataloader = train_dataloader,
                          optimizer = optimizer,
                          lr_scheduler = lr_scheduler,
                          epoch = epoch,
                          device = device)


  results_dict['train_loss'].append(train_loss)
  gc.collect()

end_time = timer()

print(f'[INFO] Total time taken: {end_time-start_time:.3f}')

# saving model
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents= True, exist_ok= True)
torch.save(obj= model.state_dict(), f = MODEL_PATH/'maskrcnn_12classes.pth')
