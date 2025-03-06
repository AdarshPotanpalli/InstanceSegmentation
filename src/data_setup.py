import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import os
from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Tuple
import random
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# Hyperparams
BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = (256,256)

def custom_augment(img: torch.Tensor,
                   masks: torch.Tensor,
                   boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
  """
  Applies the augmentation on the dataset and mask

  Inputs:
    img: image tensor to apply augmentation on
    masks: masks corresponding to the image
    boxes: boxes corresponding to the image

  Outputs:
    img: augmented image tensor
    masks: augmented masks tensor
    boxes: augmented boxes tensor
  """
  # color jitter (random contrast, hue, saturation, brightness)
  img = transforms.ColorJitter(brightness= 0.2, contrast= 0.2, hue= 0.2, saturation= 0.2)(img)

  # resize the image
  orig_w, orig_h = img.shape[-1], img.shape[-2]
  img = F.resize(img, size= IMAGE_SIZE)
  if masks is not None and len(masks) >0:
    masks = F.resize(masks, size = IMAGE_SIZE)
  if boxes is not None:
    boxes = boxes.clone()
    boxes[:, [0,2]] = boxes[:,[0,2]]*IMAGE_SIZE[1]/orig_w
    boxes[:, [1,3]] = boxes[:,[1,3]]*IMAGE_SIZE[0]/orig_h

  # random horizontal flip
  if random.random() > 0.5:
    img, masks = F.hflip(img), F.hflip(masks)
    if boxes is not None:
      boxes = boxes.clone()
      boxes[:, [0,2]] = IMAGE_SIZE[1] - boxes[:, [2,0]]

  # discard the boxes and masks if the widths or heights are invalid
  widths = boxes[:,2] - boxes[:,0]
  heights = boxes[:,3] - boxes[:,1]
  valid_indices = (widths > 0) & (heights > 0)
  boxes = boxes[valid_indices]
  masks = masks[valid_indices]

  if len(boxes) == 0:
        print("Warning: No valid boxes after augmentation, skipping this sample.")
        return img, torch.empty(0), torch.empty(0)

  return img, masks, boxes

def cat_ids_remapped(cat_ids, annotations):

  """
  Remapping the category ids to labels

  Inputs:
    cat_ids: Category IDs
    annotations: COCO annotations

  Outputs:
    A dictionary mapping category IDs to labels
  
  """
  cat_names = [cat['name'] for cat in annotations.loadCats(cat_ids)]
  return {cat_id: [remapped_cat_id+1, cat_names[remapped_cat_id]] for remapped_cat_id, cat_id in enumerate(cat_ids)}

class CocoCustomDataset(Dataset):
  def __init__(self, root_dir: str, categories: list, coco_annotations: COCO, transforms= None):
    """
    Custom dataset class for COCO dataset

    Inputs:
      root_dir: root directory path
      categories: any subset of categories available in coco dataset
      coco_annotations: COCO annotations
      transforms: transforms to be applied on image and target/mask

    """
    self.root_dir = Path(root_dir)
    self.coco_annotations = coco_annotations
    self.cat_ids = coco_annotations.getCatIds(catNms= categories) # category ids
    self.cat_ids_mapping_dict = cat_ids_remapped(self.cat_ids, self.coco_annotations) # getting the category mappings
    self.img_ids = self.get_img_ids(self.cat_ids) # image ids
    self.img_meta_data = coco_annotations.loadImgs(self.img_ids) # loading all images' meta data corresponding to image ids
    self.image_paths = [self.root_dir/img['file_name'] for img in self.img_meta_data] # paths to all images
    self.transforms = transforms # transforms

  def get_img_ids(self, cat_ids):
    img_ids = []
    for cat_id in cat_ids:
      img_ids.extend(self.coco_annotations.getImgIds(catIds= cat_id)) # gets all images containing atleast one of the cat ids
    img_ids = list(set(img_ids)) # avoids repetations of image ids
    return img_ids

  def get_img_at_index(self, i):
    image = Image.open(self.image_paths[i])
    tf = transforms.ToTensor()
    img = tf(image)
    if img.shape[0] == 1: # B/W image
      img = img.repeat(3,1,1)
    return img # returns img tensor

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, i):

    annotation_ids = self.coco_annotations.getAnnIds(
        imgIds=self.img_meta_data[i]['id'],
        catIds=self.cat_ids,
        iscrowd=None
    ) # different annotations ids in image at index i

    # image
    img = self.get_img_at_index(i)

    # mask
    anns = self.coco_annotations.loadAnns(annotation_ids)
    # mask = torch.LongTensor(np.max(np.stack([coco_annotations.annToMask(ann) * ann["category_id"] for ann in anns]), axis=0)).unsqueeze(0) # segmentation mask

    masks = []
    boxes = []
    labels = []
    iscrowd = []

    for ann in anns:
      # list binary masks for all instances in an image
      mask = self.coco_annotations.annToMask(ann)
      masks.append(torch.tensor(mask, dtype=torch.uint8))

      # list bbox for all instances
      x,y,w,h = ann['bbox']
      boxes.append([x, y, x+w, y+h]) # x_min, y_min, x_max, y_max

      # list labels for all instances
      labels.append(self.cat_ids_mapping_dict[ann['category_id']][0])

      # store is crowd per annotation
      iscrowd.append(ann['iscrowd'])

    if masks:
      masks = torch.stack(masks) # shape [num_instances, H, W]
      boxes = torch.as_tensor(boxes, dtype=torch.float32) # shape [num_instances, 4]
      labels = torch.as_tensor(labels, dtype=torch.int64) # shape [num_instances]
      iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) # shape [num_instances]

    else:
      masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
      boxes = torch.zeros((0, 4), dtype=torch.float32)
      labels = torch.zeros((0,), dtype=torch.int64)
      iscrowd = torch.zeros((0,), dtype=torch.int64)

    if self.transforms:
      img, masks, boxes = self.transforms(img, masks, boxes)

    target = {
        'boxes': boxes,
        'labels': labels,
        'masks': masks,
        'image_id': torch.tensor([i]),
        'iscrowd': iscrowd
    }

    return img, target

def custom_collate_fn(batch):
  """Unpacks the batch in the form of list"""

  images, targets = zip(*batch)
  return list(images), list(targets)

def create_dataloader(dataset: torch.utils.data.Dataset,
                      shuffle: bool,
                      subset_fraction: float = 0.1, 
                      batch_size: int = BATCH_SIZE,
                      num_workers: int = NUM_WORKERS) -> torch.utils.data.DataLoader:

  """
  Creates dataloader from datasets of give batch size

  Inputs:
    dataset: train or val dataset created
    shuffle: if you want to shuffle the dataset or not
    subset_fraction: fraction of the dataset used to create the dataloader
    batch_size: batch size for dataloader
    num_workers: number of cpu workers to create dataloader

  Outputs:
    dataloader: dataloader created from the dataset
  
  """
  total_size = len(dataset)
  # Define the percentage of data you want to use
  subset_size = int(subset_fraction * total_size)
  # Generate a random list of indices to select 10% of the dataset
  indices = torch.randperm(total_size).tolist()[:subset_size]
  # Create a new dataset with just 10% of the data
  subset = Subset(dataset, indices)

  dataloader = DataLoader(dataset=subset,
                          batch_size= batch_size,
                          num_workers= num_workers,
                          collate_fn = custom_collate_fn,
                          pin_memory= False,
                          shuffle= shuffle)
  return dataloader
