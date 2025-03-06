# Imports
import cv2
import random
import PIL
from PIL import Image
from torchvision import transforms, ops
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torch
import numpy as np
from pathlib import Path

# Hyperparams
IOU_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.7

# category names
cat_nms = [
    # People & Accessories
    'person', 'backpack', 'handbag', 'suitcase',
    # Vehicles
    'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    # Furniture
    'bench', 'chair'
]

# label id to label string mapping
id_2_label_dict = {1: 'person',2: 'bicycle',3: 'car',4: 'motorcycle',5: 'bus',6: 'train',
                  7: 'truck',8: 'bench',9: 'backpack',10: 'handbag',11: 'suitcase',12: 'chair'}

# results dict initialization
results = {'masks':[],'boxes':[],'scores':[],'labels':[]}

# Load model
def load_maskrcnn_model(path: str):
  """
  Load Mask R-CNN model from path

  Inputs:
    path: model (.pth) file path as string
  """
  model_loaded = maskrcnn_resnet50_fpn_v2(num_classes = len(cat_nms)+1)
  model_loaded.load_state_dict(torch.load(path, map_location= torch.device('cpu'))) # <----- Make changes here ------
  return model_loaded

def darken_color(color, darken_factor):
  """Darkens the color pallete(RGB) by darken_factor
  
  Inputs:
    color: color in RGB format
    darken_factor: factor to darken the color
  
  Output:
    darkened color in RGB format
  """

  return tuple(int(c*darken_factor) for c in color)

def add_masks_boxes(img_tensor, masks, boxes, labels, scores):
  """
  Applies different colors to different instances of the mask
  Draws bounding boxes with labels
  Returns the image with masks, bboxes, labels

  Inputs:
    img_tensor: image tensor
    masks: masks tensor
    boxes: bounding boxes tensor
    labels: labels tensor
    scores: scores tensor
  
  Output:
    image with masks, bboxes, labels
  """

  random.seed(100)
  masks = (masks > 0.5).squeeze(1).numpy()  # Shape: (N, H, W)
  image = (img_tensor.permute(1,2,0).numpy()*255).astype(np.uint8) # image tensor to array

  # Create color palette
  num_instances = len(boxes)
  colors = [tuple(random.choices(range(256), k=3)) for _ in range(num_instances)]

  # Overlay masks
  H,W, _ = image.shape
  font_scale = H/1000
  font_thickness = max(2, int(H/600))
  overlay = image.copy()

  for i, mask in enumerate(masks):
      color = colors[i]
      colored_mask = np.zeros_like(image, dtype=np.uint8)
      for c in range(3):  # Apply color to each channel
          colored_mask[:, :, c] = mask * color[c]

      overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.8, 0)  # Blend with transparency

      # Draw bounding box
      x1, y1, x2, y2 = map(int, boxes[i].tolist())
      cv2.rectangle(overlay, (x1, y1), (x2, y2), darken_color(color, 0.8), thickness= font_thickness)

      #add label text
      label_text = f"{id_2_label_dict[labels[i].item()]} ({scores[i]:.3f})"
      cv2.putText(overlay, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, darken_color(color, 0.6), font_thickness)

  # return the image
  return overlay


def visualize_instance_segmentation(image: PIL.Image):

  """
  Takes in a PIL image,
  performs forward pass through model to get predictions,
  applies post processing on predictions
  returns the image with masks, bboxes, labels

  Inputs:
    image: PIL image
  
  Output:
    image with masks, bboxes, labels
  """

  # 2. Transform the image and get the prediction
  img_tensor = transforms.ToTensor()(image)
  model_loaded.eval()
  with torch.inference_mode():
    pred = model_loaded(img_tensor.unsqueeze(0))

  # 3. Keep the predictions above threshold score
  indices_with_thresh_scores = pred[0]['scores'] > SCORE_THRESHOLD # indices with scores > SCORE_THRESHOLD
  results = {k:v[indices_with_thresh_scores] for k,v in pred[0].items()}

  # 4. Apply non-maximum supression on the masks based on IOU Threshold
  keep_indices = ops.nms(results['boxes'], results['scores'], IOU_THRESHOLD) # if 2 boxes have iou > IOU_THRESHOLD, one with lower score will be supressed
  results = {k: v[keep_indices] for k, v in results.items()}

  # 5. Plot instances, bboxes and labels on top of image
  result_image = add_masks_boxes(img_tensor, results['masks'], results['boxes'], results['labels'], results['scores'])
  return result_image

