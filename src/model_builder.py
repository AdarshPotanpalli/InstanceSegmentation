# pretrained model, weights and transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import List
import torch

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT

# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_model(cat_nms :List[str])-> maskrcnn_resnet50_fpn_v2:

  """
  Creates MASK RCNN model

  Inputs:
    cat_nms: list of categories for segmentations

  Outputs:
    model: Mask RCNN model
  """
  # load pretrained model
  model = maskrcnn_resnet50_fpn_v2(weights = weights).to(device)

  # freeze backbone
  for param in model.backbone.parameters():
    param.requires_grad = False

  # change the output channels in mask_predictor and box_predictor head
  model.roi_heads.box_predictor = FastRCNNPredictor(in_channels= 1024,
                                                    num_classes= len(cat_nms)+1)
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels= 256,
                                                     dim_reduced= 256,
                                                     num_classes= len(cat_nms)+1)
  return model
