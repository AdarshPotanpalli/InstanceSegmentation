# Instance Segmentation using MASK R-CNN 🎭

Instance segmentation **labels objects at the pixel level** while distinguishing between different instances of the same category.

**Mask R-CNN** with a **ResNet-50 backbone** is utilized to perform instance segmentation.

### ✅ **Supported Object Categories:**
* *People:* `person`
* *Bags & Accessories:* `backpack`, `handbag`, `suitcase`
* *Vehicles:* `bicycle`, `car`, `motorcycle`, `bus`, `train`, `truck`
* *Furniture:* `bench`, `chair`
---
---
### **Overview**: 

`data_setup.py`:

* `CocoCustomDataset`: Custom Dataset Class for Instance Segmentation on COCO dataset 
* `custom_augment`: Applies augmentation on images and corresponding masks and bboxes
* `create_dataloader`: Creates Dataloader
---
`engine.py`:

* `train_step`: Performs 1 epoch of training and updates params
* `val_step`: Performs 1 epoch over val dataloader, gets val loss
---
`model_builder.py`:

* `create_model`: creates Mask R-CNN model, ready for training
---

`train.py`:

* Contains whole training setup, trains the model and saves the model
---

`utils.py`:

* `load_maskrcnn_model`: loads the trained model
* `visualize_instance_segmentation`: performs forward pass through loaded model, returns the image with predicted masks, labels and bboxes

---
---
### **Author**:

**Adarsh Potanpalli**  
Email: [p.adarsh.24072001@gmail.com](mailto:p.adarsh.24072001@gmail.com)
