# RealWaste Image Classification with VGG16 (TensorFlow/Keras)

This repository contains a step-by-step notebook for training a VGG16-based image classifier on a RealWaste dataset. The README walks through each code block so readers can follow and reproduce the workflow end-to-end.


## Step-by-Step Code Walkthrough

> This section mirrors the notebook and explains what each cell does so newcomers can run and understand the code.

### 1) Environment & Imports
Load core libraries (TensorFlow/Keras, NumPy, Matplotlib, etc.).
<details><summary>Imports detected</summary>

```python
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
```
</details>

### 2) Configuration & Hyperparameters
Define image size, batch size, number of epochs, optimizer and learning rate.
Detected:
- **Image size**: —
- **Batch size**: 1
- **Epochs**: —
- **Optimizer**: optim.Adam
- **Learning rate**: —
- **Loss**: —
- **Metrics**: —
- **Pretrained weights**: —
- **Callbacks**: —
- **Train dir**: —
- **Val dir**: —
- **Test dir**: —
- **Model save path**: —

### 3) Data Paths & Directory Structure
Point to the dataset folders (train/val/test). Make sure your folders are laid out like:
```
dataset/
  train/
    class_a/
    class_b/
    ...
  val/
    class_a/
    class_b/
    ...
  test/
    class_a/
    class_b/
    ...
```
Update the notebook variables to your paths:
```python
train_dir = "<path-to-train>"
val_dir   = "<path-to-val>"
test_dir  = "<path-to-test>"
```

### 4) Data Loading & Augmentation
Create training/validation generators or `tf.data` pipelines with optional augmentation (e.g., flips/rotations).

Common patterns used:
- `ImageDataGenerator` with `rescale=1./255` (Keras) **or**
- `tf.keras.layers` augmentation layers such as `RandomFlip`, `RandomRotation`, `RandomZoom`.

### 5) Model: VGG16 Backbone + Custom Head
- Load `tf.keras.applications.VGG16` with `imagenet` weights and `include_top=False` to use it as a feature extractor.
- Add a classification head: `GlobalAveragePooling2D` / `Flatten` → dense layers → output layer with softmax.
- Optionally freeze base layers for transfer learning, then unfreeze selected blocks for fine-tuning.

### 6) Compile
Compile the model with:
- Loss: `categorical_crossentropy / sparse_categorical_crossentropy`
- Optimizer: `optim.Adam` (lr = `1e-4`)
- Metrics: `['accuracy']`

### 7) Callbacks
Use helpful callbacks: `EarlyStopping, ReduceLROnPlateau, ModelCheckpoint`.

### 8) Train
Fit the model using `model.fit(...)` with the specified epochs and data loaders.
Track train vs. val accuracy and loss.

### 9) Evaluate
Evaluate on the validation/test set. Optionally, produce a classification report and confusion matrix.

### 10) Interpretability (Optional)
Generate Grad-CAM heatmaps on sample images to visualize which regions influence the prediction.

### 11) Save & Export
Save weights/model to `models/vgg16_realwaste.h5` and (optionally) export to SavedModel for deployment.


## How to Run

1. **Clone** this repo and open the notebook.
2. **Install** dependencies:
   ```bash
   pip install -U tensorflow tensorflow-io-gcs-filesystem numpy matplotlib scikit-learn pandas opencv-python
   ```
3. **Place your dataset** and adjust the `train_dir`, `val_dir`, `test_dir` variables.
4. **Run cells in order**. Training settings can be tuned in the *Configuration & Hyperparameters* cell.
5. **Check outputs**: training curves, evaluation metrics, and the saved model at `models/vgg16_realwaste.h5`.


## Dataset

This notebook expects an image dataset organized by class folders under `train/`, `val/`, and `test/`.
- Replace with the RealWaste dataset link or your custom dataset.
- Ensure each class has at least a few dozen images to fine-tune VGG16 effectively.


## Results (Fill In)
Add your achieved accuracy, F1-score, and example predictions here. You can paste final training/validation metrics and a confusion matrix screenshot from the notebook.


## Repository Structure
```
.
├── realwaste-image-classification-vgg16.ipynb   # Main notebook (step-by-step)
├── README.md                                     # You're reading it
└── models/                                       # Saved models (created after training)
```


---
**Tip:** If you change hyperparameters or directory paths, update the corresponding sections above so readers can follow your exact setup.