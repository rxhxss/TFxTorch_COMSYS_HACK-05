# TFXTorch_COMSYS_HACK-05
Project repository for Comsys Hackathon 5, submission by team TFXTorch


# [TFxTorch - Gender and Face Recognition Models] - Hackathon Submission🚀

## 📌 Overview
A robust **face recognition** and **gender classification** system designed to perform in challenging environments (Fogg, Dust, Low Contrast, etc). Built with Python, PyTorch and TensorFlow.

## 🏆 Team Members
- **Shataayu Mohanty** *(Lead)* 🧠
- **Ruchita Sengupta** *(Co-Lead)* 🔍

## 🎯 Features
- 🌟 Accurate classification in difficult conditions
- ⚡ Optimized for edge devices
- 🔄 Adaptive to various visual approximations (Glow, Fog, etc.)

## 🛠️ Installation

Installation process for [Task_A](./Task_A):
* Check Requirements.txt
* Run this code in notebook:
 ```
# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision
import sys

from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import gdown
import pathlib
import requests
import zipfile
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Try to import the Task_A directory, download it from GitHub if it doesn't work
try:
    from Task_A import data_setup, engine,model_builder,evaluation_metrics,utils,load_model_weights
except:
    # Get the scripts
    print("[INFO] Couldn't find Task_A scripts... downloading them from GitHub.")
    !git clone https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05.git
    !mv TFxTorch_COMSYS_HACK-05/Task_A .
    !rm -rf TFxTorch_COMSYS_HACK-05

# Add the Task_A directory to the system path
sys.path.append('./Task_A')

from Task_A import data_setup, engine,model_builder,evaluation_metrics,utils,load_model_weights
print("[INFO] All scripts succesfully imported...")
```
---OR---
* Run this google colab notebook:
  [https://colab.research.google.com/drive/19B_VGJmK5iIU79VyTt-GhgE0_K66zwuG?usp=sharing](https://colab.research.google.com/drive/19B_VGJmK5iIU79VyTt-GhgE0_K66zwuG?usp=sharing)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Installation process for [Task_B](./Task_B):
* Check Requirements.txt
* Run this code in notebook:
```
# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision
import sys

from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import gdown
import pathlib
import requests
import zipfile
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Try to import the Task_B directory, download it from GitHub if it doesn't work
try:
    from Task_B import datasetup, model_creation1, loss_fn, train, inference_stage,utils_Task_B,load_model_weights_Task_B
except:
    # Get the scripts
    print("[INFO] Couldn't find Task_B scripts... downloading them from GitHub.")
    !git clone https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05.git
    !mv TFxTorch_COMSYS_HACK-05/Task_B .
    !rm -rf TFxTorch_COMSYS_HACK-05

# Add the Task_A directory to the system path
sys.path.append('./Task_B')

from Task_B import datasetup, model_creation1, loss_fn, train, inference_stage,utils_Task_B,load_model_weights_Task_B
print("[INFO] All scripts succesfully imported...")


```
---OR---
* Run this google colab notebook:
  https://colab.research.google.com/drive/1j_vhR1zGUIlH_hftBPnVafrmqO3wB2We?usp=sharing
  
## 📂 Project Structure

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Task_A | 91.47% | 92.52% | 97.38% | 94.89% |
| Task_B | 92.86% | 92.86% | 100% | 96.30% |
