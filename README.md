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
* Check [Requirements.txt](./Requirements.txt)
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
  [![Open In Colab](https://colab.research.google.com/drive/19B_VGJmK5iIU79VyTt-GhgE0_K66zwuG?usp=sharing)](https://colab.research.google.com/drive/19B_VGJmK5iIU79VyTt-GhgE0_K66zwuG?usp=sharing)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Installation process for [Task_B](./Task_B):
* Check [Requirements.txt](./Requirements.txt)
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

# Add the Task_B directory to the system path
sys.path.append('./Task_B')

from Task_B import datasetup, model_creation1, loss_fn, train, inference_stage,utils_Task_B,load_model_weights_Task_B
print("[INFO] All scripts succesfully imported...")


```
---OR---
* Run this google colab notebook:
  [![Open In Colab](https://colab.research.google.com/drive/1j_vhR1zGUIlH_hftBPnVafrmqO3wB2We?usp=sharing)](https://colab.research.google.com/drive/1j_vhR1zGUIlH_hftBPnVafrmqO3wB2We?usp=sharing)
  
# 📂 Project Structure
  ├── LICENSE
├── README.md
├── Technical Summary-Comsys Hackathon.pdf

├── Task_A/
│ ├── README.md
│ ├── Task_A_gender_classification_weights.pth
│ ├── data_setup.py
│ ├── engine.py
│ ├── evaluation_metrics.py
│ ├── final_train.py
│ ├── load_model_weights.py
│ ├── model_builder.py
│ ├── testing_script.py
│ └── utils.py

├── Task_B/
│ ├── README.md
│ ├── Task_B_face_recognition_weights.pth
│ ├── datasetup.py
│ ├── final_training.py
│ ├── inference_stage.py
│ ├── load_model_weights_Task_B.py
│ ├── loss_fn.py
│ ├── model_creation1.py
│ ├── testing_script_Task_B.py
│ ├── train.py
│ └── utils_Task_B.py


## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Task_A | 92.18% | 95.06% | 95.34% | 95.30% |
| Task_B | 92.86% | 92.86% | 100% | 96.30% |
