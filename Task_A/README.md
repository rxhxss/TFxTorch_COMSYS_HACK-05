# Task_A: Gender Classification 👥🔍

This folder contains all the necessary scripts and files for training and evaluating a **gender classification model** using PyTorch. Below is an overview of each file's purpose.

---

## 📂 File Structure

| File | Description |
|------|-------------|
| 🏋️ `Task_A_gender_classification_weights.pth` | Pre-trained model weights (PyTorch `.pth` format) |
| 🗃️ `data_setup.py` | Handles data loading, preprocessing, and dataset creation |
| 🚀 `engine.py` | Contains core training loop |
| 📊 `evaluation_metrics.py` | Implements evaluation metrics (e.g., accuracy, F1-score) |
| 🏗️ `final_train.py` | Main script to train the model (run this!) |
| ⚙️ `load_model_weights.py` | Utility to load saved model weights for inference |
| 🧠 `model_builder.py` | Defines the neural network architecture |
| 🛠️ `utils.py` | Saves the model to a .pth file |

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```
   pip install torch torchvision pandas numpy sklearn 
   ```
2. **Code To Load the Model Weights in Readable Format**:
   ```
   # Pass in the model and model_save_path and device['cpu' or 'cuda']
   NUM_CLASSES=2
   device="cuda" if torch.cuda.is_available() else "cpu"
   #Make sure all the scripts for Task_A are downloaded as per the installation process in the repo
   from Task_A import load_model_weights,model_builder

   #Creating model instance
   auto_transforms,model = model_builder.model_creation(NUM_CLASSES,device)
   
   # Load the saved state dict
   !wget https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05/raw/refs/heads/main/Task_A/Task_A_gender_classification_weights.pth -O Task_A_model.pth
   load_model_weights.load_model(model,"Task_A_model.pth", device)
   ```
3. **Code To Test The Model**:
   ```
   #Make sure all the scripts for Task_A are downloaded as per the installation process in the repo
   #Copy the code from testing_script.py and pass in the test directory in it to test the model
   import torch
   import torchvision
   from Task_A import model_builder,evaluation_metrics   # import your model class
   NUM_CLASSES=2
   BATCH_SIZE=32
   device="cuda" if torch.cuda.is_available() else "cpu"
   
   # Initialize your model architecture
   auto_transforms,model = model_builder.model_creation(NUM_CLASSES,device)
   
   
   # Load the saved state dict
   
   !wget https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05/raw/refs/heads/main/Task_A/Task_A_gender_classification_weights.pth -O Task_A_model.pth
   
   model.load_state_dict(torch.load("Task_A_model.pth",weights_only=False))
   #Pass the test folder path 
   test_dir=""
   
   
   # Set model to evaluation mode
   model.eval()
   metrics=evaluation_metrics.evaluate_model(model,test_dir,auto_transforms,BATCH_SIZE,device,NUM_CLASSES)
   evaluation_metrics.print_metrics(metrics)
   ```
   ## 🚀Run the test_script using this notebook (installation included)
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lQrnUYdtKuWNO71swIEYa3kiDspUsWG9?usp=sharing)

---
4. **Model Architecture & Training Details**
    
   🔹 Backbone: MobileNetV2 (pre-trained on ImageNet)

   🔹 Head: Custom classification head (2 output classes)
   
   🔹 Loss: CrossEntropyLoss (multi-class classification)
   
   🔹 Optimizer: Adam (lr=0.001)
   
   🔹 Regularization:  
   
     🔹  Dropout (p=0.2)
     
     🔹 Batch Normalization (for stability)
     
     🔹 Training: 🔄 Transfer Learning (frozen backbone → fine-tuned head)

---
