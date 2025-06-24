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
   pip install torch torchvision pandas numpy
   python final_train.py
   ```
2. **To Load the Model**:
   ```
   from Task_A import load_model_weights
   load_model_weights.load_model(final_train.model, final_train.model_save_path, final_train.device)
   ```
