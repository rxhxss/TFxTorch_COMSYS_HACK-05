# PyTorch Training Pipeline 🚀

A modular deep learning pipeline for training and inference. Structured for clarity and reproducibility.

---

## **File Structure & Workflow** 📂

| File | Purpose | Emoji |
|------|---------|-------|
| **1. Data Setup** | | 📊 |
| `datasetup.py` | Data loading, preprocessing, and augmentation. | 🔄 |
| **2. Model Creation** | | 🧠 |
| `model_creation1.py` | Defines the neural network architecture. | ⚙️ |
| **3. Loss Function** | | ⚖️ |
| `loss_fn.py` | Implements custom loss functions (Triplet Loss). | 📉 |
| **4. Training** | | 🏋️ |
| `train.py` | Main training script (epochs, validation, checkpointing). | 🔁 |
| `final_training.py` | Optimized/final training loop. | 🎯 |
| **5. Inference** | | 🔍 |
| `inference_stage.py` | Runs predictions on new data with trained models. Calculates reference embeddings and labels and matches it to test data| 🔮 |
| **Utilities** | | 🛠️ |
| `utils_Task_B.py` | Saves the model in a .pth file | 📝 |
| `load_model_weights_Task_B.py` | Loads pretrained weights into models. | ⬇️ |

---

🚀 Embedding Model with EfficientNet-B0 + Triplet Loss
This model is an embedding network designed for feature extraction and similarity learning, using:

Backbone: � EfficientNet-B0 (pretrained on ImageNet)

Projection Head: 🔧 Modified dense layer for optimal embedding space

Loss Function: 🔺 Triplet Loss (for triplet matching)

🏗 Model Architecture
🔍 Feature Extractor
Base Model: EfficientNet-B0 (fine-tuned)

Modified Layers:

Replaced the default classification head

Added a custom projection layer (e.g., 128-D embeddings)

Includes BatchNorm + Dropout for regularization

📉 Triplet Loss Training
Loss Type: Semi-hard or hard triplet mining

Margin: α = 1.0 (adjustable)

Distance Metric: Euclidean 


📦 Dependencies
Python 3.8+

PyTorch 2.0+

efficientnet_pytorch
