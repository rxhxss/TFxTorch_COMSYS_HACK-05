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


