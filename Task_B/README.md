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

1. **Code To Test The Model**:
   ```
   #Make sure all the scripts and data are downloaded and the final_training script has been run
   #Calculating the embeddings of distorted images in the test set
   from Task_B import inference_stage
   test_img_paths,test_labels=inference_stage.reference_paths_labels(test_data_path:str) #For test_data_path pass in the test folder path
   test_embeddings=inference_stage.compute_embeddings(final_training.model_1,test_img_paths,final_training.transform)
   test_image_path=""   #Pass in the test image path you want to check for match
   match,predicted_id=inference_stage.match_faces(test_image_path, test_embeddings,test_labels, final_training.model_1, final_train.transform, threshold=0.7)
   print(f"Match:{match}, Predicted Person:{predicted_id}")
   ```
