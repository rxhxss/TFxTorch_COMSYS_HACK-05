# PyTorch Training Pipeline 🚀

A modular deep learning pipeline for training and inference. Structured for clarity and reproducibility.

---

## 📂 Project Structure

| File Name                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **datasetup.py**           | 🏗️ Script for dataset preparation and preprocessing                        |
| **final_training.py**      | 🏋️‍♂️ Main script for model training with final configurations              |
| **inference_stage.py**     | 🔍 Script for running inference/predictions with trained model              |
| **load_model_weights_Task_B.py** | ⚖️ Utility for loading pre-trained model weights                          |
| **loss_fn.py**            | 📉 Custom loss function implementation [Triplet Loss]                                     |
| **model_creation1.py**     | 🧠 Neural network model architecture definition                             |
| **train.py**               | 🚂 Basic training script (likely used for initial experiments)              |
| **utils_Task_B.py**        | 🧰 Function to save our models weights                       |


---
1. **Install dependencies**:
   ```
   pip install torch torchvision pandas numpy sklearn 
   ```
2. **Code To Load the Model**:
   ```
   # Pass in the model and model_save_path and device['cpu' or 'cuda']
   from Task_B import load_model_weights_Task_B
   load_model_weights_Task_B.load_model(final_training.model_1, final_training.model_save_path, device="cpu")
3. **Code To Test The Model**:
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
4. Model Architecture:
   **Siamese Network** with **EfficientNetB0** backbone 🔄, using **Triplet Loss** (margin = 1.0)  for metric learning, optimized with **Adam** ⚡ (lr=1e-2). Features are L2-normalized before distance computation in the     embedding space.
   *(Input: 160x160 RGB images | Output: 128-dim embeddings)*
