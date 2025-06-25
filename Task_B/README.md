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
| **testing_script_Task_B.py**        | 🧪Test script/code to evaluate the model on test data                      |
| *Task_B_face_recognition_weights.pth**        | 💾 .pth file saving our model pretrained weights                       |

---
1. **Install dependencies**:
   ```
   pip install torch torchvision pandas numpy sklearn 
   ```
2. **Code To Load the Model Weights in Readable Format**:
   ```
   import torch
   import torchvision
   #Make sure all the scripts for Task_B are downloaded as per the installation process of Task_B in the ReadME file of the repo
   from Task_B import model_creation1,load_model_weights_Task_B   # import your model class and evaluation script
   
   
   device="cpu"
   
   # Initialize your model architecture
   embedding_net,model=model_creation1.model_instances(model_name='efficientnet_b0', embedding_size=128)
   
   !wget https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05/raw/refs/heads/main/Task_B/Task_B_face_recognition_weights.pth -O Task_B_model.pth
   
   #Loading our model weights
   load_model_weights_Task_B.load_model(model,"Task_B_model.pth",device)
3. **Code To Test The Model**:
   ```
   
   #Copy the code from testing_script_Task_B.py and pass in the test directory in it to test the model or use this code
   import torch
   import torchvision
   #Make sure all the scripts are downloaded as per the installation process in the ReadME file of the repo
   from Task_B import model_creation1,inference_stage   # import your model class and evaluation script
   
   
   device="cpu"
   
   # Initialize your model architecture
   embedding_net,model=model_creation1.model_instances(model_name='efficientnet_b0', embedding_size=128)
   transform=datasetup.create_transforms()
   
   
   
   # Load the saved state dict
   
   !wget https://github.com/ShataayuM/TFxTorch_COMSYS_HACK-05/raw/refs/heads/main/Task_B/Task_B_face_recognition_weights.pth -O Task_B_model.pth
   
   model.load_state_dict(torch.load("Task_B_model.pth",weights_only=False))
   
   #Setting the model to evaluation
   model.eval()
   #Calculating the embeddings of distorted images in the test set
   test_data_path=""   #Pass in the test data folder
   test_img_paths,test_labels=inference_stage.reference_paths_labels(test_data_path) #For test_data_path pass in the test folder path
   test_embeddings=inference_stage.compute_embeddings(model,test_img_paths,transform)
   
   #Evaluating our model on the given test data
   metrics=inference_stage.evaluation_metrics(model,test_data_path,test_embeddings,test_labels,transform)
   inference_stage.print_metrics(metrics)
   
   ```
OR
###  🚀Test the data using this notebook (installation included)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lgfIi9BsN47DYFf6nhHA9FmdQfogCtpK?usp=drive_link)

  ---
4. **Model Architecture**:
   **Siamese Network** with **EfficientNetB0** backbone 🔄, using **Triplet Loss** (margin = 1.0)  for metric learning, optimized with **Adam** ⚡ (lr=1e-2). Features are L2-normalized before distance computation in the     embedding space.
   *(Input: 160x160 RGB images | Output: 128-dim embeddings)*

  ---
