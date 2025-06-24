import gdown
import pathlib
import requests
import zipfile
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from PIL import Image
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

def download_and_extract_dataset():
    # Dataset URL (replace with your actual Google Drive link)
    dataset_url = "https://drive.google.com/uc?id=11EI3g783_s15QPYlpGn43NH_xEruxq8R&confirm=t"
    output_zip = 'FACECOM.zip'
    extract_to = './data'

    # Download zip file
    gdown.download(dataset_url, output_zip,fuzzy=True,quiet=False)

    # Extract zip file
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Remove zip file
    os.remove(output_zip)

    print(f"Dataset extracted to {extract_to}")

# Call this function at the start of your training script
if not os.path.exists('./data/Comsys_Hackathon5'):
    download_and_extract_dataset()

#Creating a flattened dataset with all the images in distortion folder under identity folders
# Paths
input_root = "/content/data/Comys_Hackathon5/Task_B"       # Already split into train/test
output_root = "flattened_dataset/"     # Output folder

import datasetup, model_creation1, loss_fn, train, inference_stage,utils_Task_B,load_model_weights_Task_B
datasetup.flatten_person_folders(input_root, output_root,"train")
datasetup.flatten_person_folders(input_root, output_root,"val")
print("Flattening Completed.")

#Creating transforms
transform=datasetup.create_transforms()
# Create dataset and dataloader
train_dataset = datasets.ImageFolder('flattened_dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#Create model
embedding_net, model_1 = model_creation1.model_instances(model_name='efficientnet_b0', embedding_size=128)
#Create loss function and optimizer
criterion = loss_fn.loss_fn()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)

#Training the model
train.train_model(train_loader, model_1, criterion, optimizer, num_epochs=1)

#Inference phase
#Calculating embedding for only distorted images in each identity folder
val_img_paths,val_labels=inference_stage.reference_paths_labels("/content/data/Comys_Hackathon5/Task_B/val")
val_embeddings=inference_stage.compute_embeddings(model_1,val_img_paths,transform)



#Evaluation metrics
evaluation_results=inference_stage.evaluation_metrics(model_1, "flattened_dataset/val", val_embeddings, val_labels, transform, threshold=0.7)
inference_stage.print_metrics(evaluation_results)
mode_save_path=utils_Task_B.save_model(model_1,
               target_dir="models",
               model_name="Task_B_face_recognition_weights.pth")

