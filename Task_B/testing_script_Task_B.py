#Make sure all the scripts are downloaded
#Copy the code from testing_script.py and pass in the test directory in it to test the model
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
test_data_path="/content/data/Comys_Hackathon5/Task_B/val"   #Pass in the test data folder
test_img_paths,test_labels=inference_stage.reference_paths_labels(test_data_path) #For test_data_path pass in the test folder path
test_embeddings=inference_stage.compute_embeddings(model,test_img_paths,transform)

#Evaluating our model on the given test data
metrics=inference_stage.evaluation_metrics(model,test_data_path,test_embeddings,test_labels,transform)
inference_stage.print_metrics(metrics)
