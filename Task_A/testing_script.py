import torch
import torchvision
import model_builder,evaluation_metrics# import your model class
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

