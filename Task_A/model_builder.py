
import torch
from torch import nn
import torchvision
torch.manual_seed(42)
torch.cuda.manual_seed(42)
def model_creation(num_classes,device):
  """
    Creates an MobileNetV2 model with a modified classifier head.

    Args:
        num_classes: Number of output classes for the modified classifier
        device: The device on which the model wil work
        """


  # Load MobileNetV2 with pretrained weights
  weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
  model = torchvision.models.mobilenet_v2(weights=weights).to(device)
  auto_transforms = weights.transforms()

  # (1) Freeze all layers first
  for param in model.parameters():
      param.requires_grad = False

  # (2) Unfreeze the last 10 layers
  total_layers = len(list(model.parameters()))
  layers_unfrozen = 0

  # Iterate in reverse (from last to first) and unfreeze 10 layers
  for param in reversed(list(model.parameters())):
      if layers_unfrozen >= 10:
          break
      param.requires_grad = True
      layers_unfrozen += 1
  model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),  # Keep dropout from original MobileNetV2
    torch.nn.Linear(in_features=model.last_channel, out_features=num_classes)).to(device)
  return auto_transforms,model
