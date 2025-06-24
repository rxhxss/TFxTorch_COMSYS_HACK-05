import torch

def load_model(model: torch.nn.Module,
               model_save_path: str,
               device):
    """
    Loads model weights from a .pth file into the given model.

    Args:
        model: PyTorch model architecture to load weights into
        model_save_path: Path to the saved .pth weights file
        device: The device to load the model onto
    """
    # Load saved weights into model
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Print all layer names and shapes
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
