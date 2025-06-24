import os
import shutil
from torchvision import transforms



def flatten_person_folders(input_root,output_root,split):
        # Create output directories
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

    """Merges original + distorted images for each person in train/val."""
    input_dir = os.path.join(input_root, split)
    output_dir = os.path.join(output_root, split)

    for person in os.listdir(input_dir):
        person_input = os.path.join(input_dir, person)
        person_output = os.path.join(output_dir, person)
        os.makedirs(person_output, exist_ok=True)

        # Copy original images
        for img in os.listdir(person_input):
            if img == "distortion":
                continue  # Skip the distorted folder (handled next)
            src = os.path.join(person_input, img)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(person_output, img))

        # Copy distorted images (if folder exists)
        distorted_src = os.path.join(person_input, "distortion")
        if os.path.exists(distorted_src):
            for img in os.listdir(distorted_src):
                src = os.path.join(distorted_src, img)
                dst = os.path.join(person_output, f"distorted_{img}")  # Optional: Add prefix
                shutil.copy(src, dst)


def create_transforms():
  transform = transforms.Compose([
      transforms.Resize((160, 160)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])
  return transform
