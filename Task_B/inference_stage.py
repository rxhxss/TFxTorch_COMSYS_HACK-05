
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_embeddings(model, image_paths, transform):
    """
    Compute embeddings for a list of images using the provided model.

    Args:
        model (nn.Module): Pretrained embedding model (should have an embedding_net attribute)
        image_paths (list): List of paths to image files
        transform (torchvision.transforms): Transformations to apply to images before processing

    Returns:
        numpy.ndarray: Array of embeddings with shape (num_images, embedding_dim)

    Note:
        - Sets model to evaluation mode
        - Uses inference mode for efficient computation
        - Handles RGB conversion automatically
        - Returns numpy array for easier downstream processing
    """
    model.eval()  # Set model to evaluation mode

    embeddings = []

    with torch.inference_mode():  # Disable gradient tracking for efficiency
        for img_path in image_paths:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')  # Ensure RGB format
            img = transform(img).unsqueeze(0)  # Add batch dimension

            # Compute embedding
            embedding = model.embedding_net(img)  # Forward pass through embedding network
            embeddings.append(embedding.squeeze().numpy())  # Convert to numpy and remove batch dim

    return np.array(embeddings)  # Convert list to numpy array


def reference_paths_labels(test_folder_path):
    """
    Prepare reference image paths and labels for testing from a structured directory.

    Args:
        test_folder_path (str): Path to the directory containing test images
                               Expected structure: test_folder_path/identity/*.jpg

    Returns:
        tuple: (reference_image_paths, reference_labels) where:
            - reference_image_paths (list): List of full paths to all test images
            - reference_labels (list): Corresponding identity labels for each image

    Note:
        - Assumes directory structure where each subfolder represents an identity
        - Collects all images from all identity subfolders
        - Maintains correspondence between paths and labels
    """
    reference_image_paths = []
    reference_labels = []




    # Iterate through each identity folder
    for identity in os.listdir(test_folder_path):
        identity_dir = os.path.join(test_folder_path, identity)

        # Skip if not a directory
        if not os.path.isdir(identity_dir):
            continue

        # Path to the 'distortion' subfolder
        distortion_dir = os.path.join(identity_dir, "distortion")

        # Skip if 'distortion' folder doesn't exist
        if not os.path.exists(distortion_dir):
            continue

        # Collect all images from the 'distortion' folder
        for img_name in os.listdir(distortion_dir):
            img_path = os.path.join(distortion_dir, img_name)

            # Only add if it's a file (skip subdirectories)
            if os.path.isfile(img_path):
                reference_image_paths.append(img_path)
                reference_labels.append(identity)  # Use parent folder name as label

    return reference_image_paths, reference_labels


def match_faces(test_image_path, reference_embeddings, reference_labels, model, transform, threshold=0.7):
    """
    Compare a test face image against a database of reference embeddings to find the closest match.

    Args:
        test_image_path (str): Path to the test image file to be identified
        reference_embeddings (np.ndarray): Array of reference embeddings with shape (n_samples, embedding_dim)
        reference_labels (list): List of labels corresponding to the reference embeddings
        model (nn.Module): Pretrained model for generating face embeddings
        transform (torchvision.transforms): Image transformations to apply before processing
        threshold (float, optional): Distance threshold for positive match. Default: 0.7

    Returns:
        tuple: A tuple containing:
            - match_result (int): 1 if positive match (distance < threshold), 0 otherwise
            - best_match_label (str): Label of the closest matching reference face

    Note:
        - Uses L2 distance (Euclidean distance) between embeddings
        - The threshold value is crucial and should be tuned for your specific application
        - Lower thresholds are more strict (fewer false positives but more false negatives)
        - Higher thresholds are more lenient (more false positives but fewer false negatives)
    """
    # Compute embedding for test image
    test_embedding = compute_embeddings(model, [test_image_path], transform)[0]  # [0] to get single embedding

    # Calculate pairwise Euclidean distances between test and all reference embeddings
    distances = np.linalg.norm(reference_embeddings - test_embedding, axis=1)

    # Find closest match in reference database
    min_distance = np.min(distances)  # Get smallest distance
    best_match_idx = np.argmin(distances)  # Get index of smallest distance
    best_match_label = reference_labels[best_match_idx]  # Get label of best match

    # Determine if distance is below acceptance threshold
    return (1 if min_distance < threshold else 0), best_match_label
def evaluation_metrics(model, test_folder_path, reference_embeddings, reference_labels, transform, threshold=0.7):
    print("[INFO]: Computing Evaluation Metrics for our model and embeddings....")
    y_true = []
    y_pred = []
    
    for identity in os.listdir(test_folder_path):
        identity_dir = os.path.join(test_folder_path, identity)
        if not os.path.isdir(identity_dir):
            continue
            
        # Process images directly in the identity folder
        for img_name in os.listdir(identity_dir):
            img_path = os.path.join(identity_dir, img_name)
            if os.path.isfile(img_path) :
                m, predicted_id = match_faces(img_path, reference_embeddings, reference_labels, model, transform, threshold)
                y_true.append(identity)
                y_pred.append(predicted_id)
        
        # Process images in the distortion subfolder if it exists
        distortion_dir = os.path.join(identity_dir, 'distortion')
        if os.path.exists(distortion_dir) and os.path.isdir(distortion_dir):
            for img_name in os.listdir(distortion_dir):
                img_path = os.path.join(distortion_dir, img_name)
                if os.path.isfile(img_path):
                    m, predicted_id = match_faces(img_path, reference_embeddings, reference_labels, model, transform, threshold)
                    y_true.append(identity)
                    y_pred.append(predicted_id)
    
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average="macro")
    recall = recall_score(y_true,y_pred,average="macro")
    f1 = f1_score(y_true,y_pred,average="macro")

    return {
        "Top-1 Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro Averaged F1 Score": f1,
        "Threshold": threshold
    }

def print_metrics(metrics: dict):
    """
    Prints the evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print(f"Accuracy: {metrics['Top-1 Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-score: {metrics['Macro Averaged F1 Score']:.4f}")    
