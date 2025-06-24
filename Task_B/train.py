import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from PIL import Image
import torch.nn.functional as F
from tqdm.auto import tqdm

def calculate_accuracy(anchor_emb, positive_emb, negative_emb, threshold=0.5):
    """
    Calculate accuracy based on whether positive pairs are closer than negative pairs
    """
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
    correct = (pos_dist < neg_dist).float().mean()
    return correct.item()

def train_model(train_loader, model, criterion, optimizer, num_epochs=2):
    """
    Train a siamese network using triplet loss on the provided data loader.

    The function handles:
    - Automatic triplet generation from the batch (anchor, positive, negative)
    - Training loop with forward/backward passes
    - Loss calculation and optimization
    - Progress reporting with loss and accuracy metrics

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of (images, labels)
        model (nn.Module): Siamese network model that returns embeddings for triplets
        criterion (nn.Module): Loss function (typically TripletLoss)
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        num_epochs (int, optional): Number of training epochs. Default: 2

    Returns:
        None: The function trains the model in-place and prints progress statistics

    Note:
        - Uses torch.roll to create positive/negative candidates from the same batch
        - Enforces that positives must be same class and negatives must be different class
        - Prints batch statistics every 100 batches
        - Uses tqdm for epoch progress tracking
    """
    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Create triplets (anchor, positive, negative)
            anchors = images
            positives = torch.roll(images, shifts=1, dims=0)
            negatives = torch.roll(images, shifts=2, dims=0)

            # Ensure positive is same class as anchor
            same_class = (labels == torch.roll(labels, shifts=1, dims=0))
            positives = torch.where(same_class.unsqueeze(1).unsqueeze(2).unsqueeze(3), positives, anchors)

            # Ensure negative is different class from anchor
            diff_class = (labels != torch.roll(labels, shifts=2, dims=0))
            negatives = torch.where(diff_class.unsqueeze(1).unsqueeze(2).unsqueeze(3), negatives, torch.roll(negatives, shifts=1, dims=0))

            optimizer.zero_grad()

            anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives)
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            loss.backward()
            optimizer.step()

            # Calculate accuracy
            batch_acc = calculate_accuracy(anchor_emb, positive_emb, negative_emb)

            running_loss += loss.item()
            running_acc += batch_acc * anchors.size(0)
            total_samples += anchors.size(0)

            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                avg_acc = running_acc / total_samples
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}: Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}')
                running_loss = 0.0
                running_acc = 0.0
                total_samples = 0


