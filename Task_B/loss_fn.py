import torch
from torch import nn
import torch.nn.functional as F
class TripletLoss(nn.Module):
    """
    Triplet loss function for training siamese networks.

    Takes embeddings of an anchor, positive (same class as anchor), and negative (different class)
    and computes the triplet loss using a margin. The loss encourages the distance between the
    anchor and positive to be smaller than the distance between the anchor and negative by at
    least the margin value.

    Args:
        margin (float, optional): Margin for the triplet loss. Default: 1.0
    """
    def __init__(self, margin=1.0):
        """
        Initialize the TripletLoss module.

        Args:
            margin (float, optional): Margin value for the loss calculation.
                                      Default: 1.0
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor (torch.Tensor): Embeddings of the anchor samples
            positive (torch.Tensor): Embeddings of positive samples (same class as anchor)
            negative (torch.Tensor): Embeddings of negative samples (different class from anchor)

        Returns:
            torch.Tensor: The computed triplet loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def loss_fn():
    """
    Creates and returns a TripletLoss criterion instance.

    Returns:
        TripletLoss: An instance of the TripletLoss class with default margin (1.0)
    """
    criterion = TripletLoss()
    return criterion
