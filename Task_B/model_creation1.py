import torch
import torch.nn.functional as F
from torch import nn
from timm import create_model  # Library for EfficientNet

class EfficientNetEmbedding(nn.Module):
    """
    EfficientNet-based embedding network with customizable output dimension.
    """
    def __init__(self, embedding_size=128, model_name='efficientnet_b0'):
        super(EfficientNetEmbedding, self).__init__()
        # Load pre-trained EfficientNet without classification head
        self.backbone = create_model(
            model_name,
            pretrained_cfg={'url': ''},
            num_classes=0,  # Remove final classification layer
            global_pool='avg'  # Use global average pooling
        )

        # Get the number of features from the backbone
        num_features = self.backbone.num_features

        # Projection head to get desired embedding size
        self.projection = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)  # Shape: (batch_size, num_features)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)  # L2 normalize


class SiameseNetwork(nn.Module):
    """
    Siamese network with EfficientNet backbone that can handle:
    - Single input mode (returns embeddings)
    - Pair mode (returns two embeddings)
    - Triplet mode (returns three embeddings)
    """
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            return self.embedding_net(x1)
        elif x3 is None:
            return self.embedding_net(x1), self.embedding_net(x2)
        else:
            return (self.embedding_net(x1),
                    self.embedding_net(x2),
                    self.embedding_net(x3))


def model_instances(model_name='efficientnet_b0', embedding_size=128):
    """
    Creates EfficientNet-based embedding and Siamese networks.

    Args:
        model_name (str): Name of EfficientNet variant (b0-b7)
        embedding_size (int): Dimension of output embeddings

    Returns:
        tuple: (embedding_net, siamese_net)
    """
    embedding_net = EfficientNetEmbedding(
        embedding_size=embedding_size,
        model_name=model_name
    )
    model_1 = SiameseNetwork(embedding_net)
    return embedding_net, model_1
