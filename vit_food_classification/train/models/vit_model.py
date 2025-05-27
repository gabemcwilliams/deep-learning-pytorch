"""
vit_model.py

Defines the Vision Transformer (ViT) model architecture using a pretrained ViT-B/16 backbone
from torchvision. Freezes the base layers and replaces the classification head for fine-tuning
on a custom dataset (e.g., 3 food classes: pizza, steak, sushi).

This is intended for transfer learning where only the final layer is trained.
"""


from torchvision import models
import torch.nn as nn

def build_model(num_classes: int = 3) -> nn.Module:
    """
    Loads a pretrained ViT-B/16 model and replaces the classification head.

    Args:
        num_classes (int): Number of output classes for classification. Default is 3.

    Returns:
        nn.Module: A modified Vision Transformer model ready for fine-tuning.
    """
    model = models.vit_b_16(weights="IMAGENET1K_V1")

    # Freeze all layers to retain pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification head with a custom one
    in_features = model.heads[0].in_features
    model.heads = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )

    # Unfreeze the new head for training
    for param in model.heads.parameters():
        param.requires_grad = True

    return model
