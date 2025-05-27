"""
img_prep.py

Provides image preprocessing utilities to convert uploaded files into 
PyTorch tensors compatible with Vision Transformer (ViT) models.

Used in FastAPI routes to prepare user-uploaded images for inference.
"""

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import torch
import io

# Use the default pretrained weights (e.g., ImageNet)
weights = ViT_B_16_Weights.DEFAULT

# Get the corresponding preprocessing pipeline
vit_transform = weights.transforms()


def preprocess_image(upload_file) -> torch.Tensor:
    """
    Preprocess an UploadFile into a tensor compatible with ViT.

    Args:
        upload_file (starlette.datastructures.UploadFile): The uploaded image file from FastAPI request.

    Returns:
        torch.Tensor: A 4D tensor with shape (1, 3, 224, 224) ready for inference.
    """
    image_data = upload_file.file.read()  # Read raw bytes
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB just in case
    return vit_transform(image).unsqueeze(0)  # Add batch dimension
