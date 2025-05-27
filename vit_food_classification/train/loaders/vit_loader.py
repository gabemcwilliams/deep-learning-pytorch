"""
vit_loader.py

Loads image datasets from a structured directory and returns PyTorch DataLoaders with
ImageNet-style normalization. Designed for use with ViT models trained on food classification
or similar image tasks.

Expected directory structure:
    data_dir/
        train/
            class1/
            class2/
            ...
        test/
            class1/
            class2/
            ...

Features:
- Applies train/test transforms with resizing and normalization
- Supports debug mode for fast iteration
- Returns class labels for logging and visualization
"""

import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(data_dir: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4,
              debug: bool = False) -> dict:
    """
    Loads image data from a directory and applies necessary transformations.

    Args:
        data_dir (str): Directory containing the image dataset.
        image_size (int): Resize images to this size (e.g., 224).
        batch_size (int): The batch size to use for data loading.
        num_workers (int): Number of worker threads for data loading.
        debug (bool): Whether to use a smaller subset of data for debugging.

    Returns:
        dict: Dictionary containing train and test loaders and class names.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms)

    if debug:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(100))
        test_dataset = Subset(test_dataset, range(50))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_labels = train_dataset.dataset.classes if debug else train_dataset.classes

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "class_labels": class_labels
    }
