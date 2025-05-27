# Vision Transformer Training Pipeline

This module contains the training pipeline for a PyTorch-based Vision Transformer (ViT) model using the `torchvision` `vit_b_16` architecture. The model is trained to classify images into one of three categories: pizza, steak, or sushi. Training is fully modular and integrated with MLflow for tracking experiments, metrics, and model artifacts.

---

## Features

- Uses pretrained ViT backbone from `torchvision.models`
- Supports transfer learning with frozen base and custom head
- Logs parameters, metrics, and plots to MLflow
- Tracks system metrics and gradient norms
- Automatically exports model to MLflow registry (Torch format)
- Includes signature and sample input capture
- Training includes:
  - Early stopping
  - Learning rate scheduler
  - F1, precision, recall tracking
  - Confusion matrix logging
- ONNX export logic is scaffolded but optional

---

## Directory Structure

