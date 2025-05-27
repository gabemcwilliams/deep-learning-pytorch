"""
main.py

This script serves as the primary training entry point for a Vision Transformer (ViT)-based
image classification model on the pizza/steak/sushi dataset. It orchestrates all phases of
the ML pipeline including data loading, model initialization, training, evaluation, and
optional model export to MLflow.

Key Features:
- Uses modular loader, model, and loop components.
- Integrates MLflow tracking and model artifact logging via a custom ExperimentManager.
- Includes timing decorators for stage-wise profiling.
- Supports PyTorch training with automatic CUDA device detection.

Intended to be executed as a script.

Example:
    $ python main.py
"""

# --- Standard Library ---
import os
import sys
import re
import math
import json
import time
import traceback
import threading
import warnings
from pathlib import Path

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import mlflow

from mlflow.models import infer_signature
from pydantic import BaseModel, Field, ValidationError
from colorama import Fore, Style
from sklearn.metrics import classification_report, confusion_matrix

# --- PyTorch and TorchVision ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms

# --- Optional Config Tools ---
# from omegaconf import OmegaConf
# import hydra

# --- Internal Modules ---
from utils.experiment import ExperimentManager
from loaders.vit_loader import load_data as vit_loader
from models.vit_model import build_model as vit_model
from loop import train_loop as vit_loop

# --- PyTorch Runtime Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Using device: {Fore.YELLOW}{device}{Style.RESET_ALL}")
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} PyTorch version: {Fore.GREEN}{torch.__version__}{Style.RESET_ALL}")

# Optional: limit CPU thread usage for determinism or resource control
torch.set_num_threads(32)  # Adjust to your CPU core count

# --- MLflow Experiment Setup ---
experiment_name = 'torch_vit_pizza_steak_sushi'
experiment_manager = ExperimentManager(experiment_name=experiment_name)


@experiment_manager.log_time(stage='load')
def load_data(data_dir: str):
    """
    Loads and returns training and testing dataloaders from the given directory.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        dict: Dictionary containing train/test dataloaders and class labels.
    """
    data_dict = vit_loader(data_dir=data_dir)
    return data_dict


@experiment_manager.log_time(stage='build')
def build_model():
    """
    Instantiates and returns a Vision Transformer (ViT) model instance.

    Returns:
        torch.nn.Module: The ViT model.
    """
    return vit_model()


@experiment_manager.log_time(stage='train')
def train_loop(model, train_loader, test_loader, experiment_manager):
    """
    Trains the model using the custom train loop and logs metrics.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training dataloader.
        test_loader (DataLoader): Validation/Test dataloader.
        experiment_manager (ExperimentManager): Logger and artifact handler.

    Returns:
        dict: Dictionary containing training results and final metrics.
    """
    result = vit_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        experiment_manager=experiment_manager,
        epochs=20,
        lr=0.0005
    )
    return result


@experiment_manager.log_time(stage='eval')
def evaluate_model(model, result_dict, experiment_manager):
    """
    Logs final results, generates plots, and conditionally exports the model
    to MLflow if the accuracy threshold is met.

    Args:
        model (torch.nn.Module): Trained model.
        result_dict (dict): Dictionary of training/evaluation results.
        experiment_manager (ExperimentManager): Logger and exporter.
    """
    experiment_manager.log_final_results(results=result_dict)
    experiment_manager.log_plots(results=result_dict)

    rnd_final_test_acc = round(result_dict['final_test_accuracy'], 4)
    if experiment_manager.check_accuracy_thresh(accuracy=rnd_final_test_acc, threshold_percent=90):
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Accuracy acceptable ({rnd_final_test_acc}) — exporting model.")
        experiment_manager.model_artifact_torch(
            model=model,
            registered_name=f'{experiment_name}',
            save_path='/mnt/mls/models',
            alias='production'
        )

    # ONNX export disabled due to unsupported ViT layers in current opset version.
    # Uncomment and adapt if ONNX support becomes available.


def experiment_flow():
    """
    Executes the full experiment pipeline:
    - Loads configuration
    - Loads data
    - Builds model
    - Runs training
    - Evaluates and exports the model if valid
    """
    experiment_manager.load_env('~/.secrets/.env')
    experiment_manager.check_who_am_i()
    experiment_manager.check_hostname()
    experiment_manager.check_env_vars()

    dataloader_dict = load_data(data_dir='/mnt/mls/data/pizza_steak_sushi/images')
    dataloader_sig_dict = load_data(data_dir='/mnt/mls/data/pizza_steak_sushi/images')

    experiment_manager.log_json_labels_artifact(class_labels=dataloader_dict['class_labels'])

    model = build_model()
    experiment_manager.get_signature_from_loader(data_loader=dataloader_sig_dict["train_loader"], model=model)

    result_dict = train_loop(
        model=model,
        train_loader=dataloader_dict["train_loader"],
        test_loader=dataloader_dict["test_loader"],
        experiment_manager=experiment_manager,
    )

    evaluate_model(
        model=model,
        result_dict=result_dict,
        experiment_manager=experiment_manager
    )

    experiment_manager.end_run()


if __name__ == "__main__":
    experiment_flow()
