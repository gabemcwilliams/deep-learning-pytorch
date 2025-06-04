# Deep Learning with PyTorch

This repository contains completed deep learning projects and training experiments using PyTorch. It was built alongside the [**PyTorch for Deep Learning Bootcamp**](https://www.udemy.com/course/pytorch-deep-learning/) by Daniel Bourke and significantly expanded with production-grade features, including modular training scripts, MLflow integration, ONNX export, and FastAPI inference.

## ✅ Highlights

* End-to-end CNN pipeline with evaluation, logging, and model saving
* MLflow experiment tracking with custom run logic
* ONNX export for portable model inference
* FastAPI server for real-time predictions
* Visualizations: loss curves, confusion matrices, and decision boundaries
* Reusable utilities for training, data loading, and evaluation
* Loguru-based logging for clean console + file output

## 📁 Contents

### `classification/`

Binary classification and activation experiments on synthetic data.

| File                                         | Description                                                             |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| `binary_classification_circles_mlflow.ipynb` | Train a simple neural network on 2D circle data. MLflow tracks metrics. |
| `activations_from_scratch.ipynb`             | Visualizes ReLU, Sigmoid, Tanh from scratch using raw PyTorch ops.      |
| `decision_boundary.png`                      | Decision boundary of a trained model on synthetic data.                 |

### `cnn_fashionmnist/`

Convolutional neural network trained on FashionMNIST, with full tracking and deployment hooks.

| File             | Description                                                                     |
| ---------------- | ------------------------------------------------------------------------------- |
| `train.py`       | Modular training script using PyTorch, MLflow, and Loguru.                      |
| `evaluate.py`    | Evaluation script with confusion matrix, accuracy, and precision/recall output. |
| `onnx_export.py` | Converts PyTorch model to ONNX format.                                          |
| `fastapi_infer/` | Minimal FastAPI app for image-based inference using ONNX model.                 |

## 🛠 Tools & Libraries

* `PyTorch` – Deep learning model development
* `MLflow` – Experiment tracking and artifact logging
* `Loguru` – Structured logging
* `ONNX` – Model export and cross-framework inference
* `FastAPI` – Lightweight RESTful inference backend
* `matplotlib`, `seaborn` – Visualization of results and metrics

## 🎯 Purpose

This repo is designed to show deep learning readiness with:

* Fully working training loops (not just notebooks)
* Reusable code structured for experimentation and reuse
* Deployment-focused mindset with minimal overhead

Ideal for showcasing applied PyTorch skills in MLOps, edge inference, and production-aligned model development.
