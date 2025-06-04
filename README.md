# 🧠 Deep Learning Vision Pipelines with PyTorch, ViT, and FastAPI

This repository includes complete, production-ready pipelines for image classification using PyTorch. It showcases both
classic CNN-based modeling and modern transformer-based architectures (ViT), along with integrated deployment via
FastAPI and a full-stack React + Tailwind frontend.

---

## 📦 Repository Structure

### `/cnn/`

A self-contained project for binary and multiclass classification using PyTorch.

- `fashion_mnist_cnn_mlflow.ipynb` — CNN training notebook with full MLflow integration
- `README.md` — Project-specific details and evaluation metrics

---

### `/vit_food_classification/`

A complete Vision Transformer pipeline for food classification, including training, backend inference, and frontend
visualization.

#### `backend/`

FastAPI application for real-time inference and image uploads.

- `app/api/v1/` — API routes (`predict.py`, `upload_routes.py`)
- `app/services/` — Core services (`img_prep.py`, `predict.py`, `upload_files.py`)
- `run.py` — Application entry point
- `main.py` — FastAPI initialization
- `requirements.txt` — Backend dependencies

#### `frontend/`

React + Next.js frontend for user interaction and prediction display.

- Tailwind CSS styling and layout
- Image upload interface with preview and response rendering
- Clean folder separation under `src/`, `public/`, and config files

#### `train/`

Complete training pipeline for ViT models.

- `models/vit_model.py` — Vision Transformer architecture
- `loaders/vit_loader.py` — Dataset loading and transformations
- `utils/` — Reusable modules:
    - `loop.py` — Training loop
    - `experiment.py` — MLflow tracking
    - `system_metrics.py` — Performance metrics
- `main.py` — Training orchestrator
- `requirements.txt` — Training dependencies

#### `source_data/`

- `data_gathering.ipynb` — Data acquisition and preprocessing notebook

---

## ⚙️ Tools & Frameworks

- **PyTorch** — Core deep learning framework
- **Vision Transformer (ViT)** — State-of-the-art model for image classification
- **MLflow** — Experiment tracking and model logging
- **FastAPI** — Lightweight, async-ready inference backend
- **ONNX** — Model export and serving
- **Next.js** — React framework for frontend delivery
- **Tailwind CSS** — Utility-first styling
- **Loguru** — Simplified structured logging
- **spaCy** — Used in earlier data filtering steps (optional)

---

## 💡 Features

- Fully modular architecture across training, serving, and UI layers
- MLflow-powered reproducibility with signature and artifact logging
- ONNX export path for model portability
- Secure API design with image upload and prediction routing
- Intuitive web UI for real-time image classification
- Readable logs and automated performance tracking

---

## 📊 Example Use Cases

* Food item recognition via uploaded images
* Edge deployment of transformer-based models with ONNX runtime
* Educational demos for CNN vs. ViT performance
* End-to-end ML workflows for vision projects, from data to web deployment

---

This repository demonstrates the full stack of modern deep learning—from model training to live inference with a
user-facing frontend—all using open-source tooling and portable architecture.

