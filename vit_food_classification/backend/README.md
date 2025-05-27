# FastAPI Inference Backend

This backend provides a modular inference API built with FastAPI. It loads a Vision Transformer (ViT) model via MLflow at application startup and exposes endpoints for image classification. The backend is designed for integration into web applications, automation scripts, or CLI tools.

## Features

- Loads a ViT model trained in PyTorch via MLflow
- Serves predictions using FastAPI
- Supports both file-based and base64 image input
- Includes model metadata loading (class labels)
- Designed for use with a separate frontend (e.g., Next.js)
- Optional scaffolding for PostgreSQL and Vault (not currently used)

## Requirements

- Python 3.12+
- Pip with virtual environment support
- MLflow tracking server (local or remote)

Install dependencies:

```bash
pip install -r requirements.txt
