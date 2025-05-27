# ViT Food Classification Demo (Full Stack)

This repository provides a complete stack for an image classification demo using a pretrained Vision Transformer (ViT) model. The app predicts whether an uploaded image is a **pizza**, **steak**, or **sushi**.

## Directory Structure

```
.
├── train/           # PyTorch training pipeline
├── backend/         # FastAPI service to serve the model
├── frontend/        # Next.js client for UI interaction
```

---

## 1. Model Training (`train/`)

* Trains a ViT-based image classifier using PyTorch.
* Uses torchvision's `vit_b_16` with frozen weights.
* Tracks experiments with MLflow.
* Logs final model, plots, and metrics.
* Supports TorchScript and ONNX model saving.

**Run:**

```bash
cd train
python main.py

```

---

## 2. Backend API (`backend/`)

* Loads the trained model from MLflow at startup.
* Handles image upload via REST endpoint `/api/v1/upload/image`.
* Performs preprocessing and returns prediction.
* Optional: Includes scaffolding for Vault integration and PostgreSQL.

**Run:**

```bash
cd backend
python run.py

```

---

## 3. Frontend (`frontend/`)

* Built with Next.js and Tailwind CSS.
* Uploads image and shows predicted class + confidence.
* Fetches results from FastAPI backend.
* Works locally or can be deployed as a static app.

**Run:**

```bash
cd frontend
npm install
npm run dev

```

Open [http://localhost:3000](http://localhost:3000) to access the app.

---

## Requirements

* Python 3.10+
* Node.js 18+
* Docker (optional, for containerization)
* MLflow Tracking URI set up if not using local filesystem

---

## Example Prediction Flow

1. User uploads image on frontend
2. Image is posted to `/api/v1/upload/image`
3. FastAPI backend preprocesses and sends tensor to model
4. Model returns probabilities
5. Frontend displays top class and percentage breakdown

---

## Author

Created by Gabe McWilliams as a portfolio-ready demonstration of:

* Deep learning with PyTorch
* MLOps tooling (MLflow, ONNX)
* Modern full-stack deployment with FastAPI and Next.js
