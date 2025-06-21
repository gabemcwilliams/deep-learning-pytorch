"""
main.py

FastAPI application setup and configuration.

This file contains the core setup for the FastAPI application, including middleware configurations,
routes, and the integration of machine learning models via MLOps. The application is designed to handle
API routes for file uploads, model predictions, and other necessary functionalities for AI-based services.

Key Features:
- **Model Loading**: On startup, the application loads a machine learning model from MLflow and attaches it
  to the app's state for inference.
- **API Endpoints**: Includes routes for uploading data, making predictions, and handling errors.
- **CORS and Session Management**: Configures middleware for handling CORS and session management securely.
- **Static File Serving**: Configures serving of static files like images, documents, or other assets.

To start the server, use the command:
    python run.py --host 127.0.0.1 --port 8080 --debug

This will launch the application using **Uvicorn** with the specified host and port, and it will enable
FastAPI's debug mode for development purposes.

Usage:
1. The FastAPI app loads a model during startup using **MLflow** and makes it available for inference via
   routes.
2. The application supports **CORS**, session management, and static file serving.
3. Routes are registered for handling upload tasks, predictions, and error handling.

Modules:
- **MLflow**: For model management, loading, and inference.
- **PostgreSQL**: Database connection handling for any necessary storage (not fully defined here).
- **VaultManager**: For managing secrets (e.g., API keys, database passwords).

Typical Usage:
    python run.py --host 127.0.0.1 --port 8080 --debug
    python run.py --init-db --database mydb --schema myschema

This file contains the **app creation logic** that sets up middleware, static file serving, and routes
for the application. When the app is run using **python run.py**, it starts the **Uvicorn** server with
the defined settings.

"""

from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
import asyncio

from mlflow.tracking import MlflowClient

from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.python.eager.context import async_wait

from .api.v1.upload_routes import router as upload_router
from .api.v1.predict import router as predict_router

from .utils.security.vault_mgr import VaultManager

import loguru

import json
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import mlflow
from mlflow.tracking import MlflowClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup: Loading model...")
    loop = asyncio.get_running_loop()

    client = MlflowClient()
    model_name = "torch_vit_pizza_steak_sushi"
    alias = "production"

    version = client.get_model_version_by_alias(name=model_name, alias=alias)
    print(f"Model version: {version.version}, Source: {version.source}")

    model_uri = f"models:/{model_name}@{alias}"
    model = await loop.run_in_executor(None, mlflow.pyfunc.load_model, model_uri)

    app.state.model = model
    print("Model loaded and attached to app.state.model.")

    from mlflow.artifacts import download_artifacts
    file = download_artifacts(run_id=version.run_id, artifact_path="meta/class_labels.json")

    print(f'class_labels.json loc: {file}')

    with open(file) as f:
        class_labels = json.loads(f.read())

    print(f'class_labels: {class_labels}')
    app.state.class_labels = class_labels

    yield  # REQUIRED for FastAPI to continue running

    print("Shutdown: Clean up here if needed")


def create_app():
    app = FastAPI(
        title="My ML API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,

    )

    # CORS setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the 'static' directory to serve static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Middleware for session management (optional)
    app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "supersecret"))

    # Register routes

    # --- api ---
    app.include_router(upload_router, prefix='/api/v1')
    app.include_router(predict_router, prefix='/api/v1')

    return app
