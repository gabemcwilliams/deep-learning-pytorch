from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
import asyncio

from mlflow.tracking import MlflowClient

from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.python.eager.context import async_wait

from .routes.main import router as main_router
from .routes.errors import router as errors_router
from .api.v1.upload_routes import router as upload_router
from .api.v1.predict import router as predict_router

from .utils.database.db import PostgresConnEngine
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
    # --- views ---
    app.include_router(main_router, prefix='')

    # --- api ---
    app.include_router(upload_router, prefix='/api/v1')
    app.include_router(predict_router, prefix='/api/v1')

    # --- errors ---
    app.include_router(errors_router, prefix='/errors')

    return app
