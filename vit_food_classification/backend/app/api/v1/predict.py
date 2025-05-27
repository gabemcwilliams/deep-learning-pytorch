"""
predict.py (API v1)

Handles direct JSON-based prediction requests with base64-encoded images.
"""

from fastapi import APIRouter, Request, HTTPException
from app.services.predict import predict_model
from app.services.img_prep import vit_transform  # Reuse training transform
from PIL import Image
import base64
import torch
from io import BytesIO

router = APIRouter()


@router.post("/predict")
async def predict_endpoint(request: Request, input_data: dict):
    """
    Accepts a base64-encoded image and returns a class prediction.

    Request Body:
        {
            "image_base64": "<base64-encoded image string>"
        }

    Returns:
        dict: Prediction results with class label and probabilities.
    """
    if "image_base64" not in input_data:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in input.")

    try:
        # Decode base64 string
        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess using ViT transform
        tensor = vit_transform(image).unsqueeze(0)

        # Delegate to the central prediction function
        prediction = predict_model(tensor, request)
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
