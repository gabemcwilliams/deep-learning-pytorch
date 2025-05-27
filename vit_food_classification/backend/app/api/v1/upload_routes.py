"""
upload_routes.py

Handles image upload and inference. Accepts an image file,
runs it through a ViT model served via MLflow, and returns predictions.
"""

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.services.upload_files import save_uploaded_file
from app.services.img_prep import preprocess_image
from app.services.predict import predict_model

router = APIRouter(prefix="/upload")


@router.post("/image")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to receive an image, preprocess it, and return prediction results.

    Args:
        request (Request): FastAPI request context containing app.state.
        file (UploadFile): The uploaded image file.

    Returns:
        dict: Prediction result and filename.
    """
    try:
        # Optional: Save to disk for audit/debugging purposes
        await save_uploaded_file(file)

        # Rewind stream and convert to tensor
        file.file.seek(0)
        tensor = preprocess_image(file)

        # Run inference
        prediction = predict_model(input_tensor=tensor, request=request)

        return {
            "prediction": prediction,
            "original_image": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
