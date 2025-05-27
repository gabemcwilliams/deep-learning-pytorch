from fastapi import HTTPException, APIRouter, Request
from torchvision import transforms
from PIL import Image
from io import BytesIO
import torch
import mlflow
import base64

router = APIRouter()


# Define the transform inline — same as training (ViT-friendly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_model(input_data: dict, request: Request):
    model = request.app.state.model

    if "image_base64" not in input_data:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in input.")

    try:
        # Decode base64 image and convert to RGB
        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Apply transform
        image_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return {"class": pred_class, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/predict")
async def predict_endpoint(request: Request, input_data: dict):
    return predict_model(input_data, request)
