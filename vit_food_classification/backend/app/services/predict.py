"""
predict.py

Defines prediction logic for processing image tensors through a PyTorch model loaded from MLflow.

Handles:
- Batch dimension check
- MLflow PyFunc prediction
- Softmax postprocessing
- Label resolution using app.state.class_labels
"""

import mlflow
import torch


def predict_model(input_tensor: torch.Tensor, request) -> dict:
    """
    Runs prediction on a single input tensor using the MLflow-loaded model.

    Args:
        input_tensor (torch.Tensor): A 3D or 4D image tensor ready for inference.
        request (Request): The FastAPI request context containing app state.

    Returns:
        dict: A dictionary with predicted label and class probabilities.
    """
    model = request.app.state.model

    # Convert PyTorch tensor to NumPy (MLflow models require NumPy input)
    np_input = input_tensor.numpy()

    # Ensure 4D shape (batch_size, channels, height, width)
    if len(np_input.shape) == 3:
        np_input = np_input[None, :]  # Add batch dimension

    # Run prediction using MLflow PyFunc model
    prediction = model.predict(np_input)  # Output shape: (1, num_classes)

    # Convert to PyTorch tensor for softmax
    prediction = torch.softmax(torch.tensor(prediction), dim=1)

    # Select class with highest probability
    class_id = prediction.argmax(dim=1).item()
    class_name = request.app.state.class_labels[class_id]

    # Flatten prediction to list and zip with labels
    probs = prediction.squeeze().tolist()
    class_probs = dict(zip(request.app.state.class_labels, probs))

    return {
        "predicted_label": class_name,
        "probs": class_probs
    }
