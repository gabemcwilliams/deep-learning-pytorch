"""
upload_files.py

Provides a utility function for saving uploaded files to disk.

Used in FastAPI routes where UploadFile objects are received.
"""

from fastapi import UploadFile
from pathlib import Path
import os

# Directory to save uploaded files — adjust or move to config if needed
UPLOAD_DIR = Path("app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_file(file: UploadFile) -> dict:
    """
    Saves an UploadFile to disk under UPLOAD_DIR.

    Args:
        file (UploadFile): The file received via a FastAPI endpoint.

    Returns:
        dict: Result info or error details.
    """
    try:
        file_location = UPLOAD_DIR / file.filename

        # Save file to disk
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"filename": file.filename, "status": "saved"}

    except Exception as e:
        return {"error": str(e)}
