# app.py: FastAPI service for skin-disease prediction

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # CORS support【72†L336-L344】
from pydantic import BaseModel
import uvicorn, io, os, logging, base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import json
import requests

# Set reproducibility seeds
np.random.seed(42)
tf.random.set_seed(42)

app = FastAPI()
# Enable CORS (allow cross-origin, e.g. from frontend)【72†L336-L344】【72†L348-L354】
origins = ["*"]  # TODO: restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Load model (SavedModel or .h5) and label map at startup
MODEL_PATH = "output/best_model.h5"  # specify actual path
LABEL_MAP_PATH = "output/label_map.json"  # specify actual path
try:
    if os.path.isdir(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # SavedModel directory
    else:
       model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # .h5 file
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise
try:
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    class_names = [label for label,_ in sorted(label_map.items(), key=lambda x: x[1])]
except Exception as e:
    logging.error(f"Failed to load label map: {e}")
    raise

IMG_SIZE = 224  # default image size

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0  # normalize as training
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload, run model, and return predictions."""
    # Validate file type
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")
    # Optional: check file size (e.g. <5MB)
    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(status_code=413, detail="File too large")
    # Load image from bytes
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    # Preprocess image
    img_arr = preprocess_image(image)
    # Predict probabilities
    preds = model.predict(img_arr)[0]
    # Get top-3
    top_idxs = preds.argsort()[-3:][::-1]
    results = []
    for i in top_idxs:
        label = class_names[i]
        prob = float(preds[i])
        # Apply confidence threshold (e.g. 0.5) if needed
        results.append({"label": label, "probability": prob})
    # Optional: overlay label on image and encode
    # (This code uses PIL to draw text)
    if file.filename.lower().endswith((".jpg",".jpeg",".png")):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f"Pred: {results[0]['label']} ({results[0]['probability']:.2f})"
        draw.text((10,10), text, fill="red", font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        overlay = f"data:image/png;base64,{b64}"
    else:
        overlay = None

    return JSONResponse(status_code=200, content={
        "predictions": results,
        "overlay_image": overlay
    })

# Error handlers example
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logging.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
# chatbot logic should load API keys from environment variables when enabled




   
if __name__ == "__main__":
    # Use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Start Uvicorn (or use `uvicorn` CLI)
    uvicorn.run(app, host="127.0.0.1", port=8000)
