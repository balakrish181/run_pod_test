import runpod
import base64
# import torch
import numpy as np
from PIL import Image
from io import BytesIO

# -------------------------------------------------------
# REPLACE THIS with your actual model loading logic
# e.g. YOLO, torchvision, HuggingFace, etc.
# -------------------------------------------------------
def load_model():
    # Example: ultralytics YOLO
    # from ultralytics import YOLO
    # return YOLO("yolov8n.pt")
    print("Model loaded.")
    return None  # Replace with real model

model = load_model()

# -------------------------------------------------------
# Decode base64 image string → PIL Image
# -------------------------------------------------------
def decode_image(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_bytes)).convert("RGB")

# -------------------------------------------------------
# Run inference — replace with your actual logic
# -------------------------------------------------------
def run_inference(image: Image.Image, params: dict) -> dict:
    # Example output shape — replace with real model output
    # results = model(image)
    # detections = results[0].boxes ...
    
    # --- MOCK OUTPUT (remove when using real model) ---
    return {
        "detections": [
            {
                "label": "person",
                "confidence": 0.92,
                "bbox": [120, 80, 400, 600]   # [x1, y1, x2, y2]
            },
            {
                "label": "car",
                "confidence": 0.87,
                "bbox": [500, 200, 900, 500]
            }
        ],
        "image_size": list(image.size),      # [width, height]
        "model": "yolov8n",
        #"device": "cuda" if torch.cuda.is_available() else "cpu"
        "device": "cpu"
    }

# -------------------------------------------------------
# RunPod entrypoint — receives job from the serverless queue
# -------------------------------------------------------
def handler(job):
    job_input = job["input"]

    # Validate input
    if "image" not in job_input:
        return {"error": "Missing 'image' field in input (base64 encoded)"}

    try:
        image = decode_image(job_input["image"])
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # Optional extra params (confidence threshold, etc.)
    params = {
        "confidence_threshold": job_input.get("confidence_threshold", 0.5),
        "task": job_input.get("task", "detect"),   # detect | classify | segment
    }

    try:
        result = run_inference(image, params)
        return {"status": "success", "output": result}
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})