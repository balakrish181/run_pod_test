from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CV Pipeline API", version="1.0.0")

# -------------------------------------------------------
# CORS — allow your frontend origin
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------------------------------------------------------
# Config from .env
# -------------------------------------------------------
RUNPOD_API_KEY    = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT   = os.getenv("RUNPOD_ENDPOINT_ID")   # just the ID
CLIENT_API_KEY    = os.getenv("CLIENT_API_KEY")        # key you give to your client

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/runsync"

# -------------------------------------------------------
# Auth — client must send  X-API-Key: <CLIENT_API_KEY>
# -------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Depends(api_key_header)):
    if key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# -------------------------------------------------------
# Health check (no auth needed)
# -------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------
# Main inference endpoint
# -------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    _key: str = Depends(verify_key),
):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images accepted")

    # Read and encode image
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build RunPod payload
    payload = {
        "input": {
            "image": b64_image,
            "confidence_threshold": confidence_threshold,
            "task": "detect",
        }
    }

    # Call RunPod — API key stays server-side, client never sees it
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                RUNPOD_URL,
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="RunPod timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"RunPod error: {e.response.text}")

    result = response.json()

    # RunPod wraps output in {"output": {...}}
    if "error" in result.get("output", {}):
        raise HTTPException(status_code=500, detail=result["output"]["error"])

    return JSONResponse(content=result.get("output", result))