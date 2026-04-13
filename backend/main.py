from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.detect import router as detect_router

app = FastAPI(
    title="Object Detection API",
    description="YOLOv8 and RT-DETR inference with TorchScript, OpenVINO, and ONNX CoreML backends",
    version="1.0.0",
)

# Allows the frontend at localhost:3000 to call this server at localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, prefix="/detect", tags=["Detection"])


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.get("/models", tags=["System"])
def list_models():
    return {
        "models":   ["yolov8s", "rtdetr-l"],
        "backends": ["pytorch"],
    }
