import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.detect import router as detect_router

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = ROOT / "data" / "benchmark_results.json"

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
    from backend.models.loader import _registry
    models: dict[str, list[str]] = {}
    for model_name, backend in _registry:
        models.setdefault(model_name, []).append(backend)
    return {"models": models}


@app.get("/benchmark", tags=["System"])
def get_benchmark():
    if not BENCHMARK_FILE.exists():
        return {"entries": []}
    return json.loads(BENCHMARK_FILE.read_text())
