import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


@app.post("/eval/run", tags=["System"])
async def run_eval():
    import asyncio
    import sys
    script = ROOT / "scripts" / "run_map_eval.py"
    if not (ROOT / "data" / "annotations" / "instances.json").exists():
        return JSONResponse({"error": "Annotation file not found at data/annotations/instances.json"}, status_code=400)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script),
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return JSONResponse({"error": stderr.decode()}, status_code=500)
        if BENCHMARK_FILE.exists():
            return json.loads(BENCHMARK_FILE.read_text())
        return {"entries": []}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/benchmark", tags=["System"])
async def update_benchmark(request: Request):
    new_data = await request.json()

    # Preserve existing mAP fields — only latency is overwritten by the frontend run
    existing_map: dict[tuple[str, str], dict] = {}
    if BENCHMARK_FILE.exists():
        for entry in json.loads(BENCHMARK_FILE.read_text()).get("entries", []):
            existing_map[(entry["model"], entry["backend"])] = entry

    merged = []
    for entry in new_data.get("entries", []):
        key = (entry["model"], entry["backend"])
        prev = existing_map.get(key, {})
        merged.append({
            **entry,
            "map50":    prev.get("map50"),
            "map50_95": prev.get("map50_95"),
        })

    result = {"entries": merged}
    BENCHMARK_FILE.write_text(json.dumps(result, indent=2))
    return result
