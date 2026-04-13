# Inference Optimization

CMPE 258 Homework 2 | Shamathmika | SJSU Spring 2026

Object detection inference pipeline comparing YOLOv8s and RT-DETR-L across five acceleration backends. Includes a FastAPI backend, Next.js frontend for live video/image detection, and a benchmark dashboard with mAP evaluation.

## Models

| Model | Architecture |
|---|---|
| YOLOv8s | CNN (CSP-Darknet) |
| RT-DETR-L | Transformer (hybrid encoder + decoder) |

## Backends

| Backend | Method | Device |
|---|---|---|
| PyTorch CPU | Eager mode, unoptimized baseline | CPU |
| PyTorch MPS | Eager mode, Apple Metal GPU | GPU |
| TorchScript | Traced frozen graph | CPU |
| OpenVINO | IR graph via ONNX conversion | CPU |
| ONNX + CoreML EP | ONNX Runtime + Apple Neural Engine | ANE / CPU |

## Results

### YOLOv8s
| Backend | Latency (ms) | Speedup | mAP@50 |
|---|---|---|---|
| PyTorch CPU (baseline) | 36.5 | 1.00x | 0.50 |
| PyTorch MPS | 18.3 | 1.99x | 0.50 |
| TorchScript | 46.2 | 0.79x | 0.50 |
| OpenVINO | 19.7 | 1.85x | 0.50 |
| ONNX + CoreML EP | 8.3 | 4.37x | 0.50 |

### RT-DETR-L
| Backend | Latency (ms) | Speedup | mAP@50 |
|---|---|---|---|
| PyTorch CPU (baseline) | 352.6 | 1.00x | 0.70 |
| PyTorch MPS | 31.1 | 11.34x | 0.70 |
| TorchScript | 401.2 | 0.88x | 0.00* |
| OpenVINO | 70.9 | 4.97x | 0.00* |
| ONNX + CoreML EP | 76.5 | 4.61x | 0.60 |

*RT-DETR TorchScript/OpenVINO exports produce out-of-range class IDs from the traced decoder argmax. See report.md for details.

## Requirements

- Apple Silicon Mac (MPS required for GPU backend)
- Python 3.11+
- Node.js 18+

## Setup

```bash
git clone https://github.com/shamathmika/deep-learning-assignment-2.git
cd deep-learning-assignment-2

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Export model files (run once, requires model weights in models/)
python scripts/export_models.py
```

## Running

```bash
# Terminal 1: backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:3000  
API docs: http://localhost:8000/docs

## mAP Evaluation

1. Annotate frames from `data/annotation_frames/` in Label Studio
2. Export as COCO JSON and place at `data/annotations/instances.json`
3. Click Run Eval in the Benchmark tab, or run:

```bash
python scripts/run_map_eval.py
```

## Project Structure

```
backend/
  constants.py          shared model/backend lists and COCO class map
  models/
    loader.py           loads all model/backend combos at startup
    onnx_coreml.py      ONNX inference with CoreML EP postprocessing
  routers/
    detect.py           /detect/image and /detect/video endpoints
  main.py               FastAPI app, /benchmark and /eval/run endpoints
frontend/
  app/page.tsx          detection UI and benchmark dashboard
scripts/
  export_models.py      exports .pt to TorchScript, OpenVINO, ONNX
  extract_frames.py     extracts evenly-spaced frames from video
  run_map_eval.py       computes mAP against COCO annotations
data/
  annotation_frames/    30 frames from test video
  sample_images/        sample test images
  benchmark_results.json
models/                 model weight files (not tracked in git)
```
