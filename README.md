# Inference Optimization

Object detection inference pipeline with FastAPI backend and Next.js frontend.

## Models
- YOLOv8s (CNN-based)
- RT-DETR-l (Transformer-based)

## Inference Backends
- PyTorch with MPS (Apple Metal)
- TorchScript
- OpenVINO
- ONNX Runtime with CoreML Execution Provider

## Requirements
- Apple Silicon Mac (MPS required)
- Python 3.11+
- Node.js 18+

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at `http://localhost:3000`

## Evaluation

```bash
python evaluation/map_eval.py
python evaluation/benchmark.py
```
