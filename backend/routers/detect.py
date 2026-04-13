import time
import tempfile
import os

import cv2
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

from backend.models.loader import get_model
from backend.models.onnx_coreml import OnnxCoreMLModel

router = APIRouter()

# TorchScript and OpenVINO models run on CPU.
# MPS (Apple Metal) is only used for PyTorch eager mode.
DEVICE_MAP = {
    "pytorch":     "mps",
    "pytorch-cpu": "cpu",
    "torchscript": "cpu",
    "openvino":    "cpu",
    "coreml":      "cpu",
}


def run_timed_inference(model, image_path: str, device: str) -> tuple[list, float]:
    # CoreML models return a list of dicts directly — no Ultralytics Results wrapper
    if isinstance(model, OnnxCoreMLModel):
        model.predict(image_path)  # warmup
        start = time.perf_counter()
        detections = model.predict(image_path)
        end = time.perf_counter()
        return detections, (end - start) * 1000

    # All other backends use the Ultralytics YOLO wrapper
    # First pass warms up the runtime (shader compile on MPS, graph load on others)
    model.predict(image_path, device=device, verbose=False)

    start = time.perf_counter()
    results = model.predict(image_path, device=device, verbose=False)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        # RT-DETR TorchScript/OpenVINO exports produce out-of-range class IDs
        # for some detections due to an argmax issue in the traced graph.
        # Skip any ID outside the model's known class range.
        if cls_id not in model.names:
            continue
        detections.append({
            "class":      model.names[cls_id],
            "confidence": round(float(box.conf), 3),
            "box":        [round(c) for c in box.xyxy[0].tolist()],
        })

    return detections, latency_ms


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    model_name: str  = Query(default="yolov8s"),
    backend: str     = Query(default="pytorch"),
):
    try:
        model = get_model(model_name, backend)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    device = DEVICE_MAP.get(backend, "cpu")

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        detections, latency_ms = run_timed_inference(model, tmp_path, device)
    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "model":      model_name,
        "backend":    backend,
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    })


@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    model_name: str  = Query(default="yolov8s"),
    backend: str     = Query(default="pytorch"),
    frame_interval: int = Query(default=5),
):
    try:
        model = get_model(model_name, backend)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    device = DEVICE_MAP.get(backend, "cpu")

    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_results = []
        latencies     = []
        frame_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = tmp_path + f"_frame{frame_idx}.jpg"
                cv2.imwrite(frame_path, frame)

                if frame_idx == 0:
                    model.predict(frame_path, device=device, verbose=False)

                start = time.perf_counter()
                results = model.predict(frame_path, device=device, verbose=False)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

                if isinstance(model, OnnxCoreMLModel):
                    detections = results
                else:
                    result = results[0]
                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        if cls_id not in model.names:
                            continue
                        detections.append({
                            "class":      model.names[cls_id],
                            "confidence": round(float(box.conf), 3),
                            "box":        [round(c) for c in box.xyxy[0].tolist()],
                        })

                frame_results.append({
                    "frame_index": frame_idx,
                    "latency_ms":  round(latency_ms, 2),
                    "detections":  detections,
                })

                os.unlink(frame_path)

            frame_idx += 1

        cap.release()

    finally:
        os.unlink(tmp_path)

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

    return JSONResponse({
        "model":            model_name,
        "backend":          backend,
        "fps":              round(fps, 2),
        "total_frames":     frame_idx,
        "processed_frames": len(frame_results),
        "avg_latency_ms":   avg_latency,
        "frame_results":    frame_results,
    })
