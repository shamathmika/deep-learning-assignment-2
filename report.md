# CMPE 258 Homework 2: Inference Optimization
Shamathmika | San Jose State University | Spring 2026

## 1. Overview

This project benchmarks object detection inference across multiple acceleration backends on Apple Silicon. Two models are compared: YOLOv8s (CNN) and RT-DETR-L (Transformer). Each model is evaluated across five backends: PyTorch CPU (unoptimized baseline), PyTorch MPS, TorchScript, OpenVINO, and ONNX Runtime with CoreML Execution Provider.

## 2. Hardware and Environment

| Item | Detail |
|---|---|
| Machine | MacBook Pro M5 Pro |
| RAM | 48 GB unified memory |
| GPU | Apple Metal (MPS) |
| Python | 3.14 |
| PyTorch | 2.11.0 |
| ONNX Runtime | 1.24.4 |
| OpenVINO | 2026.1.0 |

No CUDA or NVIDIA hardware is available. All acceleration is CPU or Apple Neural Engine based.

## 3. Models

**YOLOv8s** is a lightweight single-stage CNN detector. It processes images through a convolutional grid and outputs 8400 anchor predictions at 640x640 resolution.

**RT-DETR-L** is a transformer-based detector with a hybrid encoder and decoder producing 300 object queries. It achieves higher accuracy than YOLOv8s but is significantly heavier, especially on CPU.

## 4. Acceleration Methods

**PyTorch CPU (baseline):** Eager-mode PyTorch on CPU with no optimization. All speedups are relative to this.

**PyTorch MPS:** The same weights dispatched to the Apple Metal GPU. No model export needed. Speedup comes from GPU parallelism.

**TorchScript:** The model graph is traced and frozen via `torch.jit.trace`, producing a `.torchscript` file. Eliminates Python overhead. Runs on CPU.

**OpenVINO:** The model is exported to ONNX then converted to OpenVINO IR format. OpenVINO applies CPU-specific graph optimizations including operator fusion. Runs on CPU.

**ONNX + CoreML EP:** The model is exported to ONNX. At runtime, ONNX Runtime partitions the graph so that ops supported by CoreML run on the Apple Neural Engine (ANE) and the rest fall back to CPU. This is the only backend that uses the ANE. Custom pre/postprocessing handles the letterbox convention and output format differences between YOLOv8 and RT-DETR.

## 5. Results

Latency is single-image inference time in milliseconds, averaged over a warm run (one warmup pass excluded). Speedup is relative to PyTorch CPU per model.

### YOLOv8s

| Backend | Latency (ms) | Speedup | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| PyTorch CPU (baseline) | 36.5 | 1.00x | 0.50 | 0.40 |
| PyTorch MPS | 18.3 | 1.99x | 0.50 | 0.40 |
| TorchScript | 46.2 | 0.79x | 0.50 | 0.40 |
| OpenVINO | 19.7 | 1.85x | 0.50 | 0.40 |
| ONNX + CoreML EP | 8.3 | 4.37x | 0.50 | 0.40 |

### RT-DETR-L

| Backend | Latency (ms) | Speedup | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| PyTorch CPU (baseline) | 352.6 | 1.00x | 0.70 | 0.50 |
| PyTorch MPS | 31.1 | 11.34x | 0.70 | 0.50 |
| TorchScript | 401.2 | 0.88x | 0.00 | 0.00 |
| OpenVINO | 70.9 | 4.97x | 0.00 | 0.00 |
| ONNX + CoreML EP | 76.5 | 4.61x | 0.60 | 0.40 |

mAP was evaluated on 29 annotated frames from a real-world street scene video using pycocotools COCOeval.

## 6. Analysis

### Latency

YOLOv8s is already lightweight so the gains are smaller in absolute terms. CoreML EP is the fastest at 8.3ms (4.37x) by offloading to the Neural Engine. MPS and OpenVINO are close at roughly 2x and 1.85x. TorchScript is actually slower than baseline at 0.79x. With single-image inference and no batching, the compilation overhead is not amortized.

RT-DETR-L shows much larger gains because the transformer decoder is expensive on CPU at 352ms. MPS gives 11.34x speedup by parallelizing attention on the GPU. OpenVINO gets nearly 5x by fusing transformer ops. TorchScript again underperforms for the same reason.

### Accuracy

YOLOv8s mAP is identical across all backends (0.50 / 0.40). The convolutional operations are numerically stable across runtimes, so acceleration has no accuracy cost for this model.

RT-DETR-L accuracy varies significantly by backend:

PyTorch (CPU/MPS) scores 0.70 mAP@50. The Ultralytics eager-mode inference handles the decoder output correctly.

TorchScript and OpenVINO both score 0.00. When RT-DETR is traced or converted, the argmax in the decoder produces class IDs outside the valid range (>= 80). Every detection gets discarded, resulting in zero detections. This is a known issue with tracing transformer decoders that contain dynamic control flow.

ONNX + CoreML EP scores 0.60 mAP@50. Custom postprocessing was written to handle the ONNX export output format (normalized cx/cy/w/h in 640px canvas space) including letterbox reversal. This recovers most of the accuracy.

### Model Comparison

RT-DETR-L outperforms YOLOv8s at baseline (0.70 vs 0.50 mAP@50) but at 352ms vs 36ms on CPU. With MPS acceleration RT-DETR-L becomes practical at 31ms while keeping its accuracy advantage. On CPU-only hardware YOLOv8s is the better choice.

## 7. Key Findings

1. CoreML EP is fastest for YOLOv8s at 8.3ms (4.37x) using the Apple Neural Engine.
2. MPS is most effective for RT-DETR-L at 31.1ms (11.34x) via GPU attention.
3. TorchScript does not always improve latency. For single-image inference it was slower than eager mode on both models.
4. RT-DETR TorchScript and OpenVINO exports produce corrupt class predictions. The argmax in the traced decoder graph outputs out-of-range IDs. This is an export-level issue, not a runtime issue.
5. All backends preserve YOLOv8s accuracy. No accuracy cost for accelerating the CNN model.
6. ONNX RT-DETR requires custom postprocessing. The exported tensor layout differs from Ultralytics eager output and needs manual letterbox reversal.

## 8. Steps Done

1. Set up FastAPI backend with Uvicorn and a Next.js frontend.
2. Downloaded YOLOv8s and RT-DETR-L weights via Ultralytics.
3. Exported both models to TorchScript, OpenVINO IR, and ONNX formats using `scripts/export_models.py`.
4. Implemented the ONNX + CoreML EP backend with custom letterbox preprocessing and separate postprocessing branches for YOLOv8 (transposed anchor grid) and RT-DETR (normalized center-format boxes).
5. Built the detection endpoints (`/detect/image`, `/detect/video`) and a benchmark endpoint that stores latency and mAP results.
6. Built the frontend with a video detection player (canvas overlay synced to playback), image detection view, and a benchmark dashboard with live latency measurement.
7. Ran latency benchmarks across all 10 model/backend combinations using the Run All button in the dashboard.
8. Recorded a street scene video with an iPhone and extracted 30 evenly-spaced frames using `scripts/extract_frames.py`.
9. Annotated all 30 frames in Label Studio with bounding boxes across 13 object classes (car, truck, person, traffic light, train, fire hydrant, bus, umbrella, dog, parking meter, motorcycle, stop sign, potted plant). Screenshot of the annotation interface below.

![Label Studio annotation](data/assignment_screenshots/Screenshot%202026-04-13%20at%204.09.59%20PM.png)

10. Exported annotations as COCO JSON and placed at `data/annotations/instances.json`.
11. Ran mAP evaluation via `scripts/run_map_eval.py`, which calls the detection API for each frame and model/backend combo and computes mAP@50 and mAP@50:95 using pycocotools COCOeval.

## 9. Acceleration Methods Summary

| Method | Type | Status |
|---|---|---|
| TorchScript | Required | Implemented. Latency regression on both models; export issue causes zero mAP on RT-DETR |
| OpenVINO | Required | Implemented. 1.85x speedup on YOLOv8s, 4.97x on RT-DETR. Same export issue on RT-DETR accuracy |
| ONNX + CoreML EP | Bonus | Implemented. Fastest for YOLOv8s. Custom postprocessing recovers RT-DETR accuracy |
