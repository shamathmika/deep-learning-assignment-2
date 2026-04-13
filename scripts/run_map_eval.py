"""
mAP evaluation against a COCO-format ground truth annotation file.

Workflow:
  1. Annotate frames in Label Studio (export as COCO JSON).
  2. Place the exported JSON at  data/annotations/instances.json
     and the corresponding images in  data/annotations/images/
  3. Run:  python scripts/run_map_eval.py
  4. Results are written to  data/benchmark_results.json
     (latency fields are preserved; only mAP fields are updated).
"""

import json
import time
from pathlib import Path

import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT            = Path(__file__).resolve().parent.parent
ANNOTATIONS     = ROOT / "data" / "annotations" / "instances.json"
IMAGES_DIR      = ROOT / "data" / "annotation_frames"
BENCHMARK_FILE  = ROOT / "data" / "benchmark_results.json"
BACKEND_URL     = "http://localhost:8000"

# COCO category name to ID mapping (80-class COCO 2017)
COCO_NAME_TO_ID = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21,
    "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27,
    "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34,
    "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41,
    "surfboard": 42, "tennis racket": 43, "bottle": 44, "wine glass": 46,
    "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51,
    "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56,
    "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61,
    "chair": 62, "couch": 63, "potted plant": 64, "bed": 65,
    "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74,
    "remote": 75, "keyboard": 76, "cell phone": 77, "microwave": 78,
    "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84,
    "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88,
    "hair drier": 89, "toothbrush": 90,
}

MODELS_BACKENDS = [
    ("yolov8s",  "pytorch-cpu"),
    ("yolov8s",  "pytorch"),
    ("yolov8s",  "torchscript"),
    ("yolov8s",  "openvino"),
    ("yolov8s",  "coreml"),
    ("rtdetr-l", "pytorch-cpu"),
    ("rtdetr-l", "pytorch"),
    ("rtdetr-l", "torchscript"),
    ("rtdetr-l", "openvino"),
    ("rtdetr-l", "coreml"),
]


def run_inference(image_path: Path, model_name: str, backend: str) -> list[dict]:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/detect/image",
            params={"model_name": model_name, "backend": backend},
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["detections"]


def evaluate(model_name: str, backend: str, coco_gt: COCO) -> dict:
    image_ids = coco_gt.getImgIds()
    dt_results = []

    for img_id in image_ids:
        img_info  = coco_gt.loadImgs(img_id)[0]
        img_path  = IMAGES_DIR / img_info["file_name"]

        if not img_path.exists():
            print(f"  [skip] image not found: {img_path.name}")
            continue

        try:
            detections = run_inference(img_path, model_name, backend)
        except Exception as exc:
            print(f"  [error] {img_path.name}: {exc}")
            continue

        for det in detections:
            cat_id = COCO_NAME_TO_ID.get(det["class"])
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            dt_results.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO expects x,y,w,h
                "score":       det["confidence"],
            })

    if not dt_results:
        print(f"  [warn] no detections — skipping mAP for {model_name}/{backend}")
        return {"map50": None, "map50_95": None}

    coco_dt   = coco_gt.loadRes(dt_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # stats[0] = AP @IoU=0.50:0.95, stats[1] = AP @IoU=0.50
    return {
        "map50":    round(float(evaluator.stats[1]), 4),
        "map50_95": round(float(evaluator.stats[0]), 4),
    }


def load_benchmark() -> list[dict]:
    if BENCHMARK_FILE.exists():
        return json.loads(BENCHMARK_FILE.read_text()).get("entries", [])
    return []


def save_benchmark(entries: list[dict]) -> None:
    BENCHMARK_FILE.write_text(json.dumps({"entries": entries}, indent=2))
    print(f"\nSaved results to {BENCHMARK_FILE}")


def main() -> None:
    if not ANNOTATIONS.exists():
        print(f"Annotation file not found: {ANNOTATIONS}")
        print("Export your Label Studio annotations as COCO JSON and place them there.")
        return

    print(f"Loading ground truth from {ANNOTATIONS}")
    coco_gt = COCO(str(ANNOTATIONS))

    existing = {(e["model"], e["backend"]): e for e in load_benchmark()}

    for model_name, backend in MODELS_BACKENDS:
        print(f"\n{'='*50}")
        print(f"Evaluating  {model_name} / {backend}")
        t0 = time.perf_counter()

        metrics = evaluate(model_name, backend, coco_gt)

        elapsed = time.perf_counter() - t0
        print(f"  mAP@50:     {metrics['map50']}")
        print(f"  mAP@50:95:  {metrics['map50_95']}")
        print(f"  Elapsed:    {elapsed:.1f}s")

        key = (model_name, backend)
        prev = existing.get(key, {"model": model_name, "backend": backend, "avg_latency_ms": None})
        existing[key] = {**prev, **metrics}

    save_benchmark(list(existing.values()))


if __name__ == "__main__":
    main()
