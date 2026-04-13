"""
mAP evaluation against a COCO-format ground truth annotation file.

Workflow:
  1. Annotate frames in Label Studio (export as COCO JSON).
  2. Place the exported JSON at  data/annotations/instances.json
     (images are read from  data/annotation_frames/)
  3. Run:  python scripts/run_map_eval.py
  4. Results are written to  data/benchmark_results.json
     (latency fields are preserved; only mAP fields are updated).
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make backend importable when run as a script

from backend.constants import MODEL_BACKEND_PAIRS  # noqa: E402

import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNOTATIONS    = ROOT / "data" / "annotations" / "instances.json"
IMAGES_DIR     = ROOT / "data" / "annotation_frames"
BENCHMARK_FILE = ROOT / "data" / "benchmark_results.json"
BACKEND_URL    = "http://localhost:8000"


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


def _resolve_image(exported_name: str) -> Path | None:
    """
    Label Studio prefixes uploaded filenames with a hash (e.g. 'a1b2c3d4-frame_00000.jpg')
    and may export the full absolute path. Extract the frame index and match against
    the actual files in IMAGES_DIR.
    """
    basename = Path(exported_name).name  # strip any leading path
    m = re.search(r'frame[_\s]?(\d+)', basename, re.IGNORECASE)
    if not m:
        return None
    frame_idx = int(m.group(1))
    matches = list(IMAGES_DIR.glob(f"*{frame_idx:05d}*"))
    return matches[0] if matches else None


def evaluate(model_name: str, backend: str, coco_gt: COCO) -> dict:
    # Build name→id from the annotation file itself — Label Studio uses its own IDs
    name_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}

    image_ids  = coco_gt.getImgIds()
    dt_results = []

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = _resolve_image(img_info["file_name"])

        if img_path is None or not img_path.exists():
            print(f"  [skip] image not found: {img_info['file_name']}")
            continue

        try:
            detections = run_inference(img_path, model_name, backend)
        except Exception as exc:
            print(f"  [error] {img_path.name}: {exc}")
            continue

        for det in detections:
            cat_id = name_to_id.get(det["class"])
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            dt_results.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO: x,y,w,h
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

    for model_name, backend in MODEL_BACKEND_PAIRS:
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
