import time
from ultralytics import YOLO

IMAGE_PATH = "data/raw_frames/test.jpg"


def run_inference(model, image_path, label):
    # First pass compiles MPS shaders. Result is discarded.
    model.predict(image_path, device="mps", verbose=False)

    start = time.perf_counter()
    results = model.predict(image_path, device="mps", verbose=False)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    result = results[0]

    print(f"\n[{label}]")
    print(f"  Latency    : {latency_ms:.1f} ms")
    print(f"  Detections : {len(result.boxes)}")

    for box in result.boxes:
        class_id   = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        coords     = box.xyxy[0].tolist()
        print(f"    {class_name:<12} conf={confidence:.2f}  box={[round(c) for c in coords]}")

    return result, latency_ms


yolo_model = YOLO("yolov8s.pt")
run_inference(yolo_model, IMAGE_PATH, "YOLOv8s  [PyTorch/MPS baseline]")

rtdetr_model = YOLO("rtdetr-l.pt")
run_inference(rtdetr_model, IMAGE_PATH, "RT-DETR-l [PyTorch/MPS baseline]")
