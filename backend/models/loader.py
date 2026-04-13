from ultralytics import YOLO

# Models are loaded once at startup and reused across requests.
# Loading inside a request handler would add ~2s to every call.
MODELS = {
    "yolov8s":  YOLO("yolov8s.pt"),
    "rtdetr-l": YOLO("rtdetr-l.pt"),
}


def get_model(model_name: str) -> YOLO:
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
    return MODELS[model_name]
