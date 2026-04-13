from pathlib import Path
from ultralytics import YOLO
from backend.models.onnx_coreml import OnnxCoreMLModel

# loader.py lives at backend/models/loader.py — project root is two levels up
ROOT = Path(__file__).resolve().parent.parent.parent

# Registry maps (model_name, backend) to a loaded model object.
# Models are loaded once at startup and reused for every request.
# Entries are skipped silently if the file does not exist yet
# (i.e. export_models.py has not been run).
_registry: dict[tuple[str, str], YOLO | OnnxCoreMLModel] = {}


def _register_yolo(model_name: str, backend: str, path: str, task: str = "detect") -> None:
    full_path = ROOT / path
    if full_path.exists():
        _registry[(model_name, backend)] = YOLO(str(full_path), task=task)
    else:
        print(f"[loader] skipping {model_name}/{backend}: {path} not found")


def _register_coreml(model_name: str, path: str, model_type: str) -> None:
    full_path = ROOT / path
    if full_path.exists():
        _registry[(model_name, "coreml")] = OnnxCoreMLModel(str(full_path), model_type)
    else:
        print(f"[loader] skipping {model_name}/coreml: {path} not found")


_register_yolo("yolov8s",  "pytorch",     "models/yolov8s.pt")
_register_yolo("rtdetr-l", "pytorch",     "models/rtdetr-l.pt")

# Same weights as pytorch, forced to CPU. Serves as the unoptimized baseline.
_register_yolo("yolov8s",  "pytorch-cpu", "models/yolov8s.pt")
_register_yolo("rtdetr-l", "pytorch-cpu", "models/rtdetr-l.pt")

_register_yolo("yolov8s",  "torchscript", "models/yolov8s.torchscript", task="detect")
_register_yolo("rtdetr-l", "torchscript", "models/rtdetr-l.torchscript", task="detect")

_register_yolo("yolov8s",  "openvino",    "models/yolov8s_openvino_model")
_register_yolo("rtdetr-l", "openvino",    "models/rtdetr-l_openvino_model")

_register_coreml("yolov8s",  "models/yolov8s.onnx",  model_type="yolo")
_register_coreml("rtdetr-l", "models/rtdetr-l.onnx", model_type="rtdetr")


def get_model(model_name: str, backend: str) -> YOLO | OnnxCoreMLModel:
    key = (model_name, backend)
    if key not in _registry:
        available = [f"{m}/{b}" for m, b in _registry]
        raise KeyError(
            f"No model loaded for '{model_name}' with backend '{backend}'. "
            f"Available: {available}"
        )
    return _registry[key]


def available_backends(model_name: str) -> list[str]:
    return [b for m, b in _registry if m == model_name]
