"""
Single source of truth for shared constants across backend and scripts.
"""

# All supported models and their acceleration backends, in display order.
MODELS: list[str] = ["yolov8s", "rtdetr-l"]

BACKENDS: list[str] = ["pytorch-cpu", "pytorch", "torchscript", "openvino", "coreml"]

# Every valid (model, backend) pair — used by eval scripts and the loader.
MODEL_BACKEND_PAIRS: list[tuple[str, str]] = [
    (model, backend)
    for model in MODELS
    for backend in BACKENDS
]

# Maps backend name → PyTorch device string used during inference.
# TorchScript and OpenVINO run on CPU; MPS is only for PyTorch eager mode.
DEVICE_MAP: dict[str, str] = {
    "pytorch":     "mps",
    "pytorch-cpu": "cpu",
    "torchscript": "cpu",
    "openvino":    "cpu",
    "coreml":      "cpu",
}

# COCO 2017 category name → ID (80 classes).
COCO_NAME_TO_ID: dict[str, int] = {
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
