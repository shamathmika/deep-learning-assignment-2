from ultralytics import YOLO

# imgsz=640 is the standard YOLO input resolution.
# TorchScript and OpenVINO require a fixed input shape at export time,
# unlike the .pt model which accepts any size at runtime.

MODELS = {
    "yolov8s":  "models/yolov8s.pt",
    "rtdetr-l": "models/rtdetr-l.pt",
}

for name, weights in MODELS.items():
    model = YOLO(weights)

    print(f"\n{name}: exporting to TorchScript...")
    model.export(format="torchscript", imgsz=640)

    print(f"{name}: exporting to OpenVINO...")
    model.export(format="openvino", imgsz=640)

    print(f"{name}: exporting to ONNX...")
    model.export(format="onnx", imgsz=640, opset=17)

print("\nAll exports complete.")
print("Expected output files:")
print("  models/yolov8s.torchscript")
print("  models/yolov8s_openvino_model/")
print("  models/yolov8s.onnx")
print("  models/rtdetr-l.torchscript")
print("  models/rtdetr-l_openvino_model/")
print("  models/rtdetr-l.onnx")
