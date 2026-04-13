import numpy as np
import cv2
import onnxruntime as ort

# COCO dataset class names — same 80 categories that YOLOv8 and RT-DETR are trained on
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Greedy non-maximum suppression.

    Iteratively keeps the highest-scoring box and removes any remaining box
    whose IoU with it exceeds iou_threshold. This eliminates duplicate
    detections of the same object.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return keep


class OnnxCoreMLModel:
    """
    Runs an Ultralytics-exported ONNX model using ONNX Runtime with the
    CoreML Execution Provider, which offloads supported ops to Apple's
    Neural Engine on M-series chips.

    Ultralytics' YOLO wrapper does not expose a way to set ORT providers,
    so this class handles preprocessing, inference, and postprocessing
    directly against the ORT session.
    """

    INPUT_SIZE = 640  # fixed at export time with imgsz=640

    def __init__(self, onnx_path: str, model_type: str):
        """
        model_type: "yolo" or "rtdetr"
          Determines which postprocessing branch to use, since the two
          architectures produce different output tensor shapes.
        """
        self.model_type = model_type
        self.names      = {i: name for i, name in enumerate(COCO_NAMES)}

        # CoreMLExecutionProvider is listed first so ORT uses it where possible.
        # Ops not supported by CoreML fall back to CPU automatically.
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_path: str) -> tuple[np.ndarray, tuple[int, int], float, tuple[int, int]]:
        """
        Load image and apply letterbox resize to INPUT_SIZE x INPUT_SIZE.

        Letterboxing scales the image to fit within the square while preserving
        aspect ratio, then pads the remaining area with gray (value 114).
        Ultralytics ONNX exports expect this exact preprocessing. A plain
        stretch resize produces misaligned bounding boxes because the model
        never saw distorted aspect ratios during training.

        Returns:
          img_nchw    the (1, 3, 640, 640) input tensor
          original_hw (height, width) of the source image
          scale       the single scale factor applied before padding
          pad         (pad_top, pad_left) offset of the image within the canvas
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Compute uniform scale so the longer side fits in INPUT_SIZE
        scale = min(self.INPUT_SIZE / orig_h, self.INPUT_SIZE / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # Center the scaled image on a gray canvas
        pad_top  = (self.INPUT_SIZE - new_h) // 2
        pad_left = (self.INPUT_SIZE - new_w) // 2

        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = cv2.resize(img, (new_w, new_h))

        img_norm = canvas.astype(np.float32) / 255.0
        img_nchw = np.transpose(img_norm, (2, 0, 1))[np.newaxis]  # (1, 3, 640, 640)

        return img_nchw, (orig_h, orig_w), scale, (pad_top, pad_left)

    # ------------------------------------------------------------------
    # Postprocessing: YOLOv8
    # ------------------------------------------------------------------

    def _postprocess_yolo(
        self,
        raw_output: list[np.ndarray],
        scale: float,
        pad: tuple[int, int],
        conf_thresh: float = 0.25,
        iou_thresh:  float = 0.45,
    ) -> list[dict]:
        """
        YOLOv8 ONNX output shape: (1, 84, 8400)
          84 = 4 box coords (cx, cy, w, h in 640px letterboxed space) + 80 class scores
          8400 = anchor grid cells across the three detection heads

        Steps:
          1. Transpose to (8400, 84) for row-wise processing
          2. Separate box coords from class scores
          3. Filter by max class score > conf_thresh
          4. Convert cx,cy,w,h to x1,y1,x2,y2, reverse letterbox, scale to original size
          5. Apply NMS to remove duplicate detections
        """
        preds = raw_output[0][0].T  # (8400, 84)

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids  = np.argmax(class_scores, axis=1)

        mask         = max_scores > conf_thresh
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores   = max_scores[mask]
        class_ids    = class_ids[mask]

        if len(max_scores) == 0:
            return []

        # Convert cx,cy,w,h to x1,y1,x2,y2 in 640px letterboxed space
        x1_640 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1_640 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2_640 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2_640 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Reverse letterbox: subtract padding offset, then invert scale
        pad_top, pad_left = pad
        x1 = (x1_640 - pad_left) / scale
        y1 = (y1_640 - pad_top)  / scale
        x2 = (x2_640 - pad_left) / scale
        y2 = (y2_640 - pad_top)  / scale

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep       = _nms(boxes_xyxy, max_scores, iou_thresh)

        return [
            {
                "class":      self.names.get(int(class_ids[i]), str(class_ids[i])),
                "confidence": round(float(max_scores[i]), 3),
                "box":        [round(float(x1[i])), round(float(y1[i])),
                               round(float(x2[i])), round(float(y2[i]))],
            }
            for i in keep
        ]

    # ------------------------------------------------------------------
    # Postprocessing: RT-DETR
    # ------------------------------------------------------------------

    def _postprocess_rtdetr(
        self,
        raw_output: list[np.ndarray],
        scale: float,
        pad: tuple[int, int],
        conf_thresh: float = 0.25,
    ) -> list[dict]:
        """
        RT-DETR ONNX output (Ultralytics export):
          output0: (1, 300, 84)
            4 values are cx,cy,w,h normalized [0, 1] relative to the 640px canvas
            remaining 80 are class scores

        The transformer decoder outputs center-format boxes with sigmoid-activated
        coordinates. The xyxy conversion and pixel scaling that Ultralytics applies
        during eager inference are not included in the traced ONNX graph, so this
        method handles them manually.

        No NMS needed. The transformer decoder produces at most 300 non-duplicate
        detections by design (one learned query slot per object).
        """
        preds = raw_output[0][0]  # (300, 84)

        boxes_cxcywh = preds[:, :4]   # cx,cy,w,h normalized [0, 1] in 640px canvas space
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids  = np.argmax(class_scores, axis=1)

        mask         = max_scores > conf_thresh
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores   = max_scores[mask]
        class_ids    = class_ids[mask]

        pad_top, pad_left = pad
        detections = []
        for i in range(len(max_scores)):
            cx, cy, w, h = boxes_cxcywh[i]

            # Convert cx,cy,w,h to x1,y1,x2,y2 in 640px canvas space
            x1_640 = (cx - w / 2) * self.INPUT_SIZE
            y1_640 = (cy - h / 2) * self.INPUT_SIZE
            x2_640 = (cx + w / 2) * self.INPUT_SIZE
            y2_640 = (cy + h / 2) * self.INPUT_SIZE

            # Reverse letterbox: subtract padding offset, then invert scale
            x1 = round((x1_640 - pad_left) / scale)
            y1 = round((y1_640 - pad_top)  / scale)
            x2 = round((x2_640 - pad_left) / scale)
            y2 = round((y2_640 - pad_top)  / scale)

            detections.append({
                "class":      self.names.get(int(class_ids[i]), str(class_ids[i])),
                "confidence": round(float(max_scores[i]), 3),
                "box":        [x1, y1, x2, y2],
            })

        return detections

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, image_path: str, device: str = "cpu", verbose: bool = False) -> list[dict]:  # device/verbose unused; present for interface compatibility with YOLO wrapper
        """
        Run inference and return a list of detection dicts.
        The device and verbose args exist only for interface compatibility
        with the rest of the backend — ORT session provider is fixed at init.
        """
        img_input, original_hw, scale, pad = self._preprocess(image_path)
        raw_output = self.session.run(self.output_names, {self.input_name: img_input})

        if self.model_type == "yolo":
            return self._postprocess_yolo(raw_output, scale, pad)
        else:
            return self._postprocess_rtdetr(raw_output, scale, pad)
