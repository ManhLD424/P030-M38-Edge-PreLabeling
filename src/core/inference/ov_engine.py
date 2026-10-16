"""
Native OpenVINO Optimized CPU Inference Engine.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
from src.core.inference.base import BaseInferenceEngine, Detection, COCO_CLASSES


class OpenVINOInferenceEngine(BaseInferenceEngine):
    """
    Engine suy luận tối ưu hóa hiệu năng cao trên CPU sử dụng OpenVINO C++ Runtime Core.
    Hỗ trợ nạp trực tiếp file mô hình .xml (cả FP16 và INT8).
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        super().__init__(model_path, conf_thresh, iou_thresh)
        self.core = ov.Core()
        self.compiled_model = None
        self.output_layer = None
        self.load_model()

    def load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy model OpenVINO tại: {self.model_path}")

        # Tối ưu hóa cấu hình CPU: Ưu tiên chế độ LATENCY cho suy luận realtime
        model = self.core.read_model(str(path))
        self.compiled_model = self.core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1"
            }
        )
        self.output_layer = self.compiled_model.output(0)

    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Thực hiện tiền xử lý, suy luận qua OpenVINO C++ runtime và hậu xử lý NMS.
        """
        orig_h, orig_w = image.shape[:2]

        # 1. Tiền xử lý chuẩn Letterbox (giữ nguyên aspect ratio và padding 114)
        r = min(640.0 / orig_h, 640.0 / orig_w)
        new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))
        dw = (640.0 - new_unpad[0]) / 2.0
        dh = (640.0 - new_unpad[1]) / 2.0

        if (orig_w, orig_h) != new_unpad:
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]

        # 2. Suy luận qua Compiled Model
        results = self.compiled_model([input_tensor])[self.output_layer] # Shape: (1, 84, 8400)

        # 3. Hậu xử lý (NMS)
        predictions = np.transpose(results[0], (1, 0)) # Shape: (8400, 84)
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_thresh
        filtered_boxes = predictions[mask, :4]
        filtered_conf = confidences[mask]
        filtered_classes = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Chuyển đổi cx, cy, w, h sang x1, y1, x2, y2 trên tọa độ ảnh gốc
        boxes_xyxy = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            x1 = np.clip((cx - w / 2.0 - dw) / r, 0, orig_w)
            y1 = np.clip((cy - h / 2.0 - dh) / r, 0, orig_h)
            x2 = np.clip((cx + w / 2.0 - dw) / r, 0, orig_w)
            y2 = np.clip((cy + h / 2.0 - dh) / r, 0, orig_h)
            boxes_xyxy.append([int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))])

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy,
            filtered_conf.tolist(),
            score_threshold=self.conf_thresh,
            nms_threshold=self.iou_thresh
        )

        detections: List[Detection] = []
        if len(indices) == 0:
            return detections

        for i in indices.flatten():
            bx, by, bw, bh = boxes_xyxy[i]
            cls_id = int(filtered_classes[i])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            detections.append(Detection(
                x1=float(bx),
                y1=float(by),
                x2=float(bx + bw),
                y2=float(by + bh),
                confidence=float(filtered_conf[i]),
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections
