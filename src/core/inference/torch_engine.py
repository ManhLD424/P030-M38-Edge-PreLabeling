"""
PyTorch Baseline Inference Engine: Sử dụng framework Ultralytics YOLO gốc.
"""

from typing import List
import numpy as np
from ultralytics import YOLO
from src.core.inference.base import BaseInferenceEngine, Detection


class TorchInferenceEngine(BaseInferenceEngine):
    """
    Engine chạy suy luận bằng PyTorch phục vụ đo đạc baseline và đối chứng kết quả.
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45, device: str = "cpu"):
        super().__init__(model_path, conf_thresh, iou_thresh)
        self.device = device
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        self.model = YOLO(self.model_path)

    def predict(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names.get(cls_id, str(cls_id))

            detections.append(Detection(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections
