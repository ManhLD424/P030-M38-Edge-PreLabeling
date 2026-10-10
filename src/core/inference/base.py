"""
Module Base Inference Engine: Định nghĩa giao diện suy luận chuẩn hóa và cấu trúc kết quả Detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Detection:
    """
    Cấu trúc dữ liệu đại diện cho một bounding box phát hiện được.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Tọa độ [xmin, ymin, xmax, ymax]"""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def box_cvat(self) -> List[float]:
        """Định dạng tọa độ cho CVAT Bounding Box [xtl, ytl, xbr, ybr]"""
        return [self.x1, self.y1, self.x2, self.y2]


class BaseInferenceEngine(ABC):
    """
    Interface cơ sở cho mọi inference backend (PyTorch, ONNX, OpenVINO).
    Áp dụng Strategy Pattern để dễ dàng hoán đổi runtime phục vụ benchmark và pre-labeling.
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    @abstractmethod
    def load_model(self) -> None:
        """Nạp mô hình vào bộ nhớ RAM/VRAM và khởi tạo phiên thực thi."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Thực hiện suy luận trên 1 khung hình ảnh (BGR numpy array).
        Trả về danh sách đối tượng Detection đã qua lọc ngưỡng confidence và NMS.
        """
        pass
