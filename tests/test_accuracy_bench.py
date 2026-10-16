"""
Unit tests cho module AccuracyBenchmark và thuật toán IoU / mAP Drop.
"""

import pytest
import numpy as np
from src.core.benchmark.accuracy_bench import compute_iou, AccuracyBenchmark
from src.core.inference.base import Detection, BaseInferenceEngine


def test_compute_iou_identical():
    box1 = (10.0, 10.0, 50.0, 50.0)
    box2 = (10.0, 10.0, 50.0, 50.0)
    assert compute_iou(box1, box2) == 1.0


def test_compute_iou_no_overlap():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    box1 = (0.0, 0.0, 10.0, 10.0)  # Area 100
    box2 = (5.0, 0.0, 15.0, 10.0)  # Area 100, Inter = 5*10 = 50, Union = 150
    iou = compute_iou(box1, box2)
    assert abs(iou - (50.0 / 150.0)) < 1e-5


class DummyEngine(BaseInferenceEngine):
    def __init__(self, detections):
        super().__init__("dummy_path")
        self.detections = detections

    def load_model(self):
        pass

    def predict(self, image):
        return self.detections


def test_accuracy_benchmark_evaluation(tmp_path):
    # Tạo một ảnh giả để kiểm tra luồng evaluate_dataset
    import cv2
    dummy_img_path = tmp_path / "test.jpg"
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(dummy_img_path), dummy_img)

    ref_dets = [
        Detection(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=2, class_name="car"),
        Detection(x1=60, y1=10, x2=90, y2=50, confidence=0.85, class_id=0, class_name="pedestrian"),
    ]
    # Candidate có 1 hộp khớp hoàn toàn, 1 hộp lệch nhẹ
    cand_dets = [
        Detection(x1=10, y1=10, x2=50, y2=50, confidence=0.88, class_id=2, class_name="car"),
        Detection(x1=62, y1=11, x2=89, y2=49, confidence=0.80, class_id=0, class_name="pedestrian"),
    ]

    ref_eng = DummyEngine(ref_dets)
    cand_eng = DummyEngine(cand_dets)

    bench = AccuracyBenchmark()
    res = bench.evaluate_dataset(ref_eng, cand_eng, [dummy_img_path])

    assert res["total_images"] == 1
    assert res["ref_total_boxes"] == 2
    assert res["cand_total_boxes"] == 2
    assert res["map_50"] == 100.0
    assert res["mean_matched_iou"] > 0.85
    assert res["map_drop_percentage_points"] == 0.0
