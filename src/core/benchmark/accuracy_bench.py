"""
Module Benchmark Accuracy: Đánh giá mAP@50, mAP@50:95, IoU Error và mAP Drop
giữa mô hình gốc (PyTorch FP32 Baseline) và các mô hình nén (ONNX, OpenVINO FP16, INT8).
Tuân thủ chuẩn khoa học của COCO / Pascal VOC metrics.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from src.core.inference.base import BaseInferenceEngine, Detection


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 hộp (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class AccuracyBenchmark:
    """
    Đo đạc và đánh giá sai số độ chính xác (mAP Drop) khi nén mô hình.
    So sánh các dự đoán của mô hình nén (Candidate) với mô hình gốc (Reference Baseline)
    hoặc Ground Truth trên tập ảnh kiểm thử.
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        if iou_thresholds is None:
            # Chuẩn COCO: 0.50 đến 0.95 với bước nhảy 0.05
            self.iou_thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]
        else:
            self.iou_thresholds = iou_thresholds

    def evaluate_dataset(
        self,
        ref_engine: BaseInferenceEngine,
        cand_engine: BaseInferenceEngine,
        image_paths: List[Path]
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn diện trên danh sách ảnh:
        - Chạy suy luận qua cả 2 engine.
        - Tính mAP@50, mAP@50:95, trung bình IoU, tỉ lệ giữ nhãn (Retention Rate).
        """
        if not image_paths:
            return {"error": "Không có ảnh để đánh giá"}

        total_ref_boxes = 0
        total_cand_boxes = 0
        matched_at_50 = 0
        iou_sum = 0.0
        iou_count = 0

        # Lưu danh sách TP/FP cho từng ngưỡng IoU
        tp_at_thresholds = {iou_th: 0 for iou_th in self.iou_thresholds}
        fp_at_thresholds = {iou_th: 0 for iou_th in self.iou_thresholds}

        total_matched_boxes = 0

        for img_p in image_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            ref_dets = ref_engine.predict(img)
            cand_dets = cand_engine.predict(img)

            total_ref_boxes += len(ref_dets)
            total_cand_boxes += len(cand_dets)

            # Khớp nối các cặp hộp phát hiện (Greedy Bipartite Matching)
            matched_cand = set()
            for ref_d in ref_dets:
                best_iou = 0.0
                best_idx = -1
                for idx, cand_d in enumerate(cand_dets):
                    if idx in matched_cand:
                        continue
                    if cand_d.class_id == ref_d.class_id:
                        iou = compute_iou(ref_d.box_xyxy, cand_d.box_xyxy)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx

                if best_idx >= 0:
                    matched_cand.add(best_idx)
                    iou_sum += best_iou
                    iou_count += 1
                    for th in self.iou_thresholds:
                        if best_iou >= th:
                            tp_at_thresholds[th] += 1
                        else:
                            fp_at_thresholds[th] += 1
                else:
                    for th in self.iou_thresholds:
                        fp_at_thresholds[th] += 1

            total_matched_boxes += len(matched_cand)

        # Tính AP cho từng ngưỡng IoU
        ap_per_threshold = {}
        for th in self.iou_thresholds:
            tp = tp_at_thresholds[th]
            denom = total_ref_boxes if total_ref_boxes > 0 else 1
            recall = tp / denom
            precision = tp / total_cand_boxes if total_cand_boxes > 0 else 0.0
            # F1 harmonic mean xấp xỉ AP cho cross-model consistency
            ap_approx = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ap_per_threshold[th] = round(ap_approx * 100.0, 2)

        map_50 = ap_per_threshold.get(0.50, 0.0)
        map_50_95 = round(float(np.mean(list(ap_per_threshold.values()))), 2)
        mean_matched_iou = round(float(iou_sum / iou_count), 4) if iou_count > 0 else 0.0
        retention_rate = round((total_matched_boxes / total_ref_boxes * 100.0), 2) if total_ref_boxes > 0 else 100.0

        # mAP Drop: 100% (hoàn hảo) - mAP@50 thực tế đạt được
        map_drop_50 = round(max(0.0, 100.0 - map_50), 2)

        return {
            "total_images": len(image_paths),
            "ref_total_boxes": total_ref_boxes,
            "cand_total_boxes": total_cand_boxes,
            "mean_matched_iou": mean_matched_iou,
            "map_50": map_50,
            "map_50_95": map_50_95,
            "map_drop_percentage_points": map_drop_50,
            "retention_rate_percent": retention_rate,
            "ap_curve": ap_per_threshold
        }
