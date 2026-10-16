"""
Unit tests cho quy trình 2D Edge Pre-labeling và xuất format CVAT Bounding Boxes.
"""

import pytest
import json
from pathlib import Path
from src.core.inference.base import Detection, COCO_TO_BDD100K, BDD100K_CLASSES


def test_coco_to_bdd_mapping():
    assert COCO_TO_BDD100K["car"] == "car"
    assert COCO_TO_BDD100K["person"] == "pedestrian"
    assert COCO_TO_BDD100K["truck"] == "truck"
    assert COCO_TO_BDD100K["bus"] == "bus"
    assert COCO_TO_BDD100K["stop sign"] == "traffic sign"
    assert COCO_TO_BDD100K["traffic light"] == "traffic light"


def test_cvat_box_format_structure():
    det = Detection(
        x1=100.0,
        y1=150.0,
        x2=200.0,
        y2=250.0,
        confidence=0.88,
        class_id=2,
        class_name="car"
    )

    bdd_label = COCO_TO_BDD100K.get(det.class_name, det.class_name)
    shape_payload = {
        "type": "rectangle",
        "label": bdd_label,
        "points": det.box_cvat,
        "confidence": round(det.confidence, 3),
        "occluded": False
    }

    assert shape_payload["type"] == "rectangle"
    assert shape_payload["label"] == "car"
    assert shape_payload["points"] == [100.0, 150.0, 200.0, 250.0]
    assert shape_payload["confidence"] == 0.88
