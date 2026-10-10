"""
Kiểm thử đơn vị cơ bản cho BaseInferenceEngine và Detection data structure.
"""

import pytest
from src.core.inference.base import Detection


def test_detection_dataclass():
    det = Detection(
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        confidence=0.85,
        class_id=2,
        class_name="car"
    )
    assert det.box_xyxy == (10.0, 20.0, 110.0, 220.0)
    assert det.box_cvat == [10.0, 20.0, 110.0, 220.0]
    assert det.class_name == "car"
    assert det.confidence == 0.85
