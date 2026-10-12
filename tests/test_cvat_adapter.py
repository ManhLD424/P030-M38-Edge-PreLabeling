"""Tests for CVAT 3D REST API payload adapter."""
import pytest
from src.core.geometry.cuboid import Cuboid3D
from src.core.geometry.cvat_adapter import CVAT3DAdapter


def test_cvat_3d_shapes_payload():
    c1 = Cuboid3D(x=10.0, y=2.0, z=0.5, dx=4.6, dy=1.8, dz=1.5, yaw=0.5, score=0.91, class_name="vehicles")
    c2 = Cuboid3D(x=5.0, y=-1.0, z=0.0, dx=0.7, dy=0.7, dz=1.7, yaw=0.0, score=0.85, class_name="pedestrian")

    adapter = CVAT3DAdapter(label_mapping={"vehicles": 10, "pedestrian": 20})
    payload = adapter.to_shapes_payload([c1, c2], frame=0)

    assert payload["version"] == 0
    assert len(payload["shapes"]) == 2

    s1 = payload["shapes"][0]
    assert s1["type"] == "cuboid"
    assert s1["frame"] == 0
    assert s1["label_id"] == 10
    assert len(s1["points"]) == 16
    assert s1["points"][0] == 10.0
    assert s1["points"][6] == 4.6

    s2 = payload["shapes"][1]
    assert s2["label_id"] == 20
    assert s2["points"][7] == 0.7


def test_cvat_3d_tracks_payload():
    c1 = Cuboid3D(x=12.0, y=0.0, z=0.5, dx=4.8, dy=1.9, dz=1.5, yaw=0.0, score=0.88, class_name="vehicles")
    adapter = CVAT3DAdapter(label_mapping={"vehicles": 1})
    payload = adapter.to_tracks_payload({0: [c1]})

    assert "tracks" in payload
    assert len(payload["tracks"]) == 1
    t = payload["tracks"][0]
    assert t["frame"] == 0
    assert t["source"] == "semi-auto"
    assert len(t["shapes"]) == 1
    assert t["shapes"][0]["type"] == "cuboid"


def test_parse_cvat_annotations():
    adapter = CVAT3DAdapter(label_mapping={"vehicles": 1})
    c_orig = Cuboid3D(x=8.0, y=-2.5, z=0.4, dx=4.2, dy=1.8, dz=1.4, yaw=0.3, score=1.0, class_name="vehicles")
    payload = adapter.to_shapes_payload([c_orig], frame=0)

    extracted = CVAT3DAdapter.parse_cvat_annotations(payload, id_to_label={1: "vehicles"})
    assert len(extracted) == 1
    assert pytest.approx(extracted[0].x) == 8.0
    assert pytest.approx(extracted[0].y) == -2.5
    assert pytest.approx(extracted[0].dx) == 4.2
