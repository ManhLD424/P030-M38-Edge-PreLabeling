"""Tests for 3D Cuboid geometry, spatial operations, and CVAT 16-point encoding."""
import math
import numpy as np
import pytest

from src.core.geometry.cuboid import Cuboid3D, center_in_box, merge_boxes, drop_boxes_on_cars


def test_cuboid_dataclass_and_cvat_points():
    c = Cuboid3D(
        x=10.5,
        y=-3.2,
        z=0.8,
        dx=4.5,
        dy=1.9,
        dz=1.6,
        yaw=0.75,
        score=0.92,
        class_name="vehicles"
    )
    cvat_pts = c.to_cvat_points()
    assert len(cvat_pts) == 16
    assert cvat_pts[0] == 10.5
    assert cvat_pts[1] == -3.2
    assert cvat_pts[2] == 0.8
    assert cvat_pts[3] == 0.0
    assert cvat_pts[4] == 0.0
    assert cvat_pts[5] == 0.75
    assert cvat_pts[6] == 4.5
    assert cvat_pts[7] == 1.9
    assert cvat_pts[8] == 1.6
    assert cvat_pts[9:] == [0.0] * 7

    # Roundtrip from CVAT points
    reconstructed = Cuboid3D.from_cvat_points(cvat_pts, class_name="vehicles", score=0.92)
    assert pytest.approx(reconstructed.x) == c.x
    assert pytest.approx(reconstructed.y) == c.y
    assert pytest.approx(reconstructed.z) == c.z
    assert pytest.approx(reconstructed.dx) == c.dx
    assert pytest.approx(reconstructed.dy) == c.dy
    assert pytest.approx(reconstructed.dz) == c.dz
    assert pytest.approx(reconstructed.yaw) == c.yaw


def test_cuboid_corners_3d():
    c = Cuboid3D(x=0.0, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=1.0, yaw=0.0)
    corners = c.corners_3d()
    assert corners.shape == (8, 3)
    assert np.all(corners[:, 0] >= -2.0) and np.all(corners[:, 0] <= 2.0)
    assert np.all(corners[:, 1] >= -1.0) and np.all(corners[:, 1] <= 1.0)
    assert np.all(corners[:, 2] >= -0.5) and np.all(corners[:, 2] <= 0.5)


def test_center_in_box_and_merge():
    box1 = Cuboid3D(x=10.0, y=0.0, z=0.0, dx=4.5, dy=2.0, dz=1.5, yaw=0.0, score=0.90, class_name="vehicles")
    box2 = Cuboid3D(x=10.2, y=0.1, z=0.0, dx=4.5, dy=2.0, dz=1.5, yaw=0.0, score=0.80, class_name="vehicles")
    box_far = Cuboid3D(x=25.0, y=5.0, z=0.0, dx=4.5, dy=2.0, dz=1.5, yaw=0.0, score=0.85, class_name="vehicles")

    assert center_in_box(box2, box1)
    merged = merge_boxes([box1, box2, box_far])
    assert len(merged) == 2
    assert merged[0].score == 0.90
    assert merged[1].score == 0.85


def test_drop_boxes_on_cars():
    car = Cuboid3D(x=10.0, y=0.0, z=0.0, dx=4.5, dy=2.0, dz=1.5, yaw=0.0, class_name="vehicles")
    ped_inside = Cuboid3D(x=10.2, y=0.1, z=0.5, dx=0.6, dy=0.6, dz=1.7, yaw=0.0, class_name="pedestrian")
    ped_outside = Cuboid3D(x=15.0, y=3.0, z=0.0, dx=0.6, dy=0.6, dz=1.7, yaw=0.0, class_name="pedestrian")

    filtered = drop_boxes_on_cars([car, ped_inside, ped_outside])
    assert len(filtered) == 2
    class_names = [b.class_name for b in filtered]
    assert "vehicles" in class_names
    assert ped_outside in filtered
    assert ped_inside not in filtered
