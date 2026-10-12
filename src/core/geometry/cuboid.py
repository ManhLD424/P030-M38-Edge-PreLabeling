"""3D Cuboid Bounding Box dataclass and spatial manipulation functions."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Cuboid3D:
    """Represents a 3D bounding box (cuboid) in LiDAR coordinate frame.
    
    Coordinates:
        x: Center forward coordinate (meters)
        y: Center left coordinate (meters)
        z: Center vertical coordinate (meters)
        dx: Length along heading (meters)
        dy: Width across heading (meters)
        dz: Height along z (meters)
        yaw: Heading angle around z-axis in radians [-pi, pi], 0 pointing forward (+X)
        score: Detection confidence [0.0 .. 1.0]
        class_name: Class name ("vehicles", "pedestrian", "two-wheels", etc.)
        label_id: Optional CVAT label ID
    """
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float = 0.0
    score: float = 1.0
    class_name: str = "vehicles"
    label_id: Optional[int] = None

    def to_cvat_points(self) -> List[float]:
        """Converts to CVAT 3D 16-element point format:
        [x, y, z, rot_x, rot_y, rot_z, dx, dy, dz, 0, 0, 0, 0, 0, 0, 0]
        """
        return [
            float(self.x),
            float(self.y),
            float(self.z),
            0.0,           # rot_x (roll)
            0.0,           # rot_y (pitch)
            float(self.yaw), # rot_z (yaw in radians)
            float(self.dx),  # length (L)
            float(self.dy),  # width (W)
            float(self.dz),  # height (H)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding / reserved
        ]

    @classmethod
    def from_cvat_points(
        cls,
        points: List[float],
        class_name: str = "vehicles",
        score: float = 1.0,
        label_id: Optional[int] = None
    ) -> Cuboid3D:
        """Constructs a Cuboid3D from CVAT 16-element point array."""
        if len(points) < 9:
            raise ValueError(f"CVAT 3D cuboid requires at least 9 points, got {len(points)}")
        x, y, z = float(points[0]), float(points[1]), float(points[2])
        yaw = float(points[5])
        dx, dy, dz = float(points[6]), float(points[7]), float(points[8])
        return cls(
            x=x, y=y, z=z,
            dx=dx, dy=dy, dz=dz,
            yaw=yaw,
            score=score,
            class_name=class_name,
            label_id=label_id
        )

    def corners_3d(self) -> np.ndarray:
        """Computes the 8 corner vertices in 3D coordinates.
        Returns shape (8, 3).
        """
        l_half, w_half, h_half = self.dx / 2.0, self.dy / 2.0, self.dz / 2.0
        # 8 canonical corners
        canonical = np.array([
            [l_half,  w_half,  h_half],
            [l_half, -w_half,  h_half],
            [-l_half, -w_half,  h_half],
            [-l_half,  w_half,  h_half],
            [l_half,  w_half, -h_half],
            [l_half, -w_half, -h_half],
            [-l_half, -w_half, -h_half],
            [-l_half,  w_half, -h_half],
        ], dtype=np.float32)

        # Rotation matrix around Z
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        rot_mat = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        rotated = np.dot(canonical, rot_mat.T)
        rotated[:, 0] += self.x
        rotated[:, 1] += self.y
        rotated[:, 2] += self.z
        return rotated

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary."""
        d = asdict(self)
        d["cvat_points"] = self.to_cvat_points()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Cuboid3D:
        """Instantiates from dictionary."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            dx=float(data.get("dx", data.get("length", 1.0))),
            dy=float(data.get("dy", data.get("width", 1.0))),
            dz=float(data.get("dz", data.get("height", 1.0))),
            yaw=float(data.get("yaw", 0.0)),
            score=float(data.get("score", 1.0)),
            class_name=str(data.get("class_name", data.get("label", "vehicles"))),
            label_id=data.get("label_id"),
        )


def center_in_box(inner: Cuboid3D, outer: Cuboid3D, pad_m: float = 0.0) -> bool:
    """Checks whether the 2D ground-plane center of inner box lies inside outer box."""
    dx = inner.x - outer.x
    dy = inner.y - outer.y
    cos_y = math.cos(outer.yaw)
    sin_y = math.sin(outer.yaw)
    along = cos_y * dx + sin_y * dy
    across = -sin_y * dx + cos_y * dy
    return (abs(along) <= (outer.dx / 2.0 + pad_m)) and (abs(across) <= (outer.dy / 2.0 + pad_m))


def merge_boxes(boxes: List[Cuboid3D], pad_m: float = 0.0) -> List[Cuboid3D]:
    """Drops lower-scoring duplicates when centers fall inside each other's footprint."""
    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[Cuboid3D] = []
    for box in ordered:
        if any(
            other.class_name == box.class_name and (
                center_in_box(box, other, pad_m) or center_in_box(other, box, pad_m)
            )
            for other in kept
        ):
            continue
        kept.append(box)
    return kept


def drop_boxes_on_cars(boxes: List[Cuboid3D]) -> List[Cuboid3D]:
    """Removes pedestrians or cyclists whose centers sit inside vehicle cuboids."""
    cars = [b for b in boxes if b.class_name in ("vehicles", "car", "bus", "truck")]
    kept: List[Cuboid3D] = []
    for box in boxes:
        if box.class_name not in ("vehicles", "car", "bus", "truck"):
            if any(center_in_box(box, car, pad_m=0.0) for car in cars):
                continue
        kept.append(box)
    return kept
