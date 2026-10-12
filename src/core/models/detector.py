"""PointPillars 3D Object Detection Engine & Pre-labeling Pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..geometry.cuboid import Cuboid3D, merge_boxes, drop_boxes_on_cars
from ..pointcloud.ground import estimate_ground, z_to_model, z_to_source


@dataclass
class DetectorConfig:
    """Configuration for PointPillars 3D Object Detector."""
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        0.0, -39.68, -3.0, 69.12, 39.68, 1.0
    )
    voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0)
    delta: float = 1.73               # Default sensor height for KITTI preset
    score_thresh: float = 0.30
    heading_mode: str = "kitti"       # "kitti" or "forward"
    z_bottom: bool = True             # KITTI checkpoint z is bottom face
    classes: List[str] = field(default_factory=lambda: ["vehicles", "pedestrian", "two-wheels"])
    checkpoint_path: Optional[str] = None
    use_fallback_clustering: bool = True  # Allows offline/edge inference if .pth not present


class PointPillarsDetector:
    """Unified 3D Point Cloud Detector supporting PyTorch PointPillars and Edge Clustering."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempts to load PyTorch PointPillars checkpoint if available."""
        if not self.config.checkpoint_path:
            return
        ckpt_p = Path(self.config.checkpoint_path)
        if not ckpt_p.is_file():
            return

        try:
            import torch
            # Check if pointpillars package is available
            try:
                from pointpillars.model.pointpillars import PointPillars
                model = PointPillars(
                    nclasses=len(self.config.classes),
                    voxel_size=list(self.config.voxel_size),
                    point_cloud_range=list(self.config.point_cloud_range),
                )
                state = torch.load(ckpt_p, map_location=torch.device("cpu"))
                model.load_state_dict(state)
                model.score_thr = self.config.score_thresh
                model.eval()
                self.model = model
            except ImportError:
                self.model = None
        except Exception:
            self.model = None

    def detect(
        self,
        points: np.ndarray,
        z_ground: Optional[float] = None,
        full_scene: bool = False
    ) -> List[Cuboid3D]:
        """Runs 3D detection on input points (N, 3) or (N, 4).
        
        Args:
            points: Array of XYZ coordinates in scan frame.
            z_ground: Optional estimated ground level. If None, auto-estimated.
            full_scene: If True, executes front pass and rear 180-degree pass.
            
        Returns:
            List of detected 3D Cuboids in scan coordinate frame.
        """
        if len(points) == 0:
            return []

        xyz = points[:, :3]
        if z_ground is None:
            z_ground = estimate_ground(xyz)

        passes = [("identity", 1.0)]
        if full_scene:
            passes.append(("yaw180", -1.0))

        raw_boxes: List[Cuboid3D] = []

        if self.model is not None:
            raw_boxes = self._detect_pytorch(xyz, z_ground, passes)
        elif self.config.use_fallback_clustering:
            raw_boxes = self._detect_edge_clusters(xyz, z_ground)

        # Post-processing: deduplication and hierarchical filtering
        merged = merge_boxes(raw_boxes)
        final_boxes = drop_boxes_on_cars(merged)
        return final_boxes

    def _detect_edge_clusters(self, xyz: np.ndarray, z_ground: float) -> List[Cuboid3D]:
        """Edge-optimized 3D geometric cluster detector (calibrated to CVAT 3D vehicle dimensions).
        Used on CPU edge nodes when GPU/weights are not mounted.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = self.config.point_cloud_range
        delta = self.config.delta

        # Shift to model frame
        zm = z_to_model(xyz[:, 2], z_ground, delta)
        mask = (
            (xyz[:, 0] > xmin) & (xyz[:, 0] < xmax) &
            (xyz[:, 1] > ymin) & (xyz[:, 1] < ymax) &
            (zm > zmin) & (zm < zmax) &
            (xyz[:, 2] > (z_ground + 0.10))  # Filter points directly on the ground
        )
        pts = xyz[mask]
        if len(pts) < 15:
            return []

        # Grid-based spatial voxelization clustering
        voxel_res = 0.50
        coords = np.floor(pts[:, :2] / voxel_res).astype(np.int32)
        unique_voxels, inv_idx = np.unique(coords, axis=0, return_inverse=True)

        voxel_counts = np.bincount(inv_idx)
        valid_voxel_mask = voxel_counts >= 4
        valid_voxels = unique_voxels[valid_voxel_mask]

        if len(valid_voxels) == 0:
            return []

        # Simple adjacency grouping in voxel space
        clusters: List[List[int]] = []
        visited = set()
        v_set = set(map(tuple, valid_voxels))

        for vx, vy in valid_voxels:
            if (vx, vy) in visited:
                continue
            group = []
            queue = [(vx, vy)]
            visited.add((vx, vy))

            while queue:
                cx, cy = queue.pop()
                group.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nb = (cx + dx, cy + dy)
                    if nb in v_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            if len(group) >= 2:
                # Collect all point indices belonging to this voxel cluster
                group_set = set(group)
                pt_indices = [i for i, c in enumerate(coords) if tuple(c) in group_set]
                if len(pt_indices) >= 12:
                    clusters.append(pt_indices)

        boxes: List[Cuboid3D] = []
        for c_indices in clusters:
            c_pts = pts[c_indices]
            # Dimensions
            c_min = c_pts.min(axis=0)
            c_max = c_pts.max(axis=0)
            raw_dx = float(c_max[0] - c_min[0])
            raw_dy = float(c_max[1] - c_min[1])
            dz = float(c_max[2] - c_min[2])

            center_x = float((c_min[0] + c_max[0]) / 2.0)
            center_y = float((c_min[1] + c_max[1]) / 2.0)
            center_z = float((c_min[2] + c_max[2]) / 2.0)

            # Fit vehicle profile: length 2.8-5.8m, width 1.4-2.3m, height 1.1-2.2m
            if 1.8 <= max(raw_dx, raw_dy) <= 6.5 and 0.8 <= min(raw_dx, raw_dy) <= 3.0 and 0.8 <= dz <= 2.8:
                # Estimate yaw from PCA on 2D coordinates
                xy_centered = c_pts[:, :2] - [center_x, center_y]
                cov = np.cov(xy_centered, rowvar=False)
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                major_vec = eigenvecs[:, -1]
                yaw = float(math.atan2(major_vec[1], major_vec[0]))

                # Clamp dimensions to vehicle standards
                dx = float(np.clip(max(raw_dx, raw_dy), 2.8, 5.8))
                dy = float(np.clip(min(raw_dx, raw_dy), 1.4, 2.3))
                dz = float(np.clip(dz, 1.2, 2.2))

                boxes.append(Cuboid3D(
                    x=center_x,
                    y=center_y,
                    z=center_z,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    yaw=yaw,
                    score=0.88,
                    class_name="vehicles"
                ))
            elif 0.3 <= max(raw_dx, raw_dy) <= 1.2 and 1.1 <= dz <= 2.1:
                # Pedestrian profile
                boxes.append(Cuboid3D(
                    x=center_x,
                    y=center_y,
                    z=center_z,
                    dx=0.7,
                    dy=0.7,
                    dz=float(np.clip(dz, 1.3, 1.9)),
                    yaw=0.0,
                    score=0.75,
                    class_name="pedestrian"
                ))

        return boxes

    def _detect_pytorch(self, xyz: np.ndarray, z_ground: float, passes: List[Tuple[str, float]]) -> List[Cuboid3D]:
        """Inference with PyTorch PointPillars module."""
        import torch
        boxes: List[Cuboid3D] = []
        delta = self.config.delta
        xmin, ymin, zmin, xmax, ymax, zmax = self.config.point_cloud_range

        for pass_name, sign in passes:
            rows = []
            for pt in xyz:
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                if pass_name == "yaw180":
                    x, y = -x, -y
                zm = z_to_model(z, z_ground, delta)
                if xmin < x < xmax and ymin < y < ymax and zmin < zm < zmax:
                    rows.append((x, y, zm, 0.0))

            if not rows:
                continue

            cloud = torch.from_numpy(np.asarray(rows, dtype=np.float32))
            with torch.no_grad():
                result = self.model(batched_pts=[cloud], mode="test")[0]

            if not isinstance(result, dict) or "lidar_bboxes" not in result:
                continue

            for bbox, label, score in zip(result["lidar_bboxes"], result["labels"], result["scores"]):
                class_idx = int(label)
                if class_idx >= len(self.config.classes):
                    continue
                class_name = self.config.classes[class_idx]
                x_b, y_b, zm_b, width, length, height, raw_yaw = [float(v) for v in bbox]
                if self.config.z_bottom:
                    zm_b += height / 2.0

                # Coordinate transformations
                if self.config.heading_mode == "kitti":
                    yaw = math.pi / 2 - raw_yaw + math.pi
                else:
                    yaw = raw_yaw
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))

                if pass_name == "yaw180":
                    x_b, y_b = -x_b, -y_b
                    yaw = math.atan2(math.sin(yaw + math.pi), math.cos(yaw + math.pi))

                boxes.append(Cuboid3D(
                    x=x_b,
                    y=y_b,
                    z=z_to_source(zm_b, z_ground, delta),
                    dx=length,
                    dy=width,
                    dz=height,
                    yaw=yaw,
                    score=float(score),
                    class_name=class_name
                ))

        return boxes
