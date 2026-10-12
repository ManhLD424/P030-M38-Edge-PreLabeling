"""Ground estimation and height coordinates transformation for LiDAR point clouds."""
from __future__ import annotations

from typing import Union, Tuple
import numpy as np


def estimate_ground(points: Union[np.ndarray, list], bin_m: float = 0.05) -> float:
    """Estimates ground height using histogram binning of Z coordinates.
    
    Args:
        points: Array of shape (N, 3) or list of (x, y, z).
        bin_m: Histogram bin width in meters. Default 0.05m (5cm).
        
    Returns:
        Estimated ground Z height in meters.
    """
    if bin_m <= 0:
        raise ValueError("bin_m must be positive")
    if len(points) == 0:
        return 0.0

    if isinstance(points, list):
        zs = [p[2] for p in points]
    else:
        zs = points[:, 2]

    counts: dict[int, int] = {}
    for z in zs:
        bucket = int(z // bin_m)
        counts[bucket] = counts.get(bucket, 0) + 1

    if not counts:
        return 0.0

    peak_bucket = max(counts, key=counts.get)
    return float((peak_bucket + 0.5) * bin_m)


def estimate_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.06,
    max_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC plane fitting on points to find ground plane: ax + by + cz + d = 0.
    
    Returns:
        plane_params: (a, b, c, d) normalized where [a, b, c] is normal vector.
        inliers_mask: boolean mask of inlier points.
    """
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0, 0.0]), np.ones(len(points), dtype=bool)

    # Filter lowest 30% points to bias ground candidate search
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, 35)
    candidates_idx = np.where(z_vals <= z_thresh)[0]
    if len(candidates_idx) < 3:
        candidates_idx = np.arange(len(points))

    best_inliers = np.zeros(len(points), dtype=bool)
    best_plane = np.array([0.0, 0.0, 1.0, -float(np.median(z_vals))])
    max_inlier_count = 0

    pts_cand = points[candidates_idx, :3]

    for _ in range(max_iterations):
        sample_indices = np.random.choice(len(pts_cand), size=3, replace=False)
        p1, p2, p3 = pts_cand[sample_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Ground normal should be largely upright (Z component near 1 or -1)
        if abs(normal[2]) < 0.7:
            continue
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, p1)

        # Distance from all points to plane: |ax + by + cz + d|
        distances = np.abs(np.dot(points[:, :3], normal) + d)
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inliers = inliers
            best_plane = np.append(normal, d)

    return best_plane, best_inliers


def z_to_model(z_source: float, z_ground: float, delta: float) -> float:
    """Shifts source-frame Z into the model/checkpoint frame: z_model = z_source - z_ground - delta."""
    return z_source - z_ground - delta


def z_to_source(z_model: float, z_ground: float, delta: float) -> float:
    """Shifts model-frame Z back to source scan frame: z_source = z_model + delta + z_ground."""
    return z_model + delta + z_ground
