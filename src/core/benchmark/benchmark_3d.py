"""3D Point Cloud Pre-labeling Benchmarking Suite."""
from __future__ import annotations

import time
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ..pointcloud.pcd_io import read_pcd, PCDData
from ..pointcloud.ground import estimate_ground
from ..models.detector import PointPillarsDetector
from ..geometry.cvat_adapter import CVAT3DAdapter


@dataclass
class BenchmarkResult3D:
    """Benchmark results for 3D point cloud pre-labeling."""
    pcd_path: str
    num_points: int
    runs: int
    mean_total_ms: float
    p50_total_ms: float
    p90_total_ms: float
    mean_io_ms: float
    mean_ground_ms: float
    mean_detect_ms: float
    mean_cvat_ms: float
    throughput_fps: float
    peak_ram_mb: float
    detected_cuboids: int


class Benchmark3D:
    """Measures latency, throughput, and memory consumption for 3D pre-labeling."""

    def __init__(self, detector: Optional[PointPillarsDetector] = None):
        self.detector = detector or PointPillarsDetector()
        self.adapter = CVAT3DAdapter()

    def run(self, pcd_path: str | Path, runs: int = 30, warmup: int = 5) -> BenchmarkResult3D:
        """Executes repeated benchmark runs on a PCD file."""
        p = Path(pcd_path)
        if not p.is_file():
            raise FileNotFoundError(f"PCD file not found: {p}")

        # Warmup
        pcd_data = read_pcd(p)
        for _ in range(warmup):
            _ = self.detector.detect(pcd_data.xyz)

        io_times: List[float] = []
        ground_times: List[float] = []
        detect_times: List[float] = []
        cvat_times: List[float] = []
        total_times: List[float] = []

        last_cuboids = []

        proc = psutil.Process()
        ram_before = proc.memory_info().rss / (1024 * 1024)

        for _ in range(runs):
            t0 = time.perf_counter()

            # 1. IO
            pcd = read_pcd(p)
            t1 = time.perf_counter()
            io_times.append((t1 - t0) * 1000)

            # 2. Ground
            z_ground = estimate_ground(pcd.xyz)
            t2 = time.perf_counter()
            ground_times.append((t2 - t1) * 1000)

            # 3. Detect
            cuboids = self.detector.detect(pcd.xyz, z_ground=z_ground)
            t3 = time.perf_counter()
            detect_times.append((t3 - t2) * 1000)

            # 4. CVAT Formatting
            _ = self.adapter.to_shapes_payload(cuboids)
            t4 = time.perf_counter()
            cvat_times.append((t4 - t3) * 1000)

            total_times.append((t4 - t0) * 1000)
            last_cuboids = cuboids

        ram_after = proc.memory_info().rss / (1024 * 1024)
        peak_ram = max(ram_before, ram_after)

        total_arr = np.array(total_times)
        mean_total = float(np.mean(total_arr))
        p50_total = float(np.percentile(total_arr, 50))
        p90_total = float(np.percentile(total_arr, 90))
        fps = float(1000.0 / mean_total) if mean_total > 0 else 0.0

        return BenchmarkResult3D(
            pcd_path=str(p),
            num_points=pcd_data.count,
            runs=runs,
            mean_total_ms=round(mean_total, 2),
            p50_total_ms=round(p50_total, 2),
            p90_total_ms=round(p90_total, 2),
            mean_io_ms=round(float(np.mean(io_times)), 2),
            mean_ground_ms=round(float(np.mean(ground_times)), 2),
            mean_detect_ms=round(float(np.mean(detect_times)), 2),
            mean_cvat_ms=round(float(np.mean(cvat_times)), 2),
            throughput_fps=round(fps, 1),
            peak_ram_mb=round(peak_ram, 1),
            detected_cuboids=len(last_cuboids),
        )
