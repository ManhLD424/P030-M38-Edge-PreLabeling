"""Tests for PointPillars 3D detector and benchmark suite."""
from pathlib import Path
import pytest
from src.core.pointcloud.pcd_io import read_pcd
from src.core.models.detector import PointPillarsDetector, DetectorConfig
from src.core.benchmark.benchmark_3d import Benchmark3D, BenchmarkResult3D


DEMO_PCD = Path("data/pointclouds/demo.pcd")


def test_detector_on_demo_pcd():
    assert DEMO_PCD.is_file()
    pcd = read_pcd(DEMO_PCD)

    cfg = DetectorConfig(score_thresh=0.25)
    detector = PointPillarsDetector(config=cfg)

    cuboids = detector.detect(pcd.xyz)
    assert isinstance(cuboids, list)
    assert len(cuboids) > 0

    for c in cuboids:
        assert c.dx > 0
        assert c.dy > 0
        assert c.dz > 0
        assert c.score >= 0.0
        assert len(c.to_cvat_points()) == 16


def test_benchmark_3d_execution():
    assert DEMO_PCD.is_file()
    bench = Benchmark3D()
    result = bench.run(DEMO_PCD, runs=3, warmup=1)

    assert isinstance(result, BenchmarkResult3D)
    assert result.num_points == 17238
    assert result.runs == 3
    assert result.mean_total_ms > 0
    assert result.throughput_fps > 0
    assert result.peak_ram_mb > 0
