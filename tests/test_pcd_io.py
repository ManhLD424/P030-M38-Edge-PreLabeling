"""Tests for Point Cloud Data (PCD) I/O and processing."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.core.pointcloud.pcd_io import read_pcd, write_pcd, PCDData
from src.core.pointcloud.ground import estimate_ground, z_to_model, z_to_source


DEMO_PCD = Path("data/pointclouds/demo.pcd")


def test_read_demo_pcd():
    assert DEMO_PCD.is_file(), f"demo.pcd missing at {DEMO_PCD}"
    pcd = read_pcd(DEMO_PCD)
    assert isinstance(pcd, PCDData)
    assert pcd.count == 17238
    assert pcd.xyz.shape == (17238, 3)
    assert pcd.xyz.dtype == np.float32

    xmin, ymin, zmin, xmax, ymax, zmax = pcd.bounds
    assert xmin < xmax
    assert ymin < ymax
    assert zmin < zmax


def test_estimate_ground():
    pcd = read_pcd(DEMO_PCD)
    z_g = estimate_ground(pcd.xyz)
    assert isinstance(z_g, float)
    # Ground peak in this normalized scan should be roughly between -0.2 and 0.5m
    assert -0.5 <= z_g <= 0.8


def test_height_coordinate_transformation():
    z_source = 1.5
    z_ground = 0.1
    delta = 1.73

    z_model = z_to_model(z_source, z_ground, delta)
    recovered_z = z_to_source(z_model, z_ground, delta)
    assert pytest.approx(recovered_z, 1e-6) == z_source


def test_pcd_roundtrip_write_read():
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.5, 5.5, 6.5],
        [-1.0, -2.0, 0.5]
    ], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pcd = Path(tmpdir) / "test.pcd"
        write_pcd(tmp_pcd, pts, binary=True)
        assert tmp_pcd.is_file()

        loaded = read_pcd(tmp_pcd)
        assert loaded.count == 3
        np.testing.assert_allclose(loaded.xyz, pts, rtol=1e-5, atol=1e-5)
