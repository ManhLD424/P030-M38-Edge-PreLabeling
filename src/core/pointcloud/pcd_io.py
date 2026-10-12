"""Point Cloud Data (PCD) file parser and writer."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PCDData:
    """Parsed Point Cloud Data representation."""
    points: np.ndarray  # Shape (N, 3) or (N, 4)
    fields: List[str]
    count: int
    header: Dict[str, Any]
    intensity: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None

    @property
    def xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of shape (N, 3)."""
        return self.points[:, :3]

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Returns (xmin, ymin, zmin, xmax, ymax, zmax)."""
        if self.count == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        xyz = self.xyz
        return (
            float(xyz[:, 0].min()),
            float(xyz[:, 1].min()),
            float(xyz[:, 2].min()),
            float(xyz[:, 0].max()),
            float(xyz[:, 1].max()),
            float(xyz[:, 2].max()),
        )


def read_pcd(path: Union[str, Path]) -> PCDData:
    """Reads a PCD file (supports both DATA binary and DATA ascii)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PCD file not found: {p}")

    raw = p.read_bytes()
    marker_bin = b"DATA binary\n"
    marker_ascii = b"DATA ascii\n"

    is_binary = True
    marker_idx = raw.find(marker_bin)
    if marker_idx >= 0:
        header_bytes = raw[:marker_idx]
        body_bytes = raw[marker_idx + len(marker_bin):]
    else:
        marker_idx = raw.find(marker_ascii)
        if marker_idx >= 0:
            is_binary = False
            header_bytes = raw[:marker_idx]
            body_bytes = raw[marker_idx + len(marker_ascii):]
        else:
            raise ValueError(f"Unsupported or invalid PCD file (missing DATA header): {p.name}")

    header_text = header_bytes.decode("ascii", errors="replace")
    header_dict: Dict[str, Any] = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header_dict[parts[0].upper()] = parts[1:]

    fields = header_dict.get("FIELDS", ["x", "y", "z"])
    count = int(header_dict.get("POINTS", [0])[0])
    sizes = [int(s) for s in header_dict.get("SIZE", [4] * len(fields))]
    types = header_dict.get("TYPE", ["F"] * len(fields))

    record_size = sum(sizes)

    if is_binary:
        if len(body_bytes) < count * record_size:
            count = len(body_bytes) // record_size
        
        # Check standard 4-float fields (e.g. x y z rgb or x y z intensity)
        if fields[:3] == ["x", "y", "z"]:
            if len(fields) >= 4 and record_size == 16:
                arr = np.frombuffer(body_bytes[:count * 16], dtype=np.float32).reshape(-1, 4)
                points = arr[:, :3].copy()
                extra = arr[:, 3].copy()
                return PCDData(
                    points=points,
                    fields=fields,
                    count=len(points),
                    header=header_dict,
                    intensity=extra,
                )
            elif len(fields) == 3 and record_size == 12:
                points = np.frombuffer(body_bytes[:count * 12], dtype=np.float32).reshape(-1, 3).copy()
                return PCDData(
                    points=points,
                    fields=fields,
                    count=len(points),
                    header=header_dict,
                )
        
        # Generic unpack
        fmt = "<" + "".join("f" if t == "F" else "I" for t in types)
        step = struct.calcsize(fmt)
        pts = []
        for offset in range(count):
            unpacked = struct.unpack_from(fmt, body_bytes, offset * step)
            pts.append(unpacked[:3])
        points = np.asarray(pts, dtype=np.float32)
        return PCDData(
            points=points,
            fields=fields,
            count=len(points),
            header=header_dict,
        )
    else:
        # ASCII parsing
        lines = body_bytes.decode("utf-8", errors="replace").strip().splitlines()
        pts = []
        for line in lines[:count]:
            vals = [float(v) for v in line.split()[:3]]
            if len(vals) == 3:
                pts.append(vals)
        points = np.asarray(pts, dtype=np.float32)
        return PCDData(
            points=points,
            fields=fields,
            count=len(points),
            header=header_dict,
        )


def write_pcd(path: Union[str, Path], points: np.ndarray, binary: bool = True) -> None:
    """Writes point coordinates to a PCD file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "# .PCD v0.7 - Exported by M38-Edge-PreLabeling\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )
    if binary:
        xyz_bytes = points[:, :3].astype(np.float32).tobytes()
        with p.open("wb") as f:
            f.write(header.encode("ascii"))
            f.write(xyz_bytes)
    else:
        with p.open("w", encoding="utf-8") as f:
            f.write(header)
            for pt in points[:, :3]:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
