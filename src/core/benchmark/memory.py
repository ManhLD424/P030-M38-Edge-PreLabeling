"""
Module Benchmark Memory & Disk Storage Footprint.
"""

from pathlib import Path
from typing import Dict, Any, List
import psutil
import os


class MemoryBenchmark:
    """
    Đo đạc dung lượng lưu trữ trên đĩa (Disk MB) và mức tiêu hao bộ nhớ RAM (RSS MB).
    """

    @staticmethod
    def get_model_size_mb(model_path: str) -> float:
        """
        Tính tổng kích thước của mô hình trên ổ đĩa.
        Với OpenVINO IR, tính tổng cả cặp file .xml và .bin.
        """
        p = Path(model_path)
        if not p.exists():
            return 0.0

        total_bytes = 0
        if p.suffix == ".xml":
            bin_file = p.with_suffix(".bin")
            total_bytes += p.stat().st_size
            if bin_file.exists():
                total_bytes += bin_file.stat().st_size
        elif p.is_file():
            total_bytes = p.stat().st_size

        return round(total_bytes / (1024 * 1024), 2)

    @staticmethod
    def get_process_ram_mb() -> float:
        """
        Lấy dung lượng bộ nhớ RAM thực tế (Resident Set Size - RSS) của tiến trình hiện tại.
        """
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
