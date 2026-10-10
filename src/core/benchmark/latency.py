"""
Module Benchmark Latency & Throughput (FPS).
"""

import time
from typing import Dict, Any, List
import numpy as np
from src.core.inference.base import BaseInferenceEngine


class LatencyBenchmark:
    """
    Thực hiện đo đạc thời gian suy luận, độ trễ và thông lượng phần cứng chuẩn khoa học.
    """

    def __init__(self, engine: BaseInferenceEngine, warmup_runs: int = 15, test_runs: int = 100):
        self.engine = engine
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def run(self, input_image: np.ndarray) -> Dict[str, float]:
        """
        Chạy quy trình warmup và benchmark.
        Trả về dictionary chứa các chỉ số: mean_ms, p50_ms, p90_ms, p99_ms, fps.
        """
        # 1. Warmup runs để ổn định xung nhịp CPU và cache L1/L2/L3
        for _ in range(self.warmup_runs):
            _ = self.engine.predict(input_image)

        # 2. Benchmark thực nghiệm bằng time.perf_counter_ns()
        latencies_ms: List[float] = []
        for _ in range(self.test_runs):
            start = time.perf_counter_ns()
            _ = self.engine.predict(input_image)
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1_000_000.0)

        latencies = np.array(latencies_ms)
        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p90_ms = float(np.percentile(latencies, 90))
        p99_ms = float(np.percentile(latencies, 99))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        return {
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p90_ms": round(p90_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "fps": round(fps, 1),
            "runs": self.test_runs
        }
