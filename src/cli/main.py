import sys
import os
import platform
from pathlib import Path

# Đảm bảo Windows terminal hiển thị tiếng Việt UTF-8 mượt mà
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import click
import numpy as np
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(force_terminal=True)


@click.group()
def cli():
    """P030-M38: Model Compression & Edge Pre-Labeling CLI Tool."""
    pass


@cli.command()
def info():
    """Hiển thị thông số môi trường phần cứng và các thư viện cốt lõi."""
    table = Table(title="[bold green]THÔNG SỐ MÔI TRƯỜNG PHẦN CỨNG & HỆ THỐNG[/bold green]")
    table.add_column("Thành phần", style="cyan", no_wrap=True)
    table.add_column("Giá trị", style="magenta")

    table.add_row("Hệ điều hành", f"{platform.system()} {platform.release()} ({platform.architecture()[0]})")
    table.add_row("Bộ vi xử lý (CPU)", platform.processor() or "x86_64")
    table.add_row("Số nhân CPU (Cores / Threads)", f"{psutil.cpu_count(logical=False)} Cores / {psutil.cpu_count(logical=True)} Threads")
    table.add_row("Tổng bộ nhớ RAM", f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    table.add_row("Python Version", sys.version.split()[0])

    try:
        import torch
        table.add_row("PyTorch Version", torch.__version__)
    except ImportError:
        table.add_row("PyTorch Version", "Chưa cài đặt")

    try:
        import openvino as ov
        table.add_row("OpenVINO Version", ov.__version__)
    except ImportError:
        table.add_row("OpenVINO Version", "Chưa cài đặt")

    try:
        import nncf
        table.add_row("NNCF Version", nncf.__version__)
    except ImportError:
        table.add_row("NNCF Version", "Chưa cài đặt")

    try:
        import onnxruntime as ort
        table.add_row("ONNX Runtime Version", ort.__version__)
    except ImportError:
        table.add_row("ONNX Runtime Version", "Chưa cài đặt")

    console.print(table)


@cli.command()
@click.option("--weights", default="models/raw/yolov8s.pt", help="Đường dẫn checkpoint PyTorch gốc (.pt)")
@click.option("--calib-dir", default="data/calibration", help="Thư mục ảnh calibration cho NNCF")
@click.option("--subset-size", default=50, help="Số lượng ảnh calibration")
def compress(weights, calib_dir, subset_size):
    """Thực thi toàn bộ pipeline nén mô hình: FP32 -> ONNX -> OpenVINO FP16 -> OpenVINO INT8."""
    from src.core.compression.exporter import ModelExporter
    from src.core.compression.quantizer import ModelQuantizer

    console.print(Panel(f"[bold cyan]BẮT ĐẦU PIPELINE NÉN MÔ HÌNH: {weights}[/bold cyan]"))

    # 1. Xuất ONNX
    exporter = ModelExporter(weights_path=weights)
    onnx_path = exporter.export_onnx()

    # 2. Xuất OpenVINO FP16
    ov_fp16_path = exporter.export_openvino_fp16()

    # 3. Lượng tử hóa NNCF INT8
    quantizer = ModelQuantizer(str(ov_fp16_path))
    int8_path = quantizer.quantize_int8(
        output_dir="models/openvino_int8",
        calib_dir=calib_dir,
        subset_size=subset_size
    )

    console.print(Panel(f"[bold green]HOÀN THÀNH PIPELINE NÉN THÀNH CÔNG![/bold green]\n"
                        f"- ONNX Model: {onnx_path}\n"
                        f"- OpenVINO FP16: {ov_fp16_path}\n"
                        f"- OpenVINO INT8: {int8_path}"))


@cli.command()
@click.option("--runs", default=30, help="Số lượt suy luận benchmark")
def benchmark(runs):
    """Chạy đo kiểm hiệu năng so sánh đa Runtime (PyTorch FP32 vs ONNX vs OpenVINO FP16 vs INT8)."""
    from src.core.inference.torch_engine import TorchInferenceEngine
    from src.core.inference.ov_engine import OpenVINOInferenceEngine
    from src.core.inference.onnx_engine import ONNXInferenceEngine
    from src.core.benchmark.latency import LatencyBenchmark
    from src.core.benchmark.memory import MemoryBenchmark

    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    models_to_test = [
        ("PyTorch (FP32 Baseline)", TorchInferenceEngine, "models/raw/yolov8s.pt"),
        ("ONNX Runtime (FP32)", ONNXInferenceEngine, "models/onnx/yolov8s.onnx"),
        ("OpenVINO (FP16)", OpenVINOInferenceEngine, "models/openvino_fp16/yolov8s_fp16.xml"),
        ("OpenVINO (INT8 Compressed)", OpenVINOInferenceEngine, "models/openvino_int8/yolov8s_int8.xml"),
    ]

    results_table = Table(title="[bold yellow]KẾT QUẢ BENCHMARK HIỆU NĂNG THỰC NGHIỆM TRÊN CPU[/bold yellow]")
    results_table.add_column("Runtime / Phiên bản", style="cyan", no_wrap=True)
    results_table.add_column("Model Size", style="green")
    results_table.add_column("Mean Latency", style="yellow")
    results_table.add_column("P50 (Median)", style="white")
    results_table.add_column("P90", style="white")
    results_table.add_column("Throughput (FPS)", style="bold magenta")

    baseline_fps = None

    for name, engine_cls, model_path in models_to_test:
        if not Path(model_path).exists():
            console.print(f"[yellow][!] Bỏ qua {name} (Chưa tìm thấy {model_path})[/yellow]")
            continue

        console.print(f"[*] Đang đo kiểm: [bold]{name}[/bold] ({runs} lượt)...")
        size_mb = MemoryBenchmark.get_model_size_mb(model_path)
        
        try:
            engine = engine_cls(model_path=model_path)
            bench = LatencyBenchmark(engine, warmup_runs=5, test_runs=runs)
            stats = bench.run(test_img)

            fps = stats["fps"]
            if baseline_fps is None:
                baseline_fps = fps

            results_table.add_row(
                name,
                f"{size_mb} MB",
                f"{stats['mean_ms']} ms",
                f"{stats['p50_ms']} ms",
                f"{stats['p90_ms']} ms",
                f"{fps} FPS"
            )
        except Exception as e:
            console.print(f"[red][!] Lỗi khi đo {name}: {e}[/red]")

    console.print(results_table)


if __name__ == "__main__":
    cli()
