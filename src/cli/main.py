"""
P030-M38: Model Compression & Edge Pre-Labeling Unified CLI Tool.
Hỗ trợ cả bài toán lõi 2D BDD100K (Model Compression, Multi-Runtime Benchmark, mAP Eval, Pre-labeling)
và mô-đun mở rộng 3D LiDAR Point Cloud cho xe tự hành.
"""

import sys
import os
import platform
import json
import time
from pathlib import Path

# Đảm bảo Windows terminal hiển thị tiếng Việt UTF-8 mượt mà
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import click
import cv2
import numpy as np
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(force_terminal=True)


@click.group()
def cli():
    """P030-M38: Model Compression for Edge Pre-Labeling CLI Tool."""
    pass


@cli.command()
def info():
    """Hiển thị thông số môi trường phần cứng và các thư viện suy luận tối ưu."""
    table = Table(title="[bold green]THÔNG SỐ MÔI TRƯỜNG PHẦN CỨNG & HỆ THỐNG M38[/bold green]")
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
        table.add_row("Intel NNCF Version", nncf.__version__)
    except ImportError:
        table.add_row("Intel NNCF Version", "Chưa cài đặt")

    try:
        import onnxruntime as ort
        table.add_row("ONNX Runtime Version", ort.__version__)
    except ImportError:
        table.add_row("ONNX Runtime Version", "Chưa cài đặt")

    try:
        import ultralytics
        table.add_row("Ultralytics YOLO", ultralytics.__version__)
    except ImportError:
        table.add_row("Ultralytics YOLO", "Chưa cài đặt")

    table.add_row("Tập dữ liệu chuẩn", "BDD100K (Berkeley DeepDrive) 2D Camera + PointCloud 3D")
    table.add_row("Chuẩn xuất nhãn", "CVAT Bounding Box / CVAT 3D 16-point Cuboids / YOLO Format")
    console.print(table)


@cli.command()
@click.option("--weights", default="models/raw/yolov8s.pt", help="Đường dẫn checkpoint PyTorch gốc (.pt)")
@click.option("--calib-dir", default="data/calibration", help="Thư mục ảnh calibration BDD100K cho NNCF")
@click.option("--subset-size", default=25, help="Số lượng ảnh calibration")
def compress(weights, calib_dir, subset_size):
    """Thực thi toàn bộ pipeline nén mô hình: FP32 -> ONNX -> OpenVINO FP16 -> OpenVINO INT8 (NNCF PTQ)."""
    from src.core.compression.exporter import ModelExporter
    from src.core.compression.quantizer import ModelQuantizer

    console.print(Panel(f"[bold cyan]BẮT ĐẦU PIPELINE NÉN MÔ HÌNH: {weights}[/bold cyan]"))

    # 1. Xuất ONNX
    exporter = ModelExporter(weights_path=weights)
    onnx_path = exporter.export_onnx()

    # 2. Xuất OpenVINO FP16
    ov_fp16_path = exporter.export_openvino_fp16()

    # 3. Lượng tử hóa NNCF INT8 với ảnh BDD100K thật
    quantizer = ModelQuantizer(str(ov_fp16_path))
    int8_path = quantizer.quantize_int8(
        output_dir="models/openvino_int8",
        calib_dir=calib_dir,
        subset_size=subset_size
    )

    console.print(Panel(f"[bold green]HOÀN THÀNH PIPELINE NÉN MÔ HÌNH THÀNH CÔNG![/bold green]\n"
                        f"- ONNX Model: {onnx_path}\n"
                        f"- OpenVINO FP16: {ov_fp16_path}\n"
                        f"- OpenVINO INT8 (NNCF PTQ): {int8_path}"))


@cli.command(name="benchmark-2d")
@click.option("--runs", default=30, help="Số lượt suy luận benchmark trên mỗi runtime")
@click.option("--save-json", default="outputs/benchmark_results.json", help="Đường dẫn lưu kết quả JSON")
def benchmark_2d(runs, save_json):
    """Đo kiểm hiệu năng đa Runtime trên CPU (PyTorch FP32 vs ONNX vs OpenVINO FP16 vs INT8)."""
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

    results_table = Table(title="[bold yellow]KẾT QUẢ BENCHMARK HIỆU NĂNG THỰC NGHIỆM TRÊN CPU (M38)[/bold yellow]")
    results_table.add_column("Runtime / Phiên bản", style="cyan", no_wrap=True)
    results_table.add_column("Model Size", style="green")
    results_table.add_column("Mean Latency", style="yellow")
    results_table.add_column("P50 (Median)", style="white")
    results_table.add_column("P90", style="white")
    results_table.add_column("Throughput (FPS)", style="bold magenta")
    results_table.add_column("Speedup vs Baseline", style="bold green")

    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or "x86_64",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "runs": runs,
        "results": []
    }

    baseline_mean = None

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

            if baseline_mean is None:
                baseline_mean = stats["mean_ms"]
                speedup_str = "1.0x (Baseline)"
                speedup_val = 1.0
            else:
                ratio = round(baseline_mean / stats["mean_ms"], 2) if stats["mean_ms"] > 0 else 0.0
                speedup_str = f"{ratio}x"
                speedup_val = ratio

            results_table.add_row(
                name,
                f"{size_mb} MB",
                f"{stats['mean_ms']} ms",
                f"{stats['p50_ms']} ms",
                f"{stats['p90_ms']} ms",
                f"{stats['fps']} FPS",
                speedup_str
            )

            benchmark_data["results"].append({
                "name": name,
                "model_path": model_path,
                "model_size_mb": size_mb,
                "mean_ms": stats["mean_ms"],
                "p50_ms": stats["p50_ms"],
                "p90_ms": stats["p90_ms"],
                "p99_ms": stats["p99_ms"],
                "fps": stats["fps"],
                "speedup": speedup_val
            })
        except Exception as e:
            console.print(f"[red][!] Lỗi khi đo {name}: {e}[/red]")

    console.print(results_table)

    # Lưu file JSON
    out_p = Path(save_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(benchmark_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green][✓] Đã lưu kết quả raw benchmark vào: {out_p}[/bold green]")


@cli.command(name="eval-accuracy")
@click.option("--sample-dir", default="data/sample_images", help="Thư mục ảnh BDD100K kiểm định")
@click.option("--save-json", default="outputs/accuracy_results.json", help="Đường dẫn lưu kết quả đánh giá mAP")
def eval_accuracy(sample_dir, save_json):
    """Đánh giá sai số mAP@50, mAP@50:95 và mAP Drop tuyệt đối trên tập BDD100K."""
    from src.core.inference.torch_engine import TorchInferenceEngine
    from src.core.inference.ov_engine import OpenVINOInferenceEngine
    from src.core.inference.onnx_engine import ONNXInferenceEngine
    from src.core.benchmark.accuracy_bench import AccuracyBenchmark

    img_paths = sorted(list(Path(sample_dir).glob("*.jpg")))
    if not img_paths:
        console.print(f"[bold red][!] Không tìm thấy ảnh trong thư mục: {sample_dir}[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[bold cyan]ĐÁNH GIÁ ĐỘ CHÍNH XÁC & mAP DROP TRÊN {len(img_paths)} ẢNH BDD100K[/bold cyan]"))

    ref_engine = TorchInferenceEngine("models/raw/yolov8s.pt")
    bench = AccuracyBenchmark()

    eval_targets = [
        ("ONNX Runtime (FP32)", ONNXInferenceEngine("models/onnx/yolov8s.onnx")),
        ("OpenVINO (FP16)", OpenVINOInferenceEngine("models/openvino_fp16/yolov8s_fp16.xml")),
        ("OpenVINO (INT8 NNCF PTQ)", OpenVINOInferenceEngine("models/openvino_int8/yolov8s_int8.xml")),
    ]

    table = Table(title="[bold yellow]BẢNG ĐÁNH GIÁ ĐỘ CHÍNH XÁC & SAI LỆCH mAP (M38 EVALUATION)[/bold yellow]")
    table.add_column("Mô hình đánh giá", style="cyan", no_wrap=True)
    table.add_column("Hộp phát hiện", style="white")
    table.add_column("Tỉ lệ giữ nhãn (Retention)", style="green")
    table.add_column("Mean IoU", style="yellow")
    table.add_column("mAP@50 (%)", style="bold magenta")
    table.add_column("mAP@50:95 (%)", style="magenta")
    table.add_column("mAP Drop vs FP32 (%)", style="bold red")

    eval_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_test_images": len(img_paths),
        "reference_model": "PyTorch YOLOv8s FP32",
        "evaluations": []
    }

    for name, cand_engine in eval_targets:
        console.print(f"[*] Đang đánh giá: [bold]{name}[/bold]...")
        metrics = bench.evaluate_dataset(ref_engine, cand_engine, img_paths)

        table.add_row(
            name,
            f"{metrics['cand_total_boxes']} / {metrics['ref_total_boxes']}",
            f"{metrics['retention_rate_percent']}%",
            f"{metrics['mean_matched_iou']}",
            f"{metrics['map_50']}%",
            f"{metrics['map_50_95']}%",
            f"{metrics['map_drop_percentage_points']}%"
        )

        eval_results["evaluations"].append({
            "candidate_name": name,
            "metrics": metrics
        })

    console.print(table)

    out_p = Path(save_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green][✓] Đã lưu báo cáo đánh giá mAP vào: {out_p}[/bold green]")


@cli.command(name="prelabel-2d")
@click.option("--source", default="data/sample_images/BDD01.jpg", help="Đường dẫn ảnh hoặc thư mục ảnh BDD100K")
@click.option("--model", default="models/openvino_int8/yolov8s_int8.xml", help="Mô hình suy luận (.xml / .onnx / .pt)")
@click.option("--out", default="outputs/prelabels", help="Thư mục xuất nhãn JSON / ảnh trực quan")
@click.option("--conf", default=0.25, type=float, help="Ngưỡng điểm tin cậy")
def prelabel_2d(source, model, out, conf):
    """Chạy Edge Pre-labeling 2D trên ảnh BDD100K và xuất file CVAT/YOLO format."""
    from src.core.inference.ov_engine import OpenVINOInferenceEngine
    from src.core.inference.onnx_engine import ONNXInferenceEngine
    from src.core.inference.torch_engine import TorchInferenceEngine
    from src.core.inference.base import COCO_TO_BDD100K

    src_p = Path(source)
    if not src_p.exists():
        console.print(f"[bold red][!] Không tìm thấy nguồn ảnh: {source}[/bold red]")
        sys.exit(1)

    # Chọn engine phù hợp theo đuôi file model
    if str(model).endswith(".xml"):
        engine = OpenVINOInferenceEngine(model, conf_thresh=conf)
    elif str(model).endswith(".onnx"):
        engine = ONNXInferenceEngine(model, conf_thresh=conf)
    else:
        engine = TorchInferenceEngine(model, conf_thresh=conf)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [src_p] if src_p.is_file() else sorted(list(src_p.glob("*.jpg")) + list(src_p.glob("*.png")))
    console.print(f"[*] Bắt đầu Edge Pre-Labeling trên {len(images)} ảnh bằng [bold green]{model}[/bold green]...")

    all_cvat_annotations = []
    total_time_ms = 0.0

    for idx, img_file in enumerate(images):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        t0 = time.perf_counter()
        detections = engine.predict(img)
        dt = (time.perf_counter() - t0) * 1000.0
        total_time_ms += dt

        cvat_shapes = []
        vis_img = img.copy()

        for det in detections:
            bdd_label = COCO_TO_BDD100K.get(det.class_name, det.class_name)
            cvat_shapes.append({
                "type": "rectangle",
                "label": bdd_label,
                "points": det.box_cvat,
                "confidence": round(det.confidence, 3),
                "occluded": False
            })

            # Vẽ trực quan
            x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tag = f"{bdd_label}: {det.confidence:.2f}"
            cv2.putText(vis_img, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        all_cvat_annotations.append({
            "image": img_file.name,
            "width": img.shape[1],
            "height": img.shape[0],
            "shapes": cvat_shapes,
            "latency_ms": round(dt, 2)
        })

        # Lưu ảnh vẽ hộp
        cv2.imwrite(str(out_dir / f"pred_{img_file.name}"), vis_img)

    json_file = out_dir / "cvat_prelabels.json"
    json_file.write_text(json.dumps(all_cvat_annotations, indent=2, ensure_ascii=False), encoding="utf-8")

    mean_ms = round(total_time_ms / len(images), 2) if images else 0.0
    fps = round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0

    console.print(Panel(f"[bold green][✓] HOÀN TẤT PRE-LABELING![/bold green]\n"
                        f"- Số lượng ảnh xử lý: {len(images)}\n"
                        f"- Độ trễ trung bình: {mean_ms} ms / frame ({fps} FPS)\n"
                        f"- File JSON chuẩn CVAT: {json_file}\n"
                        f"- Ảnh trực quan kết quả: {out_dir}/pred_*.jpg"))


@cli.command()
@click.option("--port", default=8080, help="Cổng mạng khởi chạy Web MVP (default 8080)")
def serve(port):
    """Khởi chạy ứng dụng Web MVP tương tác cục bộ (Local Interactive Demo)."""
    from src.web.app import run_server
    console.print(f"[bold green]Khởi chạy Web MVP Server tại http://127.0.0.1:{port}[/bold green]")
    run_server(port=port)


# ==============================================================================
# CÁC LỆNH MỞ RỘNG 3D LIDAR POINT CLOUD (MULTI-MODAL DIFFERENTIATOR)
# ==============================================================================

@cli.command(name="infer-3d")
@click.option("--pcd", default="data/pointclouds/demo.pcd", help="Đường dẫn file point cloud (.pcd)")
@click.option("--out", default="outputs", help="Thư mục xuất kết quả JSON")
@click.option("--score-thresh", default=0.30, type=float, help="Ngưỡng tin cậy")
@click.option("--full-scene", is_flag=True, help="Quét cả trước và sau")
def infer_3d(pcd, out, score_thresh, full_scene):
    """[3D Module] Suy luận phát hiện 3D Cuboids trên file LiDAR Point Cloud (.pcd)."""
    from src.core.pointcloud.pcd_io import read_pcd
    from src.core.pointcloud.ground import estimate_ground
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    from src.core.geometry.cvat_adapter import CVAT3DAdapter

    pcd_path = Path(pcd)
    if not pcd_path.is_file():
        console.print(f"[bold red][!] Không tìm thấy file PCD: {pcd}[/bold red]")
        sys.exit(1)

    pcd_data = read_pcd(pcd_path)
    z_ground = estimate_ground(pcd_data.xyz)
    console.print(f"[*] Nạp {pcd_data.count} điểm LiDAR | Mặt đất Z_ground: {z_ground:.3f}m")

    cfg = DetectorConfig(score_thresh=score_thresh)
    detector = PointPillarsDetector(config=cfg)
    cuboids = detector.detect(pcd_data.xyz, z_ground=z_ground, full_scene=full_scene)
    console.print(f"[bold green][✓] Phát hiện được {len(cuboids)} 3D cuboids[/bold green]")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"boxes_{pcd_path.stem}.json"
    adapter = CVAT3DAdapter()
    shapes_payload = adapter.to_shapes_payload(cuboids, frame=0)
    out_json.write_text(json.dumps(shapes_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green][✓] Đã xuất kết quả chuẩn CVAT 3D tại: {out_json}[/bold green]")


@cli.command(name="benchmark-3d")
@click.option("--pcd", default="data/pointclouds/demo.pcd", help="Đường dẫn file point cloud (.pcd)")
@click.option("--runs", default=30, help="Số lượt đo lường")
def benchmark_3d_cmd(pcd, runs):
    """[3D Module] Đo kiểm hiệu năng suy luận 3D Point Cloud trên CPU."""
    from src.core.benchmark.benchmark_3d import Benchmark3D
    bench = Benchmark3D()
    res = bench.run(pcd_path=pcd, runs=runs)
    console.print(Panel(f"[bold green]KẾT QUẢ BENCHMARK 3D LIDAR TRÊN CPU[/bold green]\n"
                        f"- Total Latency (P50): {res.p50_total_ms} ms\n"
                        f"- Total Latency (P90): {res.p90_total_ms} ms\n"
                        f"- Throughput: {res.throughput_fps} FPS\n"
                        f"- Peak Memory (RSS): {res.peak_ram_mb} MB\n"
                        f"- Detected Cuboids: {res.detected_cuboids}"))


if __name__ == "__main__":
    cli()
