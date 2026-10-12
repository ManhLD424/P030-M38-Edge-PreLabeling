"""P030-M38: 3D Model Compression & Edge Pre-Labeling CLI Tool."""
import sys
import os
import platform
import json
from pathlib import Path

# UTF-8 terminal support on Windows
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
    """P030-M38: 3D Point Cloud LiDAR Pre-Labeling & Compression CLI Tool."""
    pass


@cli.command()
def info():
    """Hiển thị thông số môi trường phần cứng và thư viện xử lý LiDAR 3D."""
    table = Table(title="[bold green]THÔNG SỐ HỆ THỐNG & MÔI TRƯỜNG XỬ LÝ 3D LIDAR[/bold green]")
    table.add_column("Thành phần", style="cyan", no_wrap=True)
    table.add_column("Giá trị", style="magenta")

    table.add_row("Hệ điều hành", f"{platform.system()} {platform.release()} ({platform.architecture()[0]})")
    table.add_row("Bộ vi xử lý (CPU)", platform.processor() or "x86_64")
    table.add_row("Số nhân CPU (Cores / Threads)", f"{psutil.cpu_count(logical=False)} Cores / {psutil.cpu_count(logical=True)} Threads")
    table.add_row("Tổng bộ nhớ RAM", f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    table.add_row("Python Version", sys.version.split()[0])

    try:
        import numpy as np
        table.add_row("NumPy Version", np.__version__)
    except ImportError:
        table.add_row("NumPy Version", "Chưa cài đặt")

    try:
        import torch
        table.add_row("PyTorch Version", torch.__version__)
    except ImportError:
        table.add_row("PyTorch Version", "Chưa cài đặt (Dùng Edge Geometric Engine)")

    try:
        import openvino as ov
        table.add_row("OpenVINO Version", ov.__version__)
    except ImportError:
        table.add_row("OpenVINO Version", "Chưa cài đặt")

    table.add_row("Hệ tọa độ LiDAR 3D", "X (Forward), Y (Left), Z (Up) — ISO 8855")
    table.add_row("Định dạng CVAT 3D", "16-point Cuboid [x, y, z, rx, ry, rz, dx, dy, dz, ...]")
    console.print(table)


@cli.command()
@click.option("--pcd", default="data/pointclouds/demo.pcd", help="Đường dẫn file point cloud (.pcd)")
@click.option("--out", default="outputs", help="Thư mục xuất kết quả JSON / ảnh side-view")
@click.option("--score-thresh", default=0.30, type=float, help="Ngưỡng điểm tin cậy tối thiểu")
@click.option("--full-scene", is_flag=True, help="Quét cả phía trước và sau (180 deg pass)")
def infer(pcd, out, score_thresh, full_scene):
    """Chạy suy luận phát hiện 3D Cuboids trên file Point Cloud (.pcd)."""
    from src.core.pointcloud.pcd_io import read_pcd
    from src.core.pointcloud.ground import estimate_ground
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    from src.core.geometry.cvat_adapter import CVAT3DAdapter

    pcd_path = Path(pcd)
    if not pcd_path.is_file():
        console.print(f"[bold red][!] Không tìm thấy file PCD: {pcd}[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[bold cyan]BẮT ĐẦU SUY LUẬN 3D LIDAR TRÊN: {pcd_path.name}[/bold cyan]"))

    # Đọc PCD
    pcd_data = read_pcd(pcd_path)
    z_ground = estimate_ground(pcd_data.xyz)
    console.print(f"[*] Đã nạp: [bold green]{pcd_data.count}[/bold green] điểm LiDAR | Ước lượng mặt đất Z_ground: [bold yellow]{z_ground:.3f}m[/bold yellow]")

    # Khởi tạo detector
    cfg = DetectorConfig(score_thresh=score_thresh)
    detector = PointPillarsDetector(config=cfg)

    # Suy luận
    cuboids = detector.detect(pcd_data.xyz, z_ground=z_ground, full_scene=full_scene)
    console.print(f"[bold green][✓] Phát hiện được {len(cuboids)} 3D cuboids[/bold green]")

    # Xuất kết quả
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = CVAT3DAdapter()
    payload = adapter.to_shapes_payload(cuboids, frame=0)
    json_path = out_dir / f"boxes_{pcd_path.stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[*] Đã lưu định dạng CVAT 3D: [bold]{json_path}[/bold]")

    # Bảng chi tiết kết quả
    if cuboids:
        table = Table(title="[bold yellow]DANH SÁCH 3D CUBOIDS PHÁT HIỆN ĐƯỢC[/bold yellow]")
        table.add_column("STT", style="cyan", no_wrap=True)
        table.add_column("Class", style="green")
        table.add_column("Center (X, Y, Z)", style="white")
        table.add_column("Size (L × W × H)", style="magenta")
        table.add_column("Yaw (deg)", style="yellow")
        table.add_column("Score", style="bold green")

        for idx, c in enumerate(cuboids, 1):
            yaw_deg = round(np.degrees(c.yaw), 1)
            table.add_row(
                str(idx),
                c.class_name,
                f"({c.x:.2f}, {c.y:.2f}, {c.z:.2f})",
                f"{c.dx:.2f} × {c.dy:.2f} × {c.dz:.2f} m",
                f"{yaw_deg}°",
                f"{c.score:.2f}"
            )
        console.print(table)


@cli.command()
@click.option("--pcd", default="data/pointclouds/demo.pcd", help="Đường dẫn file point cloud benchmark (.pcd)")
@click.option("--runs", default=30, type=int, help="Số lượt chạy lặp lại để lấy thống kê")
def benchmark(pcd, runs):
    """Đo kiểm hiệu năng suy luận 3D Point Cloud trên CPU (Latency, Throughput, Memory)."""
    from src.core.benchmark.benchmark_3d import Benchmark3D

    pcd_path = Path(pcd)
    if not pcd_path.is_file():
        console.print(f"[bold red][!] Không tìm thấy file PCD: {pcd}[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[bold cyan]ĐO KIỂM HIỆU NĂNG 3D PRE-LABELING TRÊN CPU ({runs} LƯỢT)[/bold cyan]\nFile: {pcd_path.name}"))
    
    bench = Benchmark3D()
    res = bench.run(pcd_path, runs=runs)

    table = Table(title="[bold yellow]KẾT QUẢ ĐO KIỂM ĐỘ TRỄ & THÔNG LƯỢNG 3D TRÊN CPU[/bold yellow]")
    table.add_column("Chỉ số đo đạc", style="cyan", no_wrap=True)
    table.add_column("Kết quả đo được", style="bold green")

    table.add_row("Số lượng điểm trong PCD", f"{res.num_points:,} điểm")
    table.add_row("Thời gian nạp & giải mã PCD (I/O)", f"{res.mean_io_ms} ms")
    table.add_row("Thời gian ước lượng mặt đất (Ground)", f"{res.mean_ground_ms} ms")
    table.add_row("Thời gian suy luận 3D Cuboids", f"{res.mean_detect_ms} ms")
    table.add_row("Thời gian chuẩn hóa schema CVAT 3D", f"{res.mean_cvat_ms} ms")
    table.add_row("Tổng độ trễ trung bình (Mean Latency)", f"{res.mean_total_ms} ms")
    table.add_row("Độ trễ trung vị P50 (Median)", f"{res.p50_total_ms} ms")
    table.add_row("Độ trễ phân vị P90", f"{res.p90_total_ms} ms")
    table.add_row("Thông lượng xử lý (Throughput)", f"{res.throughput_fps} FPS (Point Clouds / sec)")
    table.add_row("Bộ nhớ RAM tiêu thụ (Peak RSS)", f"{res.peak_ram_mb} MB")
    table.add_row("Số lượng 3D Cuboids phát hiện", f"{res.detected_cuboids} boxes")

    console.print(table)


@cli.command()
@click.option("--job-id", required=True, type=int, help="ID của CVAT 3D Job cần nạp nhãn mồi")
@click.option("--pcd", default="data/pointclouds/demo.pcd", help="File PCD nguồn (hoặc kéo từ server nếu không truyền)")
@click.option("--mode", default="merge", type=click.Choice(["merge", "replace"]), help="Chế độ nạp nhãn: merge hoặc replace")
@click.option("--conf-thresh", default=0.30, type=float, help="Ngưỡng tin cậy tối thiểu")
def prelabel(job_id, pcd, mode, conf_thresh):
    """Tự động hóa nạp nhãn 3D Cuboid mồi lên CVAT 3D Job qua REST API."""
    from src.core.cvat.client import CVAT3DClient
    from src.core.pointcloud.pcd_io import read_pcd
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    from src.core.geometry.cvat_adapter import CVAT3DAdapter

    console.print(Panel(f"[bold cyan]BẮT ĐẦU QUY TRÌNH 3D PRE-LABELING LÊN CVAT JOB #{job_id}[/bold cyan]"))

    client = CVAT3DClient.from_config_or_file(
        config_path="configs/cvat_config.yaml",
        token_txt_path=r"d:\Project\VinPrj\CVAT.txt"
    )

    # 1. Kiểm tra kết nối
    ok, msg = client.ping()
    if not ok:
        console.print(f"[bold red][!] Kết nối CVAT thất bại: {msg}[/bold red]")
        sys.exit(1)
    console.print(f"[bold green][✓] Kết nối CVAT thành công: {msg}[/bold green]")

    # 2. Lấy thông tin job & nhãn
    job_info = client.get_job_info(job_id)
    labels = client.get_labels(job_id)
    label_map = {l["name"]: l["id"] for l in labels}
    console.print(f"[*] Task ID: {job_info.get('task_id')} | Dimension: {job_info.get('dimension')} | Classes: {list(label_map.keys())}")

    # 3. Nạp point cloud
    pcd_path = Path(pcd)
    if not pcd_path.is_file():
        console.print(f"[*] Đang tải point cloud frame 0 từ CVAT server...")
        pcd_path = client.download_frame_pcd(job_id, 0, f"data/pointclouds/job_{job_id}_frame0.pcd")

    pcd_data = read_pcd(pcd_path)
    console.print(f"[*] Đã nạp {pcd_data.count} điểm LiDAR từ {pcd_path.name}")

    # 4. Suy luận 3D Cuboids
    cfg = DetectorConfig(score_thresh=conf_thresh)
    detector = PointPillarsDetector(config=cfg)
    cuboids = detector.detect(pcd_data.xyz)
    console.print(f"[bold green][✓] Đã tạo {len(cuboids)} 3D cuboids tiền gán nhãn[/bold green]")

    # 5. Đẩy lên CVAT
    adapter = CVAT3DAdapter(label_mapping=label_map)
    payload = adapter.to_shapes_payload(cuboids, frame=0)

    res = client.save_annotations(job_id, payload, mode=mode)
    console.print(Panel(f"[bold green]HOÀN TẤT PRE-LABELING CVAT JOB #{job_id}![/bold green]\n"
                        f"- Số lượng 3D Cuboids đã đẩy: {len(cuboids)}\n"
                        f"- Chế độ lưu: {mode}\n"
                        f"- Trực tiếp xem tại: {client.base_url}/tasks/{job_info.get('task_id')}/jobs/{job_id}"))


if __name__ == "__main__":
    cli()
