# P030 — M38: 3D LiDAR Model Compression & Edge Pre-Labeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PointPillars 3D](https://img.shields.io/badge/Model-PointPillars_KITTI-purple.svg)](https://github.com/open-mmlab/mmdetection3d)
[![CVAT 3D Tool](https://img.shields.io/badge/Integration-CVAT_3D_REST_API-orange.svg)](https://github.com/cvat-ai/cvat)
[![Edge CPU Optimized](https://img.shields.io/badge/Inference-CPU_Optimized-green.svg)](https://github.com/openvinotoolkit/openvino)

> **Hệ thống tiền gán nhãn dữ liệu LiDAR 3D (3D Point Cloud Pre-Labeling) trên CPU máy trạm biên, trích xuất 3D Cuboids theo chuẩn ISO 8855 và tự động hóa nạp nhãn mồi vào công cụ CVAT 3D TOOL qua REST API.**

---

## 1. Giới thiệu bài toán (Problem Statement)

Trong quy trình phát triển hệ thống xe tự hành (Autonomous Vehicles / Robotaxi), khâu **gán nhãn dữ liệu 3D Point Cloud (LiDAR 3D Annotation)** là khâu phức tạp và tốn kém nhất:
- Annotator phải xoay không gian 3 chiều để chỉnh kích thước (Dx, Dy, Dz), vị trí tâm (X, Y, Z) và góc xoay (Yaw) của từng hộp **3D Cuboid**.
- Thời gian vẽ thủ công một frame 3D thường mất **3–5 phút/frame**.
- **Giải pháp M38**: Sử dụng mô hình **PointPillars 3D** (pretrained KITTI / Robotaxi) kết hợp thuật toán tối ưu hóa hình học trên CPU máy trạm để tự động tạo hộp **3D Cuboid mồi**, giảm thời gian gán nhãn xuống còn **30–45 giây/frame** (tiết kiệm ~70–80% thời gian cho annotator).

---

## 2. Kiến trúc hệ thống (System Architecture)

```
d:/Project/M38-Edge-PreLabeling/
├── configs/
│   ├── default_config.yaml         # Cấu hình PointPillars, voxel size, delta sensor, thresholds
│   └── cvat_config.yaml            # Cấu hình CVAT 3D Server, Token, Org, Label mapping
├── data/
│   └── pointclouds/                # Tập dữ liệu PCD mẫu (demo.pcd 17,238 điểm)
├── docs/
│   └── SYSTEM_SPECIFICATION.md     # Đặc tả kỹ thuật toàn diện hệ thống 3D
├── src/
│   ├── core/
│   │   ├── pointcloud/             # Module đọc/ghi PCD (Binary/ASCII), ước lượng ground
│   │   ├── geometry/               # Dataclass Cuboid3D, góc Yaw, 16-point CVAT adapter
│   │   ├── models/                 # PointPillars 3D Detector & Edge Clustering Engine
│   │   ├── cvat/                   # Client kết nối CVAT 3D REST API (shapes & tracks)
│   │   └── benchmark/              # Bộ đo kiểm độ trễ I/O, Ground, Inference, CVAT format
│   └── cli/
│       └── main.py                 # Giao diện dòng lệnh CLI (Rich UI)
├── tests/                          # Bộ kiểm thử tự động với 14 test cases
└── README.md                       # Tài liệu dự án
```

---

## 3. Định dạng dữ liệu & Chuẩn tương thích CVAT 3D

Mô hình xuất hộp **3D Cuboid** theo chuẩn **16 phần tử của CVAT 3D**:
```json
"points": [
  x, y, z,          // [0..2]: Tọa độ tâm hộp (meters)
  0.0, 0.0, yaw,    // [3..5]: Góc xoay roll, pitch, yaw (radians)
  dx, dy, dz,       // [6..8]: Kích thước dài (L), rộng (W), cao (H)
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 // [9..15]: Padding dự phòng
]
```
Tương thích 100% với định dạng của extension **CVAT 3D TOOL** (`CVAT-3D-Copilot`).

---

## 4. Hướng dẫn sử dụng CLI Tool

### 4.1. Kiểm tra môi trường hệ thống
```bash
python -m src.cli.main info
```

### 4.2. Chạy suy luận 3D Pre-Labeling trên file PCD
```bash
python -m src.cli.main infer --pcd data/pointclouds/demo.pcd --out outputs --score-thresh 0.30
```
- Tự động nạp điểm LiDAR, ước lượng cao độ mặt đường $Z_{ground}$.
- Trích xuất danh sách 3D Cuboids (`vehicles`, `pedestrian`, `two-wheels`).
- Xuất file kết quả `outputs/boxes_demo.json` chuẩn schema CVAT 3D.

### 4.3. Đo kiểm hiệu năng thực nghiệm trên CPU
```bash
python -m src.cli.main benchmark --pcd data/pointclouds/demo.pcd --runs 30
```
Thống kê chi tiết: thời gian I/O, ước lượng Ground, thời gian suy luận 3D, tổng độ trễ P50/P90, Throughput (FPS) và Peak RAM.

### 4.4. Nạp nhãn mồi tự động lên CVAT 3D Job
```bash
python -m src.cli.main prelabel --job-id 9426 --pcd data/pointclouds/demo.pcd --mode merge
```

---

## 5. Kết quả kiểm thử tự động (Unit Tests)

Chạy bộ kiểm thử tự động với `pytest`:
```bash
python -m pytest tests -v
```
Toàn bộ **14/14 test cases** đã vượt qua kiểm định:
- Đọc/ghi PCD Binary v0.7 với 17,238 điểm.
- Ước lượng ground và dịch chuyển hệ quy chiếu $Z_{source} \leftrightarrow Z_{model}$.
- Đóng gói và giải mã 16-point CVAT 3D Cuboids.
- Thuật toán loại bỏ hộp đè lấn (`merge_boxes`) và lọc người đi bộ trên nóc xe (`drop_boxes_on_cars`).
- Đóng gói shapes và tracks sequence cho CVAT 3D REST API.
