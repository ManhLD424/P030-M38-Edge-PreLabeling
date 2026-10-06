# P030 — M38: Model Compression for Edge Pre-Labeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Intel OpenVINO](https://img.shields.io/badge/Intel-OpenVINO_2024-purple.svg)](https://github.com/openvinotoolkit/openvino)
[![NNCF PTQ](https://img.shields.io/badge/Compression-NNCF_INT8-green.svg)](https://github.com/openvinotoolkit/nncf)
[![CVAT REST API](https://img.shields.io/badge/Integration-CVAT_API-orange.svg)](https://github.com/cvat-ai/cvat)

> **Hệ thống nén mô hình phát hiện đối tượng giao thông bằng lượng tử hóa sau huấn luyện (Post-Training Quantization - INT8) và tối ưu hóa suy luận trên CPU biên, tích hợp tự động hóa quy trình gán nhãn dữ liệu (Pre-labeling) lên nền tảng CVAT.**

---

## 1. Giới thiệu bài toán (Problem Statement)

Trong quy trình phát triển hệ thống thị giác máy tính cho xe tự hành (ADAS/Autonomous Vehicles), khâu **gán nhãn dữ liệu (Data Annotation)** chiếm tới 70–80% chi phí và thời gian của toàn bộ vòng đời MLOps. Giải pháp **Pre-Labeling** (sử dụng mô hình AI vẽ hộp mồi để annotator chỉ cần kiểm tra/hiệu chỉnh) giúp tăng tốc độ gán nhãn lên 3–4 lần.

Tuy nhiên, các mô hình phát hiện vật thể hiện đại đòi hỏi GPU đắt đỏ và đường truyền Cloud băng thông lớn. **Đề tài P030 — M38** tập trung giải quyết bài toán theo hướng **thuần kỹ thuật (Pure MLOps / Software Engineering)**:
- Sử dụng mô hình **Pre-trained có sẵn** (Off-the-shelf checkpoint chuẩn như YOLOv8s).
- Áp dụng kỹ thuật nén **Post-Training Static Quantization (PTQ INT8)** bằng Intel NNCF và OpenVINO Intermediate Representation (IR).
- Tối ưu hóa suy luận thời gian thực trực tiếp trên **CPU thông thường (Intel AVX-512 / VNNI)** của máy trạm annotator.
- Đóng gói công cụ dòng lệnh (CLI) khép kín kết nối trực tiếp với **CVAT REST API** để nạp nhãn mồi tự động vào Task/Job gán nhãn thực tế.

---

## 2. Kiến trúc hệ thống (System Architecture)

```
d:/Project/M38-Edge-PreLabeling/
├── configs/
│   ├── default_config.yaml         # Cấu hình ngưỡng confidence, IOU, path
│   └── cvat_config.yaml            # Cấu hình CVAT Server, Token, Org
├── data/
│   ├── calibration/                # 200-300 ảnh giao thông mẫu dùng cho NNCF PTQ
│   └── sample_images/              # Ảnh mẫu demo
├── docs/
│   ├── SYSTEM_SPECIFICATION.md     # Đặc tả kỹ thuật toàn diện
│   └── IMPLEMENTATION_PLAN.md      # Kế hoạch triển khai & RACI matrix
├── models/
│   ├── raw/                        # yolov8s.pt gốc (FP32)
│   ├── onnx/                       # yolov8s.onnx (FP32)
│   ├── openvino_fp16/              # yolov8s_fp16.xml & .bin
│   └── openvino_int8/              # yolov8s_int8.xml & .bin (Mô hình nén)
├── src/
│   ├── core/
│   │   ├── compression/            # Module xuất ONNX & lượng tử hóa NNCF INT8
│   │   ├── inference/              # Unified Inference Engine (Torch/ONNX/OpenVINO)
│   │   ├── benchmark/              # Đo Latency, FPS, RAM peak, Model size
│   │   └── cvat/                   # Client kết nối CVAT REST API
│   └── cli/
│       └── main.py                 # Giao diện dòng lệnh CLI (Rich UI)
├── tests/                          # Bộ kiểm thử tự động (Unit test)
├── requirements.txt                # Danh sách thư viện đồng bộ
├── .gitignore                      # Cấu hình loại trừ Git
└── README.md                       # Tài liệu dự án
```

---

## 3. Các chỉ tiêu kỹ thuật định lượng (Target Metrics)

| Chỉ tiêu | Mô hình gốc (FP32) | Mô hình nén (INT8 OpenVINO) | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **Kích thước mô hình (Disk)** | ~45.0 MB | **~11.5 MB** | **Giảm ~3.9x (74%)** |
| **Tốc độ suy luận CPU (Latency)** | ~65–80 ms | **~18–25 ms** | **Nhanh hơn 3x–4x** |
| **Thông lượng CPU (Throughput)** | ~12–15 FPS | **~40–55 FPS** | **Thời gian thực trên CPU** |
| **RAM Footprint (Peak RSS)** | ~1.2 GB (PyTorch) | **~350 MB (OpenVINO)** | **Giảm ~70% RAM** |
| **Chất lượng nhãn mồi** | Baseline FP32 | IoU sai lệch $< 1.5\%$ | Duy trì độ chính xác cao |
| **Thời gian gán nhãn Annotator** | ~45s / frame (Vẽ tay) | **~12s / frame (Sửa mồi)** | **Tiết kiệm ~70% thời gian** |

---

## 4. Hướng dẫn cài đặt nhanh (Quickstart)

### 4.1. Khởi tạo môi trường ảo

```bash
# Tạo môi trường ảo Python 3.10 hoặc 3.11
python -m venv .venv

# Kích hoạt trên Windows PowerShell:
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
```

### 4.2. Chạy quy trình nén mô hình (Quantization)

```bash
python -m src.cli.main compress \
    --weights models/raw/yolov8s.pt \
    --calib-dir data/calibration \
    --output-dir models/openvino_int8
```

### 4.3. Chạy đo kiểm hiệu năng (Benchmark)

```bash
python -m src.cli.main benchmark \
    --models-dir models \
    --iterations 200
```

### 4.4. Đẩy nhãn tự động lên CVAT (Edge Pre-Labeling)

```bash
python -m src.cli.main prelabel \
    --model models/openvino_int8/yolov8s_int8.xml \
    --cvat-job-id 9400 \
    --conf-thresh 0.35 \
    --mode replace
```

---

## 5. Đội ngũ thực hiện

- **Lê Đức Mạnh (Lead Developer & System Architect):** Phụ trách kiến trúc hệ thống, Unified Inference Engine đa runtime, tích hợp CVAT REST API, bộ đo hiệu năng phần cứng và hoàn thiện CLI Tool.
- **Phạm Hoàng Anh (AI Research & Optimization Specialist):** Phụ trách chuẩn bị dữ liệu calibration, cấu hình thuật toán lượng tử hóa NNCF PTQ INT8, đánh giá độ sai lệch mô hình và phân tích sai số gán nhãn.
