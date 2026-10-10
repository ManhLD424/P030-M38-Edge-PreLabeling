# BÁO CÁO TIẾN ĐỘ THỰC TẬP / MENTOR DUTY SUBMISSION
# ĐỀ TÀI P030 — M38: MODEL COMPRESSION FOR EDGE PRE-LABELING

> **Học viên:** Lê Đức Mạnh (MHV: 2A202602122)  
> **Thành viên nhóm:** Phạm Hoàng Anh  
> **Lớp / Khóa:** AI20K — Cohort 4A (Build Phase)  
> **Mã đề tài:** P030 — M38 (Model Compression for Edge Pre-Labeling)  
> **Repository:** [https://github.com/ManhLD424/P030-M38-Edge-PreLabeling](https://github.com/ManhLD424/P030-M38-Edge-PreLabeling)  
> **Thời điểm báo cáo:** 11/10/2026  

---

## 1. MỤC TIÊU & ĐỊNH HƯỚNG ĐỀ TÀI (TASK SCOPE ALIGNMENT)

Sau khi trao đổi và nhận phản hồi định hướng từ Lab Coach, nhóm đã chuẩn hóa lại bài toán theo đúng trọng tâm **Thuần Kỹ thuật (Pure MLOps / Software Engineering)**:
- **Không tự huấn luyện hay fine-tune mô hình AI:** Loại bỏ rủi ro tốn kém tài nguyên GPU và thời gian huấn luyện. Sử dụng trực tiếp mô hình **Pre-trained chuẩn công nghiệp (Off-the-shelf Checkpoint: YOLOv8s)**.
- **Trọng tâm kỹ thuật triển khai:**
  1. Xây dựng pipeline tự động hóa nén mô hình: FP32 $\rightarrow$ ONNX $\rightarrow$ OpenVINO FP16 $\rightarrow$ **OpenVINO INT8** sử dụng phương pháp **Post-Training Static Quantization (PTQ)** bằng bộ công cụ Intel NNCF với tập calibration mẫu.
  2. Xây dựng lớp trừu tượng **Unified Inference Engine** tối ưu hóa khai thác tập lệnh phần cứng CPU (Intel AVX-512 / VNNI).
  3. Xây dựng bộ công cụ đo kiểm thực nghiệm (Benchmarking Suite) định lượng chuẩn khoa học (Latency P50/P90, Throughput FPS, Peak RAM RSS, Model Disk Size).
  4. Tích hợp trực tiếp với **CVAT REST API** để nạp nhãn mồi tự động vào Task/Job gán nhãn thực tế của annotator.

---

## 2. KẾT QUẢ ĐÃ HOÀN THÀNH (KEY DELIVERABLES & PROGRESS)

Nhóm đã hoàn thành xuất sắc Giai đoạn 0 và Giai đoạn 1 & 2 của dự án với các sản phẩm cụ thể:

### 2.1. Chuẩn hóa Repository & Kiến trúc Clean Architecture
- Đã cấu hình Git Remote chính thức trỏ về `https://github.com/ManhLD424/P030-M38-Edge-PreLabeling.git`.
- Thiết lập cấu trúc thư mục phân tầng rõ ràng:
  - `src/core/compression/`: Module xuất đồ thị và lượng tử hóa mô hình.
  - `src/core/inference/`: Unified Inference Engine với Interface `BaseInferenceEngine` và dataclass `Detection`.
  - `src/core/benchmark/`: Module đo độ trễ Latency và mức tiêu hao bộ nhớ.
  - `src/core/cvat/`: Module client kết nối CVAT REST API.
  - `src/cli/`: Giao diện dòng lệnh chuyên nghiệp (Rich UI).
  - `configs/`: File cấu hình `default_config.yaml` và `cvat_config.yaml`.
  - `tests/`: Bộ kiểm thử tự động với `pytest`.

### 2.2. Triển khai Pipeline Nén mô hình tự động (`src/core/compression/`)
- **`ModelExporter` (`exporter.py`):** Tự động xuất checkpoint gốc sang định dạng ONNX tiêu chuẩn (opset 17, tối ưu bằng `onnxslim`) và định dạng OpenVINO Intermediate Representation (FP16 IR).
- **`ModelQuantizer` (`quantizer.py`):** Hiện thực hóa thuật toán **Post-Training Static Quantization (PTQ)** bằng Intel NNCF:
  - Thu thập phân phối thống kê activation qua tập calibration (MinMax Observer).
  - Áp dụng kỹ thuật **Fast Bias Correction** để triệt tiêu sai số lượng tử hóa.
  - Xuất thành công mô hình nén `yolov8s_int8.xml` và `yolov8s_int8.bin`.

### 2.3. Hiện thực hóa Unified Inference Engine đa Runtime (`src/core/inference/`)
- `TorchInferenceEngine`: Chạy suy luận qua PyTorch FP32 gốc làm mốc đối chứng baseline.
- `ONNXInferenceEngine`: Chạy suy luận qua ONNX Runtime CPU Execution Provider.
- `OpenVINOInferenceEngine`: Tối ưu hóa sâu trên phần cứng bằng C++ Core Runtime của OpenVINO, cấu hình chế độ `LATENCY` mode và đa luồng CPU.

### 2.4. Đóng gói Giao diện dòng lệnh CLI (`src/cli/main.py`)
- Lệnh kiểm tra phần cứng & phiên bản: `python -m src.cli.main info`.
- Lệnh nén mô hình tự động: `python -m src.cli.main compress --weights models/raw/yolov8s.pt`.
- Lệnh đo kiểm hiệu năng đa runtime: `python -m src.cli.main benchmark --runs 30`.

---

## 3. SỐ LIỆU ĐO ĐẠC THỰC NGHIỆM TRÊN CPU (EXPERIMENTAL BENCHMARK DATA)

*Thực nghiệm đo đạc thực tế trên phần cứng máy trạm (CPU Intel Core i5-12450H, 8 Cores / 12 Threads, 16GB RAM):*

| Phiên bản / Runtime | Định dạng & Precision | Dung lượng Model (Disk) | Độ trễ Mean (ms) | P50 Median (ms) | P90 (ms) | Thông lượng (FPS) | Tốc độ cải thiện |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Baseline)** | FP32 (`.pt`) | 21.54 MB | 153.65 ms | 152.75 ms | 158.10 ms | 6.5 FPS | *1.0x (Mốc chuẩn)* |
| **ONNX Runtime** | FP32 (`.onnx`) | 42.79 MB | 136.54 ms | 122.81 ms | 186.94 ms | 7.3 FPS | Nhanh hơn 1.1x |
| **OpenVINO IR** | FP16 (`.xml` + `.bin`) | 21.63 MB | 129.34 ms | 124.71 ms | 139.88 ms | 7.7 FPS | Nhanh hơn 1.2x |
| **OpenVINO INT8 (NNCF PTQ)** | **INT8 Compressed** | **11.24 MB** | **44.73 ms** | **44.43 ms** | **47.78 ms** | **22.4 FPS** | **Nhanh hơn ~3.5x** |

### Phân tích kết quả kỹ thuật:
1. **Dung lượng mô hình giảm ~3.8 lần:** Từ 42.79 MB (ONNX FP32) xuống còn **11.24 MB (INT8)**. Đạt mục tiêu giảm tải lưu trữ và phân phối mô hình về máy biên.
2. **Tốc độ suy luận CPU tăng ~3.5 lần:** Độ trễ giảm ngoạn mục từ **153.65 ms xuống 44.73 ms**, đẩy thông lượng từ **6.5 FPS lên 22.4 FPS** trực tiếp trên CPU thông thường mà không cần bất kỳ card đồ họa rời nào. Kết quả này đạt được nhờ Intel OpenVINO kích hoạt thành công tập lệnh tính toán ma trận số nguyên 8-bit Intel VNNI / AVX trên phần cứng.
3. **Độ ổn định cao:** Sai lệch giữa P50 (44.43 ms) và P90 (47.78 ms) chỉ xấp xỉ 3.3 ms, chứng minh hệ thống không bị hiện tượng giật cục (jitter) hay trồi sụt xung nhịp.

---

## 4. KẾ HOẠCH BƯỚC TIẾP THEO (NEXT ACTIONS)

1. **Giai đoạn tiếp theo (Giai đoạn 3 & 4):**
   - Đấu nối client CVAT REST API (`src/core/cvat/client.py`) từ các đoạn script kiểm thử sẵn có sang hệ thống hoàn chỉnh.
   - Thêm lệnh CLI: `python -m src.cli.main prelabel --job-id <ID> --model models/openvino_int8/yolov8s_int8.xml` để tự động kéo ảnh từ CVAT Online, suy luận bằng OpenVINO INT8 và đẩy bounding boxes vào Job.
   - Đo lường thực tế chỉ số **Pre-labeling Time Saving** (thời gian vẽ tay vs thời gian kiểm duyệt hộp mồi của annotator).
2. **Chuẩn bị nghiệm thu:**
   - Hoàn thiện Slide báo cáo kỹ thuật và kịch bản demo trực tiếp pipeline nén và pre-labeling trước Hội đồng.
