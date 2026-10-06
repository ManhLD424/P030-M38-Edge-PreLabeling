# KẾ HOẠCH TRIỂN KHAI DỰ ÁN CHI TIẾT TỪ A - Z
# Đề tài M38: Model Compression for Edge Pre-Labeling

> **Nhóm thực hiện:**
> - **Lê Đức Mạnh (Lead Developer & System Architect):** Phụ trách kiến trúc hệ thống, pipeline xuất mô hình, engine suy luận tối ưu CPU, tích hợp CVAT REST API, bộ đo hiệu năng phần cứng và hoàn thiện CLI.
> - **Phạm Hoàng Anh (AI Research & Optimization Specialist):** Phụ trách chuẩn bị dữ liệu BDD100K/Calibration, cấu hình thuật toán lượng tử hóa NNCF PTQ INT8, đánh giá độ chính xác mAP/Loss và phân tích sai số trên từng class.

---

## TỔNG QUAN TIẾN ĐỘ THEO CÁC GIAI ĐOẠN (ROADMAP)

```
Giai đoạn 0: Khởi tạo & Dữ liệu Baseline (Setup & Baseline Alignment)
      │
      ▼
Giai đoạn 1: Pipeline Nén mô hình sang INT8 (OpenVINO IR & NNCF PTQ)
      │
      ▼
Giai đoạn 2: Xây dựng Unified Inference Engine đa Runtime (PyTorch, ONNX, OpenVINO)
      │
      ▼
Giai đoạn 3: Bộ công cụ Benchmark thực nghiệm (Latency, FPS, RAM, Disk, mAP)
      │
      ▼
Giai đoạn 4: Tích hợp CVAT REST API & Hoàn thiện CLI Tool End-to-End
      │
      ▼
Giai đoạn 5: Hoàn thiện Báo cáo, Slide thuyết trình & Kịch bản bảo vệ
```

---

## GIAI ĐOẠN 0: KHỞI TẠO HỆ THỐNG & CHUẨN BỊ DỮ LIỆU BASELINE
*Thời lượng dự kiến: 1 – 2 ngày*

### Mục tiêu:
Thiết lập môi trường phát triển đồng bộ giữa 2 thành viên, chuẩn bị mô hình gốc YOLOv8s (FP32), tạo tập dữ liệu con (BDD100K 10 classes) và tập calibration 300 ảnh.

### Nhiệm vụ cụ thể:
- [ ] **Lê Đức Mạnh:**
  - Khởi tạo cấu trúc dự án chuẩn `d:/Project/M38-Edge-PreLabeling` theo kiến trúc Clean Architecture.
  - Cấu hình file `requirements.txt` và `pyproject.toml` với các phiên bản thư viện tương thích (`openvino>=2024.0`, `nncf>=2.8`, `ultralytics>=8.1`).
  - Viết script kiểm tra môi trường phần cứng `scripts/check_env.py` (kiểm tra tập lệnh CPU: AVX2/AVX-512/VNNI, phiên bản Python, dung lượng RAM).
  - Thiết lập Git repository nội bộ và thống nhất quy chuẩn commit (Conventional Commits).
- [ ] **Phạm Hoàng Anh:**
  - Chuẩn bị tập dữ liệu ảnh giao thông (trích xuất tập con BDD100K với 10 lớp giao thông: car, bus, truck, pedestrian, rider, motorcycle, bicycle, train, traffic light, traffic sign).
  - Tách riêng **300 ảnh không gán nhãn** lưu vào `data/calibration/` dùng cho thuật toán NNCF MinMax calibration sau này.
  - Tải checkpoint gốc `models/raw/yolov8s.pt`, chạy kiểm thử cơ bản để đo chỉ số baseline ban đầu trên tập validation ($mAP@50$ gốc).
  - Lập bảng thông số mô hình gốc (số tham số, số FLOPs, dung lượng file .pt).

**Sản phẩm bàn giao Giai đoạn 0 (Deliverables):**
- Môi trường chạy được đồng nhất trên máy cả 2 thành viên.
- Thư mục `data/calibration/` (300 ảnh) và `data/val_subset/` (~500–1000 ảnh).
- File checkpoint `models/raw/yolov8s.pt` đã xác nhận chạy tốt.

---

## GIAI ĐOẠN 1: PIPELINE NÉN MÔ HÌNH SANG INT8 (OPENVINO & NNCF PTQ)
*Thời lượng dự kiến: 2 – 3 ngày*

### Mục tiêu:
Hiện thực hóa module nén mô hình tự động, xuất thành công 4 phiên bản: PyTorch FP32, ONNX FP32, OpenVINO FP16 và OpenVINO INT8 (lượng tử hóa sau huấn luyện tĩnh).

### Nhiệm vụ cụ thể:
- [ ] **Lê Đức Mạnh:**
  - Viết module `src/core/compression/exporter.py`:
    - Hàm `export_to_onnx()`: Xuất `yolov8s.pt` sang `models/onnx/yolov8s.onnx` (opset 17, dynamic batch nếu cần).
    - Hàm `export_to_openvino_fp16()`: Sử dụng OpenVINO Model Optimizer chuyển đổi sang cặp file `models/openvino_fp16/yolov8s_fp16.xml` và `.bin`.
    - Viết unit test tự động xác minh file sau khi xuất không bị lỗi cấu trúc.
- [ ] **Phạm Hoàng Anh:**
  - Viết module `src/core/compression/quantizer.py`:
    - Xây dựng lớp `NNCFDataLoader` kế thừa từ `nncf.Dataset` để tiền xử lý chuẩn hóa 300 ảnh calibration ($640 \times 640$, RGB, normalized 0..1).
    - Áp dụng phương pháp **Post-Training Static Quantization**:
      ```python
      import nncf
      quantized_model = nncf.quantize(
          model=openvino_model,
          calibration_dataset=nncf_calibration_dataset,
          preset=nncf.QuantizationPreset.PERFORMANCE, # INT8 symmetric cho weights
          model_type=nncf.ModelType.OBJECT_DETECTION
      )
      ```
    - Lưu mô hình nén ra `models/openvino_int8/yolov8s_int8.xml` và `.bin`.
    - Kiểm tra và ghi nhận các lớp (layers) đã được chuyển đổi sang INT8 (Convolution, MatMul).

**Sản phẩm bàn giao Giai đoạn 1 (Deliverables):**
- Thư mục `models/` chứa đủ 4 phiên bản mô hình hoàn chỉnh.
- Dung lượng file: `yolov8s_int8.bin` giảm ~3–4 lần so với mô hình gốc (từ ~45MB xuống còn ~11–12MB).

---

## GIAI ĐOẠN 2: UNIFIED INFERENCE ENGINE (HỆ THỐNG SUY LUẬN ĐA RUNTIME)
*Thời lượng dự kiến: 2 ngày*

### Mục tiêu:
Xây dựng lớp giao diện suy luận trừu tượng chung để người dùng và các module khác có thể gọi `engine.predict(image)` mà không cần quan tâm bên dưới đang chạy PyTorch, ONNX Runtime hay OpenVINO.

### Nhiệm vụ cụ thể:
- [ ] **Lê Đức Mạnh:**
  - Thiết kế `src/core/inference/base.py`:
    - Định nghĩa lớp dữ liệu `Detection` (bbox: xyxy, score: float, class_id: int, class_name: str).
    - Định nghĩa interface `BaseInferenceEngine(ABC)` với phương thức chuẩn `predict(image: np.ndarray, conf_thresh: float, iou_thresh: float) -> List[Detection]`.
    - Viết hàm tiền xử lý chuẩn `letterbox()` và hậu xử lý `non_max_suppression()` chung.
  - Hiện thực `src/core/inference/ov_engine.py`:
    - Tối ưu hóa OpenVINO Core trên CPU: Cấu hình `ov::hint::performance_mode` thành `LATENCY`, thiết lập luồng tối ưu.
    - Chạy suy luận trực tiếp trên `yolov8s_int8.xml` và trả về kết quả chuẩn.
- [ ] **Phạm Hoàng Anh:**
  - Hiện thực `src/core/inference/onnx_engine.py`:
    - Sử dụng `onnxruntime.InferenceSession` với `CPUExecutionProvider`.
  - Hiện thực `src/core/inference/torch_engine.py`:
    - Chạy suy luận bằng PyTorch gốc phục vụ việc đối chiếu kết quả.
  - Viết kịch bản kiểm thử chéo (Cross-Validation Test):
    - Đưa cùng 1 ảnh vào cả 3 engine.
    - So sánh Bounding Boxes và Confidence Score giữa PyTorch và OpenVINO INT8 để đảm bảo vị trí hộp không bị sai lệch quá $2\%$.

**Sản phẩm bàn giao Giai đoạn 2 (Deliverables):**
- Module suy luận thống nhất hoạt động trơn tru trên cả 3 backend.
- Bộ test case `tests/test_inference.py` chạy qua 100% (Green).

---

## GIAI ĐOẠN 3: BỘ CÔNG CỤ BENCHMARK THỰC NGHIỆM KHOA HỌC
*Thời lượng dự kiến: 2 – 3 ngày*

### Mục tiêu:
Thu thập toàn bộ số liệu thực nghiệm khoa học chứng minh hiệu quả của việc nén mô hình trên CPU: Tốc độ (Latency, FPS), Bộ nhớ (RAM, Disk) và Độ chính xác ($mAP$).

### Nhiệm vụ cụ thể:
- [ ] **Lê Đức Mạnh (Đo đạc Hiệu năng Phần cứng & Tốc độ):**
  - Viết module `src/core/benchmark/latency_bench.py`:
    - Warmup 20 lượt để ổn định xung nhịp CPU.
    - Chạy 200 lượt suy luận liên tiếp, ghi nhận thời gian bằng `time.perf_counter_ns()`.
    - Tính toán: Mean Latency, P50 (Median), P90, P99 Latency và Throughput (FPS).
  - Viết module `src/core/benchmark/memory_bench.py`:
    - Sử dụng thư viện `psutil` đo bộ nhớ thực (Resident Set Size - RSS) trước và sau khi nạp mô hình vào RAM.
    - Đo kích thước dung lượng file trên ổ cứng (MB).
  - Viết script tự động vẽ biểu đồ trực quan (Bar chart, Line plot) bằng `matplotlib`/`seaborn` lưu vào `docs/figures/`.
- [ ] **Phạm Hoàng Anh (Đo đạc Độ chính xác & Phân tích Sai số):**
  - Viết module `src/core/benchmark/accuracy_bench.py`:
    - Chạy suy luận trên toàn bộ tập `val_subset/`.
    - Tính toán các chỉ số chuẩn: $mAP@50$, $mAP@50:95$, Precision, Recall cho từng phiên bản mô hình (FP32 vs FP16 vs INT8).
  - Phân tích độ sụt giảm độ chính xác (mAP drop):
    - Đánh giá xem class nào bị ảnh hưởng nhiều nhất sau lượng tử hóa (ví dụ: vật thể nhỏ như traffic light/pedestrian so với vật thể lớn như bus/truck).
    - Soạn thảo nội dung phân tích vào tài liệu `docs/BENCHMARK_REPORT.md`.

**Sản phẩm bàn giao Giai đoạn 3 (Deliverables):**
- Bảng số liệu so sánh toàn diện 4 mô hình:
  - PyTorch FP32
  - ONNX FP32
  - OpenVINO FP16
  - OpenVINO INT8
- Báo cáo khoa học `docs/BENCHMARK_REPORT.md` kèm 3 biểu đồ: So sánh Latency/FPS, So sánh RAM/Disk, và So sánh mAP.

---

## GIAI ĐOẠN 4: TÍCH HỢP CVAT REST API & HOÀN THIỆN CLI TOOL
*Thời lượng dự kiến: 2 ngày*

### Mục tiêu:
Đưa mô hình nén vào giải quyết đúng bài toán *Edge Pre-Labeling*: Viết công cụ dòng lệnh (CLI) tải ảnh từ CVAT $\rightarrow$ chạy model OpenVINO INT8 $\rightarrow$ đẩy trực tiếp pre-label lên Job CVAT.

### Nhiệm vụ cụ thể:
- [ ] **Lê Đức Mạnh (CVAT Integration & CLI Polish):**
  - Viết `src/core/cvat/client.py`:
    - Kế thừa và chuẩn hóa code API có sẵn (`import_to_job.py`, `export_cvat_job.py`).
    - Kết nối với server `https://cvat.note.transformerlabs.ai` bằng token.
    - Lấy thông tin Job, tự động map nhãn (ví dụ class `car` của YOLO sang nhãn `vehicles` của Job CVAT).
    - Đẩy bounding box vào CVAT qua endpoint `PATCH /api/jobs/{id}/annotations?action=create`.
  - Hoàn thiện giao diện dòng lệnh `src/cli/main.py`:
    - Hỗ trợ 3 subcommand: `compress`, `benchmark`, `annotate`.
    - Dùng thư viện `rich` để hiển thị thanh tiến trình (progress bar), bảng màu chuyên nghiệp.
- [ ] **Phạm Hoàng Anh (Thực nghiệm Pre-labeling & Đánh giá Thời gian):**
  - Lập kịch bản kiểm thử thực tế trên 1 Job CVAT cụ thể (ví dụ 50–100 ảnh).
  - Chạy tool `annotate` để gán nhãn tự động bằng mô hình INT8.
  - Phân tích định lượng **"Pre-labeling Time Saving"**:
    - Đo thời gian annotator tự vẽ hộp từ đầu (manual labeling): ~45 giây/ảnh.
    - Đo thời gian annotator chỉ cần sửa hộp do mô hình nén vẽ sẵn: ~12 giây/ảnh.
    - Đưa ra kết luận: Mô hình nén INT8 giúp **tiết kiệm ~70% thời gian gán nhãn** cho annotator.

**Sản phẩm bàn giao Giai đoạn 4 (Deliverables):**
- Công cụ CLI hoàn chỉnh chạy mượt mà từ terminal: `python -m src.cli.main ...`
- Video ngắn hoặc ảnh chụp màn hình minh chứng các hộp pre-label xuất hiện trên CVAT Online.

---

## GIAI ĐOẠN 5: TỔNG KẾT, SLIDE THUYẾT TRÌNH & BẢO VỆ ĐỒ ÁN
*Thời lượng dự kiến: 1 – 2 ngày*

### Mục tiêu:
Tổng hợp toàn bộ mã nguồn, tài liệu, slide thuyết trình và chuẩn bị kịch bản bảo vệ tự tin trước Hội đồng / Mentor.

### Phân công thuyết trình & Sản phẩm:
- [ ] **Phạm Hoàng Anh:**
  - Soạn thảo Slide phần Lý thuyết & Khoa học (5–6 slides):
    - Đặt vấn đề & Khó khăn của gán nhãn dữ liệu xe tự hành.
    - Cơ chế toán học của Lượng tử hóa sau huấn luyện (PTQ INT8).
    - Quy trình Calibration bằng NNCF và giải pháp hạn chế suy giảm mAP.
    - Phân tích chi tiết bảng kết quả độ chính xác (mAP@50).
- [ ] **Lê Đức Mạnh:**
  - Soạn thảo Slide phần Hệ thống & Thực nghiệm (5–6 slides):
    - Kiến trúc phần mềm `Edge-PreLabeler` và Unified Inference Engine.
    - Kết quả đo đạc phần cứng: Tốc độ FPS (tăng gấp bao nhiêu lần), Tiết kiệm RAM và Disk.
    - Quy trình tích hợp khép kín với CVAT REST API.
    - Demo trực tiếp (Live Demo) lệnh CLI chạy nén mô hình và nạp nhãn lên CVAT chỉ trong 3 giây.
- [ ] **Cả 2 thành viên:**
  - Ghép slide chung (Google Slides / PowerPoint chuẩn văn phong công nghệ cao).
  - Tập dượt thuyết trình (Rehearsal) và chuẩn bị bộ câu hỏi phản biện của Mentor (FAQ: Tại sao dùng PTQ mà không dùng QAT? Có bị tràn số khi lượng tử hóa không? Tại sao OpenVINO nhanh hơn ONNX trên CPU?...).

---

## BẢNG PHÂN CÔNG TỔNG HỢP & MA TRẬN TRÁCH NHIỆM (RACI MATRIX)

| Hạng mục công việc | Lê Đức Mạnh | Phạm Hoàng Anh | Kết quả đầu ra |
| :--- | :---: | :---: | :--- |
| 1. Khởi tạo Repo & Môi trường | **Accountable (A)** | Responsible (R) | Project template, `requirements.txt` |
| 2. Chuẩn bị Dataset & Calibration 300 ảnh | Consulted (C) | **Accountable (A)** | Dataset BDD100K subset, Calibration folder |
| 3. Xuất ONNX & OpenVINO FP16 | **Accountable (A)** | Consulted (C) | `yolov8s.onnx`, `yolov8s_fp16.xml` |
| 4. Lượng tử hóa NNCF Static PTQ INT8 | Consulted (C) | **Accountable (A)** | `yolov8s_int8.xml`, `.bin` |
| 5. Unified Inference Engine (Đa backend) | **Accountable (A)** | Responsible (R) | `BaseEngine`, `OpenVINOEngine`, `ONNXEngine` |
| 6. Benchmark Tốc độ, FPS, RAM, Disk | **Accountable (A)** | Consulted (C) | Biểu đồ hiệu năng, đo đạc phần cứng |
| 7. Benchmark mAP & Phân tích sai số | Consulted (C) | **Accountable (A)** | Bảng đo mAP@50, phân tích sai số lớp |
| 8. Tích hợp CVAT REST API & CLI Tool | **Accountable (A)** | Consulted (C) | Script đẩy nhãn CVAT, giao diện CLI |
| 9. Đo lường hiệu quả tiết kiệm thời gian gán nhãn | Consulted (C) | **Accountable (A)** | Số liệu Time Saving trên Job CVAT |
| 10. Báo cáo tổng kết & Slide bảo vệ | **Responsible (R)** | **Responsible (R)** | Slide trình chiếu, kịch bản bảo vệ |

*(Ghi chú: A - Chịu trách nhiệm chính/Duyệt; R - Trực tiếp thực hiện; C - Đóng góp ý kiến/Hỗ trợ)*
