# NHẬT KÝ TIẾN ĐỘ DỰ ÁN HÀNG TUẦN (WEEKLY PROGRESS LOGS)
## Đề tài P030 — M38: Model Compression for Edge Pre-Labeling (AI20K Cohort 4A)

> **Ghi chú tính xác thực:** Toàn bộ nhật ký dưới đây được tái dựng và đối chiếu trực tiếp từ lịch sử Git commits, tài liệu nộp Mentor (`MENTOR_DUTY_SUBMISSION.md`) và các mốc làm việc thực tế tại repository `https://github.com/ManhLD424/P030-M38-Edge-PreLabeling`.

---

### TUẦN 1: KHỞI TẠO HỆ THỐNG & ĐỊNH HƯỚNG BÀI TOÁN (05/10/2026 – 09/10/2026)
* **Mục tiêu tuần:** Tiếp nhận đề tài P030 - M38 (Model Compression for Edge Pre-Labeling), phân tích yêu cầu BDD100K và thiết lập kiến trúc cơ sở.
* **Hoạt động thực tế (Căn cứ Commit `6b4573c`):**
  - Khởi tạo cấu trúc Clean Architecture cho dự án.
  - Xây dựng đặc tả kỹ thuật ban đầu (`docs/SYSTEM_SPECIFICATION.md`) và kế hoạch triển khai chi tiết (`docs/IMPLEMENTATION_PLAN.md`).
  - Thiết lập môi trường Python 3.12, cấu hình `requirements.txt` (khóa `numpy<2.0`, tích hợp `ultralytics`, `openvino`, `onnxruntime`, `click`, `rich`).
* **Khó khăn & Giải pháp:**
  - *Khó khăn:* Việc tự huấn luyện hay fine-tune mô hình từ đầu trên BDD100K đòi hỏi tài nguyên GPU lớn và tốn thời gian.
  - *Giải pháp:* Tham vấn Mentor/Coach và thống nhất sử dụng checkpoint Pre-trained công nghiệp (YOLOv8s), tập trung 100% nỗ lực vào kỹ thuật nén mô hình MLOps (PTQ INT8) và tối ưu hóa suy luận CPU.

---

### TUẦN 2: TRIỂN KHAI PIPELINE NÉN & ĐA RUNTIME ENGINE (10/10/2026 – 12/10/2026)
* **Mục tiêu tuần:** Hiện thực hóa các bộ chuyển đổi mô hình (ONNX, OpenVINO FP16) và lượng tử hóa tĩnh (NNCF PTQ INT8).
* **Hoạt động thực tế (Căn cứ Commits `d93709f`, `3a17d87`, `008abd9`):**
  - Viết module `src/core/compression/exporter.py`: Hỗ trợ xuất sang ONNX opset 17 và OpenVINO FP16 IR.
  - Viết module `src/core/compression/quantizer.py`: Tích hợp bộ công cụ Intel NNCF thực hiện Post-Training Quantization sang INT8.
  - Hiện thực hóa `TorchInferenceEngine`, `ONNXInferenceEngine`, `OpenVINOInferenceEngine` tuân thủ Strategy Pattern qua interface `BaseInferenceEngine`.
  - Viết bộ đo kiểm độ trễ `src/core/benchmark/latency.py` và bộ nhớ `src/core/benchmark/memory.py`.
  - Hoàn thành báo cáo tiến độ nộp Mentor duty ngày 11/10/2026: Ghi nhận tốc độ INT8 đạt 22.4 FPS trên CPU Core i5-12450H (tăng tốc ~3.5x).

---

### TUẦN 3: MỞ RỘNG 3D LIDAR, HIỆU CHỈNH CHUẨN ĐỀ THI & HOÀN THIỆN BÀI NỘP (13/10/2026 – 16/10/2026)
* **Mục tiêu tuần:** Kết nối công cụ CVAT, mở rộng bài toán sang 3D LiDAR, đồng thời chuẩn hóa toàn diện theo đúng yêu cầu đề tài M38 để nộp bài trước hạn chót.
* **Hoạt động thực tế (Căn cứ Commit `6318f4b` và giai đoạn nước rút):**
  - *Ngày 13/10:* Phát triển mô-đun tiền gán nhãn 3D LiDAR (`pcd_io`, `ground`, `cuboid`, `cvat_adapter`, `detector`) hỗ trợ định dạng 16-point cuboids của CVAT 3D Tool.
  - *Ngày 16/10 (Audit & Rectification):*
    - Nhận diện khoảng trống so với đề M38 gốc: Khôi phục và đưa trọng tâm 2D BDD100K thành trục chính; giữ module 3D làm tính năng mở rộng sáng tạo (Multi-Modal Edge Pre-Labeling).
    - Nạp 26 ảnh giao thông thực tế BDD100K vào `data/calibration/` và `data/sample_images/`.
    - Thực hiện lượng tử hóa lại INT8 với dữ liệu BDD100K thực tế (Fast Bias Correction 63/63 layers).
    - Xây dựng module `accuracy_bench.py`, đo đạc thành công mAP@50 (77.92%) và mAP Drop (2.64% vs FP16).
    - Hợp nhất toàn bộ CLI tại `src/cli/main.py`.
    - Xây dựng Web MVP tương tác (`src/web/app.py` và `docs/mvp/index.html`).
    - Nâng cấp bộ test cases lên 20/20 tests pass 100%.
    - Soạn thảo trọn bộ hồ sơ bài nộp Build Phase (Pitch Deck, Kịch bản Video 3 phút, Benchmark Report, Testing Report).
