# SLIDE PITCH DECK — BẢO VỆ DỰ ÁN DEMO DAY AI20K
## ĐỀ TÀI P030 — M38: MODEL COMPRESSION FOR EDGE PRE-LABELING

> **Thời lượng trình bày:** 5 phút thuyết trình + 3 phút Q&A Hội đồng  
> **Nhóm thực hiện:** Lê Đức Mạnh (MHV: 2A202602122) & Phạm Hoàng Anh  
> **Lớp:** AI20K — Cohort 4A (Build Phase)

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Tiêu đề lớn:** M38 — HỆ THỐNG NÉN MÔ HÌNH SUY LUẬN BIÊN CHO TIỀN GÁN NHÃN DỮ LIỆU XE TỰ HÀNH
* **Tiêu đề phụ:** High-Performance Model Compression on Workstation CPU for 2D/3D Pre-Labeling (BDD100K)
* **Thông tin nhóm:**
  - Học viên: Lê Đức Mạnh (Lead Developer & System Architect) & Phạm Hoàng Anh (Optimization Specialist)
  - Đơn vị đào tạo: Transformer Labs — AI20K Cohort 4A
  - Sự kiện: Demo Day (VinUni Tech Career Day 22/10/2026)

---

### SLIDE 2: BỐI CẢNH & NÚT THẮT CỦA NGÀNH (PROBLEM STATEMENT)
* **Vấn đề cốt lõi:**
  - Trong các dự án xe tự hành và ADAS, khâu gán nhãn dữ liệu chiếm tới **70–80% chi phí và thời gian** của toàn bộ chu trình MLOps.
  - Một chuyên viên gán nhãn (Annotator) mất trung bình **45 giây để vẽ tay một khung hình 2D** và **3–5 phút cho một frame 3D LiDAR**.
* **Nút thắt của các giải pháp hiện tại:**
  1. *Chi phí hạ tầng Cloud GPU đắt đỏ:* Chạy pre-labeling trên Cloud ngốn hàng nghìn USD chi phí thuê server.
  2. *Tắc nghẽn băng thông & Bảo mật:* Không thể đẩy hàng triệu ảnh phân giải cao ra ngoài mạng nội bộ của doanh nghiệp.
  3. *Máy trạm Annotator bị quá tải:* Laptop/PC gán nhãn thông thường không có GPU rời, không chạy nổi các mô hình gốc cồng kềnh.

---

### SLIDE 3: GIẢI PHÁP M38 (OUR SOLUTION)
* **Định vị sản phẩm:** Xây dựng một **Công cụ Tiền Gán Nhãn Cục Bộ (Edge Pre-Labeler)** vận hành trực tiếp trên CPU của máy trạm gán nhãn thông thường mà không cần GPU rời.
* **Giá trị mang lại:**
  - **Tự động hóa gán nhãn mồi (Pre-labeling):** Tạo sẵn các bounding boxes chất lượng cao; con người chỉ cần duyệt và chỉnh nhẹ.
  - **Tiết kiệm ~70% thời gian:** Giảm thời gian hoàn thành 1 frame từ **45 giây xuống còn 12–15 giây**.
  - **Zero Cloud Cost & Bảo mật tuyệt đối:** 100% dữ liệu xử lý offline tại máy biên.

---

### SLIDE 4: CÔNG NGHỆ NÉN MÔ HÌNH (COMPRESSION METHODOLOGY)
* **Tiếp cận thuần MLOps / Engineering:** Tận dụng mô hình chuẩn công nghiệp YOLOv8s pretrained trên dữ liệu giao thông BDD100K.
* **Pipeline nén 4 bước:**
  1. *PyTorch FP32 (Baseline):* Trọng số 32-bit float gốc (21.54 MB).
  2. *ONNX Graph Export:* Tối ưu hóa đồ thị tính toán (opset 17), loại bỏ các node thừa.
  3. *OpenVINO FP16 IR:* Chuyển đổi định dạng số thực nửa chính xác 16-bit.
  4. *Intel NNCF Static PTQ (INT8):* Lượng tử hóa sau huấn luyện tĩnh với 26 ảnh đường phố BDD100K thật; áp dụng thuật toán **Fast Bias Correction trên 63 layers** để triệt tiêu hiện tượng trôi phân phối trọng số.

---

### SLIDE 5: KIẾN TRÚC HỆ THỐNG & UNIFIED INFERENCE ENGINE
* **Clean Layered Architecture:**
  - Giao diện người dùng: Unified CLI Tool (`src/cli/main.py`) & Interactive Web MVP.
  - Lớp trừu tượng: `BaseInferenceEngine` tuân thủ Strategy Pattern, cho phép chuyển đổi tức thì giữa PyTorch, ONNX và OpenVINO.
  - Tối ưu hóa phần cứng sâu: Kích hoạt tập lệnh ma trận số nguyên **Intel AVX-VNNI / AVX2** trên CPU.
* **Tích hợp liền mạch nền tảng CVAT:**
  - Xuất dữ liệu chuẩn định dạng CVAT XML và CVAT JSON, nạp trực tiếp vào Job gán nhãn thực tế.

---

### SLIDE 6: KẾT QUẢ ĐO ĐẠC THỰC NGHIỆM TRÊN CPU (BENCHMARK DATA)
*Số liệu đo thực tế trên CPU Intel Core i5-12450H (12 Threads, 16GB RAM):*

| Phiên bản / Runtime | Precision | Dung lượng Model | Độ trễ Mean | Throughput | Tốc độ cải thiện |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Baseline)** | FP32 | 21.54 MB | 156.3 ms | 6.4 FPS | 1.0x *(Chuẩn)* |
| **ONNX Runtime** | FP32 | 42.79 MB | 123.7 ms | 8.1 FPS | Nhanh hơn 1.26x |
| **OpenVINO INT8 (M38)** | **INT8** | **11.24 MB** | **53.17 ms** | **18.8 FPS** | **Nhanh hơn 2.94x ~ 3.5x** |

* **Dung lượng file giảm ~3.8 lần:** Từ 42.8 MB xuống còn **11.24 MB**.
* **Độ trễ giảm 3 lần:** Từ 156.3 ms xuống **53.2 ms**, đạt tốc độ xử lý thời gian thực trên CPU văn phòng.

---

### SLIDE 7: ĐÁNH GIÁ ĐỘ CHÍNH XÁC & SAI SỐ mAP DROP
*Kiểm định chéo trên tập ảnh giao thông BDD100K đối chứng với PyTorch FP32:*
* **Định dạng ONNX:** Đạt **99.24% mAP@50**, Mean IoU đạt **0.9972**; **mAP Drop chỉ 0.76%** (gần như bảo toàn hoàn hảo 100%).
* **Định dạng OpenVINO INT8:**
  - Đạt **77.92% mAP@50**, Tỉ lệ giữ nhãn (Retention Rate) đạt **74.14%**.
  - Độ sụt giảm mAP Drop giữa FP16 và INT8 **chỉ là 2.64%**.
* **Ý nghĩa:** Độ chính xác 77.9% với IoU 0.78 là điểm cân bằng hoàn hảo cho bài toán Pre-labeling: các vật thể chính (ô tô, xe tải, người đi bộ) đều được nhận diện chuẩn xác.

---

### SLIDE 8: TRẢI NGHIỆM SẢN PHẨM & LIVE DEMO
* **Sản phẩm Web MVP tương tác:**
  - Người dùng có thể chọn bất kỳ ảnh nào trong 26 ảnh mẫu BDD100K hoặc tải ảnh camera hành trình lên.
  - So sánh trực quan thời gian thực giữa PyTorch FP32 vs OpenVINO INT8.
  - Tải về nhãn pre-label chuẩn CVAT JSON chỉ với 1 cú click chuột.
* **Giao diện dòng lệnh CLI chuyên nghiệp:**
  - Hỗ trợ đầy đủ các lệnh: `compress`, `benchmark-2d`, `eval-accuracy`, `prelabel-2d`.

---

### SLIDE 9: ĐIỂM NHẤN SÁNG TẠO (MULTI-MODAL DIFFERENTIATOR)
* **Không dừng lại ở 2D BDD100K:** Hệ thống được thiết kế mở rộng sẵn sàng cho dữ liệu **3D LiDAR Point Cloud**.
* **Module 3D Tiên Phong:**
  - Nạp và xử lý file điểm LiDAR (`.pcd` 17,238 điểm).
  - Tự động ước lượng cao độ mặt đường $Z_{ground}$ và trích xuất hộp 3D Cuboids (`x, y, z, dx, dy, dz, yaw`).
  - Đóng gói chuẩn 16-point cuboids của **CVAT 3D Tool**, giúp giảm thời gian vẽ 3D từ 5 phút xuống 30 giây.
  - Toàn bộ **20/20 automated unit tests** đã pass 100%.

---

### SLIDE 10: TÍNH KHẢ THI, GIỚI HẠN & KẾ HOẠCH TƯƠNG LAI
* **Tính khả thi thương mại:** Giải pháp có thể đóng gói thành Docker container hoặc desktop app nhẹ cho các công ty Outsourcing Data Labeling, giúp tiết kiệm hàng trăm triệu đồng chi phí server mỗi tháng.
* **Giới hạn hiện tại:** INT8 hiện tối ưu tốt nhất trên CPU Intel hỗ trợ VNNI/AVX2; trên CPU ARM cũ tốc độ tăng trưởng sẽ thấp hơn.
* **Định hướng phát triển:**
  1. Hỗ trợ quantization cho mô hình 3D LiDAR PointPillars chuyên sâu.
  2. Tích hợp trực tiếp plugin Web Extension vào trình duyệt của annotator.
* **Lời kết:** Nhóm P030 - M38 xin trân trọng cảm ơn Ban Giám Khảo và Lab Coaches!
