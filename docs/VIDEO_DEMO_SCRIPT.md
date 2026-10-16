# Kịch Bản Video Demo 3 Phút (Video Demo Script)
## Dự Án M38: Model Compression for Edge Pre-Labeling (BDD100K)
### Cuộc Thi AI20K - Giai Đoạn Build Phase (AI20K Build Phase Checkpoint)

---

## 1. Thông Tin Tổng Quan (Metadata)

- **Tên dự án:** M38 — Model Compression for Edge Pre-Labeling
- **Thời lượng:** Đúng 3 phút 00 giây (180 giây)
- **Định dạng video:** 1920x1080 Full HD, 60fps / 30fps
- **Giọng thuyết minh:** Tiếng Việt (chuẩn, tự tin, tốc độ 130-150 từ/phút), phụ đề tiếng Anh (English Subtitles)
- **Mục tiêu video:** Chứng minh hệ thống nén mô hình BDD100K chạy thực tế trên laptop CPU, đạt tốc độ xấp xỉ 19 FPS (nhanh gấp 2.94 lần), nén 3.8 lần, mAP drop chỉ 2.64%, web demo tương tác mượt mà và khả năng tiền gán nhãn xuất ra CVAT.

---

## 2. Bảng Phân Cảnh Chi Tiết (Shot-by-Shot Timeline)

| Phân Cảnh (Scene) | Thời Gian (Timestamp) | Nội Dung Hình Ảnh (Visual / Screen Action) | Lời Thuyết Minh (Voiceover Script - Tiếng Việt) | Ghi Chú Kỹ Thuật (Production / SFX) |
|---|---|---|---|---|
| **Scene 1: Hook & Đặt Vấn Đề** | 00:00 - 00:25 (25s) | • **00:00 - 00:08**: Hiện Slide bìa / Thumbnail dự án M38 với logo AI20K & VinAI.<br>• **00:08 - 00:20**: Quay màn hình annotator đang vẽ bbox thủ công trên video giao thông đường phố; hiện đồ thị chi phí & độ trễ khi gọi Cloud API (high latency, API cost, rủi ro bảo mật dữ liệu khách hàng).<br>• **00:20 - 00:25**: Dòng chữ xuất hiện: *"Làm sao để tiền gán nhãn chạy trực tiếp trên laptop thông thường mà không cần GPU rời?"* | Xin chào Ban Giám Khảo và các Lab Coaches. Trong thị giác máy tính cho xe tự hành, gán nhãn thủ công hàng triệu khung hình BDD100K vô cùng tốn kém và mệt mỏi. Gửi ảnh lên Cloud API thì phát sinh độ trễ mạng, chi phí băng thông và nguy cơ rò rỉ dữ liệu nhạy cảm. Dự án M38 mang đến giải pháp: Nén mô hình thị giác máy tính chạy trực tiếp trên CPU laptop biên, tự động tiền gán nhãn tức thì với chi phí hạ tầng bằng không. | Nhạc nền ambient công nghệ nhẹ nhàng, âm lượng 20%. Chữ xuất hiện motion mượt mà. |
| **Scene 2: Giải Pháp & Kiến Trúc Nén** | 00:25 - 00:55 (30s) | • **00:25 - 00:40**: Chiếu sơ đồ kiến trúc hệ thống (`docs/ARCHITECTURE.md`): Luồng nén từ YOLOv8s PyTorch FP32 $\rightarrow$ ONNX Graph Optimization $\rightarrow$ NNCF Post-Training Quantization INT8 với 25 ảnh chuẩn BDD100K.<br>• **00:40 - 00:55**: Hiện biểu đồ radar & cột so sánh: Kích thước model giảm từ 42.8 MB xuống 11.2 MB (giảm 3.8x). | Chúng tôi thiết kế pipeline nén tối ưu đa tầng trên bộ dữ liệu xe tự hành BDD100K. Xuất phát từ baseline YOLOv8s, mô hình được export sang ONNX chuẩn hóa, sau đó lượng tử hóa sau huấn luyện Post-Training Quantization sang INT8 bằng Intel NNCF và OpenVINO. Đặc biệt, chúng tôi hiệu chuẩn trên tập ảnh BDD100K thực tế với Fast Bias Correction, giúp nén dung lượng mô hình 3.8 lần mà vẫn giữ nguyên độ chuẩn xác hình học. | Highlight các block: *Calibration Dataset*, *NNCF Quantizer*, *OpenVINO Engine*. |
| **Scene 3: Live Demo Web MVP (So Sánh Trực Quan)** | 00:55 - 01:45 (50s) | • **00:55 - 01:10**: Mở trình duyệt Web MVP (`docs/mvp/index.html` hoặc `http://127.0.0.1:8000`). Bấm chọn ảnh `BDD03.jpg` (đường phố đông đúc xe cộ và người đi bộ).<br>• **01:10 - 01:30**: Bấm nút **"Run Comparison"**. Màn hình chia đôi chạy suy luận đồng thời:<br>  - Bên trái: PyTorch FP32 (156 ms - 6.4 FPS)<br>  - Bên phải: OpenVINO INT8 (53 ms - 18.8 FPS).<br>  - Các bounding box phát hiện xe, người, đèn tín hiệu khớp nhau chính xác.<br>• **01:30 - 01:45**: Kéo thanh trượt Confidence Threshold từ 0.25 lên 0.45; lọc tắt lớp "traffic light" chỉ để lại "car" và "pedestrian". Bấm nút **"Export CVAT JSON"** $\rightarrow$ tải về file `cvat_prelabels.json`. | Và đây là sản phẩm thực tế: Giao diện Web Edge Pre-Labeling tương tác trực tiếp. Khi bấm so sánh, mô hình gốc PyTorch mất 156 mili-giây, trong khi mô hình OpenVINO INT8 đã nén chỉ mất 53 mili-giây — tức nhanh gấp gần 3 lần và đạt tốc độ 18.8 FPS mượt mà ngay trên CPU Intel laptop phổ thông! Bounding box phát hiện xe hơi, người đi bộ và biển báo giao thông bám sát thực tế. Annotator có thể tinh chỉnh ngưỡng tin cậy, lọc nhãn theo nhu cầu và xuất ngay file pre-label chuẩn CVAT để tiếp tục duyệt nhãn. | Âm thanh click chuột rõ ràng. Phóng to (zoom-in) vào đồng hồ Latency và FPS đo thực. |
| **Scene 4: Đo Kiểm & Khả Năng Tái Lập (CLI & Benchmarks)** | 01:45 - 02:15 (30s) | • **01:45 - 02:00**: Mở terminal, chạy lệnh benchmark 1 dòng lệnh:<br>`python -m src.cli.main benchmark-2d`<br>Màn hình terminal in bảng Rich UI hiển thị rõ P50, P95, Throughput.<br>• **02:00 - 02:15**: Chạy lệnh tính mAP:<br>`python -m src.cli.main eval-accuracy`<br>Bảng kết quả mAP@50 hiển thị: Baseline FP32 = 80.56%, INT8 = 77.92%, **mAP drop = 2.64%** (hoàn toàn nằm trong ngưỡng an toàn < 3%). | Không chỉ có giao diện, toàn bộ kết quả đo kiểm đều có thể tái lập 100% bằng giao diện dòng lệnh. Bộ benchmark đo độc lập trên CPU máy tính cá nhân cho thấy độ trễ P50 giảm từ 156ms xuống 53ms. Đánh giá độ chính xác mAP@50 theo chuẩn COCO/BDD100K cho thấy độ sụt giảm chỉ 2.64%, hoàn toàn nằm trong ngưỡng chấp nhận được đối với tác vụ tiền gán nhãn. | Hiện box cam kết: *"100% Real Benchmark - Zero Mock Data"*. |
| **Scene 5: Tính Năng Mở Rộng Độc Đáo (LiDAR 3D)** | 02:15 - 02:40 (25s) | • **02:15 - 02:30**: Chuyển sang CLI chạy lệnh 3D LiDAR:<br>`python -m src.cli.main infer-3d --pcd data/pointclouds/demo.pcd`<br>Terminal xử lý 17,238 điểm LiDAR, trích xuất 3D Cuboids.<br>• **02:30 - 02:40**: Chiếu sơ đồ tích hợp CVAT 3D Copilot: Xuất 16-point cuboids và nạp thẳng vào CVAT 3D qua REST API. | Để tạo sự khác biệt vượt trội, M38 mở rộng thêm module tiền gán nhãn đám mây điểm LiDAR 3D với mô hình PointPillars tối ưu hóa hình học. Module này tự động tách mặt phẳng đường, phát hiện cụm phương tiện và trích xuất hộp 3D Cuboid theo chuẩn 16 tọa độ của CVAT 3D Tool, giúp annotator tiết kiệm tới 75% thời gian vẽ hộp không gian 3 chiều. | B-roll hình ảnh đám mây điểm 3D và bounding cuboids xoay 360 độ. |
| **Scene 6: Tổng Kết & Kêu Gọi Hành Động (CTA)** | 02:40 - 03:00 (20s) | • **02:40 - 02:50**: Chiếu slide tổng kết thành tựu: Pipeline 2D BDD100K nén 3.8x, tăng tốc 2.94x, 20/20 Unit Tests passing, Web MVP & Live GitHub Pages.<br>• **02:50 - 03:00**: Hiện QR code dẫn đến GitHub repository `ManhLD424/P030-M38-Edge-PreLabeling` và link Demo Day. | Tóm lại, M38 mang đến một giải pháp hoàn thiện, trung thực và thực dụng: Nén mô hình chuẩn xác cho BDD100K trên máy tính biên, giảm triệt để chi phí gán nhãn xe tự hành. Toàn bộ mã nguồn, dữ liệu và hướng dẫn chạy đã sẵn sàng trên GitHub. Cảm ơn các bạn đã theo dõi và hẹn gặp lại tại Demo Day! | Nhạc nền crescendo kết thúc, hiện thông tin team và link GitHub. |

---

## 3. Lời Thoại Tiếng Anh (English Subtitles Track)

- **[00:00 - 00:25]**: "Welcome judges and lab coaches! In autonomous driving computer vision, manually annotating millions of BDD100K frames is tedious and costly. Relying on Cloud APIs introduces network latency, high infrastructure bills, and privacy risks. Project M38 provides the answer: Compressing vision models to run directly on commodity laptop CPUs for instant edge pre-labeling at zero server cost."
- **[00:25 - 00:55]**: "We designed a multi-stage compression pipeline on the autonomous driving BDD100K dataset. Starting from YOLOv8s baseline, the model is exported to ONNX and quantized to INT8 using Intel NNCF and OpenVINO Post-Training Quantization with Fast Bias Correction on real BDD100K calibration data, shrinking model size by 3.8x from 42.8 MB to 11.2 MB."
- **[00:55 - 01:45]**: "Here is our live Web MVP: An interactive edge pre-labeling studio. Comparing side-by-side on an Intel Core i5 laptop CPU, baseline PyTorch takes 156 milliseconds, whereas our OpenVINO INT8 model completes inference in just 53 milliseconds — nearly 3x faster and delivering smooth 18.8 FPS! Bounding boxes for cars, pedestrians, and traffic signs match accurately. Annotators can adjust confidence thresholds, filter categories, and instantly export standard CVAT JSON annotations."
- **[01:45 - 02:15]**: "Beyond the web UI, all results are 100% reproducible via our CLI. Our offline benchmark confirms P50 latency dropped from 156ms to 53ms. Accuracy evaluation shows an mAP@50 drop of only 2.64%, well within the safe operational threshold for pre-labeling."
- **[02:15 - 02:40]**: "As a high-value differentiator, M38 extends to 3D LiDAR point clouds via geometry-optimized PointPillars, extracting 16-point 3D Cuboids fully compatible with CVAT 3D Tool to save up to 75% annotation time."
- **[02:40 - 03:00]**: "In summary, M38 delivers a verified, production-ready edge pre-labeling pipeline for BDD100K with 20/20 unit tests passing. Full code, docs, and benchmarks are available on GitHub. Thank you, and see you at Demo Day!"

---

## 4. Hướng Dẫn Quay & Dựng Video (Screen Recording & Editing Checklist)

1. **Chuẩn bị môi trường quay:**
   - Đặt độ phân giải màn hình về **1920x1080**.
   - Tắt thông báo hệ thống (Windows Focus Assist / Do Not Disturb).
   - Đổi theme terminal sang Dark Mode (hoặc dùng Windows Terminal với font Cascadia Code).
2. **Các lệnh quay màn hình:**
   ```bash
   # Cửa sổ 1: Chạy Web MVP Server
   python -m src.cli.main serve --host 127.0.0.1 --port 8000
   
   # Cửa sổ 2: Chạy benchmark CLI
   python -m src.cli.main benchmark-2d --runs 20
   python -m src.cli.main eval-accuracy
   python -m src.cli.main infer-3d --pcd data/pointclouds/demo.pcd
   ```
3. **Thao tác trên Web UI (`http://127.0.0.1:8000`):**
   - Chọn ảnh đông xe `BDD03.jpg`.
   - Bấm nút **"Run Comparison (PyTorch FP32 vs OpenVINO INT8)"**.
   - Di chuột qua lại hai ảnh để khán giả thấy bounding boxes vẽ mượt mà.
   - Thử kéo slider `Confidence: 0.35`.
   - Bấm nút **"Export CVAT JSON"** để xuất hiện thông báo tải file thành công.
4. **Hậu kỳ (Post-Production):**
   - Cắt gọt các đoạn chờ hoặc load thừa, đảm bảo từng giây đều có thông tin hữu ích.
   - Chèn callout zoom 120% vào bảng so sánh tốc độ (53.17 ms / 18.8 FPS).
   - Thêm nhạc nền không bản quyền (YouTube Audio Library / Epidemic Sound).
   - Xuất file định dạng `H.264 MP4`, bitrate 12-16 Mbps.
