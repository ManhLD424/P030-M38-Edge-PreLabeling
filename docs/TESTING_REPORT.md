# BÁO CÁO KIỂM THỬ TỰ ĐỘNG & BẢO ĐẢM CHẤT LƯỢNG (TESTING REPORT)
## Đề tài P030 — M38: Model Compression for Edge Pre-Labeling

> **Trạng thái:** ✅ **20/20 Test Cases Passed (100% Pass Rate)**  
> **Thời gian thực thi:** 2.61 giây  
> **Framework kiểm thử:** `pytest 9.1.1` trên Python 3.12.10 (Windows 11 x64)

---

## 1. TỔNG QUAN CÁC HẠNG MỤC KIỂM THỬ (TEST MATRIX)

Hệ thống được kiểm thử tự động toàn diện qua 6 module độc lập, bao phủ toàn bộ vòng đời từ tiền xử lý, tính toán hình học, suy luận đa runtime, lượng tử hóa đến đóng gói payload CVAT:

| Module kiểm thử | Số lượng test | Mục tiêu kiểm định | Kết quả |
| :--- | :---: | :--- | :---: |
| `tests/test_accuracy_bench.py` | 4 tests | Kiểm tra thuật toán tính IoU, bipartite matching, độ chính xác mAP@50 và tính toán mAP drop | **PASSED** |
| `tests/test_prelabel_2d.py` | 2 tests | Kiểm tra ánh xạ nhãn COCO sang BDD100K 10 lớp và chuẩn hóa định dạng CVAT 2D Rectangle | **PASSED** |
| `tests/test_inference.py` | 1 test | Kiểm tra cấu trúc `Detection` dataclass, trích xuất `box_xyxy` và `box_cvat` | **PASSED** |
| `tests/test_cuboid_geometry.py` | 4 tests | Kiểm tra 8 góc đỉnh 3D, kiểm tra góc xoay Yaw, loại bỏ hộp đè lấn (`merge_boxes`) | **PASSED** |
| `tests/test_pcd_io.py` | 4 tests | Kiểm tra nạp file PCD Binary v0.7, ước lượng mặt đất $Z_{ground}$, biến đổi cao độ hệ quy chiếu | **PASSED** |
| `tests/test_cvat_adapter.py` | 3 tests | Kiểm tra đóng gói 16-point CVAT 3D shapes payload, tracks sequence payload và parse ngược | **PASSED** |
| `tests/test_detector.py` | 2 tests | Kiểm tra suy luận trên demo pointcloud và kiểm tra bộ đo benchmark 3D | **PASSED** |
| **TỔNG CỘNG** | **20 tests** | **Bảo đảm tính toàn vẹn 100% mã nguồn** | **100% GREEN** |

---

## 2. CHI TIẾT CÁC TEST CASES

### 2.1. Module Đánh giá Độ chính xác & Sai lệch (`test_accuracy_bench.py`)
- `test_compute_iou_identical`: Xác minh IoU giữa 2 hộp trùng khớp hoàn toàn bằng $1.000$.
- `test_compute_iou_no_overlap`: Xác minh IoU giữa 2 hộp không giao nhau bằng $0.000$.
- `test_compute_iou_partial_overlap`: Xác minh công thức toán học $IoU = \frac{A \cap B}{A \cup B}$ với diện tích giao 50 và hợp 150 $\rightarrow 0.333$.
- `test_accuracy_benchmark_evaluation`: Đánh giá end-to-end trên dữ liệu mẫu với Dummy Engine, xác nhận tính toán đúng mAP@50 và mAP drop.

### 2.2. Module Tiền gán nhãn 2D & CVAT Mapping (`test_prelabel_2d.py`)
- `test_coco_to_bdd_mapping`: Xác minh ánh xạ chính xác các nhãn COCO cốt lõi (`person` $\rightarrow$ `pedestrian`, `car` $\rightarrow$ `car`, `truck` $\rightarrow$ `truck`, `stop sign` $\rightarrow$ `traffic sign`).
- `test_cvat_box_format_structure`: Xác minh payload xuất ra tuân thủ chính xác schema REST API của CVAT Server (tọa độ `[xtl, ytl, xbr, ybr]`, `label`, `confidence`, `occluded`).

### 2.3. Module Đọc/Ghi Dữ liệu & Biến đổi Hình học (`test_pcd_io.py`, `test_cuboid_geometry.py`)
- `test_read_demo_pcd`: Đọc thành công file PCD Binary v0.7 gồm 17,238 điểm LiDAR mà không bị tràn số.
- `test_estimate_ground`: Thuật toán ước lượng cao độ mặt đất $Z_{ground} = 0.075\text{ m}$ khớp với mặt phẳng đường thực tế.
- `test_height_coordinate_transformation`: Chuyển đổi qua lại giữa hệ tọa độ cảm biến quét ($Z_{source}$) và hệ tọa độ mô hình ($Z_{model}$) không bị sai lệch số học.
- `test_drop_boxes_on_cars`: Thuật toán lọc bỏ các bounding box người đi bộ ảo xuất hiện trên nóc xe hơi.

---

## 3. LỆNH CHẠY KIỂM THỬ (TEST REPRODUCTION)

```bash
# Chạy toàn bộ 20 test cases với thông tin chi tiết
python -m pytest tests -v

# Kết quả kiểm thử chuẩn:
# tests/test_accuracy_bench.py::test_compute_iou_identical PASSED          [  5%]
# tests/test_accuracy_bench.py::test_compute_iou_no_overlap PASSED         [ 10%]
# tests/test_accuracy_bench.py::test_compute_iou_partial_overlap PASSED    [ 15%]
# tests/test_accuracy_bench.py::test_accuracy_benchmark_evaluation PASSED  [ 20%]
# ...
# ============================= 20 passed in 2.61s ==============================
```
