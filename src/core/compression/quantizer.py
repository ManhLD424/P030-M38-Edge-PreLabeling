"""
Module Quantizer: Lượng tử hóa sau huấn luyện tĩnh (Static PTQ INT8) bằng Intel NNCF.
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import List, Optional, Callable
import numpy as np
import cv2
import openvino as ov
import nncf


class ModelQuantizer:
    """
    Quản lý quy trình lượng tử hóa OpenVINO IR sang INT8 sử dụng NNCF.
    """

    def __init__(self, fp16_xml_path: str):
        self.xml_path = Path(fp16_xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Không tìm thấy OpenVINO model tại: {self.xml_path}")

        self.core = ov.Core()
        self.model = self.core.read_model(str(self.xml_path))

    def _prepare_calibration_data(self, calib_dir: Optional[str] = None, subset_size: int = 100) -> List[np.ndarray]:
        """
        Nạp ảnh calibration thực tế hoặc tạo tập ảnh giả lập (synthetic) chuẩn hóa nếu thư mục trống.
        """
        data = []
        if calib_dir and Path(calib_dir).exists():
            image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(calib_dir).glob(ext)))

            for img_path in image_files[:subset_size]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
                img = np.expand_dims(img, axis=0)  # BCHW
                data.append(img)

        # Nếu không có đủ ảnh thật, sinh dữ liệu synthetic để đảm bảo calibration chạy được
        if len(data) < 10:
            print(f"[i] Tập calibration có {len(data)} ảnh thật. Bổ sung dữ liệu synthetic ({subset_size} mẫu)...")
            data = [np.random.uniform(0.0, 1.0, (1, 3, 640, 640)).astype(np.float32) for _ in range(subset_size)]

        return data

    def quantize_int8(self, output_dir: str, calib_dir: Optional[str] = None, subset_size: int = 100) -> Path:
        """
        Thực hiện Post-Training Quantization (PTQ) sang INT8 và lưu model.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("[*] Đang chuẩn bị tập dữ liệu Calibration cho NNCF...")
        calib_data = self._prepare_calibration_data(calib_dir, subset_size=subset_size)

        def transform_fn(data_item):
            return data_item

        calibration_dataset = nncf.Dataset(calib_data, transform_fn)

        print("[*] Đang tiến hành lượng tử hóa NNCF Static PTQ (Preset: PERFORMANCE / INT8 Symmetric)...")
        quantized_model = nncf.quantize(
            model=self.model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.CPU,
            subset_size=subset_size
        )

        dest_xml = out_path / f"{self.xml_path.stem.replace('_fp16', '')}_int8.xml"
        ov.save_model(quantized_model, str(dest_xml))
        print(f"[+] Lượng tử hóa INT8 thành công! Đã lưu tại: {dest_xml}")
        return dest_xml
