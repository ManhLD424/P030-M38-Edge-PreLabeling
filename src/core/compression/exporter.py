"""
Module Exporter: Chuyển đổi mô hình Pre-trained PyTorch sang ONNX và OpenVINO FP16.
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import os
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO
import openvino as ov


class ModelExporter:
    """
    Quản lý quy trình xuất đồ thị tính toán từ checkpoint PyTorch gốc (.pt).
    """

    def __init__(self, weights_path: str, output_root: Optional[str] = "models"):
        self.weights_path = Path(weights_path)
        self.output_root = Path(output_root)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def export_onnx(self, output_dir: Optional[str] = None, opset: int = 17) -> Path:
        """
        Xuất mô hình sang định dạng ONNX tiêu chuẩn.
        """
        target_dir = Path(output_dir) if output_dir else self.output_root / "onnx"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Đang xuất ONNX (opset {opset}) từ: {self.weights_path}...")
        exported_path = self.model.export(
            format="onnx",
            opset=opset,
            dynamic=False,
            simplify=True,
            imgsz=640
        )
        dest_path = target_dir / f"{self.weights_path.stem}.onnx"
        if Path(exported_path) != dest_path:
            import shutil
            shutil.move(str(exported_path), str(dest_path))

        print(f"[+] Xuất ONNX thành công: {dest_path}")
        return dest_path

    def export_openvino_fp16(self, output_dir: Optional[str] = None) -> Path:
        """
        Chuyển đổi sang định dạng OpenVINO Intermediate Representation (FP16).
        """
        target_dir = Path(output_dir) if output_dir else self.output_root / "openvino_fp16"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Đang chuyển đổi sang OpenVINO IR (FP16)...")
        exported_path = self.model.export(
            format="openvino",
            half=True,
            imgsz=640
        )
        exported_dir = Path(exported_path)
        
        # Di chuyển các artifact (.xml, .bin) vào target_dir
        dest_xml = target_dir / f"{self.weights_path.stem}_fp16.xml"
        dest_bin = target_dir / f"{self.weights_path.stem}_fp16.bin"

        for file in exported_dir.glob("*.xml"):
            file.replace(dest_xml)
        for file in exported_dir.glob("*.bin"):
            file.replace(dest_bin)

        print(f"[+] Xuất OpenVINO FP16 thành công tại: {dest_xml}")
        return dest_xml
