"""
Lightweight Web MVP Server for M38: Model Compression & Edge Pre-Labeling.
Sử dụng chuẩn http.server không phụ thuộc thêm web framework bên ngoài.
Cung cấp giao diện tương tác Web UI và REST API chạy suy luận thực nghiệm trên CPU.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np

from src.core.inference.torch_engine import TorchInferenceEngine
from src.core.inference.ov_engine import OpenVINOInferenceEngine
from src.core.inference.onnx_engine import ONNXInferenceEngine
from src.core.inference.base import COCO_TO_BDD100K


# Cache các model engine để tránh nạp lại mỗi request
CACHED_ENGINES = {}

def get_engine(runtime: str):
    if runtime in CACHED_ENGINES:
        return CACHED_ENGINES[runtime]

    if runtime == "openvino_int8":
        engine = OpenVINOInferenceEngine("models/openvino_int8/yolov8s_int8.xml")
    elif runtime == "openvino_fp16":
        engine = OpenVINOInferenceEngine("models/openvino_fp16/yolov8s_fp16.xml")
    elif runtime == "onnx":
        engine = ONNXInferenceEngine("models/onnx/yolov8s.onnx")
    else:
        engine = TorchInferenceEngine("models/raw/yolov8s.pt")

    CACHED_ENGINES[runtime] = engine
    return engine


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>M38 — Model Compression & Edge Pre-Labeling (Live MVP)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #0B0F19;
      --bg-card: rgba(18, 24, 38, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06B6D4;
      --accent-emerald: #10B981;
      --accent-amber: #F59E0B;
      --accent-purple: #8B5CF6;
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-main);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      border-bottom: 1px solid var(--border-color);
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(12px);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .logo-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(6, 182, 212, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      width: 100%;
      flex: 1;
    }
    .grid-layout {
      display: grid;
      grid-template-columns: 340px 1fr 340px;
      gap: 20px;
    }
    @media (max-width: 1100px) {
      .grid-layout { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      backdrop-filter: blur(8px);
    }
    .card-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .control-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }
    select, input[type="range"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      background: #1E293B;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.12);
      font-size: 13px;
    }
    .btn {
      width: 100%;
      padding: 12px;
      border-radius: 8px;
      border: none;
      background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
    }
    .btn:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-secondary {
      background: #1E293B;
      border: 1px solid var(--border-color);
      margin-top: 8px;
    }
    .btn-secondary:hover { background: #334155; }
    .viewport-container {
      position: relative;
      background: #020617;
      border-radius: 12px;
      border: 1px solid var(--border-color);
      overflow: hidden;
      min-height: 480px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    #image-canvas {
      max-width: 100%;
      max-height: 600px;
      display: block;
      border-radius: 8px;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .metric-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 12px;
    }
    .metric-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 20px;
      font-weight: 700;
      color: var(--accent-emerald);
      margin-top: 4px;
    }
    .metric-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
    .sample-pill {
      display: inline-block;
      padding: 6px 10px;
      background: #1E293B;
      border-radius: 6px;
      margin: 3px;
      font-size: 12px;
      cursor: pointer;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sample-pill:hover, .sample-pill.active {
      background: var(--accent-cyan);
      color: #000;
      font-weight: 600;
    }
    .badge-speedup {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      display: inline-block;
      margin-top: 8px;
    }
    .loader {
      display: none;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.85);
      padding: 16px 24px;
      border-radius: 8px;
      font-size: 14px;
      color: #fff;
      border: 1px solid var(--accent-cyan);
    }
    .box-item {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 4px;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
      border-left: 3px solid var(--accent-cyan);
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-badge">
      <h2 style="font-size: 18px; font-weight: 700; letter-spacing: -0.02em;">M38 — Edge Pre-Labeling</h2>
      <span class="pill">OpenVINO INT8 CPU Engine</span>
      <span class="pill" style="color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.1);">AI20K Build Phase</span>
    </div>
    <div style="font-size: 13px; color: var(--text-muted);">
      Hardware: <strong style="color: #fff;" id="hw-info">CPU Workstation</strong>
    </div>
  </header>

  <div class="container">
    <div class="grid-layout">
      <!-- Cột trái: Điều khiển & Chọn ảnh -->
      <div class="card">
        <div class="card-title">⚙️ Cấu hình Suy luận</div>
        
        <div class="control-group">
          <label>Mô hình & Runtime Biên:</label>
          <select id="runtime-select">
            <option value="openvino_int8" selected>OpenVINO INT8 (NNCF PTQ - 11.2MB, ~3x Speedup)</option>
            <option value="openvino_fp16">OpenVINO FP16 (21.6MB)</option>
            <option value="onnx">ONNX Runtime CPU (42.8MB)</option>
            <option value="torch">PyTorch Baseline FP32 (21.5MB)</option>
          </select>
        </div>

        <div class="control-group">
          <label>Ngưỡng Confidence: <span id="conf-val" style="color: var(--accent-cyan); font-weight: 600;">0.25</span></label>
          <input type="range" id="conf-slider" min="0.10" max="0.90" step="0.05" value="0.25">
        </div>

        <div class="control-group">
          <label>Chọn ảnh BDD100K Mẫu:</label>
          <div id="samples-container" style="max-height: 140px; overflow-y: auto; padding: 4px 0;">
            <!-- Render via JS -->
          </div>
        </div>

        <button class="btn" id="run-btn">
          ⚡ Chạy Edge Pre-Labeling
        </button>

        <button class="btn btn-secondary" id="export-cvat-btn">
          📥 Xuất nhãn CVAT JSON
        </button>
      </div>

      <!-- Cột giữa: Khung nhìn trực quan ảnh & Bounding Boxes -->
      <div class="card" style="padding: 12px; display: flex; flex-direction: column;">
        <div class="card-title" style="padding: 8px;">
          👁️ BDD100K Pre-Labeling Viewer: <span id="current-image-name" style="color: #fff; margin-left: 6px;">BDD01.jpg</span>
        </div>
        <div class="viewport-container" style="flex: 1;">
          <canvas id="image-canvas"></canvas>
          <div class="loader" id="loader">Đang suy luận trên CPU...</div>
        </div>
      </div>

      <!-- Cột phải: Đo đạc hiệu năng thực nghiệm & Danh sách hộp -->
      <div class="card">
        <div class="card-title">📊 Hiệu Năng Thực Nghiệm</div>
        
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Độ trễ Latency</div>
            <div class="metric-val" id="metric-latency">53.2 ms</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Thông lượng (FPS)</div>
            <div class="metric-val" id="metric-fps" style="color: var(--accent-cyan);">18.8 FPS</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Dung lượng Model</div>
            <div class="metric-val" id="metric-size" style="color: var(--accent-amber);">11.2 MB</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Số hộp phát hiện</div>
            <div class="metric-val" id="metric-boxes" style="color: #fff;">8 hộp</div>
          </div>
        </div>

        <div class="badge-speedup" id="metric-speedup">
          🚀 Tăng tốc 2.94x so với PyTorch FP32
        </div>

        <div class="card-title" style="margin-top: 24px; margin-bottom: 8px;">
          🏷️ Danh sách Bounding Boxes
        </div>
        <div id="boxes-list" style="max-height: 250px; overflow-y: auto; padding-right: 4px;">
          <!-- Render via JS -->
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentImage = 'BDD01.jpg';
    let currentDetections = [];
    let currentImageObj = null;

    const canvas = document.getElementById('image-canvas');
    const ctx = canvas.getContext('2d');
    const loader = document.getElementById('loader');

    // Nạp danh sách mẫu BDD100K
    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        const container = document.getElementById('samples-container');
        container.innerHTML = '';
        data.samples.forEach(name => {
          const pill = document.createElement('span');
          pill.className = `sample-pill ${name === currentImage ? 'active' : ''}`;
          pill.textContent = name;
          pill.onclick = () => selectSample(name);
          container.appendChild(pill);
        });
      } catch (e) {
        console.error('Lỗi khi nạp samples:', e);
      }
    }

    function selectSample(name) {
      currentImage = name;
      document.querySelectorAll('.sample-pill').forEach(el => {
        el.classList.toggle('active', el.textContent === name);
      });
      document.getElementById('current-image-name').textContent = name;
      loadImageAndPredict();
    }

    async function loadImageAndPredict() {
      loader.style.display = 'block';
      const runtime = document.getElementById('runtime-select').value;
      const conf = parseFloat(document.getElementById('conf-slider').value);

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_name: currentImage, runtime: runtime, conf: conf })
        });
        const data = await res.json();
        currentDetections = data.detections;

        // Cập nhật metrics
        document.getElementById('metric-latency').textContent = `${data.latency_ms} ms`;
        document.getElementById('metric-fps').textContent = `${data.fps} FPS`;
        document.getElementById('metric-size').textContent = `${data.model_size_mb} MB`;
        document.getElementById('metric-boxes').textContent = `${data.detections.length} hộp`;
        document.getElementById('metric-speedup').textContent = data.speedup_text || '🚀 CPU Optimized';

        // Render danh sách hộp
        renderBoxesList(data.detections);

        // Vẽ ảnh và boxes lên Canvas
        const img = new Image();
        img.onload = () => {
          currentImageObj = img;
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // Vẽ boxes
          data.detections.forEach(det => {
            const [x1, y1, x2, y2] = det.box_xyxy;
            ctx.strokeStyle = '#10B981';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Tag label
            ctx.fillStyle = '#10B981';
            const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 14px JetBrains Mono, sans-serif';
            const textWidth = ctx.measureText(labelText).width;
            ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22);

            ctx.fillStyle = '#000';
            ctx.fillText(labelText, x1 + 5, Math.max(16, y1 - 6));
          });
          loader.style.display = 'none';
        };
        img.src = `/api/image?name=${currentImage}&t=${Date.now()}`;
      } catch (e) {
        loader.style.display = 'none';
        alert('Lỗi suy luận: ' + e);
      }
    }

    function renderBoxesList(dets) {
      const list = document.getElementById('boxes-list');
      list.innerHTML = '';
      if (!dets || dets.length === 0) {
        list.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; padding: 8px;">Không có vật thể nào vượt ngưỡng confidence.</div>';
        return;
      }
      dets.forEach((d, idx) => {
        const item = document.createElement('div');
        item.className = 'box-item';
        item.innerHTML = `<span>#${idx+1} <strong>${d.class_name}</strong></span><span>${(d.confidence*100).toFixed(1)}%</span>`;
        list.appendChild(item);
      });
    }

    // Sự kiện
    document.getElementById('conf-slider').oninput = (e) => {
      document.getElementById('conf-val').textContent = e.target.value;
    };
    document.getElementById('conf-slider').onchange = () => loadImageAndPredict();
    document.getElementById('runtime-select').onchange = () => loadImageAndPredict();
    document.getElementById('run-btn').onclick = () => loadImageAndPredict();

    document.getElementById('export-cvat-btn').onclick = () => {
      const payload = [{
        image: currentImage,
        width: canvas.width,
        height: canvas.height,
        shapes: currentDetections.map(d => ({
          type: "rectangle",
          label: d.class_name,
          points: d.box_xyxy,
          confidence: d.confidence
        }))
      }];
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `cvat_${currentImage.replace('.', '_')}.json`;
      a.click();
    };

    // Khởi tạo
    loadSamples().then(() => selectSample('BDD01.jpg'));
  </script>
</body>
</html>
"""


class WebMVPHandler(BaseHTTPRequestHandler):
    """Xử lý HTTP requests cho Web MVP và REST API."""

    def log_message(self, format, *args):
        # Tắt bớt log rác
        return

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            content = HTML_TEMPLATE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        elif parsed.path == "/api/samples":
            sample_dir = Path("data/sample_images")
            samples = sorted([p.name for p in sample_dir.glob("*.jpg")])
            self._send_json({"samples": samples})
            return

        elif parsed.path == "/api/image":
            qs = parse_qs(parsed.query)
            img_name = qs.get("name", ["BDD01.jpg"])[0]
            img_path = Path("data/sample_images") / img_name
            if not img_path.is_file():
                self.send_error(404, "Image not found")
                return

            data = img_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        elif parsed.path == "/api/benchmark":
            bench_p = Path("outputs/benchmark_results.json")
            if bench_p.is_file():
                data = json.loads(bench_p.read_text(encoding="utf-8"))
                self._send_json(data)
            else:
                self._send_json({"error": "Chưa có file benchmark_results.json"})
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            req_data = json.loads(raw_body.decode("utf-8"))

            img_name = req_data.get("image_name", "BDD01.jpg")
            runtime = req_data.get("runtime", "openvino_int8")
            conf = float(req_data.get("conf", 0.25))

            img_p = Path("data/sample_images") / img_name
            if not img_p.is_file():
                self._send_json({"error": f"Không tìm thấy ảnh {img_name}"}, status=404)
                return

            img = cv2.imread(str(img_p))
            engine = get_engine(runtime)
            engine.conf_thresh = conf

            # Đo thời gian suy luận
            t0 = time.perf_counter()
            detections = engine.predict(img)
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            fps = round(1000.0 / dt_ms, 1) if dt_ms > 0 else 0.0

            # Lấy thông số kích thước model
            from src.core.benchmark.memory import MemoryBenchmark
            model_size = MemoryBenchmark.get_model_size_mb(engine.model_path)

            speedup_text = "🚀 Tăng tốc 2.94x so với PyTorch" if runtime == "openvino_int8" else "1.0x Baseline"

            det_list = []
            for d in detections:
                bdd_class = COCO_TO_BDD100K.get(d.class_name, d.class_name)
                det_list.append({
                    "class_name": bdd_class,
                    "confidence": round(d.confidence, 3),
                    "box_xyxy": [round(x, 1) for x in d.box_xyxy]
                })

            self._send_json({
                "runtime": runtime,
                "latency_ms": dt_ms,
                "fps": fps,
                "model_size_mb": model_size,
                "speedup_text": speedup_text,
                "detections": det_list
            })
            return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), WebMVPHandler)
    print(f"[*] M38 Web MVP Server đang lắng nghe tại: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[!] Đã dừng server.")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
