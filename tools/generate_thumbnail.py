"""
Script sinh Thumbnail đại diện dự án chuẩn kích thước 1200x630 (16:9 / OpenGraph standard)
cho cổng nộp bài Demo Day AI20K Build Phase.
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(12, 6.3), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor('#0B0F19')
ax.set_xlim(0, 1200)
ax.set_ylim(0, 630)
ax.axis('off')

# Gradient background effect
for i in range(630):
    alpha = 0.85 - (i / 630.0) * 0.4
    ax.axhline(i, color='#111827', alpha=alpha, linewidth=1.5)

# Glowing accent shapes
rect_glow = patches.FancyBboxPatch((40, 40), 1120, 550, boxstyle="round,pad=10,rounding_size=20",
                                   edgecolor='#06B6D4', facecolor='#131D31', alpha=0.6, linewidth=2)
ax.add_patch(rect_glow)

# Top Bar Badges
badge1 = patches.FancyBboxPatch((80, 520), 220, 36, boxstyle="round,pad=5,rounding_size=8",
                                edgecolor='#10B981', facecolor='#064E3B', alpha=0.9, linewidth=1.5)
ax.add_patch(badge1)
ax.text(190, 538, "AI20K BUILD PHASE", color='#10B981', fontsize=12, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')

badge2 = patches.FancyBboxPatch((320, 520), 240, 36, boxstyle="round,pad=5,rounding_size=8",
                                edgecolor='#06B6D4', facecolor='#083344', alpha=0.9, linewidth=1.5)
ax.add_patch(badge2)
ax.text(440, 538, "TOPIC P030 — M38", color='#06B6D4', fontsize=12, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')

# Main Title & Subtitle
ax.text(80, 440, "EDGE PRE-LABELING", color='#FFFFFF', fontsize=38, fontweight='bold', fontfamily='sans-serif')
ax.text(80, 385, "High-Performance Model Compression on Workstation CPU", color='#38BDF8', fontsize=20, fontweight='medium', fontfamily='sans-serif')
ax.text(80, 345, "Dataset: BDD100K 2D Camera Object Detection + 3D LiDAR Point Cloud Extension", color='#94A3B8', fontsize=14, fontfamily='sans-serif')

# 3 Metric Cards
# Card 1: Speedup
c1 = patches.FancyBboxPatch((80, 160), 320, 140, boxstyle="round,pad=10,rounding_size=12",
                           edgecolor='#334155', facecolor='#0F172A', linewidth=1.5)
ax.add_patch(c1)
ax.text(100, 260, "TỐC ĐỘ SUY LUẬN CPU", color='#94A3B8', fontsize=12, fontweight='bold', fontfamily='sans-serif')
ax.text(100, 205, "2.94x ~ 3.5x", color='#10B981', fontsize=32, fontweight='bold', fontfamily='sans-serif')
ax.text(100, 175, "Latency giảm từ 156ms xuống 53ms", color='#64748B', fontsize=11, fontfamily='sans-serif')

# Card 2: Model Size
c2 = patches.FancyBboxPatch((440, 160), 320, 140, boxstyle="round,pad=10,rounding_size=12",
                           edgecolor='#334155', facecolor='#0F172A', linewidth=1.5)
ax.add_patch(c2)
ax.text(460, 260, "DUNG LƯỢNG MÔ HÌNH", color='#94A3B8', fontsize=12, fontweight='bold', fontfamily='sans-serif')
ax.text(460, 205, "11.2 MB", color='#F59E0B', fontsize=32, fontweight='bold', fontfamily='sans-serif')
ax.text(460, 175, "Nén 3.8x so với ONNX (42.8MB)", color='#64748B', fontsize=11, fontfamily='sans-serif')

# Card 3: Accuracy
c3 = patches.FancyBboxPatch((800, 160), 320, 140, boxstyle="round,pad=10,rounding_size=12",
                           edgecolor='#334155', facecolor='#0F172A', linewidth=1.5)
ax.add_patch(c3)
ax.text(820, 260, "mAP@50 RETENTION", color='#94A3B8', fontsize=12, fontweight='bold', fontfamily='sans-serif')
ax.text(820, 205, "99.2% ONNX", color='#818CF8', fontsize=32, fontweight='bold', fontfamily='sans-serif')
ax.text(820, 175, "mAP drop chỉ 0.76% (ONNX) / 2.6% (INT8)", color='#64748B', fontsize=11, fontfamily='sans-serif')

# Footer
ax.text(80, 80, "Tác giả: Lê Đức Mạnh (2A202602122) & Phạm Hoàng Anh • AI20K Cohort 4A", color='#94A3B8', fontsize=13, fontfamily='sans-serif')
ax.text(1120, 80, "Demo Day 22/10/2026 • VinUni", color='#06B6D4', fontsize=13, fontweight='bold', ha='right', fontfamily='sans-serif')

out_path = Path("docs/figures/thumbnail.png")
plt.savefig(out_path, dpi=100)
plt.close()
print(f"[+] Đã tạo thumbnail đại diện tại: {out_path}")
