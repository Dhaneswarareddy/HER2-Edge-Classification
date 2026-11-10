"""
Generate publication-quality composite figure of the physical Android App running
on the Realme 6 (RMX2001, MediaTek Helio G90T, Android 11) for FYP3 documentation.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREEN_DIR = Path("/Users/dhaneswarareddy/Downloads/FYP3/deployment/app_screenshots/realme_6")
OUT_DIR = Path("/Users/dhaneswarareddy/Downloads/FYP3/final_report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

screens = [
    ("01_app_overview_hardware_and_controls.png", "A: Device and configuration\n(Helio G90T • 2× A76 + 6× A55 • Android 11)"),
    ("04_app_inference_score_2_equivocal_fp16.png", "B: Reference tile 2+ (FP16)\n(equivocal: reflex ISH required)"),
    ("05_app_inference_score_3_positive_fp16.png", "C: Reference tile 3+ (FP16)\n(HER2-positive; pathologist review)"),
    ("06_app_inference_score_3_int8_quantized.png", "D: INT8 selected\n(failed accuracy gate: speed test only)"),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 10.5))

for ax, (fname, title) in zip(axes, screens):
    img_bgr = cv2.imread(str(SCREEN_DIR / fname))
    # Crop status bar (top 115px) and nav bar (bottom 115px)
    img_cropped = img_bgr[115:2285, :]
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12, color="#0d3b66")
    ax.axis("off")

fig.suptitle("Figure 10: Android app v1.1 on Realme 6 (MediaTek Helio G90T, Android 11), screenshots 10 Nov 2026", 
             fontsize=15, fontweight="bold", y=0.98, color="#051b2e")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_file = OUT_DIR / "fig10_realme6_app_interface_and_inference.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)

print(f"[SUCCESS] Composite publication figure saved to: {out_file}")
