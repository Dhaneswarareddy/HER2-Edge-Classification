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
    ("01_app_overview_hardware_and_controls.png", "A: Hardware Platform & Control Panel\n(Helio G90T • 2× A76 + 6× A55 • Android 11)"),
    ("02_app_inference_score_0_negative_fp16.png", "B: Score 0 (Negative Biopsy)\n(No Overexpression • 100.0% P0)"),
    ("04_app_inference_score_2_equivocal_fp16.png", "C: Score 2+ (Equivocal / Borderline)\n(ASCO/CAP Reflex Dual-Probe FISH Mandated)"),
    ("05_app_inference_score_3_positive_fp16.png", "D: Score 3+ (Positive Overexpressed)\n(Targeted Anti-HER2 Therapy Indicated)"),
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

fig.suptitle("Figure 10: Physical Point-of-Care Android Application Deployment (Realme 6 • MediaTek Helio G90T • Android 11)", 
             fontsize=15, fontweight="bold", y=0.98, color="#051b2e")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_file = OUT_DIR / "fig10_realme6_app_interface_and_inference.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)

print(f"[SUCCESS] Composite publication figure saved to: {out_file}")
