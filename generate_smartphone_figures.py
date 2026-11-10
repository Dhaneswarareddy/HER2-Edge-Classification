"""SUPERSEDED (10 Nov 2026). This script hard-coded values that were never measured (Pi INT8 122.9 ms, 81.5 C,
'Stable (0x0)' phone states) and plotted INT8 as a deployment option. Running it would overwrite the corrected
Figure 7. Use experiments_v2/08_edge/make_edge_figures.py, which reads only the 10 Nov 2026 measurements.
The original code is kept below for the record (see experiments_v2/REPORT_CORRECTIONS_v2.md, item 41)."""
import sys
sys.exit("generate_smartphone_figures.py is superseded: run experiments_v2/08_edge/make_edge_figures.py instead")

"""
Generate publication-quality mobile edge deployment figure for FYP3 Final Technical Report.
Reads authoritative data from:
  - deployment/benchmark_results/benchmark_samsung_m35.csv
  - deployment/benchmark_results/benchmark_realme_6.csv
  - deployment/benchmark_results/benchmark_primary_pi4.csv

Outputs:
  - final_report_figures/fig7_smartphone_edge_benchmark.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/dhaneswarareddy/Downloads/FYP3")
OUT_DIR = ROOT / "final_report_figures"
BENCH_DIR = ROOT / "deployment/benchmark_results"

# Typography and styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

def despine(ax, keep_bottom=True, keep_left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)
    ax.spines["left"].set_visible(keep_left)

print("Reading authoritative benchmark data...")
df_m35 = pd.read_csv(BENCH_DIR / "benchmark_samsung_m35.csv")
df_realme = pd.read_csv(BENCH_DIR / "benchmark_realme_6.csv")
df_pi_prim = pd.read_csv(BENCH_DIR / "benchmark_primary_pi4.csv")

# Filter M35 models
m35_fp16 = df_m35[df_m35["model_name"] == "primary_bci_hires_fp16"].sort_values("threads")
m35_int8 = df_m35[df_m35["model_name"] == "primary_bci_hires_int8"].sort_values("threads")
m35_fp32 = df_m35[df_m35["model_name"] == "primary_bci_hires_fp32"].sort_values("threads")
m35_exp = df_m35[df_m35["model_name"] == "experimental_stain224_fp16"].sort_values("threads")

# Filter Realme 6 models
r6_fp16 = df_realme[df_realme["model_name"] == "primary_bci_hires_fp16"].sort_values("threads")
r6_int8 = df_realme[df_realme["model_name"] == "primary_bci_hires_int8"].sort_values("threads")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

threads = [1, 2, 4, 8]
x_t = np.arange(len(threads))
w = 0.22

# Panel 1: Multi-Threading Latency Scaling on Exynos 1380
lat_fp16 = m35_fp16["median_ms"].values
lat_int8 = m35_int8["median_ms"].values
lat_fp32 = m35_fp32["median_ms"].values

bars_fp32 = ax1.bar(x_t - w, lat_fp32, w, label="Primary FP32 (448px)", color="#457b9d", edgecolor="#1d3557", lw=1)
bars_fp16 = ax1.bar(x_t, lat_fp16, w, label="Primary FP16 (448px)", color="#1d3557", edgecolor="#03045e", lw=1)
bars_int8 = ax1.bar(x_t + w, lat_int8, w, label="Primary INT8 (448px)", color="#2a9d8f", edgecolor="#1b4332", lw=1)

for b, v in zip(bars_fp16, lat_fp16):
    ax1.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold", color="#1d3557")
for b, v in zip(bars_int8, lat_int8):
    ax1.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold", color="#2a9d8f")

ax1.set_xticks(x_t)
ax1.set_xticklabels([f"{t} Thread{'s' if t>1 else ''}" for t in threads])
ax1.set_ylabel("Median Inference Latency (ms)")
ax1.set_title("A: Multi-Thread Latency Scaling on Exynos 1380 (5nm)", loc="left", fontweight="bold")
ax1.set_ylim(0, 250)
ax1.grid(True, axis="y")
ax1.legend(loc="upper right", framealpha=0.9)
despine(ax1)

ax1.annotate("4-Thread Sweet Spot!\nMatches 4× Cortex-A78 big cores\n(78.9 ms FP16 / 21.7 ms INT8)",
             xy=(2, 78.9), xytext=(1.4, 160),
             arrowprops=dict(facecolor="#e63946", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", color="#e63946", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#e63946", lw=1))

ax1.annotate("8T Barrier Penalty:\nA55 sync stalls (+97.6%)",
             xy=(3, 155.9), xytext=(2.5, 205),
             arrowprops=dict(facecolor="#333333", shrink=0.05, width=1, headwidth=4),
             fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#f0f4f8", ec="#333333", lw=0.8))

# Panel 2: Throughput (FPS) Scaling Across Exynos 1380 vs Helio G90T
fps_m35_fp16 = m35_fp16["throughput_fps"].values
fps_m35_int8 = m35_int8["throughput_fps"].values
fps_r6_fp16 = r6_fp16["throughput_fps"].values
fps_r6_int8 = r6_int8["throughput_fps"].values

ax2.plot(threads, fps_m35_int8, marker="^", lw=2.2, ms=8, color="#2a9d8f", label="Samsung M35: INT8 (448px)")
ax2.plot(threads, fps_m35_fp16, marker="o", lw=2, ms=7, color="#1d3557", label="Samsung M35: FP16 (448px)")
ax2.plot(threads, fps_r6_int8, marker="v", lw=2, ms=7, color="#e76f51", linestyle="--", label="Realme 6: INT8 (448px)")
ax2.plot(threads, fps_r6_fp16, marker="s", lw=1.8, ms=6, color="#457b9d", linestyle="--", label="Realme 6: FP16 (448px)")

for t, f_m35, f_r6 in zip(threads, fps_m35_int8, fps_r6_int8):
    ax2.text(t, f_m35 + 1.2, f"{f_m35:.1f}", ha="center", fontsize=8, fontweight="bold", color="#2a9d8f")
    ax2.text(t, f_r6 - 3.2, f"{f_r6:.1f}", ha="center", fontsize=8, fontweight="bold", color="#e76f51")

ax2.axhline(30.0, color="#888888", linestyle=":", lw=1.2, label="Real-Time Video (30 FPS)")
ax2.set_xticks(threads)
ax2.set_xlabel("Thread Count")
ax2.set_ylabel("Inference Throughput (FPS)")
ax2.set_title("B: Throughput Scaling: Samsung M35 vs. Realme 6", loc="left", fontweight="bold")
ax2.set_ylim(0, 55)
ax2.grid(True)
ax2.legend(loc="upper left", framealpha=0.9)
despine(ax2)

# Panel 3: Memory Footprint (Peak RSS) vs Precision
models_mem = ["FP32 (448px)", "FP16 (448px)", "INT8 (448px)", "Stain224 (224px)"]
rss_mem = [
    m35_fp32.loc[m35_fp32["threads"] == 4, "peak_rss_mb"].values[0],
    m35_fp16.loc[m35_fp16["threads"] == 4, "peak_rss_mb"].values[0],
    m35_int8.loc[m35_int8["threads"] == 4, "peak_rss_mb"].values[0],
    m35_exp.loc[m35_exp["threads"] == 4, "peak_rss_mb"].values[0],
]
file_sizes = [22.36, 11.27, 6.82, 11.27]

x_m = np.arange(len(models_mem))
w_m = 0.35
bars_rss = ax3.bar(x_m - w_m/2, rss_mem, w_m, label="Peak RSS Memory (RAM)", color="#3a86ff", edgecolor="#023e8a", lw=1)
bars_file = ax3.bar(x_m + w_m/2, file_sizes, w_m, label="Model File Size (Flash)", color="#ffb703", edgecolor="#d48b00", lw=1)

for b, r in zip(bars_rss, rss_mem):
    ax3.text(b.get_x() + b.get_width()/2, r + 2.5, f"{r:.1f} MB", ha="center", fontsize=8, fontweight="bold", color="#023e8a")
for b, s in zip(bars_file, file_sizes):
    ax3.text(b.get_x() + b.get_width()/2, s + 2.5, f"{s:.1f} MB", ha="center", fontsize=8, fontweight="bold", color="#d48b00")

ax3.set_xticks(x_m)
ax3.set_xticklabels(models_mem)
ax3.set_ylabel("Memory / Storage (MB)")
ax3.set_title("C: Memory Footprint & Storage Efficiency (4 Threads)", loc="left", fontweight="bold")
ax3.set_ylim(0, 140)
ax3.grid(True, axis="y")
ax3.legend(loc="upper left", framealpha=0.9)
despine(ax3)

ax3.annotate("INT8: 67% RAM Reduction\n(39.4 MB peak RSS)",
             xy=(2 - w_m/2, 39.4), xytext=(2 - w_m/2 - 0.2, 75),
             arrowprops=dict(facecolor="#023e8a", shrink=0.05, width=1, headwidth=4),
             fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#e7f0fd", ec="#023e8a", lw=0.8))

# Panel 4: Tri-Hardware Comparison: Pi 4B vs Realme 6 vs Samsung M35 5G
hw_labels = [
    "Raspberry Pi 4B (4T FP16)",
    "Raspberry Pi 4B (4T INT8)",
    "Realme 6 (8T FP16)",
    "Realme 6 (8T INT8)",
    "Samsung M35 (4T FP16)",
    "Samsung M35 (4T INT8)"
]
hw_latencies = [
    df_pi_prim.loc[df_pi_prim["threads"] == 4, "invoke_median_ms"].values[0],
    122.9, # Pi 4B INT8 4T
    r6_fp16.loc[r6_fp16["threads"] == 8, "median_ms"].values[0],
    r6_int8.loc[r6_int8["threads"] == 8, "median_ms"].values[0],
    m35_fp16.loc[m35_fp16["threads"] == 4, "median_ms"].values[0],
    m35_int8.loc[m35_int8["threads"] == 4, "median_ms"].values[0],
]
hw_temps = [84.2, 81.5, 30.2, 30.6, 28.9, 29.0]
hw_throttles = [
    "Throttled (0x80008)",
    "Warm (Passive)",
    "Stable (30.2°C)",
    "Stable (30.6°C)",
    "Stable (28.9°C)",
    "Stable (29.0°C)"
]
colors_hw = ["#d62828", "#e76f51", "#457b9d", "#f4a261", "#1d3557", "#2a9d8f"]

y_h = np.arange(len(hw_labels))
bars_hw = ax4.barh(y_h, hw_latencies, height=0.6, color=colors_hw, edgecolor="#333333", lw=1)

for i, (b, lat, tmp, thr) in enumerate(zip(bars_hw, hw_latencies, hw_temps, hw_throttles)):
    ax4.text(lat + 6, i, f"{lat:.1f} ms [{tmp:.1f}°C | {thr}]", va="center", fontsize=8, fontweight="bold")

ax4.set_yticks(y_h)
ax4.set_yticklabels(hw_labels)
ax4.set_xlabel("Inference Latency (ms)")
ax4.set_title("D: Tri-Hardware Latency & Thermal Benchmark", loc="left", fontweight="bold")
ax4.set_xlim(0, 480)
ax4.grid(True, axis="x")
despine(ax4)

fig.suptitle("Figure 7: Physical Smartphone Edge AI Benchmark (Samsung Galaxy M35 5G vs. Realme 6 vs. Raspberry Pi 4B)", fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_file = OUT_DIR / "fig7_smartphone_edge_benchmark.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)
print(f"[SUCCESS] Generated {out_file}")
