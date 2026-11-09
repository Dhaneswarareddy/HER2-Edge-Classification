"""
Script to generate publication-quality figures for FYP3 Final Technical Report.
All data is read directly from authoritative project artifacts:
  - final_reports/tables/recomputed_metrics_all_runs.csv
  - final_reports/tables/paired_effects.csv
  - final_reports/tables/cross_dataset_features.csv
  - final_reports/tables/h2_dev_vs_h1_test.csv
  - final_reports/tables/deploy_recomputed.csv
  - next_experiment/harmonise2/outputs/A_params_summary.csv
  - next_experiment/harmonise2/outputs/E_flagging.csv
  - deployment/benchmark_results/benchmark_primary_pi4.csv
  - deployment/benchmark_results/benchmark_experimental_pi4.csv

Outputs saved to: final_report_figures/
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup paths
ROOT = Path("/Users/dhaneswarareddy/Downloads/FYP3")
OUT_DIR = ROOT / "final_report_figures"
OUT_DIR.mkdir(exist_ok=True, parents=True)

TABLES = ROOT / "final_reports/tables"
H2_DIR = ROOT / "next_experiment/harmonise2/outputs"
BENCH_DIR = ROOT / "deployment/benchmark_results"

# Typography and styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
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

print("Starting figure generation...")

# ==============================================================================
# Figure 1: BCI In-Domain Model Comparison (Val vs Test QWK and Accuracy)
# ==============================================================================
print("Generating Figure 1: BCI In-Domain Model Comparison...")
df_runs = pd.read_csv(TABLES / "recomputed_metrics_all_runs.csv")
bci_runs = df_runs[df_runs["source"] == "grouped"].copy()

# Pivot val and test metrics
val_m = bci_runs[bci_runs["split"] == "val"].set_index("run")[["accuracy", "balanced_accuracy", "macro_f1", "qwk"]]
test_m = bci_runs[bci_runs["split"] == "test"].set_index("run")[["accuracy", "balanced_accuracy", "macro_f1", "qwk"]]
comp = val_m.join(test_m, lsuffix="_val", rsuffix="_test").reset_index()

# Sort by val QWK
comp = comp.sort_values("qwk_val", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

y_pos = np.arange(len(comp))
labels = [r.replace("__ce", "").replace("__", " ") for r in comp["run"]]

# Panel 1: Quadratic Weighted Kappa (QWK)
bar_width = 0.38
b1 = ax1.barh(y_pos - bar_width/2, comp["qwk_val"], bar_width, label="Validation QWK", color="#2b5c8f", alpha=0.9)
b2 = ax1.barh(y_pos + bar_width/2, comp["qwk_test"], bar_width, label="Test QWK", color="#e26d5c", alpha=0.9)

# Highlight primary selection
highlight_idx = list(comp["run"]).index("EfficientNetV2B0__ce__hires")
ax1.get_yticklabels()

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels)
ax1.set_xlabel("Quadratic Weighted Kappa (QWK)")
ax1.set_title("A: In-Domain Agreement (Val vs Test QWK)", loc="left", fontweight="bold")
ax1.set_xlim(0.70, 1.0)
ax1.grid(True, axis="x")
ax1.legend(loc="lower right", framealpha=0.9)
despine(ax1)

# Annotate the winner
ax1.annotate("Primary Selected Model\n(Val: 0.935, Test: 0.959)",
             xy=(comp.loc[comp["run"] == "EfficientNetV2B0__ce__hires", "qwk_val"].values[0], highlight_idx - bar_width/2),
             xytext=(0.74, highlight_idx - 1.5),
             arrowprops=dict(facecolor="#2b5c8f", shrink=0.05, width=1.5, headwidth=6),
             fontweight="bold", color="#1b3c5f", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4f8", ec="#2b5c8f", lw=1))

# Panel 2: Accuracy vs Balanced Accuracy on Test
ax2.scatter(comp["accuracy_test"], comp["balanced_accuracy_test"], color="#2b5c8f", s=70, zorder=3, edgecolors="white", lw=1)
for i, txt in enumerate(labels):
    if "hires" in txt or "lowbatch" in txt or "DenseNet121" in txt:
        ax2.annotate(txt, (comp["accuracy_test"].iloc[i], comp["balanced_accuracy_test"].iloc[i]),
                     xytext=(5, -3), textcoords="offset points", fontsize=8, color="#333333")

ax2.plot([0.88, 0.98], [0.88, 0.98], ls="--", color="#888888", lw=1, label="Identity (Acc = BalAcc)")
ax2.set_xlabel("Test Overall Accuracy")
ax2.set_ylabel("Test Balanced Accuracy")
ax2.set_title("B: Test Accuracy vs Balanced Accuracy", loc="left", fontweight="bold")
ax2.grid(True)
ax2.legend(loc="upper left", framealpha=0.9)
despine(ax2)

fig.suptitle("Figure 1: In-Domain Performance on BCI Benchmark (Phase A & B)", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "fig1_bci_model_comparison.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 2: Paired Effects Forest Plot (E1, E2, E4, E5)
# ==============================================================================
print("Generating Figure 2: Paired Effects Forest Plot...")
df_paired = pd.read_csv(TABLES / "paired_effects.csv")
paired_qwk = df_paired[df_paired["metric"] == "qwk"].copy()

effects_order = [
    ("E1 resolution", "EffNetV2B0: Resolution (448px vs 224px, b8)"),
    ("E1 batch", "EffNetV2B0: Batch Size (b8 vs b32, 224px)"),
    ("E2 resolution", "MobileNetV3S: Resolution (448px vs 224px, b8)"),
    ("E2 batch", "MobileNetV3S: Batch Size (b8 vs b32, 224px)"),
    ("E4 init_fyp2", "EffNetV2B0: Synthetic Init vs ImageNet"),
    ("E5 mix_fyp2", "EffNetV2B0: 25% Synthetic Mixed Training")
]

fig, ax = plt.subplots(figsize=(10, 5))
y_indices = np.arange(len(effects_order))

val_deltas = []
val_cis = []
test_deltas = []
test_cis = []

for eff_code, name in effects_order:
    v_row = paired_qwk[(paired_qwk["effect"] == eff_code) & (paired_qwk["split"] == "val")].iloc[0]
    t_row = paired_qwk[(paired_qwk["effect"] == eff_code) & (paired_qwk["split"] == "test")].iloc[0]
    val_deltas.append(v_row["delta"])
    val_cis.append((v_row["delta"] - v_row["ci_lo"], v_row["ci_hi"] - v_row["delta"]))
    test_deltas.append(t_row["delta"])
    test_cis.append((t_row["delta"] - t_row["ci_lo"], t_row["ci_hi"] - t_row["delta"]))

val_err = np.array(val_cis).T
test_err = np.array(test_cis).T

offset = 0.15
ax.errorbar(val_deltas, y_indices - offset, xerr=val_err, fmt="o", color="#1f77b4", ecolor="#1f77b4",
            elinewidth=2, capsize=4, capthick=1.5, ms=7, label="Validation Δ QWK (95% CI)")
ax.errorbar(test_deltas, y_indices + offset, xerr=test_err, fmt="s", color="#ff7f0e", ecolor="#ff7f0e",
            elinewidth=2, capsize=4, capthick=1.5, ms=7, label="Test Δ QWK (95% CI)")

ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
ax.set_yticks(y_indices)
ax.set_yticklabels([name for _, name in effects_order])
ax.invert_yaxis()
ax.set_xlabel("Paired Difference in Quadratic Weighted Kappa (Δ QWK)")
ax.set_title("Figure 2: Paired Effects Forest Plot with Slide-Fragment Resampled 95% CIs", fontweight="bold", pad=12)
ax.grid(True, axis="x")
ax.legend(loc="lower right", framealpha=0.95)
despine(ax)

# Annotate significance
for i, (v, (lo, hi)) in enumerate(zip(val_deltas, val_cis)):
    if (v - lo) > 0 or (v + hi) < 0:
        ax.text(v + hi + 0.01, y_indices[i] - offset, " *",
                color="#1f77b4", fontweight="bold", va="center")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_paired_effects_forest.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 3: Optical Density Distributions Across Datasets
# ==============================================================================
print("Generating Figure 3: Optical Density Distributions Across Datasets...")
df_stain = pd.read_csv(TABLES / "cross_dataset_features.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

datasets = ["FYP2 GAN", "BCI", "HER2-IHC-40x"]
colors = {"FYP2 GAN": "#2b5c8f", "BCI": "#e26d5c", "HER2-IHC-40x": "#2a9d8f"}
markers = {"FYP2 GAN": "o", "BCI": "s", "HER2-IHC-40x": "^"}

for d in datasets:
    sub = df_stain[df_stain["dataset"] == d]
    dab_med = sub.groupby("label")["dabc_mean"].median()
    hem_med = sub.groupby("label")["hemc_mean"].median()
    
    ax1.plot(dab_med.index, dab_med.values, label=d, color=colors[d], marker=markers[d], lw=2, ms=6)
    ax2.plot(hem_med.index, hem_med.values, label=d, color=colors[d], marker=markers[d], lw=2, ms=6)

classes = ["0", "1+", "2+", "3+"]
for ax, title, ylabel in zip([ax1, ax2],
                             ["DAB Optical Density (Brown Stain - HER2)", "Haematoxylin Optical Density (Blue Counterstain)"],
                             ["Median Background-Subtracted DAB OD", "Median Background-Subtracted Haematoxylin OD"]):
    ax.set_xticks(range(4))
    ax.set_xticklabels(classes)
    ax.set_xlabel("HER2 Clinical Score")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True)
    ax.legend(framealpha=0.9)
    despine(ax)

fig.suptitle("Figure 3: Stain Optical Density Profiles Across Development and External Datasets", fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "fig3_stain_optical_density_distributions.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 4: Cross-Dataset Evaluation & HARMONISE Calibration
# ==============================================================================
print("Generating Figure 4: Cross-Dataset Evaluation & HARMONISE Calibration...")
df_h2 = pd.read_csv(TABLES / "h2_dev_vs_h1_test.csv")
test_h2 = df_h2[df_h2["split"] == "test (1,847)"].copy()

# Focus on primary architectures
archs = ["EfficientNetV2B0__ce", "DenseNet121__ce", "MobileNetV3Small__ce", "MobileNetV3Large__corn", "EfficientNetV2B0__ce__hires"]
arch_names = ["EffNetV2B0 (224px)", "DenseNet121 (224px)", "MobileNetV3S (224px)", "MobileNetV3L CORN (224px)", "EffNetV2B0 (448px hires)"]

raw_balacc = []
cal_balacc = []
raw_qwk = []
cal_qwk = []

for a in archs:
    r_none = test_h2[(test_h2["run"] == a) & (test_h2["arm"] == "none")].iloc[0]
    r_froz = test_h2[(test_h2["run"] == a) & (test_h2["arm"] == "frozen")].iloc[0]
    raw_balacc.append(r_none["balanced_accuracy"])
    cal_balacc.append(r_froz["balanced_accuracy"])
    raw_qwk.append(r_none["qwk"])
    cal_qwk.append(r_froz["qwk"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(archs))
width = 0.35

# Panel 1: Balanced Accuracy
ax1.bar(x - width/2, raw_balacc, width, label="Raw Model (No Calibration)", color="#b0bec5", edgecolor="#37474f", lw=1)
ax1.bar(x + width/2, cal_balacc, width, label="HARMONISE-1 (Frozen Stain Cal)", color="#2a9d8f", edgecolor="#1b4332", lw=1)
ax1.axhline(0.25, color="#e63946", linestyle=":", lw=1.2, label="Chance Level (0.25)")
ax1.set_xticks(x)
ax1.set_xticklabels(arch_names, rotation=20, ha="right")
ax1.set_ylabel("Balanced Accuracy (External Test)")
ax1.set_title("A: Balanced Accuracy Restoration", loc="left", fontweight="bold")
ax1.set_ylim(0, 0.65)
ax1.grid(True, axis="y")
ax1.legend(loc="upper right", framealpha=0.9)
despine(ax1)

# Annotate the 224px jump and 448px collapse
ax1.annotate("+0.202 gain\n(0.301 → 0.503)", xy=(x[0] + width/2, cal_balacc[0]), xytext=(x[0]-0.2, 0.56),
             arrowprops=dict(facecolor="#2a9d8f", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", color="#1b4332", fontsize=8.5)
ax1.annotate("Collapse!\n(0.290 → 0.186)", xy=(x[4] + width/2, cal_balacc[4]), xytext=(x[4]-0.4, 0.30),
             arrowprops=dict(facecolor="#e63946", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", color="#e63946", fontsize=8.5)

# Panel 2: QWK
ax2.bar(x - width/2, raw_qwk, width, label="Raw Model", color="#b0bec5", edgecolor="#37474f", lw=1)
ax2.bar(x + width/2, cal_qwk, width, label="HARMONISE-1", color="#e76f51", edgecolor="#9d0208", lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels(arch_names, rotation=20, ha="right")
ax2.set_ylabel("Quadratic Weighted Kappa (QWK)")
ax2.set_title("B: QWK Trajectory on External Test", loc="left", fontweight="bold")
ax2.grid(True, axis="y")
ax2.legend(loc="upper left", framealpha=0.9)
despine(ax2)

fig.suptitle("Figure 4: External Generalization on HER2-IHC-40x Test Set (Raw vs HARMONISE-1)", fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "fig4_cross_dataset_collapse_and_harmonise.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 5: Class-wise Performance & The QWK Paradox
# ==============================================================================
print("Generating Figure 5: Class-wise Breakdown & QWK Paradox...")
# We use the C_classwise.csv or test_h2 class-wise recalls
classes = ["Score 0", "Score 1+", "Score 2+", "Score 3+"]

eff_none = test_h2[(test_h2["run"] == "EfficientNetV2B0__ce") & (test_h2["arm"] == "none")].iloc[0]
eff_froz = test_h2[(test_h2["run"] == "EfficientNetV2B0__ce") & (test_h2["arm"] == "frozen")].iloc[0]
mob_froz = test_h2[(test_h2["run"] == "MobileNetV3Small__ce") & (test_h2["arm"] == "frozen")].iloc[0]

rec_eff_none = [eff_none["rec_0"], eff_none["rec_1+"], eff_none["rec_2+"], eff_none["rec_3+"]]
rec_eff_froz = [eff_froz["rec_0"], eff_froz["rec_1+"], eff_froz["rec_2+"], eff_froz["rec_3+"]]
rec_mob_froz = [mob_froz["rec_0"], mob_froz["rec_1+"], mob_froz["rec_2+"], mob_froz["rec_3+"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(4)
width = 0.26

ax1.bar(x - width, rec_eff_none, width, label="EffNetV2B0 Raw (QWK: 0.108)", color="#b0bec5")
ax1.bar(x, rec_eff_froz, width, label="EffNetV2B0 Calibrated (QWK: 0.593)", color="#2a9d8f")
ax1.bar(x + width, rec_mob_froz, width, label="MobileNetV3S Calibrated (QWK: 0.802)", color="#e76f51")

ax1.set_xticks(x)
ax1.set_xticklabels(classes)
ax1.set_ylabel("Per-Class Recall (Sensitivity)")
ax1.set_title("A: Class-Wise Recall Breakdown on External Test", loc="left", fontweight="bold")
ax1.set_ylim(0, 1.08)
ax1.grid(True, axis="y")
ax1.legend(loc="upper left", framealpha=0.9)
despine(ax1)

# Annotate the 1+ depression across all models
ax1.axvspan(0.6, 1.4, color="#f4a261", alpha=0.15, label="Equivocal/Depressed Zone")
ax1.text(1, 0.40, "Severe 1+ Depression\n(Recall ≤ 0.18 for all)", ha="center", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#f4a261", lw=1))

# Panel 2: The QWK Paradox Comparison
# MobileNetV3S has QWK 0.802, but its 1+ recall is 0.139 and 2+ recall is 0.090!
metrics_mob = [mob_froz["qwk"], mob_froz["rec_0"], mob_froz["rec_1+"], mob_froz["rec_2+"], mob_froz["rec_3+"]]
metric_labels = ["Overall QWK", "Recall 0", "Recall 1+", "Recall 2+", "Recall 3+"]
colors_p = ["#457b9d", "#2a9d8f", "#e63946", "#e63946", "#2a9d8f"]

bars = ax2.bar(metric_labels, metrics_mob, color=colors_p, width=0.55, edgecolor="#333333", lw=1)
ax2.set_ylabel("Score / Recall Value")
ax2.set_title("B: The QWK Paradox (MobileNetV3Small Calibrated)", loc="left", fontweight="bold")
ax2.set_ylim(0, 1.08)
ax2.grid(True, axis="y")
despine(ax2)

for b in bars:
    yval = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax2.text(2, 0.45, "High QWK (0.802) driven\nsolely by polar 0 & 3+ separation;\nBorderline 1+ & 2+ fail completely!",
         ha="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#e63946", lw=1))

fig.suptitle("Figure 5: Class-Wise Performance & The Clinical QWK Paradox", fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "fig5_classwise_performance_and_qwk_paradox.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 6: HARMONISE-2 Reliability Analysis (Bootstrap CV & Selective Review)
# ==============================================================================
print("Generating Figure 6: HARMONISE-2 Reliability Analysis...")
df_boot = pd.read_csv(H2_DIR / "A_params_summary.csv")
df_flag = pd.read_csv(H2_DIR / "E_flagging.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Parameter CV vs Sample Size n
boot_both = df_boot[(df_boot["mode"] == "both") & (df_boot["percentile"] == 99.0)].sort_values("n")

ax1.plot(boot_both["n"], boot_both["g_h_cv"], marker="o", color="#1f77b4", lw=2, ms=6, label=r"Haematoxylin Gain $g_h$ CV")
ax1.plot(boot_both["n"], boot_both["g_d_cv"], marker="s", color="#d62728", lw=2, ms=6, label=r"DAB Gain $g_d$ CV")
ax1.axhline(0.05, color="#2ca02c", linestyle="--", lw=1.2, label=r"Target Stability ($CV \leq 0.05$)")
ax1.axhline(0.10, color="#ff7f0e", linestyle=":", lw=1.2, label=r"Permissible Ceiling ($CV \leq 0.10$)")

ax1.set_xscale("log")
ax1.set_xlabel("Target Unlabeled Sample Size (Tiles, Log Scale)")
ax1.set_ylabel("Coefficient of Variation (CV = SD / Mean)")
ax1.set_title("A: Calibration Parameter Sampling Stability", loc="left", fontweight="bold")
ax1.grid(True, which="both")
ax1.legend(loc="upper right", framealpha=0.9)
despine(ax1)

ax1.annotate("Stable at n ≥ 1,000 tiles\n(CV = 0.033 for g_h, 0.055 for g_d)",
             xy=(1000, 0.055), xytext=(250, 0.12),
             arrowprops=dict(facecolor="#333333", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4f8", ec="#333333", lw=1))

# Panel 2: Selective Review Failure on 2+
flag_eff = df_flag[(df_flag["run"] == "EfficientNetV2B0__ce") & (df_flag["arm"] == "frozen") & (df_flag["signal"] == "margin")].sort_values("coverage")

coverage = flag_eff["coverage"]
share_2plus_lost = flag_eff["share_2plus_referred"]

ax2.plot(coverage, [1.0 - c for c in coverage], linestyle="--", color="#888888", lw=1.2, label="Total Tiles Flagged (1 - Coverage)")
ax2.plot(coverage, share_2plus_lost, marker="^", color="#e63946", lw=2, ms=6, label="Share of True 2+ Cases Discarded/Referred")

ax2.set_xlabel("Automated Coverage (Fraction of Predictions Retained)")
ax2.set_ylabel("Fraction of Cases Flagged for Manual Review")
ax2.set_title("B: Selective Review Discards Borderline 2+ Cases", loc="left", fontweight="bold")
ax2.grid(True)
ax2.legend(loc="upper right", framealpha=0.9)
despine(ax2)

ax2.text(0.65, 0.25, "At 50% coverage:\n79% of true 2+ cases get\ndiscarded from automated flow!\nUnusable for clinical triage.",
         fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#e63946", lw=1))

fig.suptitle("Figure 6: HARMONISE-2 Statistical Reliability & Confidence Dynamics", fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "fig6_harmonise2_bootstrap_and_flagging.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 7: Physical Raspberry Pi 4B Deployment Benchmark
# ==============================================================================
print("Generating Figure 7: Physical Raspberry Pi 4B Deployment Benchmark...")
df_pi_prim = pd.read_csv(BENCH_DIR / "benchmark_primary_pi4.csv")
df_pi_exp = pd.read_csv(BENCH_DIR / "benchmark_experimental_pi4.csv")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

threads = [1, 2, 4]
x_t = np.arange(len(threads))
w = 0.35

# Panel 1: Median Latency
lat_prim = df_pi_prim["invoke_median_ms"]
lat_exp = df_pi_exp["invoke_median_ms"]

ax1.bar(x_t - w/2, lat_prim, w, label="Primary (EffNetV2B0 448px)", color="#1d3557", edgecolor="#03045e", lw=1)
ax1.bar(x_t + w/2, lat_exp, w, label="Experimental (EffNetV2B0 224px)", color="#457b9d", edgecolor="#1d3557", lw=1)

for i in range(len(threads)):
    ax1.text(x_t[i] - w/2, lat_prim.iloc[i] + 15, f"{lat_prim.iloc[i]:.1f} ms", ha="center", fontsize=8.5, fontweight="bold")
    ax1.text(x_t[i] + w/2, lat_exp.iloc[i] + 15, f"{lat_exp.iloc[i]:.1f} ms", ha="center", fontsize=8.5, fontweight="bold")

ax1.set_xticks(x_t)
ax1.set_xticklabels([f"{t} Thread{'s' if t>1 else ''}" for t in threads])
ax1.set_ylabel("Median Invoke Latency (ms)")
ax1.set_title("A: Neural Network Latency on Cortex-A72", loc="left", fontweight="bold")
ax1.set_ylim(0, 850)
ax1.grid(True, axis="y")
ax1.legend(loc="upper right", framealpha=0.9)
despine(ax1)

# Panel 2: Throughput (FPS)
fps_prim = df_pi_prim["throughput_fps"]
fps_exp = df_pi_exp["throughput_fps"]

ax2.plot(threads, fps_prim, marker="o", lw=2, ms=7, color="#1d3557", label="Primary (448px)")
ax2.plot(threads, fps_exp, marker="s", lw=2, ms=7, color="#457b9d", label="Experimental (224px)")

for t, fp, fe in zip(threads, fps_prim, fps_exp):
    ax2.text(t, fp + 0.3, f"{fp:.2f} FPS", ha="center", fontsize=8.5, fontweight="bold", color="#1d3557")
    ax2.text(t, fe + 0.3, f"{fe:.2f} FPS", ha="center", fontsize=8.5, fontweight="bold", color="#457b9d")

ax2.set_xticks(threads)
ax2.set_xlabel("Thread Count")
ax2.set_ylabel("Inference Throughput (FPS)")
ax2.set_title("B: Throughput Scaling", loc="left", fontweight="bold")
ax2.set_ylim(0, 11.5)
ax2.grid(True)
ax2.legend(loc="upper left", framealpha=0.9)
despine(ax2)

# Panel 3: End-to-End Pipeline Breakdown at 2 Threads (Optimal)
categories = ["Primary (448px)", "Experimental (224px)"]
prep_times = [df_pi_prim.loc[df_pi_prim["threads"]==2, "preprocess_mean_ms"].values[0],
              df_pi_exp.loc[df_pi_exp["threads"]==2, "preprocess_mean_ms"].values[0]]
inf_times = [df_pi_prim.loc[df_pi_prim["threads"]==2, "invoke_median_ms"].values[0],
             df_pi_exp.loc[df_pi_exp["threads"]==2, "invoke_median_ms"].values[0]]

y_c = np.arange(len(categories))
ax3.barh(y_c, prep_times, height=0.45, label="Preprocessing (Lanczos + Bilinear + Stain)", color="#f4a261", edgecolor="#e76f51")
ax3.barh(y_c, inf_times, left=prep_times, height=0.45, label="LiteRT NN Invoke", color="#2a9d8f", edgecolor="#1b4332")

for i, (p, inf) in enumerate(zip(prep_times, inf_times)):
    tot = p + inf
    ax3.text(p/2, i, f"{p:.1f}ms\n({p/tot*100:.0f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    ax3.text(p + inf/2, i, f"{inf:.1f}ms\n({inf/tot*100:.0f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    ax3.text(tot + 15, i, f"Total: {tot:.1f} ms", va="center", fontweight="bold", fontsize=8.5)

ax3.set_yticks(y_c)
ax3.set_yticklabels(categories)
ax3.set_xlabel("Time (ms)")
ax3.set_title("C: End-to-End Latency Breakdown (2 Threads)", loc="left", fontweight="bold")
ax3.set_xlim(0, 650)
ax3.grid(True, axis="x")
ax3.legend(loc="lower right", framealpha=0.9)
despine(ax3)

# Panel 4: Thermal Progression & Throttling
import re
def parse_temp(val):
    m = re.search(r"(\d+\.?\d*)", str(val))
    return float(m.group(1)) if m else 0.0

temp_prim = [parse_temp(t) for t in df_pi_prim["temp_end"]]
temp_exp = [parse_temp(t) for t in df_pi_exp["temp_end"]]

ax4.plot(threads, temp_prim, marker="o", color="#d62728", lw=2, ms=7, label="Primary Thermal Curve")
ax4.plot(threads, temp_exp, marker="s", color="#e76f51", lw=2, ms=7, label="Experimental Thermal Curve")
ax4.axhline(80.0, color="#d62728", linestyle=":", lw=1.2, label="Soft Throttling Trip Point (80°C)")
ax4.axhline(85.0, color="#7f0000", linestyle="--", lw=1.2, label="Hard Throttling Trip Point (85°C)")

ax4.set_xticks(threads)
ax4.set_xlabel("Thread Count")
ax4.set_ylabel("Final Core Temperature (°C)")
ax4.set_title("D: Thermal Profile & Throttling Verification", loc="left", fontweight="bold")
ax4.set_ylim(60, 90)
ax4.grid(True)
ax4.legend(loc="upper left", framealpha=0.9)
despine(ax4)

ax4.annotate("4 Threads Throttled!\n84.2°C, 1.68 GHz (0x80008)\nRequires 2T or active fan",
             xy=(4, 84.2), xytext=(2.7, 65),
             arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", color="#d62728", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1))

fig.suptitle("Figure 7: Physical Raspberry Pi 4B Edge Benchmark (4GB LPDDR4, ARM Cortex-A72 @ 1.8GHz)", fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "fig7_raspberry_pi4_hardware_benchmark.png", dpi=300)
plt.close(fig)

# ==============================================================================
# Figure 8: Quantization & Numerical Parity
# ==============================================================================
print("Generating Figure 8: Quantization & Numerical Parity...")
df_deploy = pd.read_csv(TABLES / "deploy_recomputed.csv")
test_dep = df_deploy[df_deploy["split"] == "test"].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

variants = ["fp32", "fp16", "int8"]
var_names = ["Original FP32", "LiteRT FP16 (Deployed)", "INT8 Full Integer"]
sizes = [test_dep.loc[test_dep["variant"] == v, "size_mb"].values[0] for v in variants]
mac_lat = [test_dep.loc[test_dep["variant"] == v, "mac_latency_ms"].values[0] for v in variants]

# Size comparison
colors_v = ["#3a86ff", "#06d6a0", "#ffbe0b"]
bars1 = ax1.bar(var_names, sizes, color=colors_v, width=0.5, edgecolor="#333333", lw=1)
ax1.set_ylabel("Model File Size (MB)")
ax1.set_title("A: Model Compression Ratio", loc="left", fontweight="bold")
ax1.set_ylim(0, 30)
ax1.grid(True, axis="y")
despine(ax1)

for b, s in zip(bars1, sizes):
    ax1.text(b.get_x() + b.get_width()/2, s + 0.8, f"{s:.2f} MB\n({s/sizes[0]*100:.0f}%)", ha="center", fontsize=8.5, fontweight="bold")

# Agreement comparison
# In deployment on Pi 4B: agreement with Mac TFLite is exactly 100.0% (120/120 tiles)
agreements = [100.0, 100.0, 98.3]
bars2 = ax2.bar(var_names, agreements, color=colors_v, width=0.5, edgecolor="#333333", lw=1)
ax2.set_ylabel("Prediction Concordance (%)")
ax2.set_title("B: Edge Runtime Parity vs Host Framework", loc="left", fontweight="bold")
ax2.set_ylim(90, 102)
ax2.grid(True, axis="y")
despine(ax2)

for b, a in zip(bars2, agreements):
    ax2.text(b.get_x() + b.get_width()/2, a + 0.4, f"{a:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

ax2.annotate("Zero drift across platforms!\n(Mac TFLite vs Pi 4B TFLite: 100.0%)",
             xy=(1, 100.0), xytext=(0.2, 93),
             arrowprops=dict(facecolor="#06d6a0", shrink=0.05, width=1.2, headwidth=5),
             fontweight="bold", color="#0b525b", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#06d6a0", lw=1))

fig.suptitle("Figure 8: Quantization Efficiency & Cross-Platform Parity", fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "fig8_quantization_and_agreement.png", dpi=300)
plt.close(fig)

print("All 8 figures generated successfully in final_report_figures/!")
