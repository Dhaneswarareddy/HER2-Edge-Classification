"""Edge figures and tables from the 10 Nov 2026 measurements only (no hard-coded values).

Inputs (all written by the scripts in this folder):
  outputs/android_samsung_m35/benchmark.csv, outputs/android_realme_6/benchmark.csv   (android_benchmark.py)
  outputs/pi4/results_pi.csv                                                          (pi_bundle/pi_precision_benchmark.py)
  outputs/ondevice_<device>/summary.json                                              (ondevice_validation.py)
  outputs/precision_accuracy.csv                                                      (mac_prepare.py)
Outputs:
  outputs/edge_latency_table.csv
  outputs/fig_edge_benchmark.png  (also copied to final_report_figures/fig7_smartphone_edge_benchmark.png)
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ROOT = HERE.parents[1]
DEV = {"samsung_m35": "Samsung Galaxy M35 5G", "realme_6": "Realme 6"}
MODEL_NAME = {"primary_448_fp32": "448 px FP32", "primary_448_fp16": "448 px FP16 (deployed)",
              "primary_448_int8_REJECTED": "448 px INT8 (rejected)", "experimental_224_fp16": "224 px FP16"}


def load():
    rows = []
    for d, name in DEV.items():
        f = OUT / f"android_{d}" / "benchmark.csv"
        if not f.exists():
            continue
        b = pd.read_csv(f)
        for _, r in b.iterrows():
            t0, t1 = json.loads(r.temp_start), json.loads(r.temp_end)
            rows.append(dict(device=name, model=MODEL_NAME[r.model], threads=int(r.threads), median_ms=r.median_ms,
                             p95_ms=r.p95_ms, mean_ms=r.mean_ms, n=int(r.n), peak_mem_mb=r.mem_overall_mb,
                             file_mib=r.file_mb, temp_note=f"battery {t0['battery_temp_c']}->{t1['battery_temp_c']} C",
                             throttle="not observable on Android"))
    f = OUT / "pi4" / "results_pi.csv"
    if f.exists():
        p = pd.read_csv(f)
        for _, r in p.iterrows():
            rows.append(dict(device="Raspberry Pi 4B", model=MODEL_NAME[{"fp32": "primary_448_fp32", "fp16": "primary_448_fp16",
                                                                     "int8": "primary_448_int8_REJECTED"}[r.precision]],
                             threads=int(r.threads), median_ms=r.invoke_median_ms, p95_ms=r.invoke_p95_ms,
                             mean_ms=r.invoke_mean_ms, n=int(r.invoke_n), peak_mem_mb=r.peak_rss_mb, file_mib=r.model_mb,
                             temp_note=f"SoC {r.temp_start} -> {r.temp_end}".replace("temp=", ""),
                             throttle=f"{r.throttled_start} -> {r.throttled_end}".replace("throttled=", "")))
    return pd.DataFrame(rows)


def main():
    t = load()
    t.to_csv(OUT / "edge_latency_table.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    col = {"Samsung Galaxy M35 5G": "#1d3557", "Realme 6": "#e76f51", "Raspberry Pi 4B": "#2a9d8f"}
    # A: deployed model latency vs threads on every device
    ax = axes[0]
    for dev, c in col.items():
        d = t[(t.device == dev) & (t.model == "448 px FP16 (deployed)")].sort_values("threads")
        if len(d):
            ax.errorbar(d.threads, d.median_ms, yerr=[np.zeros(len(d)), d.p95_ms - d.median_ms], fmt="o-", c=c, capsize=3, label=dev)
            for _, r in d.iterrows():
                ax.annotate(f"{r.median_ms:.0f}", (r.threads, r.median_ms), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=c)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("CPU threads (no core pinning)"); ax.set_ylabel("Network latency per 448 px tile (ms; median, bar to p95)")
    ax.set_title("A: Deployed model (448 px FP16)", loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # B: best configuration per device and model
    ax = axes[1]
    models = ["448 px FP32", "448 px FP16 (deployed)", "448 px INT8 (rejected)", "224 px FP16"]
    devs = [d for d in col if d in set(t.device)]
    x = np.arange(len(models)); w = 0.8 / max(len(devs), 1)
    for k, dev in enumerate(devs):
        vals = [t[(t.device == dev) & (t.model == m)].median_ms.min() if len(t[(t.device == dev) & (t.model == m)]) else np.nan for m in models]
        bars = ax.bar(x + (k - (len(devs) - 1) / 2) * w, vals, w, color=col[dev], label=dev)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x, [m.replace(" (", "\n(") for m in models], fontsize=8)
    ax.set_ylabel("Fastest median latency over thread counts (ms, log)")
    ax.set_title("B: Fastest configuration per model (INT8: speed only)", loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    # C: accuracy that the app actually delivers on the phone
    ax = axes[2]
    prec = pd.read_csv(OUT / "precision_accuracy.csv").set_index("precision")
    labels, acc, agr = ["Mac TFLite FP16\n(training path)"], [prec.loc["fp16", "test_accuracy"]], [prec.loc["fp16", "test_agreement_with_keras"]]
    for d, name in DEV.items():
        f = OUT / f"ondevice_{d}" / "summary.json"
        if f.exists():
            s = json.load(open(f))["results"]["test"]
            labels.append(f"{name}\n(app v1.1, on device)"); acc.append(s["accuracy"]); agr.append(s["agreement_with_keras"])
    xx = np.arange(len(labels))
    b1 = ax.bar(xx - 0.18, acc, 0.36, color="#457b9d", label="Accuracy vs labels")
    b2 = ax.bar(xx + 0.18, agr, 0.36, color="#06d6a0", label="Agreement with Keras")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_ylim(0.9, 1.01); ax.set_xticks(xx, labels, fontsize=8)
    ax.set_title("C: BCI test split (758 tiles), 448 px FP16", loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Edge deployment, measured 10 Nov 2026 (same protocol on every device; no power measured; "
                 "phone temperatures are battery temperatures)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_edge_benchmark.png", dpi=300)
    shutil.copyfile(OUT / "fig_edge_benchmark.png", ROOT / "final_report_figures" / "fig7_smartphone_edge_benchmark.png")
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
