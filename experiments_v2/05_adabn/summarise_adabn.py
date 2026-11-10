"""Cross-network summary for experiment 5: metric tables (development R_eval and external test), paired test
differences, and the clinical error taxonomy / workflow cost of experiment 1 for every arm (same definitions)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "01_clinical_risk"))
import v2lib as L  # noqa: E402
import run_clinical_risk as R  # noqa: E402

OUT = L.out_dir("05_adabn")
RUNS = ["EfficientNetV2B0__ce", "MobileNetV3Small__ce", "MobileNetV3Large__corn", "DenseNet121__ce", "EfficientNetV2B0__ce__hires"]
ARMS = ["raw", "stain_p99", "adabn", "adabn_p99"]
MET = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "recall_0", "recall_1+", "recall_2+", "recall_3+", "precision_2+", "off_by_2plus"]


def main():
    test, dev, pairs, checks = [], [], [], []
    for run in RUNS:
        f = OUT / f"adabn_{run}.json"
        if not f.exists():
            print("missing", run); continue
        a = json.load(open(f))
        z = np.load(OUT / f"preds_{run}.npz", allow_pickle=True)
        y = z["y"].astype(int)
        for arm in ARMS:
            m = L.class_metrics(y, z[f"test_{arm}"])
            test.append(dict(run=run, arm=arm, **{k: m[k] for k in MET}, **R.taxonomy(m["confusion"]),
                             **R.costs(m["confusion"])))
        dev += [dict(run=run, arm=t["arm"], subset=t["subset"], **{k: t[k] for k in MET}) for t in a["dev"]]
        pairs += a["paired_test"]
        checks.append(dict(run=run, input_size=a["input_size"], fresh_vs_stored_raw=a["check_raw_vs_stored"],
                           fresh_vs_stored_p99=a["check_p99_vs_stored"], bn_layers=a["adabn_bn_layers"],
                           dropouts_off=a["adabn_dropouts_disabled"],
                           mean_abs_change_moving_mean=a["adabn_mean_abs_change_moving_mean"],
                           mean_abs_change_moving_mean_p99=a["adabn_p99_mean_abs_change_moving_mean"]))
    test, dev, pairs, checks = map(pd.DataFrame, (test, dev, pairs, checks))
    for name, df in [("summary_test.csv", test), ("summary_dev.csv", dev), ("summary_paired_test.csv", pairs), ("checks.csv", checks)]:
        L.write_csv(OUT / name, df)
    pd.set_option("display.width", 250)
    print(checks.to_string())
    piv = lambda df, m: df.pivot(index="run", columns="arm", values=m)[ARMS].round(3)
    print("TEST balanced accuracy\n", piv(test, "balanced_accuracy"))
    print("DEV R_eval balanced accuracy\n", piv(dev[dev.subset == "R_eval"], "balanced_accuracy"))
    print("TEST QWK\n", piv(test, "qwk"))
    print("TEST S2 workflow cost\n", piv(test, "cost_S2_workflow") if "cost_S2_workflow" in test else test.filter(like="cost").head())
    cols = [c for c in test.columns if c.startswith("n_")]
    print(test[["run", "arm"] + cols].to_string())
    print(pairs.round(3).to_string())
    figure(test, dev)


def figure(test, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"raw": "#7f7f7f", "stain_p99": "#b03a2e", "adabn": "#7d3c98", "adabn_p99": "#16a085"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    panels = [(dev[dev.subset == "R_eval"], "balanced_accuracy", "Development R_eval (2,402 tiles): balanced accuracy"),
              (test, "balanced_accuracy", "External test (exploratory): balanced accuracy"),
              (test, "macro_f1", "External test (exploratory): macro-F1")]
    runs = [r for r in RUNS if r in set(test.run)]
    x = np.arange(len(runs))
    for ax, (df, m, title) in zip(axes, panels):
        for k, arm in enumerate(ARMS):
            v = [df[(df.run == r) & (df.arm == arm)][m].values[0] for r in runs]
            ax.bar(x + (k - 1.5) * 0.2, v, 0.2, color=col[arm], label=arm)
        ax.set_xticks(x, [r.replace("__", "\n") for r in runs], fontsize=7)
        ax.set_title(title, fontsize=9); ax.axhline(0.25, ls=":", c="k", lw=0.8)
    axes[0].legend(fontsize=8); axes[0].set_ylabel("score (0.25 = chance balanced accuracy)")
    fig.tight_layout()
    fig.savefig(L.guard(OUT / "fig_adabn.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
