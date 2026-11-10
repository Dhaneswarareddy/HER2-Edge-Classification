"""Experiment 7 summary: backbone-freezing / head ablation and the existing resolution / batch arms, BCI grouped split.

Arms (all EfficientNetV2B0 CE, seed 42, one run each):
  full_ft (baseline)   results/runs/EfficientNetV2B0__ce                   224 px, batch 32, all layers fine-tuned
  frozen_linear        experiments_v2/07_ablation/runs/frozen_linear       backbone frozen, linear head
  frozen_mlp           experiments_v2/07_ablation/runs/frozen_mlp          backbone frozen, Dense(256)-ReLU-Dropout head
  partial              experiments_v2/07_ablation/runs/partial             block6 + top unfrozen
  lowbatch (existing)  results/runs/EfficientNetV2B0__ce__lowbatch         224 px, batch 8   (E1 control)
  hires (existing)     results/runs/EfficientNetV2B0__ce__hires            448 px, batch 8   (E1)
Also MobileNetV3Small: baseline / lowbatch / hires (E2), to show whether the resolution effect generalises.
Decision set = validation (727 tiles); test (758) is report-only. Paired differences use the fragment-cluster
bootstrap (2,000 draws) on the same tiles. External-domain numbers are not part of this ablation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

OUT = L.out_dir("07_ablation")
ABL = L.V2 / "07_ablation" / "runs"
ARMS = {
    "full_ft (baseline)": L.RUNS / "EfficientNetV2B0__ce",
    "frozen_linear": ABL / "frozen_linear",
    "frozen_mlp": ABL / "frozen_mlp",
    "partial (block6+top)": ABL / "partial",
    "lowbatch 224 px, batch 8": L.RUNS / "EfficientNetV2B0__ce__lowbatch",
    "hires 448 px, batch 8": L.RUNS / "EfficientNetV2B0__ce__hires",
    "MNV3S full_ft (baseline)": L.RUNS / "MobileNetV3Small__ce",
    "MNV3S lowbatch 224 px, batch 8": L.RUNS / "MobileNetV3Small__ce__lowbatch",
    "MNV3S hires 448 px, batch 8": L.RUNS / "MobileNetV3Small__ce__hires",
}
PAIRS = [("frozen_linear", "full_ft (baseline)"), ("frozen_mlp", "full_ft (baseline)"),
         ("partial (block6+top)", "full_ft (baseline)"), ("frozen_mlp", "frozen_linear"),
         ("hires 448 px, batch 8", "lowbatch 224 px, batch 8"), ("hires 448 px, batch 8", "full_ft (baseline)"),
         ("MNV3S hires 448 px, batch 8", "MNV3S lowbatch 224 px, batch 8"),
         ("MNV3S hires 448 px, batch 8", "MNV3S full_ft (baseline)")]
FUN = {"accuracy": lambda y, p: (y == p).mean(),
       "balanced_accuracy": lambda y, p: recall_score(y, p, average="macro", labels=range(4), zero_division=0),
       "macro_f1": lambda y, p: f1_score(y, p, average="macro", labels=range(4), zero_division=0),
       "qwk": lambda y, p: cohen_kappa_score(y, p, weights="quadratic", labels=range(4))}


def load(d):
    z = np.load(d / "predictions.npz")
    s = json.load(open(d / "summary.json"))
    h = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
    return z, s, h


def main():
    rows, preds = [], {}
    for arm, d in ARMS.items():
        if not (d / "summary.json").exists():
            print("missing", arm); continue
        z, s, h = load(d)
        preds[arm] = {sp: (z[f"y_{sp}"].astype(int), z[sp]) for sp in ["val", "test"]}
        w = d / "model.weights.h5"
        r = dict(arm=arm, run_dir=str(d.relative_to(L.ROOT)), image_size=s.get("image_size", 224), batch=s.get("batch"),
                 params=s.get("params"), trainable_stage2=s.get("trainable_params_stage2"),
                 best_epoch=s.get("best_epoch"), epochs_run=s.get("epochs_run"),
                 ended_at_epoch_cap=bool(h is not None and len(h) >= 45),  # 5 head + 40 fine-tune (or 45 head)
                 train_min=round(s.get("train_time_s", np.nan) / 60, 1), weights_mib=round(w.stat().st_size / 2**20, 2))
        for sp in ["val", "test"]:
            m = L.class_metrics(*preds[arm][sp])
            r.update({f"{sp}_{k}": m[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "qwk"] +
                      [f"recall_{c}" for c in L.LEVELS]})
        rows.append(r)
    tab = pd.DataFrame(rows)
    L.write_csv(OUT / "ablation_table.csv", tab)
    pairs = []
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        for sp in ["val", "test"]:
            y, pa = preds[a][sp]; y2, pb = preds[b][sp]
            assert (y == y2).all()
            pa, pb = pa.argmax(1), pb.argmax(1)
            g = L.fragments(sp)
            draws = list(L.cluster_draws(g, 2000, L.SEED))
            r = dict(a=a, b=b, split=sp, role="decision" if sp == "val" else "report-only")
            for k, f in FUN.items():
                r[f"delta_{k}"] = f(y, pa) - f(y, pb)
                r[f"ci95_{k}"] = L.ci95([f(y[i], pa[i]) - f(y[i], pb[i]) for i in draws])
            pairs.append(r)
    pairs = pd.DataFrame(pairs)
    L.write_csv(OUT / "ablation_paired.csv", pairs)
    probe = json.load(open(L.ROOT / "results" / "analysis" / "frozen_probe.json"))["BCI"]
    L.write_json(OUT / "reference_frozen_probe.json", dict(
        note="existing src/analysis result: frozen ImageNet EfficientNetB0 (not V2) features + logistic regression, "
             "no augmentation; split as stored by that script", **{k: v for k, v in probe.items() if not isinstance(v, list)}))
    pd.set_option("display.width", 250)
    print(tab[["arm", "image_size", "batch", "trainable_stage2", "best_epoch", "epochs_run", "train_min",
               "val_accuracy", "val_macro_f1", "val_qwk", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_qwk"]].round(4).to_string())
    print(pairs[["a", "b", "split", "delta_accuracy", "ci95_accuracy", "delta_macro_f1", "ci95_macro_f1", "delta_qwk", "ci95_qwk"]].round(4).to_string())
    figure(tab)


def figure(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tab[~tab.arm.str.startswith("MNV3S")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, sp in zip(axes, ["val", "test"]):
        x = np.arange(len(t))
        for k, (m, c) in enumerate([("accuracy", "#1f5fa8"), ("macro_f1", "#d9822b"), ("qwk", "#2e8b57")]):
            ax.bar(x + (k - 1) * 0.27, t[f"{sp}_{m}"], 0.27, color=c, label=m)
        ax.set_xticks(x, [a.replace(" (", "\n(").replace(", batch", "\nbatch") for a in t.arm], fontsize=7)
        ax.set_ylim(0.5, 1.0); ax.set_title(f"EfficientNetV2B0 ablation, BCI {sp}{' (decision set)' if sp == 'val' else ' (report only)'}", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(L.guard(OUT / "fig_ablation.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
