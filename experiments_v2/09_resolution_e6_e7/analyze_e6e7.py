"""E6 / E7 paired analysis (BCI grouped split), same method as 07_ablation/analyze_ablation.py.

E6  DenseNet121: 448 px batch 4 (hires) vs 224 px batch 4 (lowbatch, the batch-matched control) -> DECISION pair.
    Also vs the original 224 px batch 16 run (results/runs/DenseNet121__ce) for context.
E7  MobileNetV3Small: 896 px input from native 1024 px, batch 2 (hires1024) vs 448 px batch 8 (hires, E2) -> DECISION pair.
    Also vs 224 px batch 8 (lowbatch) and 224 px batch 32 (original) for context.
Pre-registered rule (NEXT_EXPERIMENTS.md): an effect counts only if the paired 95% fragment-cluster bootstrap CI of
delta VALIDATION QWK excludes 0. Test (758 tiles) is report-only. One seed per arm.
  python analyze_e6e7.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import v2lib as L  # noqa: E402

OUT = L.out_dir("09_resolution_e6_e7")
NEW = HERE / "runs"
ARMS = {
    "DN121 hires 448 px, batch 4": NEW / "DenseNet121__ce__hires",
    "DN121 lowbatch 224 px, batch 4": NEW / "DenseNet121__ce__lowbatch",
    "DN121 original 224 px, batch 16": L.RUNS / "DenseNet121__ce",
    "MNV3S hires1024 896 px, batch 2": NEW / "MobileNetV3Small__ce__hires1024",
    "MNV3S hires 448 px, batch 8": L.RUNS / "MobileNetV3Small__ce__hires",
    "MNV3S lowbatch 224 px, batch 8": L.RUNS / "MobileNetV3Small__ce__lowbatch",
    "MNV3S original 224 px, batch 32": L.RUNS / "MobileNetV3Small__ce",
}
PAIRS = [  # (a, b, experiment, role of the val comparison)
    ("DN121 hires 448 px, batch 4", "DN121 lowbatch 224 px, batch 4", "E6", "pre-registered decision"),
    ("DN121 hires 448 px, batch 4", "DN121 original 224 px, batch 16", "E6", "context"),
    ("DN121 lowbatch 224 px, batch 4", "DN121 original 224 px, batch 16", "E6", "context (batch effect)"),
    ("MNV3S hires1024 896 px, batch 2", "MNV3S hires 448 px, batch 8", "E7", "pre-registered decision"),
    ("MNV3S hires1024 896 px, batch 2", "MNV3S lowbatch 224 px, batch 8", "E7", "context"),
    ("MNV3S hires1024 896 px, batch 2", "MNV3S original 224 px, batch 32", "E7", "context"),
]
DECISION = {(a, b) for a, b, _, r in PAIRS if "decision" in r}
FUN = {"accuracy": lambda y, p: (y == p).mean(),
       "balanced_accuracy": lambda y, p: recall_score(y, p, average="macro", labels=range(4), zero_division=0),
       "macro_f1": lambda y, p: f1_score(y, p, average="macro", labels=range(4), zero_division=0),
       "qwk": lambda y, p: cohen_kappa_score(y, p, weights="quadratic", labels=range(4))}


def main():
    rows, preds = [], {}
    for arm, d in ARMS.items():
        z = np.load(d / "predictions.npz")
        s = json.load(open(d / "summary.json"))
        h = pd.read_csv(d / "history.csv")
        preds[arm] = {sp: (z[f"y_{sp}"].astype(int), z[sp]) for sp in ["val", "test"]}
        r = dict(arm=arm, run_dir=str(d.relative_to(L.ROOT)), image_size=s.get("image_size") or 224,
                 batch=s.get("batch"), params=s.get("params"), best_epoch=s.get("best_epoch"),
                 epochs_run=s.get("epochs_run"), ended_at_epoch_cap=bool(len(h) >= 45),
                 train_min=round(s.get("train_time_s", np.nan) / 60, 1))
        for sp in ["val", "test"]:
            m = L.class_metrics(*preds[arm][sp])
            r.update({f"{sp}_{k}": m[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "qwk"] +
                      [f"recall_{c}" for c in L.LEVELS]})
        rows.append(r)
    tab = pd.DataFrame(rows)
    L.write_csv(OUT / "e6e7_table.csv", tab)

    pairs = []
    for a, b, exp, role in PAIRS:
        for sp in ["val", "test"]:
            y, pa = preds[a][sp]
            y2, pb = preds[b][sp]
            assert (y == y2).all(), "tile order differs between runs"
            pa, pb = pa.argmax(1), pb.argmax(1)
            draws = list(L.cluster_draws(L.fragments(sp), 2000, L.SEED))
            r = dict(experiment=exp, a=a, b=b, split=sp, role=role if sp == "val" else "report-only")
            for k, f in FUN.items():
                r[f"delta_{k}"] = f(y, pa) - f(y, pb)
                r[f"ci95_{k}"] = L.ci95([f(y[i], pa[i]) - f(y[i], pb[i]) for i in draws])
            if sp == "val" and role == "pre-registered decision":
                lo, hi = r["ci95_qwk"]
                r["rule_met"] = bool(lo > 0 or hi < 0)
            pairs.append(r)
    pairs = pd.DataFrame(pairs)
    L.write_csv(OUT / "e6e7_paired.csv", pairs)
    L.write_json(OUT / "e6e7_decision.json", {
        f"{p.experiment}: {p.a} vs {p.b}": dict(delta_val_qwk=round(p.delta_qwk, 4),
                                               ci95=[round(v, 4) for v in p.ci95_qwk], rule_met=p.rule_met)
        for p in pairs[pairs.role == "pre-registered decision"].itertuples()})
    pd.set_option("display.width", 250)
    print(tab[["arm", "image_size", "batch", "best_epoch", "epochs_run", "ended_at_epoch_cap", "train_min",
               "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_qwk",
               "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_qwk"]].round(4).to_string())
    print(tab[["arm"] + [f"{sp}_recall_{c}" for sp in ["val", "test"] for c in L.LEVELS]].round(3).to_string())
    show = pairs.copy()
    for k in FUN:
        show[f"ci95_{k}"] = show[f"ci95_{k}"].map(lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]")
    print(show[["experiment", "a", "b", "split", "role", "delta_accuracy", "ci95_accuracy", "delta_balanced_accuracy",
                "ci95_balanced_accuracy", "delta_qwk", "ci95_qwk"]].round(4).to_string())
    figure(pairs)


def figure(pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    lab = [f"{p.experiment}: {p.a.split(',')[0]}\n vs {p.b.split(',')[0]}" for p in pairs[pairs.split == "val"].itertuples()]
    for ax, sp in zip(axes, ["val", "test"]):
        d = pairs[pairs.split == sp].reset_index(drop=True)
        yy = np.arange(len(d))[::-1]
        for i, p in d.iterrows():
            lo, hi = p.ci95_qwk
            dec = (p.a, p.b) in DECISION
            col = "#1f5fa8" if p.experiment == "E6" else "#d9822b"
            ax.plot([lo, hi], [yy[i]] * 2, color=col, lw=2.5 if dec else 1.2)
            ax.plot(p.delta_qwk, yy[i], "o" if dec else "s", color=col, ms=8 if dec else 5)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(yy, lab, fontsize=7)
        ax.set_xlabel("Δ QWK (a − b), 95% fragment-cluster bootstrap CI")
        ax.set_title(f"BCI {sp} ({'decision set' if sp == 'val' else 'report only'}); thick = pre-registered pair",
                     fontsize=9)
    fig.suptitle("E6 (DenseNet121) and E7 (MobileNetV3Small): resolution effects, one seed per arm", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.guard(OUT / "fig_e6e7_paired.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
