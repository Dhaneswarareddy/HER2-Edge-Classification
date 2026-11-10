"""Summaries and data-efficiency curves for experiment 6, with zero-shot, HARMONISE (stain_p99) and AdaBN references
on the same sets (R_eval development tiles; external test split = exploratory second look)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

OUT = L.out_dir("06_fewshot")
MET = ["balanced_accuracy", "macro_f1", "qwk", "accuracy", "recall_0", "recall_1+", "recall_2+", "recall_3+", "precision_2+"]
RUN = "EfficientNetV2B0__ce"


def references():
    rows = []
    ri = pd.read_csv(L.H2OUT / "R_index.csv"); ev = np.where(ri.subset == "R_eval")[0]; y = ri.label.values.astype(int)
    for arm, nm in [("none", "zero-shot raw"), ("frozen", "HARMONISE stain_p99")]:
        p = np.load(L.H2OUT / "preds" / "C" / f"{RUN}__{arm}.npz")["prob"]
        rows.append(dict(method=nm, set="R_eval", **L.class_metrics(y[ev], p[ev])))
    names, yt, pt = L.ext_raw("BCI", RUN); rows.append(dict(method="zero-shot raw", set="ext_test", **L.class_metrics(yt, pt)))
    names, yt, pt = L.ext_h1(RUN); rows.append(dict(method="HARMONISE stain_p99", set="ext_test", **L.class_metrics(yt, pt)))
    f = L.V2 / "05_adabn" / "outputs" / f"adabn_{RUN}.json"
    if f.exists():
        a = json.load(open(f))
        for t in a["test"]:
            if t["arm"] in ("adabn", "adabn_p99"):
                rows.append(dict(method=f"AdaBN ({t['arm']})", set="ext_test", **t))
        for t in a["dev"]:
            if t["arm"] in ("adabn", "adabn_p99") and t["subset"] == "R_eval":
                rows.append(dict(method=f"AdaBN ({t['arm']})", set="R_eval", **t))
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(OUT / "final_results.csv")
    fs = d[d.regime != "zero_shot"]
    agg = fs.groupby(["set", "regime", "k"])[MET].agg(["mean", "std", "min", "max"]).round(4)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    L.write_csv(OUT / "final_summary_by_k.csv", agg)
    ref = references()
    L.write_csv(OUT / "reference_methods.csv", ref[["method", "set"] + MET])
    # paired, per seed: few-shot - zero-shot on the external test split (same tiles)
    z = d[(d.regime == "zero_shot") & (d.set == "ext_test")].iloc[0]
    fs_t = fs[fs.set == "ext_test"]
    gain = fs_t.assign(**{f"gain_{m}": fs_t[m] - z[m] for m in ["balanced_accuracy", "macro_f1", "qwk", "recall_2+"]})
    g = gain.groupby(["regime", "k"])[[c for c in gain.columns if c.startswith("gain_")]].agg(["mean", "min", "max"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    L.write_csv(OUT / "gain_over_zero_shot_test.csv", g.reset_index())
    print(L.md_table(agg[agg.set == "ext_test"][["regime", "k", "balanced_accuracy_mean", "balanced_accuracy_std", "macro_f1_mean",
                                                    "qwk_mean", "recall_1+_mean", "recall_2+_mean", "precision_2+_mean"]]))
    print(L.md_table(ref[["method", "set", "balanced_accuracy", "macro_f1", "qwk", "recall_1+", "recall_2+", "precision_2+"]]))
    figure(agg, ref)


def figure(agg, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mets = [("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("qwk", "QWK"), ("recall_2+", "2+ recall")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    colors = {"head": "#1f5fa8", "partial": "#d9822b", "full": "#2e8b57"}
    refc = {"zero-shot raw": "k", "HARMONISE stain_p99": "#b03a2e", "AdaBN (adabn)": "#7d3c98", "AdaBN (adabn_p99)": "#16a085"}
    for r, s in enumerate(["R_eval", "ext_test"]):
        for c, (m, lab) in enumerate(mets):
            ax = axes[r, c]
            for reg, col in colors.items():
                a = agg[(agg.set == s) & (agg.regime == reg)].sort_values("k")
                ax.errorbar(a.k, a[f"{m}_mean"], yerr=a[f"{m}_std"], fmt="o-", c=col, capsize=3, label=f"few-shot {reg} (5 seeds)")
            for _, rr in ref[ref.set == s].iterrows():
                ax.axhline(rr[m], ls="--", lw=1.2, c=refc.get(rr["method"], "grey"), label=rr["method"])
            ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20], ["1", "5", "10", "20"])
            ax.set_title(f"{lab}: {'development R_eval' if s == 'R_eval' else 'external test (exploratory)'}", fontsize=9)
            if r == 1:
                ax.set_xlabel("Labelled target tiles per class (k)")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Few-shot adaptation of EfficientNetV2B0 224 px to HER2-IHC-40x (mean ± SD over 5 support draws)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(L.guard(OUT / "fig_fewshot_data_efficiency.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
