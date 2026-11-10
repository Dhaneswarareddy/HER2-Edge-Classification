"""Experiment 1 (v2): clinical error taxonomy and cost-sensitivity analysis. No inference, no training.

Inputs (read-only): stored BCI predictions of all 20 grouped runs (val + test), stored external HER2-IHC-40x test
predictions (13 raw networks, results/external/her2_40x) and HARMONISE-1 calibrated predictions (5 networks,
next_experiment/outputs/test).

What is objective here: confusion matrices, per-class precision/recall/F1, QWK, and counts of each error TYPE, where
types follow the ASCO/CAP IHC categories (0 and 1+ negative, 2+ equivocal -> reflex ISH, 3+ positive).
What is NOT objective: the cost values. They are illustrative scenarios, chosen before looking at any result, and a
sensitivity analysis checks whether conclusions survive when every cost is perturbed. No scenario is a validated
clinical utility, and nothing here says what treatment a patient should receive.

Run: <project python> run_clinical_risk.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

OUT = L.out_dir("01_clinical_risk")
N_BOOT = 2000
N_SENS = 2000

# ---- error taxonomy (true -> predicted), fixed before any result was computed
TAXONOMY = {
    "positive_missed (3+ -> 0/1+)": [(3, 0), (3, 1)],
    "equivocal_missed (2+ -> 0/1+)": [(2, 0), (2, 1)],
    "ish_bypassed (2+ -> 3+)": [(2, 3)],
    "false_positive (0/1+ -> 3+)": [(0, 3), (1, 3)],
    "unnecessary_ish (0/1+ -> 2+)": [(0, 2), (1, 2)],
    "positive_undercalled (3+ -> 2+)": [(3, 2)],
    "zero_vs_low (0 <-> 1+)": [(0, 1), (1, 0)],
}
DENOM = {  # true classes whose tiles can make each error type (for rates)
    "positive_missed (3+ -> 0/1+)": [3], "equivocal_missed (2+ -> 0/1+)": [2], "ish_bypassed (2+ -> 3+)": [2],
    "false_positive (0/1+ -> 3+)": [0, 1], "unnecessary_ish (0/1+ -> 2+)": [0, 1],
    "positive_undercalled (3+ -> 2+)": [3], "zero_vs_low (0 <-> 1+)": [0, 1],
}

# ---- illustrative cost parameters (fixed a priori; see README for the reasoning)
NOMINAL = dict(c_low=1.0, c_extra_ish=2.0, c_under=2.0, c_bypass=4.0, c_miss_ish=6.0, c_fp=8.0, c_fn=10.0)


def workflow_matrix(c):
    return np.array([
        [0, c["c_low"], c["c_extra_ish"], c["c_fp"]],
        [c["c_low"], 0, c["c_extra_ish"], c["c_fp"]],
        [c["c_miss_ish"], c["c_miss_ish"], 0, c["c_bypass"]],
        [c["c_fn"], c["c_fn"], c["c_under"], 0]], float)


SCENARIOS = {
    "S0_linear": np.abs(np.subtract.outer(range(4), range(4))).astype(float),
    "S1_quadratic": np.subtract.outer(range(4), range(4)).astype(float) ** 2,
    "S2_workflow": workflow_matrix(NOMINAL),
    "S3_workflow_her2low": workflow_matrix({**NOMINAL, "c_low": 4.0}),
}


def taxonomy(conf):
    conf = np.asarray(conf)
    out = {}
    for name, cells in TAXONOMY.items():
        k = sum(conf[i, j] for i, j in cells)
        n = sum(conf[i].sum() for i in DENOM[name])
        out[f"n_{name}"] = int(k)
        out[f"rate_{name}"] = float(k / n) if n else np.nan
    off = np.abs(np.subtract.outer(range(4), range(4)))
    errs = conf.sum() - np.trace(conf)
    out["adjacent_errors"] = int(conf[off == 1].sum())
    out["nonadjacent_errors"] = int(conf[off >= 2].sum())
    out["nonadjacent_share_of_errors"] = float(conf[off >= 2].sum() / errs) if errs else np.nan
    return out


def costs(conf):
    conf = np.asarray(conf, float)
    return {f"cost_{s}": float((M * conf).sum() / conf.sum()) for s, M in SCENARIOS.items()}


def row(source, run, split, y, prob, groups=None):
    m = L.class_metrics(y, prob)
    r = dict(source=source, run=run, split=split, **{k: v for k, v in m.items() if k != "confusion"},
             **taxonomy(m["confusion"]), **costs(m["confusion"]), confusion=str(m["confusion"]))
    if groups is not None:
        p = prob.argmax(1)
        boot = {s: [] for s in SCENARIOS}
        for i in L.cluster_draws(groups, N_BOOT, L.SEED):
            c = L.confusion(y[i], p[i])
            for s, M in SCENARIOS.items():
                boot[s].append((M * c).sum() / c.sum())
        for s in SCENARIOS:
            r[f"cost_{s}_ci95"] = str(np.round(L.ci95(boot[s]), 4).tolist())
    return r


def paired_cost(y, pa, pb, groups, M):
    ca, cb = M[y, pa], M[y, pb]
    d = [ca[i].mean() - cb[i].mean() for i in L.cluster_draws(groups, N_BOOT, L.SEED)]
    return float(ca.mean() - cb.mean()), L.ci95(d)


def sensitivity(conf_by_name, rng):
    """Draw every workflow cost uniformly in [0.5x, 1.5x] its nominal value; record which entry has the lowest cost."""
    names = list(conf_by_name)
    wins = np.zeros(len(names))
    for _ in range(N_SENS):
        c = {k: v * rng.uniform(0.5, 1.5) for k, v in NOMINAL.items()}
        M = workflow_matrix(c)
        cs = [(M * np.asarray(conf_by_name[n], float)).sum() / np.asarray(conf_by_name[n]).sum() for n in names]
        wins[int(np.argmin(cs))] += 1
    return dict(zip(names, (wins / N_SENS).round(4).tolist()))


def heatmaps(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = SCENARIOS["S2_workflow"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.9))
    for ax, (title, conf) in zip(axes, panels):
        conf = np.asarray(conf, float)
        rown = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                txt = f"{rown[i, j]:.2f}\n(n={int(conf[i, j])})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if rown[i, j] > 0.55 else "black",
                        fontweight="bold" if (i != j and M[i, j] >= 6 and conf[i, j] > 0) else "normal")
                if i != j and M[i, j] >= 6:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#c0392b", lw=1.6))
        ax.set_xticks(range(4), L.LEVELS); ax.set_yticks(range(4), L.LEVELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ec = (M * conf).sum() / conf.sum()
        ax.set_title(f"{title}\nS2 expected cost/tile = {ec:.2f}", fontsize=9)
    fig.suptitle("Row-normalised confusion; red boxes = high-cost cells in illustrative scenario S2 "
                 "(missed 3+, missed 2+, 0/1+ called 3+)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(L.guard(path), dpi=300)
    plt.close(fig)


def bars(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [c for c in df.columns if c.startswith("rate_")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(df)
    for k, (_, r) in enumerate(df.iterrows()):
        ax.bar(np.arange(len(cols)) + k * w, [r[c] for c in cols], w, label=r["label"])
    ax.set_xticks(np.arange(len(cols)) + 0.4 - w / 2, [c[5:].split(" (")[0].replace("_", " ") for c in cols],
                  rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Rate (share of tiles of the true class(es) concerned)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Clinical error taxonomy", fontsize=10)
    fig.tight_layout()
    fig.savefig(L.guard(path), dpi=300)
    plt.close(fig)


def main():
    cfg = dict(experiment="01_clinical_risk", seed=L.SEED, n_boot=N_BOOT, n_sensitivity=N_SENS, taxonomy=TAXONOMY,
               nominal_costs=NOMINAL, scenarios={k: v.tolist() for k, v in SCENARIOS.items()},
               sensitivity_rule="each workflow cost drawn uniformly in [0.5, 1.5] x nominal, independently",
               bci_ci="fragment-cluster bootstrap", external_ci="filename-run bootstrap (no slide IDs: optimistic)",
               test_status="BCI test report-only; external test = exploratory second look (previously examined)")
    L.write_json(OUT / "config.json", cfg)

    rows = []
    for run in L.bci_runs():
        for split in ["val", "test"]:
            y, prob = L.bci_preds(run, split)
            rows.append(row("BCI", run, split, y, prob, L.fragments(split)))
    for f in sorted(L.EXT.glob("*__*.csv")):
        source, run = f.stem.split("__", 1)
        names, y, prob = L.ext_raw(source, run)
        rows.append(row(f"external_raw_{source}-trained", run, "ext_test", y, prob, L.filename_groups(names)))
    for f in sorted(L.H1TEST.glob("*.csv")):
        names, y, prob = L.ext_h1(f.stem)
        rows.append(row("external_HARMONISE1_stain_p99", f.stem, "ext_test", y, prob, L.filename_groups(names)))
    df = pd.DataFrame(rows)
    L.write_csv(OUT / "risk_table_all.csv", df)

    # ---- paired comparisons fixed in advance
    pairs = []
    for split in ["val", "test"]:
        g = L.fragments(split)
        for a, b in [("EfficientNetV2B0__ce__hires", "EfficientNetV2B0__ce"),
                     ("EfficientNetV2B0__ce__hires", "EfficientNetV2B0__ce__lowbatch"),
                     ("MobileNetV3Large__corn", "MobileNetV3Large__ce"),
                     ("DenseNet121__corn", "DenseNet121__ce"), ("EfficientNetV2B0__corn", "EfficientNetV2B0__ce")]:
            y, pa = L.bci_preds(a, split); _, pb = L.bci_preds(b, split)
            for s in ["S2_workflow", "S1_quadratic"]:
                d, ci = paired_cost(y, pa.argmax(1), pb.argmax(1), g, SCENARIOS[s])
                pairs.append(dict(domain="BCI", split=split, a=a, b=b, scenario=s, delta_cost=d, ci95=str(np.round(ci, 4).tolist()),
                                  ci_excludes_0=bool(ci[0] > 0 or ci[1] < 0)))
    for run in [f.stem for f in sorted(L.H1TEST.glob("*.csv"))]:
        names, y, pc = L.ext_h1(run)
        n2, y2, pr = L.ext_raw("BCI", run)
        order = pd.Series(range(len(n2)), index=n2)[names].values
        assert (y2[order] == y).all()
        g = L.filename_groups(names)
        for s in ["S2_workflow", "S1_quadratic"]:
            d, ci = paired_cost(y, pc.argmax(1), pr[order].argmax(1), g, SCENARIOS[s])
            pairs.append(dict(domain="external", split="ext_test", a=f"{run} + stain_p99", b=f"{run} raw", scenario=s,
                              delta_cost=d, ci95=str(np.round(ci, 4).tolist()), ci_excludes_0=bool(ci[0] > 0 or ci[1] < 0)))
    L.write_csv(OUT / "paired_cost_differences.csv", pd.DataFrame(pairs))

    # ---- sensitivity: decisions only on BCI val; external report-only
    rng = np.random.default_rng(L.SEED)
    bv = df[(df.source == "BCI") & (df.split == "val")].set_index("run")
    sens = {"BCI_val_all_runs": sensitivity({r: eval(bv.loc[r, "confusion"]) for r in bv.index}, rng)}
    ext = {}
    for run in [f.stem for f in sorted(L.H1TEST.glob("*.csv"))]:
        ext[f"{run} raw"] = eval(df[(df.source == "external_raw_BCI-trained") & (df.run == run)].confusion.iloc[0])
        ext[f"{run} stain_p99"] = eval(df[(df.source == "external_HARMONISE1_stain_p99") & (df.run == run)].confusion.iloc[0])
    for r in ["ResNet50__ce", "MobileNetV2__ce", "EfficientNetV2B0__ce"]:
        ext[f"FYP2 {r} raw"] = eval(df[(df.source == "external_raw_FYP2-trained") & (df.run == r)].confusion.iloc[0])
    sens["external_test_all_entries"] = sensitivity(ext, rng)
    L.write_json(OUT / "cost_sensitivity.json", sens)

    # ---- figures
    def conf_of(source, run, split):
        return eval(df[(df.source == source) & (df.run == run) & (df.split == split)].confusion.iloc[0])
    heatmaps([("BCI test: EfficientNetV2B0 448 px (primary)", conf_of("BCI", "EfficientNetV2B0__ce__hires", "test")),
              ("BCI test: EfficientNetV2B0 224 px", conf_of("BCI", "EfficientNetV2B0__ce", "test")),
              ("External: EffNetV2B0 224 raw", conf_of("external_raw_BCI-trained", "EfficientNetV2B0__ce", "ext_test")),
              ("External: EffNetV2B0 224 + stain_p99", conf_of("external_HARMONISE1_stain_p99", "EfficientNetV2B0__ce", "ext_test")),
              ("External: MobileNetV3Small + stain_p99", conf_of("external_HARMONISE1_stain_p99", "MobileNetV3Small__ce", "ext_test"))],
             OUT / "fig_risk_heatmaps.png")
    sel = [("BCI", "EfficientNetV2B0__ce__hires", "test", "BCI test 448 px"),
           ("BCI", "EfficientNetV2B0__ce", "test", "BCI test 224 px"),
           ("external_raw_BCI-trained", "EfficientNetV2B0__ce", "ext_test", "Ext. 224 raw"),
           ("external_HARMONISE1_stain_p99", "EfficientNetV2B0__ce", "ext_test", "Ext. 224 + stain_p99"),
           ("external_HARMONISE1_stain_p99", "MobileNetV3Small__ce", "ext_test", "Ext. MNV3S + stain_p99"),
           ("external_raw_FYP2-trained", "ResNet50__ce", "ext_test", "Ext. FYP2 ResNet50 raw")]
    bdf = pd.concat([df[(df.source == s) & (df.run == r) & (df.split == sp)].assign(label=lab) for s, r, sp, lab in sel])
    bars(bdf, OUT / "fig_error_taxonomy.png")

    keep = ["source", "run", "split", "accuracy", "balanced_accuracy", "macro_f1", "qwk", "recall_2+", "precision_2+",
            "rate_positive_missed (3+ -> 0/1+)", "rate_equivocal_missed (2+ -> 0/1+)", "rate_false_positive (0/1+ -> 3+)",
            "nonadjacent_share_of_errors", "cost_S2_workflow", "cost_S2_workflow_ci95"]
    L.write_csv(OUT / "risk_table_key.csv", bdf[keep])
    print(L.md_table(bdf[keep].drop(columns=["source"])))
    print("sensitivity:", sens)


if __name__ == "__main__":
    main()
