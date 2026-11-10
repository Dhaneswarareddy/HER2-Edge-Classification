"""Experiment 2 (v2): confidence, calibration and selective prediction -- only the gaps left by HARMONISE-2 (E).

Already done elsewhere (reused, not repeated): ECE, NLL, temperature scaling and review flagging for 5 networks on
HER2-IHC-40x development tiles (next_experiment/harmonise2/outputs/E_*.csv, R_cal fits, R_eval reports).

New here (stored predictions only, no inference):
  A. In-domain BCI, all 20 runs: ECE/NLL/Brier on val and test; temperature T fitted on val (NLL), applied to test.
  B. In-domain selective prediction for every run: confidence thresholds set on BCI VAL for target coverages,
     applied unchanged to BCI TEST; accuracy kept, share of each class referred, error-detection AUROC,
     fragment-cluster bootstrap CI of the accuracy gain for the two deployment models.
  C. External (exploratory second look at a previously examined test split): T and thresholds fitted on R_cal
     development tiles (HARMONISE-2 predictions), applied unchanged to the 1,847-tile test split, for the raw and
     stain_p99 predictions of the five HARMONISE panel networks.
  D. Figures: risk-coverage curves, reliability diagrams, per-class confidence distributions.

Softmax confidence is a heuristic score, not a statistically validated uncertainty; nothing here claims coverage
guarantees.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

OUT = L.out_dir("02_uncertainty")
BINS = 15
COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95]
T_GRID = np.geomspace(0.2, 20, 120)
N_BOOT = 2000
DEPLOY = ["EfficientNetV2B0__ce__hires", "EfficientNetV2B0__ce"]
PANEL = ["EfficientNetV2B0__ce", "MobileNetV3Small__ce", "MobileNetV3Large__corn", "DenseNet121__ce", "EfficientNetV2B0__ce__hires"]


def temp(p, T):
    q = np.exp(np.log(np.maximum(p, 1e-12)) / T)
    return q / q.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def brier(p, y):
    return float(((p - np.eye(4)[y]) ** 2).sum(1).mean())


def ece(p, y, bins=BINS):
    conf, a = p.max(1), p.argmax(1)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    rows = [dict(bin=b, n=int((idx == b).sum()), confidence=float(conf[idx == b].mean()) if (idx == b).any() else np.nan,
                 accuracy=float((a[idx == b] == y[idx == b]).mean()) if (idx == b).any() else np.nan) for b in range(bins)]
    t = pd.DataFrame(rows)
    v = t[t.n > 0]
    return float((v.n / v.n.sum() * (v.confidence - v.accuracy).abs()).sum()), t


def fit_T(p, y):
    return float(T_GRID[np.argmin([nll(temp(p, t), y) for t in T_GRID])])


def signals(p):
    s = np.sort(p, 1)
    return dict(max_prob=p.max(1), margin=s[:, -1] - s[:, -2], neg_entropy=(p * np.log(np.maximum(p, 1e-12))).sum(1))


def selective(p_fit, p_rep, y_rep, groups=None, tag=None):
    rows = []
    wrong = p_rep.argmax(1) != y_rep
    for sig, v_fit in signals(p_fit).items():
        v = signals(p_rep)[sig]
        auc = float(roc_auc_score(wrong, -v)) if 0 < wrong.mean() < 1 else np.nan
        for cov in COVERAGES:
            th = float(np.quantile(v_fit, 1 - cov))
            k = v >= th
            m = L.class_metrics(y_rep[k], p_rep[k]) if k.sum() > 0 else {}
            r = dict(tag, signal=sig, target_coverage=cov, threshold=th, coverage=float(k.mean()),
                     accuracy_all=float(1 - wrong.mean()), accuracy_kept=m.get("accuracy"), error_auroc=auc,
                     macro_f1_kept=m.get("macro_f1"), qwk_kept=m.get("qwk"))
            for c, nm in enumerate(L.LEVELS):
                r[f"share_{nm}_referred"] = float((~k[y_rep == c]).mean()) if (y_rep == c).any() else np.nan
            r["share_errors_referred"] = float((~k[wrong]).mean()) if wrong.any() else np.nan
            r["accuracy_gain"] = r["accuracy_kept"] - r["accuracy_all"] if r["accuracy_kept"] is not None else np.nan
            if groups is not None and sig == "max_prob":
                g = []
                for i in L.cluster_draws(groups, N_BOOT, L.SEED):
                    kk = k[i]
                    g.append((p_rep[i][kk].argmax(1) == y_rep[i][kk]).mean() - (p_rep[i].argmax(1) == y_rep[i]).mean()
                             if kk.any() else np.nan)
                r["accuracy_gain_ci95"] = str(np.round(L.ci95(g), 4).tolist())
            rows.append(r)
    return rows


def main():
    L.write_json(OUT / "config.json", dict(experiment="02_uncertainty", seed=L.SEED, bins=BINS, coverages=COVERAGES,
                                           temperature_grid="geomspace(0.2, 20, 120), NLL minimised",
                                           bci_fit_split="val", bci_report_split="test",
                                           external_fit="HARMONISE-2 R_cal development tiles (2,191)",
                                           external_report="1,847-tile WSI test split, exploratory second look",
                                           signals=["max_prob", "margin", "neg_entropy"]))
    # ---------------- A + B: BCI
    cal_rows, sel_rows, curves = [], [], []
    for run in L.bci_runs():
        yv, pv = L.bci_preds(run, "val"); yt, pt = L.bci_preds(run, "test")
        T = fit_T(pv, yv)
        for split, y, p in [("val", yv, pv), ("test", yt, pt)]:
            for lab, q, tt in [("raw", p, 1.0), ("temp_scaled", temp(p, T), T)]:
                e, c = ece(q, y)
                cal_rows.append(dict(run=run, split=split, probs=lab, T=tt, ece=e, nll=nll(q, y), brier=brier(q, y),
                                     accuracy=float((q.argmax(1) == y).mean()), mean_confidence=float(q.max(1).mean())))
                if run in DEPLOY and split == "test":
                    curves.append(c.assign(run=run, domain="BCI test", probs=lab))
        sel_rows += selective(pv, pt, yt, L.fragments("test") if run in DEPLOY else None,
                              dict(domain="BCI", run=run, fit="val", report="test"))
    L.write_csv(OUT / "bci_calibration_all_runs.csv", pd.DataFrame(cal_rows))
    L.write_csv(OUT / "bci_selective_prediction_all_runs.csv", pd.DataFrame(sel_rows))

    # ---------------- C: external (R_cal -> test)
    r = pd.read_csv(L.H2OUT / "R_index.csv")
    cal = np.where(r.subset == "R_cal")[0]
    ycal = r.label.values[cal].astype(int)
    ext_cal, ext_sel = [], []
    for run in PANEL:
        for arm, src in [("none", "raw"), ("frozen", "stain_p99")]:
            z = np.load(L.H2OUT / "preds" / "C" / f"{run}__{arm}.npz")
            assert (z["idx"] == np.arange(len(r))).all()
            pc = z["prob"][cal].astype(np.float64)
            names, yt, pt = (L.ext_raw("BCI", run) if arm == "none" else L.ext_h1(run))
            T = fit_T(pc, ycal)
            for lab, q, tt in [("raw", pt, 1.0), ("temp_scaled_on_R_cal", temp(pt, T), T)]:
                e, c = ece(q, yt)
                ext_cal.append(dict(run=run, input=src, probs=lab, T=tt, ece=e, nll=nll(q, yt), brier=brier(q, yt),
                                    accuracy=float((q.argmax(1) == yt).mean()), mean_confidence=float(q.max(1).mean())))
                if run == "EfficientNetV2B0__ce":
                    curves.append(c.assign(run=run, domain=f"external test ({src})", probs=lab))
            ext_sel += selective(pc, pt, yt, None, dict(domain="external_test_exploratory", run=run, input=src,
                                                        fit="R_cal", report="ext_test"))
    L.write_csv(OUT / "external_calibration_test.csv", pd.DataFrame(ext_cal))
    L.write_csv(OUT / "external_selective_prediction_test.csv", pd.DataFrame(ext_sel))
    L.write_csv(OUT / "reliability_curves.csv", pd.concat(curves))
    figures(pd.DataFrame(sel_rows), pd.DataFrame(ext_sel), pd.concat(curves))

    s = pd.DataFrame(sel_rows)
    show = s[(s.run.isin(DEPLOY)) & (s.signal == "max_prob")][["run", "target_coverage", "coverage", "accuracy_all",
                                                                "accuracy_kept", "accuracy_gain_ci95", "share_2+_referred",
                                                                "share_errors_referred", "error_auroc"]]
    print(L.md_table(show))
    c = pd.DataFrame(cal_rows)
    print(L.md_table(c[c.run.isin(DEPLOY)]))
    e = pd.DataFrame(ext_sel)
    print(L.md_table(e[(e.signal == "max_prob") & (e.run == "EfficientNetV2B0__ce")][
        ["input", "target_coverage", "coverage", "accuracy_all", "accuracy_kept", "share_2+_referred", "share_errors_referred"]]))
    print(L.md_table(pd.DataFrame(ext_cal)))


def figures(sel, ext, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # risk-coverage (dense curve on BCI test from val-free sorting is NOT used: we plot the val-fixed points)
    ax = axes[0]
    for run, lab in [("EfficientNetV2B0__ce__hires", "448 px (BCI test)"), ("EfficientNetV2B0__ce", "224 px (BCI test)")]:
        d = sel[(sel.run == run) & (sel.signal == "max_prob")].sort_values("coverage")
        ax.plot(d.coverage, d.accuracy_kept, "o-", label=f"{lab}: accuracy kept")
        ax.plot(d.coverage, d["share_2+_referred"], "s--", label=f"{lab}: share of 2+ referred")
    d = ext[(ext.run == "EfficientNetV2B0__ce") & (ext.input == "stain_p99") & (ext.signal == "max_prob")].sort_values("coverage")
    ax.plot(d.coverage, d.accuracy_kept, "o-", c="#b03a2e", label="224 px + stain_p99 (ext. test): accuracy kept")
    ax.plot(d.coverage, d["share_2+_referred"], "s--", c="#b03a2e", label="224 px + stain_p99 (ext. test): 2+ referred")
    ax.set_xlabel("Coverage (share of tiles kept; thresholds fixed on val / R_cal)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5); ax.set_title("Selective prediction (max softmax probability)", fontsize=9.5)
    ax = axes[1]
    show = [("EfficientNetV2B0__ce__hires", "BCI test", "raw"), ("EfficientNetV2B0__ce", "BCI test", "raw"),
            ("EfficientNetV2B0__ce", "external test (stain_p99)", "raw"),
            ("EfficientNetV2B0__ce", "external test (stain_p99)", "temp_scaled_on_R_cal")]
    for run, dom, pr in show:
        c = curves[(curves.run == run) & (curves.domain == dom) & (curves.probs == pr) & (curves.n >= 10)]
        ax.plot(c.confidence, c.accuracy, "o-", ms=3, label=f"{'448' if 'hires' in run else '224'} px, {dom}, {pr}")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=6.5); ax.set_title("Reliability diagrams (15 bins; bins with n >= 10 only)", fontsize=9.5)
    ax = axes[2]
    y, p = L.bci_preds("EfficientNetV2B0__ce__hires", "test")
    names, ye, pe = L.ext_h1("EfficientNetV2B0__ce")
    data, labels = [], []
    for c in range(4):
        data.append(p[y == c].max(1)); labels.append(f"BCI 448\n{L.LEVELS[c]}")
    for c in range(4):
        data.append(pe[ye == c].max(1)); labels.append(f"Ext 224+cal\n{L.LEVELS[c]}")
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.tick_params(axis="x", labelsize=6.5); ax.set_ylabel("Max softmax probability")
    ax.set_title("Confidence by true class", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(L.guard(OUT / "fig_uncertainty.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
