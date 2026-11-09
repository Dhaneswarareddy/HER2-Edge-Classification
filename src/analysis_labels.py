"""Diagnosis B: label noise, scan type and near-duplicates, from saved predictions (no TensorFlow).

For every finished run (results/runs/*/predictions.npz), on val and on test:

  1. Label noise. BCI labels are slide-level. 'refined_label' (patch-level, distributed with BCI in
     BCI_metadata_pairs_refined.csv, method undocumented) exists for ~72% of tiles. Split those tiles into
       clean = refined == filename label,   noisy = refined != filename label
     and report accuracy on each, plus how often a prediction that is "wrong" against the filename label
     equals the refined label. If the model is right on clean tiles and its "errors" on noisy tiles
     agree with the refined label, much of the measured error is label noise, not model failure.
     (Refined labels are an independent second opinion, not ground truth.)
  2. Scan type. Background brightness = mean grey level of near-unsaturated pixels (HSV S < 8/255).
     Bright vs grey scans are split by an Otsu threshold fitted on the TRAIN split (the earlier FYP
     project used 201.0 with the same definition). Accuracy / macro-F1 / per-class recall per scan type.
  3. Near-duplicates. Test metrics with the 15 test tiles that are pHash near-duplicates of training
     tiles removed.
  4. Prediction drivers (both datasets, CE baselines, test split): cross-validated R^2 of each model's
     predicted score from groups of hand-crafted statistics (DAB, haematoxylin, colour, background,
     texture), next to the R^2 for the true label. Shows which simple cues the outputs track.
  5. Scan type x class (val+test pooled): BCI's grey scans are more common in higher classes, so a model
     could use background brightness as a shortcut. Accuracy and mean bias (expected score - label) per
     class and scan type; a shortcut would over-score grey and under-score bright tiles of the same class.
  6. Error drivers (val+test pooled): OLS of each run's error (expected score - label) on standardised
     illumination-corrected DAB, background brightness, corrected haematoxylin, tissue fraction and log
     sharpness, with class fixed effects; 95% CIs from a fragment-cluster bootstrap. Separates "the model
     follows the tile's own stain" from "the model follows scan brightness" (correlational; the causal
     test is the perturbation analysis in analysis_models.py).

Development decisions use the val numbers only; test numbers are reported.
Outputs (in config.ANALYSIS_DIR, i.e. results/analysis or results/analysis_official): label_noise.csv,
         scan_type.csv, scan_by_class.csv, error_drivers.csv, near_dup.csv (official split only),
         prediction_drivers.csv, figures/figD3_label_noise*.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import config as C

OUT = C.ANALYSIS_DIR
FIG = C.RESULTS_DIR / "figures"
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"


def otsu(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    hist, edges = np.histogram(v, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = ((hist * mids).sum() - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    return float(mids[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def basic(y, p):
    if len(y) == 0:
        return dict(n=0)
    return dict(n=len(y), accuracy=float((p == y).mean()), macro_f1=float(f1_score(y, p, average="macro")),
                qwk=float(cohen_kappa_score(y, p, weights="quadratic")) if len(set(y)) > 1 else np.nan,
                **{f"recall_{k}": float((p[y == k] == k).mean()) if (y == k).any() else np.nan for k in range(4)})


def load_runs():
    runs = {}
    for p in sorted(C.RUNS_DIR.glob("*/predictions.npz")):
        d = np.load(p)
        runs[p.parent.name] = d
    return runs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(C.MANIFEST)
    feat = pd.read_csv(C.SHARED_ANALYSIS / "data_features.csv")
    feat = feat[feat.dataset == "BCI (real)"].set_index("path")
    for c in ERROR_DRIVERS + ["dab_mean"]:
        man[c] = man.path.map(feat[c])
    thr = otsu(man.bg_brightness[man.split == "train"])
    man["scan"] = np.where(man.bg_brightness >= thr, "bright", "grey")
    man.loc[man.bg_brightness.isna(), "scan"] = "no background"
    if "fragment" not in man:                                # official manifest
        man["fragment"] = man.path.map(pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv").set_index("path").fragment)
    splits = {s: man[man.split == s].reset_index(drop=True) for s in ["val", "test"]}

    noise_rows, scan_rows, dup_rows, class_rows, driver_rows = [], [], [], [], []
    for run, d in load_runs().items():
        pooled = pd.concat([splits[s].assign(pred=d[s].argmax(1), escore=d[s] @ np.arange(C.NUM_CLASSES))
                            for s in ["val", "test"]], ignore_index=True)
        pooled["ok"], pooled["bias"] = pooled.pred == pooled.label, pooled.escore - pooled.label
        for (k, scan), g in pooled[pooled.scan != "no background"].groupby(["label", "scan"]):
            class_rows.append(dict(run=run, label=k, scan=scan, n=len(g), accuracy=float(g.ok.mean()),
                                   mean_bias=float(g.bias.mean())))
        driver_rows += [dict(run=run, **r) for r in error_drivers(pooled)]
        for s, key in [("val", "val"), ("test", "test")]:
            m = splits[s]
            y, p = d[f"y_{key}"], d[key].argmax(1)
            assert (y == m.label.values).all(), f"{run}: prediction order does not match manifest ({s})"
            ref = m.refined_label.values
            has = ~np.isnan(ref)
            clean, noisy = has & (ref == y), has & (ref != y)
            wrong = p != y
            noise_rows.append(dict(
                run=run, split=s, n=len(y), accuracy=float((p == y).mean()),
                n_with_refined=int(has.sum()), refined_disagree_rate=float(noisy.sum() / has.sum()),
                acc_clean=float((p[clean] == y[clean]).mean()), n_clean=int(clean.sum()),
                acc_noisy_vs_filename=float((p[noisy] == y[noisy]).mean()), n_noisy=int(noisy.sum()),
                acc_noisy_vs_refined=float((p[noisy] == ref[noisy]).mean()),
                acc_vs_refined_all=float((p[has] == ref[has]).mean()),
                errors_with_refined=int((wrong & has).sum()),
                share_errors_matching_refined=float((p[wrong & has] == ref[wrong & has]).mean())
                if (wrong & has).any() else np.nan,
                share_errors_on_noisy_tiles=float(noisy[wrong & has].mean()) if (wrong & has).any() else np.nan,
                macro_f1_clean=float(f1_score(y[clean], p[clean], average="macro"))))
            for scan in ["bright", "grey"]:
                k = (m.scan == scan).values
                scan_rows.append(dict(run=run, split=s, scan=scan, **basic(y[k], p[k])))
            if s == "test" and "test_near_dup_of_train" in m:      # official split only (pHash flags)
                keep = ~m.test_near_dup_of_train.values
                dup_rows.append(dict(run=run, **{f"all_{k}": v for k, v in basic(y, p).items()},
                                     **{f"no_dup_{k}": v for k, v in basic(y[keep], p[keep]).items()}))

    nz, sc, dp = pd.DataFrame(noise_rows), pd.DataFrame(scan_rows), pd.DataFrame(dup_rows)
    cl, dr = pd.DataFrame(class_rows), pd.DataFrame(driver_rows)
    nz.to_csv(OUT / "label_noise.csv", index=False)
    sc.to_csv(OUT / "scan_type.csv", index=False)
    if len(dp):                                               # official split only
        dp.to_csv(OUT / "near_dup.csv", index=False)
    cl.to_csv(OUT / "scan_by_class.csv", index=False)
    dr.to_csv(OUT / "error_drivers.csv", index=False)
    counts = {f"{a}/{b}": int(n) for (a, b), n in man.groupby(["split", "scan"]).size().items()}
    known = man[man.scan != "no background"]
    json.dump({"otsu_threshold_train": thr, "scan_counts": counts,
               "grey_share_by_class": known.groupby("label").scan.apply(lambda s: float((s == "grey").mean())).to_dict(),
               "dabc_median_by_class_scan": {f"{k}/{s}": float(v) for (k, s), v in
                                             known.groupby(["label", "scan"]).dabc_mean.median().items()}},
              open(OUT / "scan_type_threshold.json", "w"), indent=2, default=str)
    pd.set_option("display.width", 250)
    print(f"Otsu background threshold (train): {thr:.1f}")
    print(nz.round(3).to_string(index=False))
    print(sc.round(3).to_string(index=False))
    print(cl.round(3).to_string(index=False))
    print(dr.round(3).to_string(index=False))
    if len(dp):
        print(dp[["run", "all_accuracy", "no_dup_accuracy", "all_macro_f1", "no_dup_macro_f1"]].round(4).to_string(index=False))
    fig_noise(nz)
    prediction_drivers()


ERROR_DRIVERS = ["dabc_mean", "bg_brightness", "hemc_mean", "tissue_frac", "sharpness"]


def error_drivers(m):
    """Standardised OLS coefficients of (expected score - label) on ERROR_DRIVERS with class fixed effects
    (score units per 1 SD), with fragment-cluster bootstrap 95% CIs."""
    import evaluate_all as E
    X = m[ERROR_DRIVERS].assign(bg_brightness=m.bg_brightness.fillna(m.bg_brightness.median()),
                                sharpness=np.log1p(m.sharpness))
    X = (X - X.mean()) / X.std()
    A = np.column_stack([X.values, pd.get_dummies(m.label, dtype=float).values])
    t, k = m.bias.values, len(ERROR_DRIVERS)
    beta = np.linalg.lstsq(A, t, rcond=None)[0][:k]
    boots = np.array([np.linalg.lstsq(A[i], t[i], rcond=None)[0][:k] for i in E.cluster_draws(m.fragment.values, C.SEED)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return [dict(feature=f, coef=float(b), ci_lo=float(a), ci_hi=float(c), n=len(m))
            for f, b, a, c in zip(ERROR_DRIVERS, beta, lo, hi)]


DRIVER_GROUPS = {
    "DAB, raw OD": ["dab_mean", "dab_median", "dab_p90", "dab_frac015", "dab_frac030"],
    "DAB, corrected": ["dabc_mean", "dabc_median", "dabc_p90", "dabc_frac015", "dabc_frac030"],
    "haematoxylin": ["hem_mean", "hemc_mean"],
    "mean colour": ["r_mean", "g_mean", "b_mean", "sat_mean", "val_mean"],
    "background": ["bg_brightness", "tissue_frac"],
    "texture": ["sharpness", "hf_share"],
}


def prediction_drivers():
    """How much of each CE baseline model's predicted score (expected class under its probabilities) on
    the TEST split is explained by each group of hand-crafted image statistics, compared with how much
    of the true label the same statistics explain. 5-fold CV R^2 of a linear model, per group and all."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat = pd.read_csv(C.SHARED_ANALYSIS / "data_features.csv")
    sources = [("FYP2", "FYP2 (GAN)", C.FYP2_ROOT / "results" / "runs", C.FYP2_ROOT / "splits" / "split_manifest.csv"),
               ("BCI", "BCI (real)", C.RUNS_DIR, C.MANIFEST)]
    cv = KFold(5, shuffle=True, random_state=C.SEED)
    rows = []
    for ds, ds_label, runs_root, manifest in sources:
        f = feat[feat.dataset == ds_label].set_index("path")
        test = pd.read_csv(manifest)
        test = test[test.split == "test"].reset_index(drop=True)
        X_all = f.loc[test.path]
        X_all = X_all.fillna(X_all.median(numeric_only=True))
        y = test.label.values.astype(float)
        targets = {"true label": y}
        for model in C.MODELS:
            p = runs_root / f"{model}__ce" / "predictions.npz"
            if p.exists():
                targets[model] = np.load(p)["test"] @ np.arange(C.NUM_CLASSES)
        for tname, t in targets.items():
            row = dict(dataset=ds, target=tname,
                       spearman_dab=float(spearmanr(t, X_all.dab_mean).correlation),
                       spearman_dabc=float(spearmanr(t, X_all.dabc_mean).correlation),
                       spearman_bg=float(spearmanr(t, X_all.bg_brightness).correlation),
                       spearman_sharp=float(spearmanr(t, X_all.sharpness).correlation))
            for g, cols in list(DRIVER_GROUPS.items()) + [("all", sum(DRIVER_GROUPS.values(), []))]:
                pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X_all[cols].values, t, cv=cv)
                row[f"r2_{g}"] = float(1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "prediction_drivers.csv", index=False)
    print(d.round(3).to_string(index=False))


def fig_noise(nz):
    t = nz[nz.split == "val"].copy()
    if t.empty:
        return
    t = t.sort_values("acc_clean")
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(t) + 1.5))
    y = np.arange(len(t))
    bars = [("acc_clean", "Clean tiles (refined = filename)", "#2a78d6", 0.27),
            ("acc_noisy_vs_filename", "Noisy tiles, scored against filename label", "#eb6834", 0.0),
            ("acc_noisy_vs_refined", "Noisy tiles, scored against refined label", "#1baf7a", -0.27)]
    for col, lab, c, dy in bars:
        ax.barh(y + dy, t[col], height=0.25, color=c, label=lab)
        for yi, v in zip(y + dy, t[col]):
            ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, t.run.str.replace("__", " / "), fontsize=8.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Validation accuracy")
    ax.set_title("Where do the errors come from? Slide-level vs patch-level labels", loc="left",
                 fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"figD3_label_noise{C.SUFFIX}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
