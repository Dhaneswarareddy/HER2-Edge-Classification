"""Diagnosis A: model-free comparison of FYP2 (GAN) and BCI (real) HER2 IHC images.

Question: are the four classes separable by simple, interpretable image statistics in each dataset?
If FYP2's classes are nearly separable by global colour/stain alone while BCI's are not, that is
direct evidence the two tasks differ in difficulty, independently of any CNN.

Per image (both datasets at the same 256x256 resolution):
  * colour deconvolution (Ruifrok & Johnston 2001 H-E-DAB matrix, as in skimage.rgb2hed) on
    tissue pixels: DAB optical density mean / median / p90, fraction of pixels with DAB OD > 0.15
    and > 0.30, haematoxylin OD mean. Two versions:
      raw (dab_*, hem_mean): OD relative to 255, as skimage does. A grey background cast then counts
          as stain (a neutral grey of OD g adds 0.88 g to DAB and 1.21 g to haematoxylin), and BCI
          scans have grey backgrounds of varying brightness.
      illumination-corrected (dabc_*, hemc_mean): OD relative to the image's own background colour
          I0 (Beer-Lambert with the actual illumination), so the cast cancels.
  * background brightness: mean grey level of near-unsaturated pixels (HSV S < 8/255), the
    definition used in the earlier FYP project; tissue fraction = 1 - background fraction
  * mean RGB, mean HSV saturation and value
  * sharpness: variance of the Laplacian; high-frequency share of the grey-level power spectrum

Separability: 5-fold CV on the DEVELOPMENT split only (train+val; test is not used), logistic
regression (standardised) and random forest, on feature subsets: all, DAB only, colour only (mean
RGB), background only. Folds respect leakage groups (FYP2: pHash twins; BCI: slide fragments from
analysis_slides.py); for BCI an image-level CV is also reported to show the effect of slide cues.

Outputs: results/analysis/data_features.csv, data_separability.csv, figures/figD1_*.png, figD2_*.png
Runs on CPU with numpy/OpenCV only (safe to run alongside training).
"""
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

OUT = C.SHARED_ANALYSIS
FIG = C.RESULTS_DIR / "figures"
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
DS_COLOR = {"FYP2 (GAN)": "#2a78d6", "BCI (real)": "#eb6834"}

# Ruifrok & Johnston stain vectors (rows: haematoxylin, eosin, DAB); same as skimage.rgb2hed
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)

FEATURE_SETS = {
    "all": None,
    "DAB, raw OD": ["dab_mean", "dab_median", "dab_p90", "dab_frac015", "dab_frac030"],
    "DAB, illumination-corrected": ["dabc_mean", "dabc_median", "dabc_p90", "dabc_frac015", "dabc_frac030"],
    "mean colour only": ["r_mean", "g_mean", "b_mean"],
    "background only": ["bg_brightness", "tissue_frac"],
}


def rgb2hed(rgb, I0=255.0):
    """Stain concentrations (rows: pixels; columns: H, E, DAB), OD relative to illumination I0."""
    od = -np.log(np.maximum(rgb.astype(np.float32) / np.asarray(I0, np.float32), 1e-6))
    return od.reshape(-1, 3) @ _HED_FROM_RGB


def stain_stats(hed, tissue, prefix):
    dab, hem = np.clip(hed[tissue, 2], 0, None), np.clip(hed[tissue, 0], 0, None)
    return {f"{prefix}_mean": dab.mean(), f"{prefix}_median": np.median(dab), f"{prefix}_p90": np.percentile(dab, 90),
            f"{prefix}_frac015": (dab > 0.15).mean(), f"{prefix}_frac030": (dab > 0.30).mean(),
            f"{prefix.replace('dab', 'hem')}_mean": hem.mean()}


def image_features(path):
    rgb = np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.LANCZOS))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1].reshape(-1), hsv[..., 2].reshape(-1)
    bg = sat < 8
    tissue = ~bg if (~bg).sum() > 100 else np.ones_like(bg)
    I0 = rgb.reshape(-1, 3)[bg].mean(0) if bg.sum() > 100 else np.percentile(rgb.reshape(-1, 3), 99, axis=0)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean()))) ** 2
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - 128, xx - 128)
    return dict(
        **stain_stats(rgb2hed(rgb), tissue, "dab"), **stain_stats(rgb2hed(rgb, np.maximum(I0, 1.0)), tissue, "dabc"),
        bg_brightness=float(rgb.reshape(-1, 3)[bg].mean()) if bg.sum() > 100 else np.nan,
        tissue_frac=1 - bg.mean(),
        r_mean=rgb[..., 0].mean(), g_mean=rgb[..., 1].mean(), b_mean=rgb[..., 2].mean(),
        sat_mean=sat.mean(), val_mean=val.mean(),
        sharpness=cv2.Laplacian(grey, cv2.CV_32F).var(),
        hf_share=spec[r > 64].sum() / spec.sum(),
    )


def collect():
    fyp2 = pd.read_csv(C.FYP2_ROOT / "splits" / "split_manifest.csv")
    fyp2_dir = C.FYP2_ROOT / "HER2 SCORES DATA" / "Generated_HER2_images"
    bci = pd.read_csv(C.OFFICIAL_MANIFEST)
    rows = []
    for ds, df, root in [("FYP2 (GAN)", fyp2, fyp2_dir), ("BCI (real)", bci, C.DATA_DIR)]:
        for r in df.itertuples():
            f = image_features(root / r.path)
            rows.append(dict(dataset=ds, path=r.path, label=r.label, split=r.split,
                             refined_label=getattr(r, "refined_label", np.nan), **f))
    return pd.DataFrame(rows)


META = ("dataset", "path", "label", "split", "refined_label", "group")


def attach_split(feat):
    """Split and leakage group per image from the CURRENT manifests (features are cached without them):
    FYP2 -> its own split and pHash groups; BCI -> the FYP3 split in use and slide fragments."""
    feat = feat.drop(columns=[c for c in ("split", "refined_label", "group") if c in feat])
    fyp2 = pd.read_csv(C.FYP2_ROOT / "splits" / "split_manifest.csv").set_index("path")
    bci = pd.read_csv(C.MANIFEST).set_index("path")
    frag = pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv").set_index("path").fragment
    is_f = feat.dataset == "FYP2 (GAN)"
    feat["split"] = np.where(is_f, feat.path.map(fyp2.split), feat.path.map(bci.split))
    feat["group"] = np.where(is_f, "f" + feat.path.map(fyp2.group).astype(str),
                             "b" + feat.path.map(frag).astype(str))
    feat["refined_label"] = np.where(is_f, np.nan, feat.path.map(bci.refined_label))
    return feat


def separability(feat):
    """5-fold CV on the development splits. 'grouped': folds respect leakage groups (FYP2 pHash twins,
    BCI slide fragments), as the main split does. 'image-level' (BCI only): folds ignore fragments, so
    neighbouring tiles of one slide can be in train and test folds; the gap to 'grouped' measures how
    much slide-level cues inflate a model-free classifier."""
    rows = []
    for ds, d in feat[feat.split != "test"].groupby("dataset"):
        y = d.label.values
        schemes = [("grouped", StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED), d.group.values)]
        if ds == "BCI (real)":
            schemes.append(("image-level", StratifiedKFold(5, shuffle=True, random_state=C.SEED), None))
        for set_name, cols in FEATURE_SETS.items():
            cols = cols or [c for c in feat.columns if c not in META]
            X = d[cols].fillna(d[cols].median()).values
            for clf_name, clf in [("logistic", make_pipeline(StandardScaler(),
                                                             LogisticRegression(max_iter=2000))),
                                  ("random forest", RandomForestClassifier(300, random_state=C.SEED,
                                                                           n_jobs=2))]:
                for cv_name, cv, groups in schemes:
                    pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
                    rows.append(dict(dataset=ds, cv=cv_name, features=set_name, classifier=clf_name, n=len(y),
                                     accuracy=(pred == y).mean(), macro_f1=f1_score(y, pred, average="macro"),
                                     qwk=cohen_kappa_score(y, pred, weights="quadratic")))
    return pd.DataFrame(rows)


def fig_distributions(feat):
    cols = [("dab_mean", "Mean DAB OD, raw (vs 255)"), ("dabc_mean", "Mean DAB OD, illumination-corrected"),
            ("bg_brightness", "Background brightness"), ("sharpness", "Sharpness (Laplacian variance, log)"),
            ("hf_share", "High-frequency share of spectrum")]
    d = feat[feat.split != "test"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15.5, 3.4))
    for ax, (c, lab) in zip(axes, cols):
        for j, (ds, col) in enumerate(DS_COLOR.items()):
            data = [d[(d.dataset == ds) & (d.label == k)][c].dropna().values for k in range(4)]
            pos = np.arange(4) + (j - 0.5) * 0.36
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                            medianprops=dict(color=SURF, lw=1.5))
            for b in bp["boxes"]:
                b.set(facecolor=col, edgecolor=col)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color=col)
        if c == "sharpness":
            ax.set_yscale("log")
        ax.set_xticks(range(4), ["0", "1+", "2+", "3+"])
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=ds) for ds, c in DS_COLOR.items()], loc="upper right",
               ncol=2, frameon=False)
    fig.supxlabel("HER2 score (filename label)", fontsize=10, color=INK2)
    fig.suptitle("Image statistics per class, development splits", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "figD1_stain_distributions.png", dpi=160)
    plt.close(fig)


def fig_separability(sep):
    rf = sep[sep.classifier == "random forest"]
    bars = [("FYP2 (GAN)", "grouped", "FYP2 (GAN), grouped CV", DS_COLOR["FYP2 (GAN)"], None),
            ("BCI (real)", "grouped", "BCI (real), grouped CV (by slide fragment)", DS_COLOR["BCI (real)"], None),
            ("BCI (real)", "image-level", "BCI (real), image-level CV (fragments split)", "#f6b49a", "///")]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(FEATURE_SETS))[::-1]
    h = 0.26
    for j, (ds, cvn, lab, col, hatch) in enumerate(bars):
        v = rf[(rf.dataset == ds) & (rf.cv == cvn)].set_index("features").macro_f1.reindex(list(FEATURE_SETS))
        yy = y + (1 - j) * h
        ax.barh(yy, v, height=h * 0.92, color=col, label=lab, hatch=hatch, edgecolor="white" if hatch else None)
        for yi, x in zip(yy, v):
            ax.text(x + 0.01, yi, f"{x:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0.25, color=INK2, lw=0.8, ls=":")
    ax.text(0.255, y[0] + 0.5, "chance", fontsize=7.5, color=INK2)
    ax.set_yticks(y, list(FEATURE_SETS))
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Macro-F1, 5-fold CV on development data\n(random forest on hand-crafted features)")
    ax.set_title("How separable are the classes WITHOUT a CNN?", loc="left", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), frameon=False, fontsize=8, ncol=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "figD2_handcrafted_separability.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cache = OUT / "data_features.csv"
    feat = pd.read_csv(cache) if cache.exists() else None
    if feat is None or "dabc_mean" not in feat:          # older cache without the corrected stain features
        feat = collect()
        feat.to_csv(cache, index=False)
    feat = attach_split(feat)
    sep = separability(feat)
    sep.to_csv(OUT / "data_separability.csv", index=False)
    fig_distributions(feat)
    fig_separability(sep)
    print(sep.round(3).to_string(index=False))
    d = feat[feat.split != "test"]
    summ = d.groupby(["dataset", "label"])[["dab_mean", "dabc_mean", "dab_frac015", "dabc_frac015", "hem_mean",
                                            "hemc_mean", "bg_brightness", "tissue_frac", "sharpness",
                                            "hf_share"]].median().round(4)
    print(summ.to_string())
    # how much of the raw stain measures is scan brightness? (within-class Spearman with background)
    conf = pd.DataFrame([dict(dataset=ds, label=k, **{f"rho_bg_{c}": g[["bg_brightness", c]].corr("spearman").iloc[0, 1]
                                                       for c in ["dab_mean", "dabc_mean", "hem_mean", "hemc_mean"]})
                         for (ds, k), g in d.groupby(["dataset", "label"])]).round(3)
    conf.to_csv(OUT / "stain_vs_background.csv", index=False)
    print(conf.to_string(index=False))
    # label noise visible in the stain? filename-2+ BCI tiles split by their refined label
    b = d[(d.dataset == "BCI (real)") & (d.label == 2)].dropna(subset=["refined_label"])
    noise = b.groupby("refined_label")[["dab_mean", "dabc_mean", "dab_frac015"]].agg(["median", "count"]).round(4)
    print("\nBCI filename-2+ tiles by refined label:\n", noise.to_string())
    json.dump({"per_class_median": summ.reset_index().to_dict("records"),
               "bci_2plus_by_refined": noise.reset_index().to_json()},
              open(OUT / "data_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
