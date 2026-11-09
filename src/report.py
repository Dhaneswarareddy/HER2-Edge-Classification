"""Step 6 (FYP3): build results/REPORT.md + figures from everything produced so far.

Works with partial results. The written interpretation (what is confirmed, what is a hypothesis)
lives in results/findings.md and is inserted verbatim as section 0; everything else is generated
from the result files.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import paired as P
import select_models as S

FIG = C.RESULTS_DIR / "figures"
AN = C.ANALYSIS_DIR
SHARED = C.SHARED_ANALYSIS
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LOSS_COLOR = dict(ce=SERIES[0], corn=SERIES[1])
DS_COLOR = {"FYP2": SERIES[0], "BCI": SERIES[1]}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MODEL_ORDER = list(C.MODELS)
LEVELS = C.BCI_LEVELS

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.titlecolor": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False})


def fmt(x, d=3):
    return "-" if x is None or pd.isna(x) else f"{x:.{d}f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def order_key(df):
    return df.model.map({m: i for i, m in enumerate(MODEL_ORDER)})


def run_label(r):
    v = "" if r.variant in (None, "baseline") or pd.isna(r.variant) else f" + {r.variant}"
    return f"{r.model} / {r.loss.upper()}{v}"


# ------------------------------------------------------------------------- data from FYP2

def fyp2_runs():
    rows = []
    for p in sorted((C.FYP2_ROOT / "results" / "runs").glob("*/summary.json")):
        s = json.load(open(p))
        rows.append(dict(model=s["model"], loss=s["loss"], tier=s["tier"],
                         **{f"test_{k}": v for k, v in s["test"].items() if k != "confusion_matrix"},
                         **{f"val_{k}": v for k, v in s["val"].items() if k != "confusion_matrix"}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- figures

def fig_fyp2_vs_bci(cmp):
    c = cmp.sort_values("order")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(c) + 1.4))
    y = np.arange(len(c))[::-1]
    for yi, (_, r) in zip(y, c.iterrows()):
        ax.plot([r.bci_test_acc, r.fyp2_test_acc], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(c.fyp2_test_acc, y, s=60, color=DS_COLOR["FYP2"], edgecolor=SURF, lw=1.5, zorder=3,
               label="FYP2 (GAN) test accuracy")
    ax.scatter(c.bci_test_acc, y, s=60, color=DS_COLOR["BCI"], edgecolor=SURF, lw=1.5, zorder=3,
               label="BCI (real) test accuracy")
    ax.scatter(c.bci_val_acc, y, s=46, facecolor=SURF, edgecolor=DS_COLOR["BCI"], lw=1.6, zorder=3,
               label="BCI (real) validation accuracy")
    for yi, (_, r) in zip(y, c.iterrows()):
        ax.annotate(f"{(r.bci_test_acc - r.fyp2_test_acc) * 100:+.1f} pts", (r.bci_test_acc, yi),
                    xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, c.model)
    ax.set_xlim(max(0.0, min(c.bci_test_acc.min(), c.bci_val_acc.min()) - 0.12), 1.01)
    ax.set_xlabel("Test accuracy, CE loss, identical pipeline")
    ax.set_title("Same pipeline, same architectures: FYP2 vs BCI")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "figR1_fyp2_vs_bci.png", dpi=160)
    plt.close(fig)


def fig_curves(runs):
    base = runs[runs.variant == "baseline"]
    models = [m for m in MODEL_ORDER if m in set(base.model)]
    rows = int(np.ceil(len(models) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(10, 2.4 * rows), sharey=True, squeeze=False)
    lo = 1.0
    for ax, m in zip(axes.flat, models):
        for loss in ["ce", "corn"]:
            p = C.RUNS_DIR / f"{m}__{loss}" / "history.csv"
            if p.exists():
                h = pd.read_csv(p)
                lo = min(lo, h.val_qwk.min())
                ax.plot(h.epoch, h.val_qwk, color=LOSS_COLOR[loss], lw=2, label=loss.upper())
        ax.axvline(C.HEAD_EPOCHS + 0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(m, fontsize=10)
        grid(ax, "y")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    axes[0, 0].set_ylim(max(0.0, lo - 0.02), 1.0)
    axes[0, 0].legend(loc="lower right")
    fig.supxlabel("Epoch (dotted line: backbone unfrozen)", color=INK2, fontsize=10)
    fig.supylabel("Validation QWK", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figR2_learning_curves.png", dpi=160)
    plt.close(fig)


def fig_generalisation(ov):
    o = ov.assign(order=order_key(ov)).sort_values(["order", "loss", "variant"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.8, 0.4 * len(o) + 1.4))
    y = np.arange(len(o))[::-1]
    for i, r in o.iterrows():
        ax.plot([r.train_acc, r.val_acc, r.test_acc], [y[i] + 0.22, y[i], y[i] - 0.22], color=GRID, lw=1.5, zorder=1)
    for col, lab, mk, c, dy in [("train_acc", "Train (clean, no augmentation)", "s", SERIES[0], 0.22),
                                ("val_acc", "Validation", "D", SERIES[1], 0.0),
                                ("test_acc", "Test", "o", SERIES[2], -0.22)]:
        ax.scatter(o[col], y + dy, s=36, marker=mk, color=c, edgecolor=SURF, lw=1.2, zorder=2, label=lab)
    ax.set_yticks(y, [run_label(r) for _, r in o.iterrows()], fontsize=8.5)
    ax.set_xlabel("Accuracy (restored best weights)")
    ax.set_title("Generalisation: train vs validation vs test accuracy")
    ax.set_xlim(o[["train_acc", "val_acc", "test_acc"]].values.min() - 0.02, 1.005)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "figR3_generalisation_gap.png", dpi=160)
    plt.close(fig)


def fig_confusions(metrics, runs_shown):
    runs_shown = [r for r in runs_shown if r in metrics]
    if not runs_shown:
        return
    rows = int(np.ceil(len(runs_shown) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3.0 * rows), squeeze=False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", [SURF] + BLUES)
    for ax, run in zip(axes.flat, runs_shown):
        cm = np.array(metrics[run]["test"]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color=SURF if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(4), LEVELS, fontsize=8)
        ax.set_yticks(range(4), LEVELS, fontsize=8)
        ax.set_title(run.replace("__", " / "), fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.flat[len(runs_shown):]:
        ax.axis("off")
    fig.supxlabel("Predicted (shading = row-normalised recall)", color=INK2, fontsize=10)
    fig.supylabel("True (filename label)", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figR4_confusion_matrices_test.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------- sections

def section_data():
    split = open(C.SPLIT_REPORT).read()
    tbl = split[split.index("       Score_0"):split.index("Leakage check")].strip("\n")
    leak = split[split.index("Leakage check"):split.index("Refined")].strip()
    ref = split[split.index("Refined"):split.index("{")].strip()
    return ["## 1. Data", "",
            "- **Dataset.** BCI (Breast Cancer Immunohistochemical image generation), Liu et al., 2022 "
            "(arXiv:2204.11425). It has 4,873 paired H&E/IHC 1024×1024 tiles, 3,896 official train and 977 official test. "
            "The licence is non-commercial research only; cite the BCI paper. The raw files in "
            "`FYP/data/raw/BCI_dataset` were only read.",
            "- **Image type: IHC.** FYP2's images are HER2 IHC (DAB-stained membrane) patches. Its pipeline, augmentation "
            "and report are all built around DAB stain intensity. So FYP3 uses `BCI_dataset/IHC`; H&E is not used.",
            "- **Labels: the BCI filename convention** `NNNNN_{train|test}_{0|1+|2+|3+}.png`. Per the BCI README, this is the "
            "score of the whole slide the tile was cut from, not of the tile itself. A second, patch-level "
            "label exists for 72% of tiles (`BCI_metadata_pairs_refined.csv`, method undocumented). It is used for "
            "analysis only and never for training or model selection.",
            "- **Split (main): grouped, leakage-controlled.** This follows FYP2's recipe (`make_split.py`): StratifiedGroupKFold "
            "70/15/15 over all 4,873 tiles. The groups are *slide fragments*: tiles cut side by side from one slide, "
            "reconstructed from the continuity of their borders (`analysis_slides.py`, section 1b). FYP2's pHash groups are "
            "not used for BCI because they link background-dominated tiles from different slides (section 1b).",
            "- **Split (reference): the official BCI split.** This is the official train/test split, with validation drawn from "
            "15% of official train. It is kept for comparison with published BCI results, but it leaks (section 1b). "
            "Two models were trained on it (`results/runs_official/`).",
            "- **Resolution.** Tiles are downsampled once to 256×256 (Lanczos), so the networks see FYP2's geometry: "
            "256 px source → 224 px input.", "",
            "```", tbl, "", leak, "", ref, "```", ""]


def section_leakage():
    p = SHARED / "slide_fragments.json"
    if not p.exists():
        return []
    f = json.load(open(p))
    L = ["## 1b. Slide leakage in the official BCI split", "",
         "**Method.** BCI has no slide IDs, but tiles cut side by side from one slide share a continuous border. "
         "For every tile, the tile whose border best continues its right/bottom edge is found. A link is accepted only "
         "if it is mutual, unambiguous (ratio test) and as smooth as the tile's own adjacent pixel columns "
         "(`analysis_slides.py`).",
         f"- **Validation of the method.** {f['accepted_links']:,} links were accepted. "
         f"{f['accepted_links_same_label'] * 100:.2f}% of them join tiles with the same slide label, against "
         f"{f['random_pair_same_label'] * 100:.0f}% expected for random pairs. {f['fragments_label_pure'] * 100:.1f}% of the "
         f"{f['n_fragments']} multi-tile fragments are label-pure, and the largest has {f['largest_fragment']} tiles. "
         "The links therefore join neighbouring tiles of the same slide.",
         f"- **Confirmed leakage.** {f['official_train_vs_test']['tiles_of_b_in_shared']} of "
         f"{f['official_train_vs_test']['tiles_of_b']} official test tiles belong to slide fragments that also contain "
         "official training tiles.",
         "- **pHash grouping (FYP2's rule) does not transfer to BCI.** 39 of 53 pHash groups mix slide labels, even 6-bit "
         "pairs agree on label only 50% of the time, and linked tiles are mostly background. These are look-alike "
         "backgrounds, not duplicates.", "", "![](figures/figD8_edge_matches.png)", ""]
    pe, ps, pp, ph = (SHARED / f"leakage_{k}.csv" for k in ["exposure", "by_size", "paired", "headline"])
    if ph.exists():
        hd, ex, sz, pa = (pd.read_csv(p) for p in (ph, pe, ps, pp))
        ex, pa = ex[ex.split == "test"], pa[pa.subset != "unlinked"]
        L += ["**Effect on accuracy** (`analysis_leakage.py`). MobileNetV3Small and MobileNetV2 were trained on both splits. "
              "All 95% CIs come from a cluster bootstrap over slide fragments, because tiles from one fragment are correlated.", "",
              md_table(pd.DataFrame({"Run": hd.run.str.replace("__", " / "), "Split design": hd.split_design,
                                     "Held-out set": hd.split, "Tiles": hd.n,
                                     "Accuracy [95% CI]": ci_cells(hd.accuracy, hd.accuracy_ci_lo, hd.accuracy_ci_hi),
                                     "Macro-F1 [95% CI]": ci_cells(hd.macro_f1, hd.macro_f1_ci_lo, hd.macro_f1_ci_hi),
                                     "QWK [95% CI]": ci_cells(hd.qwk, hd.qwk_ci_lo, hd.qwk_ci_hi)})), "",
              "These headline numbers are not comparable across split designs: each scores different tiles, and difficulty "
              "varies strongly between slide fragments (section 5c). The fair test scores both models on the **same tiles**: "
              "official test tiles that the grouped split also holds out. Neither model trained on them; only the "
              "official-split model trained on their slide neighbours. For tiles in grouped *val*, the grouped model "
              "early-stopped on them, which favours it, so that difference is conservative.", "",
              md_table(pd.DataFrame({"Run": pa.run.str.replace("__", " / "), "Tiles": pa.tile_set,
                                     "Subset": pa.subset, "n (fragments)": [f"{a} ({b})" for a, b in zip(pa.n, pa.n_fragments)],
                                     "Acc, official-split model": pa.acc_official_model.map(fmt),
                                     "Acc, grouped-split model": pa.acc_grouped_model.map(fmt),
                                     "Difference [95% CI]": [f"{d:+.3f} [{a:+.3f}, {b:+.3f}]" for d, a, b in
                                                             zip(pa["diff"], pa.diff_ci_lo, pa.diff_ci_hi)]})), "",
              "**The simple exposure comparison is confounded.** The official-split model scores much better on official "
              "test tiles whose fragment was in training:", "",
              md_table(pd.DataFrame({"Run": ex.run.str.replace("__", " / "), "Exposure": ex.exposure, "n": ex.n,
                                     "Accuracy": ex.accuracy.map(fmt), "Macro-F1": ex.macro_f1.map(fmt)})), "",
              "Only multi-tile fragments can be \"seen\", however, and large fragments are easier for every model. The "
              "grouped-split model, which saw none of these fragments, shows the same rise with fragment size:", "",
              md_table(size_table(sz)), "", "![](figures/figD9_leakage.png)", ""]
    return L


def ci_cells(v, lo, hi):
    return [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(v, lo, hi)]


def size_table(sz):
    sz = sz.assign(col=sz.split_design.str.cat(sz.split, sep=" model, ").str.replace("official model, test",
                                                                                     "official model, official test")
                   .str.replace("grouped model, ", "grouped model, grouped "))
    t = sz.pivot_table(index=["run", "size_bin"], columns="col", values="accuracy", sort=False).apply(lambda c: c.map(fmt))
    seen = sz[sz.split_design == "official"].set_index(["run", "size_bin"]).seen_fragment_share
    t.insert(0, "% official test seen", seen.reindex(t.index).map(lambda v: f"{v:.0%}"))
    t = t.reset_index()
    return t.assign(run=t.run.str.replace("__", " / ")).rename(columns={"run": "Run", "size_bin": "Fragment size"})


def section_setup():
    return ["## 2. Training setup", "",
            "**Identical to FYP2:** the 9 ImageNet backbones, 224×224 input with preprocessing in the graph, "
            f"stage 1 frozen ({C.HEAD_EPOCHS} epochs at {C.HEAD_LR}), stage 2 full fine-tuning (≤ {C.FT_EPOCHS} epochs, "
            f"Adam {C.FT_LR}, warm-up and cosine), BatchNorm frozen, dropout {C.DROPOUT}, early stopping on validation QWK "
            f"(patience {C.PATIENCE}) with best weights restored, the same augmentation, seed {C.SEED}, CE and CORN losses.", "",
            "**Deliberate differences, all forced by the data or the scope:**", "",
            "1. **Training images are streamed from disk** instead of being held in memory. The training split is twice FYP2's "
            "and the Mac has 8 GB. The pixels are identical.",
            "2. **Leakage groups are slide fragments, not pHash twins.** The split recipe is FYP2's; only the grouping unit "
            "differs, because the leakage mechanism differs (section 1b).",
            "3. **CE for all 9 models; CORN only for the best CE model per tier (by validation QWK).** "
            "In FYP2, CE vs CORN was a tie (3 vs 3 wins of 9), so repeating all 18 runs (~35 h here) was not justified. "
            "One paired comparison per tier still answers whether the ordinal loss helps on noisy real labels.",
            "4. **No quantisation or deployment** (out of scope). Selection ties go to the smaller model "
            "(FYP2 used quantised latency).",
            "5. **Improvement experiments** (section 6) run on top of the baseline, never instead of it.", ""]


def section_results(runs, sel, fyp2):
    L = ["## 3. Results on BCI", ""]
    t = runs.assign(order=order_key(runs)).sort_values(["order", "loss", "variant"])
    L += [md_table(pd.DataFrame({
        "Run": [run_label(r) for _, r in t.iterrows()], "Tier": t.tier, "Params (M)": (t.params / 1e6).round(2),
        "Input": t.image_size, "Best ep. / run": t.best_epoch.astype(str) + " / " + t.epochs_run.astype(str),
        "Train (min)": t.train_min.round(0).astype(int),
        "Val acc": t.val_accuracy.map(fmt), "Val F1": t.val_macro_f1.map(fmt), "Val QWK": t.val_qwk.map(fmt),
        "Test acc": t.test_accuracy.map(fmt), "Test F1": t.test_macro_f1.map(fmt), "Test QWK": t.test_qwk.map(fmt),
        "Test bal. acc": t.test_balanced_accuracy.map(fmt),
        "Test recall 0/1+/2+/3+": [f"{r.test_recall_0:.2f} / {r.test_recall_1:.2f} / {r.test_recall_2:.2f} / "
                                   f"{r.test_recall_3:.2f}" for _, r in t.iterrows()]})), "",
          "![](figures/figR2_learning_curves.png)", ""]
    if sel:
        cv = sel["ce_vs_corn"]
        if cv["models_compared"]:
            L += [f"**CE vs CORN** ({cv['models_compared']} paired models): CORN higher validation QWK on "
                  f"{cv['corn_better']}, CE on {cv['ce_better']}. Loss chosen per model: " +
                  ", ".join(f"{m}: {l.upper()}" for m, l in sel["chosen_loss_per_model"].items()) + ".", ""]
            rows = []
            for m in [m for m in MODEL_ORDER if ((runs.model == m) & (runs.loss == "corn")).any()]:
                dv = P.compare(f"{m}__corn", f"{m}__ce", "val", ("qwk", "macro_f1"))
                dt = P.compare(f"{m}__corn", f"{m}__ce", "test", ("accuracy", "qwk"))
                rows.append({"Model": m, "Δ val QWK [95% CI]": P.cell(dv["qwk"]),
                             "Δ val macro-F1 [95% CI]": P.cell(dv["macro_f1"]),
                             "Δ test acc [95% CI]": P.cell(dt["accuracy"]), "Δ test QWK [95% CI]": P.cell(dt["qwk"])})
            L += ["Paired difference CORN − CE on the same tiles (fragment-cluster CI):", "",
                  md_table(pd.DataFrame(rows)), ""]
        rows = []
        for key, lab in [("tier_winners_baseline", "FYP2 pipeline (baseline)"),
                         ("tier_best_with_improvements", "Best incl. improvements")]:
            for tier, w in sel.get(key, {}).items():
                rest = runs[(runs.tier == tier) & (runs.run != w["run"])]
                if key == "tier_winners_baseline":
                    rest = rest[rest.variant == "baseline"]
                ru = rest.sort_values("val_qwk").iloc[-1].run if len(rest) else None
                rows.append({"Selection": lab, "Tier": tier, "Run": w["run"].replace("__", " / "),
                             "Val QWK": fmt(w["val_qwk"]), "Val F1": fmt(w["val_macro_f1"]),
                             "Test acc": fmt(w["test_accuracy"]), "Test F1": fmt(w["test_macro_f1"]),
                             "Test QWK": fmt(w["test_qwk"]), "Runner-up": ru.replace("__", " / ") if ru else "-",
                             "Δ val QWK vs runner-up [95% CI]":
                                 P.cell(P.compare(w["run"], ru, "val", ("qwk",))["qwk"]) if ru else "-"})
        L += ["**Best model per tier** (chosen on validation QWK, see `select_models.py`; runs within 0.005 count as "
              "equal and the smaller one wins). The last column is the paired validation difference to the next-best "
              "run of the tier; a CI that includes 0 means validation cannot separate the two:", "",
              md_table(pd.DataFrame(rows)), ""]
    # FYP2 comparison
    b = runs[(runs.variant == "baseline") & (runs.loss == "ce")].set_index("model")
    f = fyp2[fyp2.loss == "ce"].set_index("model")
    common = [m for m in MODEL_ORDER if m in b.index and m in f.index]
    if common:
        cmp = pd.DataFrame({"model": common, "order": range(len(common)),
                            "fyp2_test_acc": [f.loc[m, "test_accuracy"] for m in common],
                            "bci_test_acc": [b.loc[m, "test_accuracy"] for m in common],
                            "bci_val_acc": [b.loc[m, "val_accuracy"] for m in common]})
        fig_fyp2_vs_bci(cmp)
        mpath = C.RESULTS_DIR / f"metrics_all{C.SUFFIX}.csv"
        ci = {}
        if mpath.exists():
            ma = pd.read_csv(mpath)
            ma = ma[(ma.split == "test") & (ma.loss == "ce") & ma.variant.isna()].set_index("model")
            ci = {m: f" [{ma.loc[m, 'accuracy_ci_lo']:.3f}, {ma.loc[m, 'accuracy_ci_hi']:.3f}]" for m in common if m in ma.index}
        L += ["## 4. FYP2 vs BCI, same pipeline (CE)", "",
              "FYP2's test set is 360 GAN images, which every model scores almost perfectly. For BCI, both held-out sets are "
              "shown, because they differ by several points for the same model (section 5c). The 95% CI is a "
              "fragment-cluster bootstrap.", "", md_table(pd.DataFrame({
            "Model": common,
            "FYP2 test acc": [fmt(f.loc[m, "test_accuracy"]) for m in common],
            "BCI val acc": [fmt(b.loc[m, "val_accuracy"]) for m in common],
            "BCI test acc [95% CI]": [fmt(b.loc[m, "test_accuracy"]) + ci.get(m, "") for m in common],
            "Δ test acc": [f"{(b.loc[m, 'test_accuracy'] - f.loc[m, 'test_accuracy']) * 100:+.1f} pts" for m in common],
            "FYP2 macro-F1": [fmt(f.loc[m, "test_macro_f1"]) for m in common],
            "BCI macro-F1": [fmt(b.loc[m, "test_macro_f1"]) for m in common],
            "FYP2 QWK": [fmt(f.loc[m, "test_qwk"]) for m in common],
            "BCI QWK": [fmt(b.loc[m, "test_qwk"]) for m in common],
            "BCI recall 0": [fmt(b.loc[m, "test_recall_0"], 2) for m in common],
            "BCI recall 2+": [fmt(b.loc[m, "test_recall_2"], 2) for m in common]})), "",
              "![](figures/figR1_fyp2_vs_bci.png)", ""]
    return L


def section_metrics():
    mpath, opath = C.RESULTS_DIR / f"metrics_all{C.SUFFIX}.csv", C.RESULTS_DIR / f"overfitting{C.SUFFIX}.csv"
    if not (mpath.exists() and opath.exists()):
        return ["## 5. Full metrics and overfitting check", "",
                "*Pending: run `python src/evaluate_all.py`.*", ""] + section_heldout()
    import evaluate_all as E
    ma, ov = pd.read_csv(mpath), pd.read_csv(opath)
    ma["variant"] = ma.variant.fillna("baseline")
    ov["variant"] = ov.variant.fillna("baseline")
    metrics = {p.stem: json.load(open(p)) for p in E.METRICS_DIR.glob("*.json")}
    fig_generalisation(ov)
    sp = C.RESULTS_DIR / f"selection{C.SUFFIX}.json"
    sel = json.load(open(sp)) if sp.exists() else {}
    shown = [w["run"] for w in sel.get("tier_winners_baseline", {}).values()] + \
            [w["run"] for w in sel.get("tier_best_with_improvements", {}).values()]
    fig_confusions(metrics, list(dict.fromkeys(shown)))
    L = ["## 5. Full metrics and overfitting check", "",
         "These are the restored best weights. 95% confidence intervals come from 2,000 cluster-bootstrap resamples of "
         "whole slide fragments (FYP2 resampled images; BCI tiles of one fragment are correlated, so a tile-level bootstrap "
         "would be too narrow). Per-class precision, recall, F1 and specificity and the confusion matrices are in "
         f"`results/metrics_all{C.SUFFIX}.csv` and `results/metrics{C.SUFFIX}/<run>.json`.", ""]
    t = ma[ma.split == "test"].assign(order=order_key(ma[ma.split == "test"])).sort_values(["order", "loss", "variant"])
    ci = lambda lo, hi: [f"[{a:.3f}, {b:.3f}]" for a, b in zip(lo, hi)]
    n_test = int(np.array(json.loads(t.confusion_matrix.iloc[0])).sum())
    L += [f"**Test split ({n_test} tiles)**", "", md_table(pd.DataFrame({
        "Run": [run_label(r) for _, r in t.iterrows()], "Acc": t.accuracy.map(fmt),
        "Acc 95% CI": ci(t.accuracy_ci_lo, t.accuracy_ci_hi), "Bal acc": t.balanced_accuracy.map(fmt),
        "Macro-F1": t.macro_f1.map(fmt), "F1 95% CI": ci(t.macro_f1_ci_lo, t.macro_f1_ci_hi),
        "QWK": t.qwk.map(fmt), "MCC": t.mcc.map(fmt), "MAE": t.mae.map(fmt), "Off-by-2+": t.off_by_2plus.map(fmt),
        "Clinical acc": t.clinical_accuracy.map(fmt), "AUC": t.macro_auc.map(fmt), "Log-loss": t.log_loss.map(fmt),
        "ECE": t.ece.map(fmt)})), "",
          "**Per-class test precision / recall (0, 1+, 2+, 3+)**", "", md_table(pd.DataFrame({
              "Run": [run_label(r) for _, r in t.iterrows()],
              **{f"P {lv}": t[f"precision_{k}"].map(lambda v: fmt(v, 2)) for k, lv in enumerate(LEVELS)},
              **{f"R {lv}": t[f"recall_{k}"].map(lambda v: fmt(v, 2)) for k, lv in enumerate(LEVELS)}})), "",
          "![](figures/figR4_confusion_matrices_test.png)", ""]
    o = ov.assign(order=order_key(ov)).sort_values(["order", "loss", "variant"])
    counts = lambda col: ", ".join(f"{v}: {n} run(s)" for v, n in o[col].value_counts().items())
    L += ["### 5b. Overfitting check (same rules as FYP2)", "",
          f"Each run's best weights are re-evaluated on clean training tiles (no augmentation). A run counts as "
          f"**not overfitted** only if all three hold: train − test accuracy ≤ {E.GAP_T:.2f}; |validation − test accuracy| ≤ "
          f"{E.VALTEST_T:.2f}; test − train log-loss ≤ {E.NLL_T:.2f}. Failing one check gives \"mild\"; failing two "
          "or more gives \"overfit\".", "",
          "These thresholds were set for FYP2's near-perfect accuracy, and on BCI they sit below the sampling noise of one "
          "held-out set. The **noise-aware verdict** therefore counts the accuracy-gap check as failed only if train "
          "accuracy is also above the test CI. It counts the validation check as failed only if validation is *better* than "
          "test, with a CI excluding zero, since overfitting the selection makes validation look better, not worse. The "
          "log-loss check is unchanged.", "",
          f"**Result, FYP2 rule:** {counts('verdict')}. **Noise-aware:** {counts('verdict_noise_aware')}.", "",
          md_table(pd.DataFrame({
              "Run": [run_label(r) for _, r in o.iterrows()],
              "Best ep. / run": o.best_epoch.astype(str) + " / " + o.epochs_run.astype(str),
              "Train acc": o.train_acc.map(fmt), "Val acc": o.val_acc.map(fmt), "Test acc": o.test_acc.map(fmt),
              "Test acc 95% CI": ci(o.test_acc_ci_lo, o.test_acc_ci_hi), "Train−test": o.gap_train_test_acc.map(fmt),
              "Val−test [95% CI]": [f"{d:+.3f} [{a:+.3f}, {b:+.3f}]" for d, a, b in
                                    zip(o.val_minus_test_acc, o.val_minus_test_ci_lo, o.val_minus_test_ci_hi)],
              "Train LL": o.train_log_loss.map(fmt), "Test LL": o.test_log_loss.map(fmt), "LL gap": o.nll_gap.map(fmt),
              "Val-loss rise after best": o.val_loss_rise_after_best.map(fmt), "Test ECE": o.test_ece.map(fmt),
              "Verdict (FYP2 rule)": o.verdict, "Verdict (noise-aware)": o.verdict_noise_aware})), "",
          "![](figures/figR3_generalisation_gap.png)", ""]
    return L + section_heldout()


def section_heldout():
    pc, pf = SHARED / "heldout_concentration.csv", SHARED / "heldout_fragments.csv"
    if C.SPLIT != "grouped" or not pc.exists():
        return []
    hc, hf = pd.read_csv(pc), pd.read_csv(pf)
    hc = hc.assign(order=hc.run.str.split("__").str[0].map({m: i for i, m in enumerate(MODEL_ORDER)}))
    hc = hc.sort_values(["order", "run", "split"])
    hard = (hf[hf.n >= 5].groupby(["split", "fragment"])
            .agg(label=("label", "first"), n=("n", "first"), runs=("run", "nunique"), mean_acc=("accuracy", "mean"),
                 errors_per_run=("errors", "mean"), refined_agree=("refined_agree", "first"),
                 dab_pct=("dab_mean_pct_in_class", "first"), dabc_pct=("dabc_mean_pct_in_class", "first"),
                 sharp_pct=("sharpness_pct_in_class", "first"), bg_pct=("bg_brightness_pct_in_class", "first"))
            .sort_values("errors_per_run", ascending=False).head(8).reset_index())
    return ["### 5c. How precise is one held-out split?", "",
            "Errors cluster in a few large slide fragments, so one fragment can move a split's accuracy by several points. "
            "\"Top-3\" is the three fragments with the most errors in that split for that run "
            "(`analysis_leakage.py`, `results/analysis/heldout_*.csv`).", "",
            md_table(pd.DataFrame({"Run": hc.run.str.replace("__", " / "), "Split": hc.split, "Tiles": hc.n,
                                   "Accuracy": hc.accuracy.map(fmt), "Errors": hc.errors,
                                   "Top-3 fragments (tiles)": [f"{f} ({n})" for f, n in zip(hc.top3_fragments, hc.top3_tiles)],
                                   "Share of errors in top-3": hc.top3_error_share.map(lambda v: f"{v:.0%}"),
                                   "Accuracy without top-3": hc.accuracy_without_top3.map(fmt)})), "",
            "**Hardest fragments** (≥ 5 held-out tiles, averaged over runs). Percentiles place the fragment's median tile "
            "within its own class in the grouped training split. *Raw* DAB is measured against 255, so a dark grey scan "
            "raises it; *corrected* DAB is measured against the tile's own background (section 6a).", "",
            md_table(pd.DataFrame({"Split": hard.split, "Fragment": hard.fragment,
                                   "Slide label": hard.label.map(lambda k: LEVELS[k]), "Tiles": hard.n, "Runs": hard.runs,
                                   "Mean accuracy": hard.mean_acc.map(fmt), "Errors per run": hard.errors_per_run.round(1),
                                   "Patch labels agreeing": hard.refined_agree.map(lambda v: "-" if pd.isna(v) else f"{v:.0%}"),
                                   "DAB pct, raw": hard.dab_pct.map(lambda v: f"{v:.0%}"),
                                   "DAB pct, corrected": hard.dabc_pct.map(lambda v: f"{v:.0%}"),
                                   "Sharpness pct": hard.sharp_pct.map(lambda v: f"{v:.0%}"),
                                   "Background pct": hard.bg_pct.map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")})), ""]


def section_diagnosis():
    L = ["## 6. Diagnosis: evidence", "",
         "The interpretation, with each finding marked confirmed or hypothesis, is in section 0. This section holds the evidence.", ""]
    p = SHARED / "data_separability.csv"
    if p.exists():
        s = pd.read_csv(p)
        s = s[s.classifier == "random forest"]
        n = s.groupby("dataset").n.first()
        piv = s.pivot_table(index="features", columns=["dataset", "cv"], values="macro_f1", sort=False).round(3)
        piv.columns = [f"{d}, {cv} CV" for d, cv in piv.columns]
        L += ["### 6a. Model-free: how separable are the classes from simple image statistics?", "",
              f"5-fold CV on the development splits only (FYP2 train+val, {n['FYP2 (GAN)']:,} images; BCI train+val, "
              f"{n['BCI (real)']:,} tiles). Random forest on hand-crafted features (`analysis_data.py`); macro-F1, chance 0.25. "
              "*Grouped* folds keep leakage groups whole (FYP2 pHash twins, BCI slide fragments), as the main split does. "
              "*Image-level* folds (BCI only) ignore fragments, so the gap between the two measures how much slide-level "
              "cues inflate a model-free classifier.", "",
              md_table(piv.reset_index().rename(columns={"features": "Feature set (macro-F1)"})), "",
              "![](figures/figD1_stain_distributions.png)", "", "![](figures/figD2_handcrafted_separability.png)", ""]
    p = AN / "label_noise.csv"
    if p.exists():
        nz = pd.read_csv(p)
        nz["order"] = nz.run.str.split("__").str[0].map({m: i for i, m in enumerate(MODEL_ORDER)})
        nz = nz.sort_values(["split", "order", "run"])
        L += ["### 6b. Label noise: slide-level vs patch-level labels", "",
              "*Clean* tiles have refined = filename label. *Noisy* tiles disagree. \"Errors matching refined\" is the share of "
              "the model's errors (against the filename label) where the prediction equals the patch-level label.", "",
              md_table(pd.DataFrame({
                  "Run": nz.run.str.replace("__", " / "), "Split": nz.split, "Acc (filename)": nz.accuracy.map(fmt),
                  "Noisy tiles": nz.n_noisy.astype(str) + " / " + nz.n_with_refined.astype(str),
                  "Acc clean": nz.acc_clean.map(fmt), "Acc noisy vs filename": nz.acc_noisy_vs_filename.map(fmt),
                  "Acc noisy vs refined": nz.acc_noisy_vs_refined.map(fmt),
                  "Errors matching refined": nz.share_errors_matching_refined.map(fmt),
                  "Errors on noisy tiles": nz.share_errors_on_noisy_tiles.map(fmt)})), "",
              f"![](figures/figD3_label_noise{C.SUFFIX}.png)", ""]
    p = AN / "scan_type.csv"
    if p.exists():
        sc = pd.read_csv(p)
        thr = json.load(open(AN / "scan_type_threshold.json"))["otsu_threshold_train"]
        piv = sc.pivot_table(index=["run", "split"], columns="scan", values=["accuracy", "macro_f1", "recall_3"])
        piv.columns = [f"{a} {b}" for a, b in piv.columns]
        piv = piv.reset_index()
        L += [f"### 6c. Scan type (background brightness ≥ {thr:.0f} = bright)", "",
              md_table(piv.assign(run=piv.run.str.replace("__", " / ")).round(3)), ""]
        p = AN / "scan_by_class.csv"
        if p.exists():
            info = json.load(open(AN / "scan_type_threshold.json"))
            grey = info.get("grey_share_by_class", {})
            dabc = info.get("dabc_median_by_class_scan", {})
            cl = pd.read_csv(p)
            cl = cl.assign(order=cl.run.str.split("__").str[0].map({m: i for i, m in enumerate(MODEL_ORDER)}))
            cl = cl.sort_values(["order", "run", "label"])
            t = cl.pivot_table(index=["run", "label"], columns="scan", values=["n", "accuracy", "mean_bias"], sort=False)
            t = t.reset_index()
            L += ["**By class.** Grey scans are more common in higher classes, so background brightness could act as a "
                  "shortcut. A shortcut would over-score grey tiles and under-score bright tiles of the same class "
                  "(positive vs negative mean bias, expected score − label; val+test pooled). Corrected DAB differs between "
                  "scan types within a class too, so this table alone cannot separate the two explanations (see 6e and 6f).", "",
                  md_table(pd.DataFrame({
                      "Run": t.run.str.replace("__", " / "), "Class": t.label.map(lambda k: LEVELS[k]),
                      "Grey share (all tiles)": t.label.map(lambda k: f"{grey.get(str(k), grey.get(k, np.nan)):.0%}"),
                      "Corrected DAB, bright / grey": t.label.map(
                          lambda k: f"{dabc.get(f'{k}/bright', np.nan):.3f} / {dabc.get(f'{k}/grey', np.nan):.3f}"),
                      "n bright / grey": [f"{int(a)} / {int(b)}" for a, b in zip(t[("n", "bright")], t[("n", "grey")])],
                      "Acc bright": t[("accuracy", "bright")].map(fmt), "Acc grey": t[("accuracy", "grey")].map(fmt),
                      "Bias bright": t[("mean_bias", "bright")].map(lambda v: f"{v:+.2f}"),
                      "Bias grey": t[("mean_bias", "grey")].map(lambda v: f"{v:+.2f}")})), ""]
    p = AN / "near_dup.csv"
    if p.exists():
        d = pd.read_csv(p)
        L += ["### 6d. Test without the 15 near-duplicates of training tiles", "",
              md_table(pd.DataFrame({"Run": d.run.str.replace("__", " / "), "Test acc (all)": d.all_accuracy.map(fmt),
                                     "Test acc (no near-dups)": d.no_dup_accuracy.map(fmt),
                                     "Macro-F1 (all)": d.all_macro_f1.map(fmt),
                                     "Macro-F1 (no near-dups)": d.no_dup_macro_f1.map(fmt)})), ""]
    p = AN / "prediction_drivers.csv"
    if p.exists():
        d = pd.read_csv(p)
        r2 = [c for c in d.columns if c.startswith("r2_")]
        L += ["### 6e. Which simple image cues do the predictions track?", "",
              "This is the cross-validated R² of a linear model predicting each CE model's expected score on the test split "
              "from one group of hand-crafted statistics. The *true label* row is the reference: how much of the label those "
              "same statistics explain. A model row well above the label row means the model's output follows that cue "
              "more closely than the labels do.", "",
              md_table(pd.DataFrame({"Dataset": d.dataset, "Target": d.target,
                                     **{c[3:]: d[c].map(fmt) for c in r2},
                                     "ρ(DAB raw)": d.spearman_dab.map(lambda v: fmt(v, 2)),
                                     "ρ(DAB corrected)": d.get("spearman_dabc", pd.Series(dtype=float)).map(lambda v: fmt(v, 2)),
                                     "ρ(background)": d.spearman_bg.map(lambda v: fmt(v, 2)),
                                     "ρ(sharpness)": d.spearman_sharp.map(lambda v: fmt(v, 2))})), ""]
    p = AN / "error_drivers.csv"
    if p.exists():
        e = pd.read_csv(p)
        e = e.assign(cell=[f"{c:+.3f} [{a:+.3f}, {b:+.3f}]" for c, a, b in zip(e.coef, e.ci_lo, e.ci_hi)])
        t = e.pivot(index="run", columns="feature", values="cell").reset_index()
        t = t.assign(order=t.run.str.split("__").str[0].map({m: i for i, m in enumerate(MODEL_ORDER)})).sort_values("order")
        L += ["**What drives the errors?** Linear regression of each run's error (expected score − label, val+test pooled) "
              "on standardised tile statistics with class fixed effects. Coefficients are in score units per standard "
              "deviation, with 95% fragment-cluster bootstrap CIs (`analysis_labels.py`). A positive DAB coefficient means "
              "the model scores a tile higher than its slide label when the tile itself carries more stain.", "",
              md_table(t[["run"] + [c for c in ["dabc_mean", "bg_brightness", "hemc_mean", "tissue_frac", "sharpness"]
                                    if c in t]].rename(columns={
                  "run": "Run", "dabc_mean": "DAB (corrected)", "bg_brightness": "Background brightness",
                  "hemc_mean": "Haematoxylin (corrected)", "tissue_frac": "Tissue fraction",
                  "sharpness": "Sharpness (log)"}).assign(Run=lambda x: x.Run.str.replace("__", " / "))), ""]
    p = AN / "model_summary.csv"
    if p.exists():
        ms = pd.read_csv(p)
        L += ["### 6f. What the networks learned (`analysis_models.py`)", "",
              "- **Frozen ImageNet probe:** a linear classifier on generic EfficientNetB0 features, measuring dataset separability with no training.",
              "- **kNN / silhouette / scatter ratio:** class separation in each trained model's own embedding (test split).",
              "- **Cross-domain:** the model trained on one dataset, scored on the other's test split. Balanced accuracy "
              "(mean recall over the four scores) is shown because BCI's test set is imbalanced: chance is 0.25 for both "
              "datasets, whereas plain accuracy on BCI is 0.43 for a model that always answers 2+.",
              "- **Grad-CAM DAB ratio:** attention-weighted DAB divided by mean DAB, with DAB measured against each tile's "
              "own background. Above 1 means the model looks at stained regions.",
              "- **Background ratio:** attention share on background divided by background area. Below 1 means it avoids background.", "",
              md_table(pd.DataFrame({
                  "Network": ms.arch, "Trained/eval on": ms.dataset, "Test acc": ms.test_accuracy.map(fmt),
                  "Test bal. acc": ms.get("test_balanced_accuracy", pd.Series(dtype=float)).map(fmt),
                  "Test QWK": ms.test_qwk.map(fmt), "kNN acc": ms.get("knn_accuracy", pd.Series(dtype=float)).map(fmt),
                  "Silhouette": ms.silhouette.map(fmt), "Scatter ratio": ms.scatter_ratio.map(fmt),
                  "Cross-domain bal. acc": ms.get("cross_domain_balanced_accuracy", pd.Series(dtype=float)).map(fmt),
                  "Cross-domain QWK": ms.get("cross_domain_qwk", pd.Series(dtype=float)).map(fmt),
                  "CAM DAB ratio": ms.get("cam_dab_ratio", pd.Series(dtype=float)).map(fmt),
                  "CAM bg ratio": ms.get("cam_bg_ratio", pd.Series(dtype=float)).map(fmt)})), "",
              "![](figures/figD7_cross_domain.png)", "", "![](figures/figD5_cka.png)", ""]
        cams = [FIG / f"figD6_gradcam_{w['model']}.png"
                for w in (json.load(open(C.RESULTS_DIR / "selection.json"))["tier_winners_baseline"].values()
                          if (C.RESULTS_DIR / "selection.json").exists() else [])]
        for f in sorted(FIG.glob("figD4_tsne_*.png")) + [f for f in cams if f.exists()]:
            L += [f"![](figures/{f.name})", ""]
        L += ["Grad-CAM examples for the other six networks are in `results/figures/figD6_gradcam_<network>.png`.", ""]
    p = AN / "cka.csv"
    if p.exists():
        ck = pd.read_csv(p)
        ck = ck[ck.on == "BCI"].pivot(index="arch", columns="depth_index")
        ck = ck.loc[[m for m in MODEL_ORDER if m in ck.index]]
        last = ck.columns.get_level_values(1).max()
        L += ["**Linear CKA on BCI test images** at five evenly spaced depths (1 = early, 5 = last conv block); "
              "every value is in `results/analysis/cka.csv`:", "",
              md_table(pd.DataFrame({
                  "Network": ck.index,
                  "FYP2-net vs BCI-net, depth 1 / 2 / 3 / 4 / 5": [" / ".join(f"{v:.2f}" for v in r)
                                                                  for r in ck["fyp2_vs_bci"].values],
                  "FYP2-net vs ImageNet start, depth 5": ck[("fyp2_vs_init", last)].map(lambda v: f"{v:.2f}").values,
                  "BCI-net vs ImageNet start, depth 5": ck[("bci_vs_init", last)].map(lambda v: f"{v:.2f}").values})), ""]
    L += section_stain()
    L += section_haem_cue()
    return L


# Pre-registered decision rules for the gated variants (variants.py): variant -> (perturbation, the two scales)
GATES = {"illum": ("illumination", (0.85, 1.15)), "hed": ("haematoxylin", (0.8, 1.25))}
GATE_THRESHOLD = 0.10


def gate_values():
    """For each gated variant: max over the two directions of the median over BCI-trained networks of
    |mean shift of the expected score|, as the rules in variants.py define it."""
    p = AN / "stain_sensitivity.csv"
    if not p.exists():
        return {}
    st = pd.read_csv(p)
    st = st[st.dataset == "BCI"]
    return {v: max(st[(st.stain == stain) & np.isclose(st.scale, s)].mean_delta_score.abs().median() for s in scales)
            for v, (stain, scales) in GATES.items()}


def section_stain():
    p = AN / "stain_sensitivity.csv"
    if not p.exists():
        return []
    st = pd.read_csv(p)
    rows = []
    for stain in ["DAB", "haematoxylin", "all stains", "illumination"]:
        for s in sorted(st[st.stain == stain].scale.unique()):
            cell = {}
            for ds in ["FYP2", "BCI"]:
                v = st[(st.stain == stain) & np.isclose(st.scale, s) & (st.dataset == ds)].mean_delta_score
                cell[ds] = f"{v.median():+.3f} [{v.min():+.3f}, {v.max():+.3f}]" if len(v) else "-"
            rows.append({"Perturbation": "scan brightness" if stain == "illumination" else stain, "Scale": f"×{s:g}",
                         "FYP2-trained on FYP2 test: median [range]": cell["FYP2"],
                         "BCI-trained on BCI test: median [range]": cell["BCI"]})
    gv = gate_values()
    L = ["### 6g. Stain and scan-brightness perturbation (counterfactual)", "",
         "Each test tile is altered in one controlled way and re-scored. Stain concentrations are scaled in "
         "Ruifrok–Johnston optical-density space, relative to the tile's own background, so the glass keeps its colour. "
         "Alternatively, the illumination itself is scaled, which mimics a brighter or darker scan. The table gives the mean "
         "shift of the expected score (Σ k·p_k, 0 to 3) per network: the median over networks, with the range in brackets. "
         "A network that reads HER2 from DAB should respond to DAB, and hardly at all to haematoxylin or scan brightness. "
         "The Ruifrok–Johnston stain vectors are generic, not calibrated for these scanners, so the stain channels are only approximately separated.", "",
         "![](figures/figD10_stain_sensitivity.png)", "", md_table(pd.DataFrame(rows)), ""]
    if gv:
        L += ["**Pre-registered decision rules** (`variants.py`, fixed before this analysis was run): a gated variant "
              f"is trained only if the median over BCI-trained networks of |shift| reaches {GATE_THRESHOLD:.2f} "
              "in at least one of its two directions.", ""]
        L += [f"- `{v}` ({GATES[v][0]}, ×{GATES[v][1][0]:g} or ×{GATES[v][1][1]:g}): {gv[v]:.3f} → "
              f"**{'met: trained' if gv[v] >= GATE_THRESHOLD else 'not met: not trained'}**" for v in GATES]
        L += [""]
    return L


def section_haem_cue():
    p, d = AN / "haem_cue.csv", AN / "haem_cue_decision.json"
    if not (p.exists() and d.exists()):
        return []
    t, dec = pd.read_csv(p), json.load(open(d))
    pct = lambda a, b: f"{a:.0%} → {b:.0%}"
    rows = [{"FYP2-trained network": r.arch,
             "A: FYP2 0/2+/3+ tiles, H ×0.2": pct(r.A_before, r.A_after),
             "B: FYP2 1+ tiles, H ×5": pct(r.B_before, r.B_after),
             "C: BCI tiles, H ×5": pct(r.C_before, r.C_after),
             "C: BCI balanced accuracy": f"{r.C_balanced_accuracy_before:.3f} → {r.C_balanced_accuracy_after:.3f}"}
            for r in t.itertuples()]
    dC = float((t.C_after - t.C_before).median())
    dBal = float((t.C_balanced_accuracy_after - t.C_balanced_accuracy_before).median())
    return ["### 6h. Do FYP2-trained networks read low haematoxylin as 1+? (`analysis_haem_cue.py`)", "",
            "FYP2's 1+ images carry a fifth to a quarter of the haematoxylin of its other classes (median corrected OD 0.037 "
            "vs 0.15–0.21), and every BCI class is as low (0.02–0.05). FYP2-trained networks call most BCI tiles 1+ "
            "(section 6f). Section 6g scaled haematoxylin only ×0.6–1.5, too little to test this, so here it is "
            "scaled to the other group's level and the share of tiles predicted 1+ is compared before → after.", "",
            md_table(pd.DataFrame(rows)), "",
            f"**Decision rule (fixed before the run):** the cue counts as confirmed if the median network moves the "
            f"predicted-1+ share by ≥ {dec['threshold'] * 100:.0f} percentage points in the expected direction in both "
            f"A (up) and B (down). Median change: A {dec['median_change_A'] * 100:+.0f} points, "
            f"B {dec['median_change_B'] * 100:+.0f} points → "
            f"**{'confirmed' if dec['confirmed'] else 'not confirmed'}**.", "",
            "- Removing haematoxylin (A, visually plausible: nuclei fade, DAB untouched) turns only a minority of "
            "tiles into 1+. Low haematoxylin is at most one of several 1+ cues.",
            "- Adding haematoxylin (B) removes nearly all 1+ answers, but ×5 also creates visible artefacts (yellow speckle on the "
            "background, figure D11). That direction therefore cannot be attributed to haematoxylin alone.",
            f"- On BCI tiles (C), adding haematoxylin cuts the 1+ share by a median {-dC * 100:.0f} points but does not make the "
            f"FYP2 networks more accurate (balanced accuracy {dBal:+.3f}). The haematoxylin difference changes *which* wrong "
            "answer they give, not *whether* they fail.", "",
            "![](figures/figD11_haem_examples.png)", ""]


def section_improvements(runs):
    v = runs[runs.variant != "baseline"]
    if v.empty:
        return ["## 7. Improvement experiments", "", "*Pending.*", ""]
    L = ["## 7. Improvement experiments", "",
         "Each variant is compared with its own baseline (same model and loss) on the same tiles: Δ = variant − baseline, "
         "with a 95% fragment-cluster bootstrap CI (`paired.py`). A variant counts as an improvement only if the CI of Δ "
         "validation QWK excludes 0. Decisions use validation; test is shown only to confirm. Variant definitions are in "
         "`variants.py`.", ""]
    rows = []
    for _, r in v.assign(order=order_key(v)).sort_values(["order", "loss", "variant"]).iterrows():
        b = runs[(runs.model == r.model) & (runs.loss == r.loss) & (runs.variant == "baseline")]
        if b.empty:
            continue
        b = b.iloc[0]
        dv = P.compare(r.run, b.run, "val", ("qwk", "macro_f1", "recall_0"))
        dt = P.compare(r.run, b.run, "test", ("accuracy", "qwk"))
        rows.append({"Model / loss": f"{r.model} / {r.loss.upper()}", "Variant": r.variant,
                     "Val QWK (base → var)": f"{b.val_qwk:.3f} → {r.val_qwk:.3f}",
                     "Δ val QWK [95% CI]": P.cell(dv["qwk"]), "Δ val macro-F1 [95% CI]": P.cell(dv["macro_f1"]),
                     "Δ val recall 0 [95% CI]": P.cell(dv["recall_0"], 2),
                     "Test acc (base → var)": f"{b.test_accuracy:.3f} → {r.test_accuracy:.3f}",
                     "Δ test acc [95% CI]": P.cell(dt["accuracy"]), "Δ test QWK [95% CI]": P.cell(dt["qwk"]),
                     "Train (min)": int(round(r.train_min))})
    L += [md_table(pd.DataFrame(rows)), ""]
    return L


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = S.load_runs()
    fyp2 = fyp2_runs()
    sp = C.RESULTS_DIR / f"selection{C.SUFFIX}.json"
    sel = json.load(open(sp)) if sp.exists() else {}
    fig_curves(runs)
    L = ["# FYP3: the FYP2 pipeline reproduced on real HER2 IHC (BCI)", "",
         "A separate experiment. FYP2 remains the main project and was not modified; its results and weights were only read.", ""]
    findings = C.RESULTS_DIR / "findings.md"
    L += [findings.read_text().strip(), ""] if findings.exists() else ["## 0. Findings", "", "*Pending.*", ""]
    L += section_data() + section_leakage() + section_setup() + section_results(runs, sel, fyp2) + section_metrics() + \
        section_diagnosis() + section_improvements(runs)
    L += ["## 8. Limitations", "",
          "- **No slide or patient IDs.** The grouped split keeps edge-connected tiles together, but 1,645 tiles have no edge "
          "link (the file numbering is shuffled, so it gives no slide order either). Tiles of one slide that are not "
          "edge-connected can still cross splits, so grouped-split numbers may remain slightly optimistic for unseen "
          "patients. For the same reason the fragment-cluster CIs are lower bounds on the true uncertainty.",
          "- **One held-out split.** Grouped test accuracy is 2.4–7.3 points above validation accuracy for every one of the 9 "
          "baselines: the validation tiles are harder for every model (section 5c). Comparisons between models should use paired "
          "differences on the same tiles; absolute accuracies carry several points of split-to-split uncertainty.",
          "- **Labels are slide-level.** The patch-level labels used for diagnosis have an undocumented origin; they are a second "
          "opinion, not ground truth.",
          "- **Single seed per configuration.** Differences of about 1–2 points between runs are within run-to-run and bootstrap noise "
          "(see the CIs).",
          "- **Unknown relative magnification.** FYP2 and BCI images are compared at the same pixel size (256 px), but the "
          "physical magnification of the GAN images is unknown, so cross-domain results mix stain, texture *and* scale "
          "differences.",
          "- **Resolution and batch size change together.** The 448 px (hires) runs train with a quarter of the batch size "
          "(8 instead of 32) to fit in memory, so their gain cannot be attributed to resolution alone.",
          "- **Some steps ran on the CPU** to avoid faults in the Mac GPU driver (tensorflow-metal): ConvNeXtTiny's "
          "network analysis (its Grad-CAM gradient deadlocks on the GPU) and the evaluation of the two 448 px runs "
          "(inference hangs). Rebuilt networks reproduce the saved GPU predictions to within about 1e-6.", ""]
    (C.RESULTS_DIR / f"REPORT{C.SUFFIX}.md").write_text("\n".join(L))
    print(f"wrote {C.RESULTS_DIR / f'REPORT{C.SUFFIX}.md'}")


if __name__ == "__main__":
    main()
