"""Diagnosis E: does slide-fragment leakage in the official BCI split inflate accuracy, and how precise is
one held-out split?

Uses saved predictions only (no TensorFlow). For every architecture trained on BOTH splits
(results/runs_official/<run> and results/runs/<run>):

  1. Exposure. Official-split model on official val/test: accuracy of tiles whose slide fragment also has
     training tiles ("seen fragment"), whose fragment has none ("unseen fragment"), and of tiles with no edge
     link ("unlinked"). This comparison is confounded: only multi-tile fragments can be "seen", and the larger
     the fragment the more likely it is. Control: accuracy by fragment size for the grouped-split model too,
     which saw no held-out fragment.
  2. Paired comparison on IDENTICAL tiles: official test tiles that the grouped split also holds out.
     Neither model trained on them; only the official-split model trained on their slide neighbours.
       grouped test  clean comparison
       grouped val   the grouped model early-stopped on these tiles, which favours it, so the difference
                     (official - grouped) is if anything underestimated
     Each set is also split by exposure. 95% CIs: cluster bootstrap over slide fragments.
  3. Headline: each model on its own split, with cluster-bootstrap CIs.
  4. Error concentration (every grouped-split run): per-fragment accuracy on grouped val and test, with
     each fragment's stain profile as a percentile within its class in the grouped training split.

Outputs: results/analysis/leakage_exposure.csv, leakage_by_size.csv, leakage_paired.csv,
         leakage_headline.csv, heldout_fragments.csv, heldout_concentration.csv, figures/figD9_leakage.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import config as C
import evaluate_all as E

OUT = C.SHARED_ANALYSIS
FIG = C.RESULTS_DIR / "figures"
OFF_RUNS = C.RESULTS_DIR / "runs_official"
GRP_RUNS = C.RESULTS_DIR / "runs"
INK2, MUTED, GRID, SURF = "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SIZE_BINS, SIZE_LABELS = [0, 1, 3, 10, 10_000], ["1 (unlinked)", "2-3", "4-10", ">10"]
PROFILE = ["dab_mean", "dabc_mean", "hem_mean", "bg_brightness", "sharpness"]
TILE_SETS = {"test": "official test ∩ grouped test", "val": "official test ∩ grouped val"}


def preds(run_dir, split):
    d = np.load(run_dir / "predictions.npz")
    return d[f"y_{split}"], d[split]


def manifests():
    frag = pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv").set_index("path")
    off = pd.read_csv(C.OFFICIAL_MANIFEST)
    off["fragment"], off["fragment_size"] = off.path.map(frag.fragment), off.path.map(frag.fragment_size)
    grp = pd.read_csv(C.SPLIT_DIR / "split_manifest_grouped.csv")
    for m in (off, grp):
        m["size_bin"] = pd.cut(m.fragment_size, SIZE_BINS, labels=SIZE_LABELS)
    return off, grp


def split_rows(m, split):
    return m[m.split == split].reset_index(drop=True)


def diff_ci(a, b, groups):
    """Cluster-bootstrap 95% CI of mean(a) - mean(b) for paired correctness vectors."""
    return E.ci95([a[i].mean() - b[i].mean() for i in E.cluster_draws(groups, C.SEED)])


def exposure(m, train_frags):
    return np.where(m.fragment.isin(train_frags), "seen fragment",
                    np.where(m.fragment_size > 1, "unseen fragment", "unlinked"))


def leakage(run, off, grp, train_frags):
    exp_rows, size_rows, pair_rows, head_rows = [], [], [], []
    for split in ["val", "test"]:
        m = split_rows(off, split)
        y, pr = preds(OFF_RUNS / run, split)
        assert (y == m.label.values).all()
        ok, cat = pr.argmax(1) == y, exposure(m, train_frags)
        for c in ["seen fragment", "unseen fragment", "unlinked"]:
            k = cat == c
            exp_rows.append(dict(run=run, split=split, exposure=c, n=int(k.sum()), accuracy=float(ok[k].mean()),
                                 macro_f1=float(f1_score(y[k], pr[k].argmax(1), average="macro"))))
    # 1. control: accuracy by fragment size, exposed (official) vs unexposed (grouped) model
    for design, root, man, splits in [("official", OFF_RUNS, off, ["test"]), ("grouped", GRP_RUNS, grp, ["val", "test"])]:
        for split in splits:
            m = split_rows(man, split)
            y, pr = preds(root / run, split)
            ok = pr.argmax(1) == y
            seen = m.fragment.isin(train_frags).values & (design == "official")
            for b in SIZE_LABELS:
                k = (m.size_bin == b).values
                size_rows.append(dict(run=run, split_design=design, split=split, size_bin=b, n=int(k.sum()),
                                      accuracy=float(ok[k].mean()), seen_fragment_share=float(seen[k].mean())))
    # 2. paired, identical tiles
    mo = split_rows(off, "test")
    yo, po = preds(OFF_RUNS / run, "test")
    io_all = mo.reset_index().set_index("path")["index"]
    for gsplit, tile_set in TILE_SETS.items():
        mg = split_rows(grp, gsplit)
        yg, pg = preds(GRP_RUNS / run, gsplit)
        common = sorted(set(mo.path) & set(mg.path))
        io = io_all.loc[common].values
        ig = mg.reset_index().set_index("path").loc[common, "index"].values
        y = yo[io]
        assert (y == yg[ig]).all()
        co, cg = po[io].argmax(1) == y, pg[ig].argmax(1) == y
        frags, cat = mo.fragment.values[io], exposure(mo.iloc[io], train_frags)
        for subset in ["all", "seen fragment", "unlinked"]:
            k = np.ones(len(y), bool) if subset == "all" else cat == subset
            lo, hi = diff_ci(co[k], cg[k], frags[k])
            pair_rows.append(dict(run=run, tile_set=tile_set, grouped_split=gsplit, subset=subset, n=int(k.sum()),
                                  n_fragments=len(np.unique(frags[k])),
                                  acc_official_model=float(co[k].mean()), acc_grouped_model=float(cg[k].mean()),
                                  diff=float(co[k].mean() - cg[k].mean()), diff_ci_lo=lo, diff_ci_hi=hi,
                                  only_official_right=int((co & ~cg)[k].sum()),
                                  only_grouped_right=int((~co & cg)[k].sum())))
    # 3. headline, each model on its own split
    for design, root, man in [("official", OFF_RUNS, off), ("grouped", GRP_RUNS, grp)]:
        for split in ["val", "test"]:
            m = split_rows(man, split)
            y, pr = preds(root / run, split)
            ci = E.bootstrap_ci(y, pr, m.fragment.values)
            head_rows.append(dict(run=run, split_design=design, split=split, n=len(y),
                                  accuracy=float((pr.argmax(1) == y).mean()),
                                  macro_f1=float(f1_score(y, pr.argmax(1), average="macro")),
                                  qwk=float(cohen_kappa_score(y, pr.argmax(1), weights="quadratic")),
                                  **{f"{k[:-5]}_ci_{s}": v[j] for k, v in ci.items() for j, s in enumerate(["lo", "hi"])}))
    return exp_rows, size_rows, pair_rows, head_rows


def fragment_profiles(grp):
    """Median stain/scan statistics of each fragment, as a percentile within its class in grouped train."""
    feat = pd.read_csv(C.SHARED_ANALYSIS / "data_features.csv")
    feat = feat[feat.dataset == "BCI (real)"].set_index("path")[PROFILE]
    m = grp.join(feat, on="path")
    assert m.dab_mean.notna().all()            # bg_brightness is NaN for tiles with no background
    med = m.groupby("fragment")[PROFILE].median()
    lab = m.groupby("fragment").label.first()
    tr = m[m.split == "train"]
    pct = {c: [float((tr.loc[tr.label == lab[f], c].dropna() < med.loc[f, c]).mean())
               if pd.notna(med.loc[f, c]) else np.nan for f in med.index] for c in PROFILE}
    return med.add_suffix("_median").join(pd.DataFrame(pct, index=med.index).add_suffix("_pct_in_class"))


def concentration(grp):
    prof = fragment_profiles(grp)
    rows, summ = [], []
    for run in sorted(p.name for p in GRP_RUNS.glob("*") if (p / "predictions.npz").exists()):
        for split in ["val", "test"]:
            m = split_rows(grp, split)
            y, pr = preds(GRP_RUNS / run, split)
            assert (y == m.label.values).all()
            m = m.assign(ok=pr.argmax(1) == y, pred=pr.argmax(1),
                         refined_agree=np.where(m.refined_label.notna(), m.refined_label == m.label, np.nan))
            f = m.groupby("fragment").agg(fragment_size=("fragment_size", "first"), label=("label", "first"),
                                          n=("ok", "size"), accuracy=("ok", "mean"), errors=("ok", lambda s: int((~s).sum())),
                                          pred_mode=("pred", lambda s: int(s.mode().iloc[0])),
                                          refined_n=("refined_label", "count"), refined_agree=("refined_agree", "mean"))
            f = f.join(prof).reset_index().assign(run=run, split=split)
            rows.append(f)
            top = f.nlargest(3, "errors")
            keep = ~m.fragment.isin(top.fragment)
            summ.append(dict(run=run, split=split, n=len(m), accuracy=float(m.ok.mean()), errors=int((~m.ok).sum()),
                             top3_fragments=", ".join(map(str, top.fragment)), top3_tiles=int(top.n.sum()),
                             top3_error_share=float(top.errors.sum() / max((~m.ok).sum(), 1)),
                             accuracy_without_top3=float(m.ok[keep].mean())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summ)


def main():
    off, grp = manifests()
    train_frags = set(off.fragment[(off.split == "train") & (off.fragment_size > 1)])
    runs = sorted(p.name for p in OFF_RUNS.glob("*") if (p / "predictions.npz").exists()
                  and (GRP_RUNS / p.name / "predictions.npz").exists())
    parts = [leakage(run, off, grp, train_frags) for run in runs]
    ex, sz, pa, hd = (pd.DataFrame([r for p in parts for r in p[i]]) for i in range(4))
    hf, hc = concentration(grp)
    for df, name in [(ex, "leakage_exposure"), (sz, "leakage_by_size"), (pa, "leakage_paired"),
                     (hd, "leakage_headline"), (hf, "heldout_fragments"), (hc, "heldout_concentration")]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 250)
    for df in (sz, pa, hd, hc):
        print(df.round(3).to_string(index=False), "\n")
    if len(pa):
        fig_leakage(pa, sz, runs)


def fig_leakage(pa, sz, runs):
    fig, axes = plt.subplots(1, 1 + len(runs), figsize=(5.8 + 3.3 * len(runs), 4.6),
                             gridspec_kw=dict(width_ratios=[1.5] + [1] * len(runs)))
    ax = axes[0]
    rows = [(r, s) for r in runs for s in ["all", "seen fragment"]]
    y = np.arange(len(rows))[::-1]
    for j, (gsplit, c, lab) in enumerate([("test", BLUE, "Tiles in grouped test (clean)"),
                                          ("val", ORANGE, "Tiles in grouped val (conservative: the grouped\n"
                                                          "model early-stopped on them)")]):
        d = pa[pa.grouped_split == gsplit].set_index(["run", "subset"]).loc[rows]
        yy, pts = y + (0.15 - 0.3 * j), d["diff"] * 100
        ax.errorbar(pts, yy, xerr=[pts - d.diff_ci_lo * 100, d.diff_ci_hi * 100 - pts], fmt="o", ms=6,
                    color=c, ecolor=c, elinewidth=1.6, capsize=0, mec=SURF, mew=1.2, label=lab)
        for yi, x, n in zip(yy, pts, d.n):
            ax.annotate(f"{x:+.1f} (n={n})", (x, yi), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [f"{r.split('__')[0]}\n{'all tiles' if s == 'all' else 'seen fragment'}" for r, s in rows],
                  fontsize=8.5)
    ax.set_xlabel("Official-split minus grouped-split model accuracy\n(points; 95% cluster-bootstrap CI)", fontsize=9)
    ax.set_title("Leakage effect on identical official-test tiles", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), fontsize=8, frameon=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for ax, run in zip(axes[1:], runs):
        d = sz[sz.run == run]
        x = np.arange(len(SIZE_LABELS))
        for (design, split), c, mk, lab in [(("official", "test"), ORANGE, "o", "Official-split model, official test"),
                                            (("grouped", "test"), BLUE, "s", "Grouped-split model, grouped test"),
                                            (("grouped", "val"), GREEN, "D", "Grouped-split model, grouped val")]:
            s = d[(d.split_design == design) & (d.split == split)].set_index("size_bin").loc[SIZE_LABELS]
            ax.plot(x, s.accuracy, color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, label=lab)
        seen = d[d.split_design == "official"].set_index("size_bin").loc[SIZE_LABELS].seen_fragment_share
        ax.set_xticks(x, [f"{b.split()[0]}\n{v:.0%}" for b, v in zip(SIZE_LABELS, seen)], fontsize=8)
        ax.set_ylim(0.7, 1.0)
        ax.set_xlabel("Fragment size (tiles)\n% official test seen in training", fontsize=8.5)
        ax.set_title(f"Accuracy by fragment size\n{run.split('__')[0]}", loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.3), fontsize=8, frameon=False)
    for a in axes:
        a.set_axisbelow(True)
        a.tick_params(colors=INK2)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "figD9_leakage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
