"""Paired comparison of two runs on the same held-out tiles, with fragment-cluster bootstrap CIs.

Absolute val/test numbers on BCI carry several points of split-to-split noise (REPORT section 5c), but
two models scored on the SAME tiles share that noise, so their difference is far more precise. Each
bootstrap resample draws whole slide fragments and scores both runs on the identical tiles.

  python paired.py RUN_A RUN_B [--split val]      # prints metric(A) - metric(B) with 95% CIs
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

import config as C
import evaluate_all as E

METRICS = {"qwk": lambda y, p: cohen_kappa_score(y, p, weights="quadratic"),
           "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
           "accuracy": lambda y, p: np.mean(y == p),
           "recall_0": lambda y, p: recall_score(y, p, labels=[0], average="macro", zero_division=0)}


def fragments(split):
    m = pd.read_csv(C.MANIFEST)
    frag = pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv").set_index("path").fragment
    return m[m.split == split].path.map(frag).values


def compare(run_a, run_b, split="val", metrics=tuple(METRICS)):
    """{metric: {diff, ci_lo, ci_hi}} for metric(run_a) - metric(run_b) on `split`."""
    da, db = (np.load(C.RUNS_DIR / r / "predictions.npz") for r in (run_a, run_b))
    y = da[f"y_{split}"]
    assert (y == db[f"y_{split}"]).all()
    pa, pb = da[split].argmax(1), db[split].argmax(1)
    draws = list(E.cluster_draws(fragments(split), C.SEED))
    out = {}
    for m in metrics:
        f = METRICS[m]
        lo, hi = E.ci95([f(y[i], pa[i]) - f(y[i], pb[i]) for i in draws])
        out[m] = dict(diff=float(f(y, pa) - f(y, pb)), ci_lo=lo, ci_hi=hi)
    return out


def cell(d, digits=3):
    """'+0.012 [-0.010, +0.035]' for report tables."""
    return f"{d['diff']:+.{digits}f} [{d['ci_lo']:+.{digits}f}, {d['ci_hi']:+.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    for m, d in compare(args.run_a, args.run_b, args.split).items():
        print(f"{m:9s} {args.run_a} - {args.run_b} on {args.split}: {cell(d)}")


if __name__ == "__main__":
    main()
