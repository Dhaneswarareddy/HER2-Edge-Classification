"""Shared helpers for experiments_v2. Read-only access to every earlier artifact; all writes go under experiments_v2/.

Conventions used by every experiment here:
  * class indices 0..3 = HER2 IHC 0, 1+, 2+, 3+;
  * BCI = grouped split (splits/split_manifest_grouped.csv); 'val' is used for every choice, 'test' is report-only;
  * HER2-IHC-40x: 'dev' tiles come from the WSI-based training archive (86 slides); the 1,847-tile WSI test split
    (21 slides) was examined in earlier diagnostics and evaluated once by HARMONISE-1, so any result on it here is an
    EXPLORATORY SECOND LOOK, never an untouched test result;
  * seeds are fixed and written into each experiment's config.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parent
ROOT = V2.parent
SRC = ROOT / "src"
NEXT = ROOT / "next_experiment"
H2OUT = NEXT / "harmonise2" / "outputs"
RUNS = ROOT / "results" / "runs"
EXT = ROOT / "results" / "external" / "her2_40x"
H1TEST = NEXT / "outputs" / "test"
EXT_TEST_DIR = (ROOT.parent / "FYP" / "data" / "raw" / "HER2-IHC-40x" / "wsi_test").resolve()
LEVELS = ["0", "1+", "2+", "3+"]
SEED = 2027

sys.path.insert(0, str(SRC))


def out_dir(exp):
    d = V2 / exp / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def guard(path):
    """Refuse to write anywhere outside experiments_v2."""
    p = Path(path).resolve()
    assert V2 in p.parents or p == V2, f"refusing to write outside experiments_v2: {p}"
    return p


def write_json(path, obj):
    p = guard(path)
    json.dump(obj, open(p, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def write_csv(path, df):
    df.to_csv(guard(path), index=False)


# ------------------------------------------------------------------ BCI data
def manifest():
    return pd.read_csv(ROOT / "splits" / "split_manifest_grouped.csv")


def fragments(split):
    m = manifest()
    frag = pd.read_csv(ROOT / "results" / "analysis" / "slide_fragments.csv").set_index("path").fragment
    return m[m.split == split].path.map(frag).values


def bci_preds(run, split):
    z = np.load(RUNS / run / "predictions.npz")
    return z[f"y_{split}"].astype(int), z[split].astype(np.float64)


def bci_runs():
    return sorted(p.name for p in RUNS.iterdir() if (p / "summary.json").exists())


# ------------------------------------------------------------------ external test (exploratory second look)
def ext_label(path):
    return LEVELS.index(path.split("/")[0][len("class_"):])


def ext_raw(source, run):
    d = pd.read_csv(EXT / f"{source}__{run}.csv")
    return d.path.values, d.label.values.astype(int), d[[f"p{k}" for k in range(4)]].values


def ext_h1(run):
    d = pd.read_csv(H1TEST / f"{run}.csv")
    return d.path.values, np.array([ext_label(p) for p in d.path]), d[[f"p{k}" for k in range(4)]].values


def filename_groups(names):
    """Consecutive filename numbers within a score (same rule as harmonise_lib.filename_groups). NOT slides:
    the release has no slide IDs, so CIs built on these groups are optimistic."""
    import re
    rows = sorted((re.search(r"her2-(.+?)-score", n).group(1), int(re.search(r"_(\d+)\.png$", n).group(1)), i)
                  for i, n in enumerate(names))
    g = np.zeros(len(names), int); c = 0
    for j, (sc, num, i) in enumerate(rows):
        if j and (sc != rows[j - 1][0] or num != rows[j - 1][1] + 1):
            c += 1
        g[i] = c
    return g


# ------------------------------------------------------------------ metrics
def confusion(y, p):
    c = np.zeros((4, 4), int)
    np.add.at(c, (y, p), 1)
    return c


def class_metrics(y, prob_or_pred):
    from sklearn.metrics import cohen_kappa_score
    p = prob_or_pred.argmax(1) if np.ndim(prob_or_pred) == 2 else np.asarray(prob_or_pred)
    c = confusion(y, p)
    rec = np.diag(c) / np.maximum(c.sum(1), 1)
    pre = np.diag(c) / np.maximum(c.sum(0), 1)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    present = c.sum(1) > 0
    m = dict(n=int(len(y)), accuracy=float((p == y).mean()), balanced_accuracy=float(rec[present].mean()),
             macro_f1=float(f1.mean()), qwk=float(cohen_kappa_score(y, p, weights="quadratic", labels=range(4))),
             mae=float(np.abs(p - y).mean()), off_by_2plus=float((np.abs(p - y) >= 2).mean()))
    for k, nm in enumerate(LEVELS):
        m.update({f"precision_{nm}": float(pre[k]), f"recall_{nm}": float(rec[k]), f"f1_{nm}": float(f1[k])})
    m["confusion"] = c.tolist()
    return m


def cluster_draws(groups, n, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    _, start, count = np.unique(groups[order], return_index=True, return_counts=True)
    members = [order[s:s + c] for s, c in zip(start, count)]
    for _ in range(n):
        yield np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def ci95(v):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def tf_setup(cpu=False):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    if cpu:
        tf.config.set_visible_devices([], "GPU")
    return tf


def md_table(df, digits=3):
    f = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.values])
