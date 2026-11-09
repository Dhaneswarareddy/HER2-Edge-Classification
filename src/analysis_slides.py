"""Diagnosis D: reconstruct slide fragments from tile edges, to test for slide overlap across splits.

BCI has no slide or patient IDs. Its tiles are 1024x1024 crops of whole-slide images, so two tiles that
were side by side on the slide share a boundary: the last pixel column of one continues into the first
column of the other. Matching edges therefore links tiles of the same slide, with no metadata.

Method (on the 256 px cache; CPU only):
  1. For every tile keep its four 1-px borders (256 x 3 values each).
  2. For each tile, find the tile whose LEFT border best continues its RIGHT border (mean squared
     difference), and likewise bottom->top. Accept a match only if it is
       - mutual (each is the other's best match),
       - unambiguous (best distance < RATIO x second best; Lowe's ratio test), and
       - close: distance < DIST_FACTOR x the tile's own median distance between adjacent INTERNAL
         columns/rows (a true neighbour should look like the next pixel column inside a tile).
  3. Linked tiles form connected components = slide fragments.

Validation of the method (independent of the hypothesis being tested): all tiles of one slide share one
filename label, while random tile pairs share a label only ~33% of the time (class proportions). Label
purity of matched pairs therefore measures how many matches are real.

Outputs: results/analysis/slide_fragments.csv (path, fragment, size), slide_fragments.json (summary),
         figures/figD8_edge_matches.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config as C

OUT = C.SHARED_ANALYSIS
FIG = C.RESULTS_DIR / "figures"
RATIO = 0.6
DIST_FACTOR = 3.0
INK2, GRID = "#52514e", "#e4e3df"


def borders(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    internal_cols = np.mean((a[:, 1:] - a[:, :-1]) ** 2, axis=(0, 2))       # per adjacent column pair
    internal_rows = np.mean((a[1:] - a[:-1]) ** 2, axis=(1, 2))
    return (a[:, -1].ravel(), a[:, 0].ravel(), a[-1].ravel(), a[0].ravel(),
            float(np.median(internal_cols)), float(np.median(internal_rows)))


def sqdist(A, B):
    """Mean squared difference between every row of A and every row of B."""
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0) / A.shape[1]


def match(A, B, scale):
    """Mutual, unambiguous, close matches between A[i] (e.g. right borders) and B[j] (left borders)."""
    D = sqdist(A.astype(np.float64), B.astype(np.float64))
    np.fill_diagonal(D, np.inf)
    best_j = D.argmin(1)
    best_i = D.argmin(0)
    srt = np.sort(D, axis=1)
    d1, d2 = srt[:, 0], srt[:, 1]
    rows = []
    for i, j in enumerate(best_j):
        ok = best_i[j] == i and d1[i] < RATIO * d2[i] and d1[i] < DIST_FACTOR * scale[i]
        rows.append(dict(i=i, j=int(j), d1=float(d1[i]), d2=float(d2[i]), rel=float(d1[i] / scale[i]), accepted=ok))
    return pd.DataFrame(rows)


def main():
    man = pd.read_csv(C.OFFICIAL_MANIFEST)     # fragments are split-independent
    R, L, Bo, T, sc_c, sc_r = zip(*[borders(C.DATA_DIR / p) for p in man.path])
    R, L, Bo, T = map(np.stack, (R, L, Bo, T))
    sc_c, sc_r = np.array(sc_c), np.array(sc_r)
    horiz = match(R, L, sc_c).assign(kind="right->left")
    vert = match(Bo, T, sc_r).assign(kind="bottom->top")
    m = pd.concat([horiz, vert], ignore_index=True)
    lab = man.label.values
    m["same_label"] = lab[m.i] == lab[m.j]
    acc = m[m.accepted]

    # connected components over accepted links
    parent = list(range(len(man)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(acc.i, acc.j):
        parent[root(i)] = root(j)
    comp = np.array([root(i) for i in range(len(man))])
    man["fragment"] = pd.factorize(comp)[0]
    man["fragment_size"] = man.fragment.map(man.fragment.value_counts())
    frag = man[man.fragment_size > 1]
    purity = frag.groupby("fragment").label.agg(lambda s: s.value_counts().iloc[0] / len(s))
    p_random = float(np.sum((np.bincount(lab) / len(lab)) ** 2))

    # do fragments cross the split boundaries?
    def crosses(col, a, b):
        in_frag = man.fragment_size > 1
        shared = set(man.fragment[(man[col] == a) & in_frag]) & set(man.fragment[(man[col] == b) & in_frag])
        return dict(fragments_shared=len(shared),
                    tiles_of_b_in_shared=int(((man[col] == b) & man.fragment.isin(shared)).sum()),
                    tiles_of_b=int((man[col] == b).sum()))

    summary = dict(
        n_tiles=len(man), candidate_pairs=len(m), accepted_links=int(m.accepted.sum()),
        accepted_links_same_label=float(acc.same_label.mean()) if len(acc) else None,
        rejected_best_matches_same_label=float(m[~m.accepted].same_label.mean()),
        random_pair_same_label=p_random,
        tiles_in_fragments=int((man.fragment_size > 1).sum()), n_fragments=int(frag.fragment.nunique()),
        largest_fragment=int(man.fragment_size.max()),
        fragment_size_quantiles=man[man.fragment_size > 1].drop_duplicates("fragment").fragment_size
        .quantile([0.5, 0.9, 1.0]).to_dict(),
        fragments_label_pure=float((purity == 1).mean()) if len(purity) else None,
        train_vs_val=crosses("split", "train", "val"), train_vs_test=crosses("split", "train", "test"),
        official_train_vs_test=crosses("official_split", "train", "test"),
        params=dict(ratio=RATIO, dist_factor=DIST_FACTOR))
    OUT.mkdir(parents=True, exist_ok=True)
    man[["path", "split", "official_split", "label", "fragment", "fragment_size"]].to_csv(
        OUT / "slide_fragments.csv", index=False)
    m.to_csv(OUT / "edge_matches.csv", index=False)
    json.dump(summary, open(OUT / "slide_fragments.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

    # figure: distance of best match relative to internal adjacent-pixel distance, by label agreement
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.logspace(-1, 3, 60)
    for same, c, lab_ in [(True, "#2a78d6", "best match has the SAME filename label"),
                          (False, "#eb6834", "best match has a DIFFERENT label")]:
        ax.hist(m[m.same_label == same].rel.clip(0.1, 1000), bins=bins, color=c, alpha=0.75, label=lab_)
    ax.axvline(DIST_FACTOR, color=INK2, ls=":", lw=1)
    ax.text(DIST_FACTOR * 1.08, ax.get_ylim()[1] * 0.9, "acceptance\nthreshold", fontsize=8, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Best edge-match distance / tile's own adjacent-pixel distance (log)")
    ax.set_ylabel("Tile borders")
    ax.set_title("Edge continuity between tiles", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "figD8_edge_matches.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
