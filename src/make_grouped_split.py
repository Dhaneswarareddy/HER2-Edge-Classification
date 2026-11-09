"""Step 1b (FYP3): the main, leakage-controlled split, using FYP2's recipe with slide fragments as groups.

Why not the official BCI split: tiles cut side by side from one slide share a continuous border.
analysis_slides.py links such tiles (99.96% of links join tiles with the same slide label, against 33%
for random pairs), and finds that 619 of the 977 official test tiles belong to slide fragments that
also contain official training tiles. The official split is therefore not slide-independent.

Why not FYP2's pHash groups: in BCI, pHash <= 12 links tiles of mostly background from different slides
(39 of 53 pHash groups mix slide labels; even 6-bit pairs agree on label only 50% of the time), so they
are not near-duplicates and would chain unrelated slides together.

Recipe (as FYP2/src/make_split.py): StratifiedGroupKFold, 20 folds over all 4,873 tiles, groups = edge
fragments; 3 folds test (15%), 3 folds val (15%), 14 folds train (70%); best of 50 seeds by how closely
each split's class proportions match the whole dataset.

Residual risk (cannot be removed without slide IDs): tiles of one slide that are not edge-connected
(1,645 tiles have no accepted edge link) can still fall in different splits.

Outputs: splits/split_manifest_grouped.csv, splits/split_report_grouped.txt
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C

N_FOLDS = 20


def main():
    off = pd.read_csv(C.OFFICIAL_MANIFEST)
    fr = pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv")
    assert (off.path.values == fr.path.values).all(), "fragment table out of order"
    df = off.drop(columns=["split", "group", "test_near_dup_of_train"]).copy()
    df["fragment"], df["fragment_size"] = fr.fragment.values, fr.fragment_size.values
    prop_all = np.bincount(df.label, minlength=C.NUM_CLASSES) / len(df)

    best = None
    for seed in range(50):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED + seed)
        fold = np.zeros(len(df), int)
        for k, (_, idx) in enumerate(sgkf.split(df, df.label, df.fragment)):
            fold[idx] = k
        split = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
        dev = 0.0
        for s, target in [("train", 1 - C.VAL_FRAC - C.TEST_FRAC), ("val", C.VAL_FRAC), ("test", C.TEST_FRAC)]:
            m = split == s
            dev += np.abs(np.bincount(df.label[m], minlength=C.NUM_CLASSES) / m.sum() - prop_all).sum()
            dev += abs(m.mean() - target)
        if best is None or dev < best[0]:
            best = (dev, split, seed)
    df["split"] = best[1]

    # ---- verification ----
    assert df[df.fragment_size > 1].groupby("fragment").split.nunique().max() == 1, "fragment spans splits"
    links = pd.read_csv(C.SHARED_ANALYSIS / "edge_matches.csv")
    links = links[links.accepted]
    sp = df.split.values
    crossing = int((sp[links.i.values] != sp[links.j.values]).sum())
    assert crossing == 0, f"{crossing} accepted edge links cross splits"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(df.md5[df.split == a]) & set(df.md5[df.split == b]), f"identical files in {a}/{b}"

    emb = np.load(C.RESULTS_DIR / "features" / "imagenet_effb0_bci.npz", allow_pickle=True)
    E = pd.DataFrame(emb["emb"], index=emb["path"]).loc[df.path].values
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    tr = sp == "train"
    lines = ["Leakage check: nearest-neighbour similarity of each eval tile to the TRAIN set",
             "(frozen ImageNet EfficientNetB0 cosine) and accepted edge links across splits",
             f"  accepted edge links crossing any split boundary: {crossing} (of {len(links)})"]
    report = {"n_images": len(df), "n_fragments_multi_tile": int(df[df.fragment_size > 1].fragment.nunique()),
              "tiles_without_edge_link": int((df.fragment_size == 1).sum()), "split_seed_offset": best[2],
              "edge_links_crossing_splits": crossing}
    for name in ["val", "test"]:
        m = sp == name
        nn = (En[m] @ En[tr].T).max(1)
        lines.append(f"  {name:5s}->train  cos max={nn.max():.3f} p99={np.percentile(nn, 99):.3f} median={np.median(nn):.3f}")
        report[f"{name}_to_train_median_cos"] = float(np.median(nn))
    S = En[tr] @ En[tr].T
    np.fill_diagonal(S, -1)
    lines.append(f"  train->train cos max={S.max():.3f} median={np.median(S.max(1)):.3f}")
    off_te = (df.official_split == "test").values
    lines.append(f"  official-test tiles per grouped split: " +
                 ", ".join(f"{s} {int((off_te & (sp == s)).sum())}" for s in ["train", "val", "test"]))

    ref = df.dropna(subset=["refined_label"])
    lines += ["", "Refined (patch-level) vs filename (slide-level) label agreement per split: " +
              ", ".join(f"{s} {(ref[ref.split == s].refined_label == ref[ref.split == s].label).mean():.3f}"
                        for s in ["train", "val", "test"])]
    report["refined_agreement"] = float((ref.refined_label == ref.label).mean())

    df.to_csv(C.SPLIT_DIR / "split_manifest_grouped.csv", index=False)
    counts = pd.crosstab(df.split, df.label, margins=True)
    counts.columns = C.CLASS_NAMES + ["total"]
    with open(C.SPLIT_DIR / "split_report_grouped.txt", "w") as f:
        f.write(__doc__ + "\n" + counts.to_string() + "\n\n" + "\n".join(lines) + "\n\n")
        f.write(json.dumps(report, indent=2, default=str))
    print(counts)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
