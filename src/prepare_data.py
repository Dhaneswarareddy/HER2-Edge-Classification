"""Step 1 (FYP3): BCI IHC images -> 256 px cache, labels, leakage-safe train/val split.

Mirrors FYP2/src/make_split.py, with three differences forced by the data:

  * Labels come from the BCI filename convention 'NNNNN_{train|test}_{0|1+|2+|3+}.png'. Per the BCI
    README this is the score of the whole slide (WSI) the tile came from, not of the tile itself.
  * The OFFICIAL BCI test split (977 tiles) is the test set, so results are comparable with published
    BCI work. Validation is 15% of the official training split.
  * Tiles are 1024x1024. They are downsampled once to 256x256 (Lanczos) so the networks see the same
    geometry as FYP2 (256 px source -> 224 px input).

Leakage control is the same as FYP2: pHash (<= 12 bits) near-duplicate groups never straddle the
train/val boundary. Near-duplicates between the official train and test splits cannot be moved; they
are counted and flagged (column 'test_near_dup_of_train') so test metrics can be re-checked without them.
There are no patient/slide IDs in BCI, so this split is NOT slide-independent: different tiles of one
slide can sit in train and val.

The patch-level 'her2_level_refined' column from BCI_metadata_pairs_refined.csv is merged in for
analysis only (column 'refined_label'). It is never used for training or model selection.

Outputs: data/ihc_256/{train,test}/*.png, splits/split_manifest_official.csv, splits/split_report_official.txt,
         results/features/imagenet_effb0_bci.npz (frozen ImageNet embeddings, reused by the analysis)
"""
import hashlib
import json
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import imagehash
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import config as C

PHASH_T = 12           # same as FYP2
N_FOLDS = 20           # 3 of 20 folds of the official train split -> 15% validation
NAME_RE = re.compile(r"^(\d{5})_(train|test)_(0|1\+|2\+|3\+)\.png$")


def find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def list_images():
    rows = []
    for part in ["train", "test"]:
        for f in sorted(os.listdir(C.BCI_IHC / part)):
            if not f.endswith(".png"):
                continue
            m = NAME_RE.match(f)
            assert m, f"unexpected file name {f}"
            assert m.group(2) == part, f"partition mismatch {f}"
            rows.append(dict(path=f"{part}/{f}", pair_id=m.group(1), official_split=part,
                             label=C.BCI_LEVELS.index(m.group(3))))
    return pd.DataFrame(rows)


def refined_labels(df):
    """Patch-level labels distributed with BCI (origin undocumented). Non-standard values -> NaN."""
    r = pd.read_csv(C.BCI_REFINED_CSV, dtype=str)
    r["key"] = r.partition + "/" + r.file_name_ihc
    mp = {lv: i for i, lv in enumerate(C.BCI_LEVELS)}
    r["refined_label"] = r.her2_level_refined.map(mp)       # '1', '2', '0+' -> NaN (4 rows)
    return df.path.map(r.set_index("key").refined_label)


def main():
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df = list_images()
    print(f"{len(df)} IHC tiles; official train {sum(df.official_split == 'train')}, "
          f"test {sum(df.official_split == 'test')}")
    print(pd.crosstab(df.official_split, df.label))

    # ---- 256 px cache (raw files are only read) ----
    for part in ["train", "test"]:
        (C.DATA_DIR / part).mkdir(parents=True, exist_ok=True)
    md5 = []
    for p in df.path:
        src, dst = C.BCI_IHC / p, C.DATA_DIR / p
        md5.append(hashlib.md5(open(src, "rb").read()).hexdigest())
        if not dst.exists():
            Image.open(src).convert("RGB").resize((C.SOURCE_SIZE, C.SOURCE_SIZE), Image.LANCZOS).save(dst)
    df["md5"] = md5
    df["refined_label"] = refined_labels(df)
    paths = [C.DATA_DIR / p for p in df.path]

    # ---- near-duplicate groups (pHash), exactly as FYP2 ----
    ph = np.array([imagehash.phash(Image.open(p)).hash.flatten() for p in paths])
    df["phash"] = [str(imagehash.ImageHash(h.reshape(8, 8))) for h in ph]
    H = np.concatenate([(ph[k:k + 400, None, :] != ph[None, :, :]).sum(-1).astype(np.int16)
                        for k in range(0, len(ph), 400)])        # chunked: 4873^2 x 64 would need 1.5 GB
    np.fill_diagonal(H, 999)
    I, J = np.where(np.triu(H <= PHASH_T, 1))
    parent = list(range(len(df)))
    for i, j in zip(I, J):
        parent[find_root(parent, i)] = find_root(parent, j)
    roots = [find_root(parent, i) for i in range(len(df))]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    df["group"] = [remap[r] for r in roots]
    gsize = df.group.map(df.group.value_counts())
    print(f"pHash<= {PHASH_T} linked pairs: {len(I)}  groups: {df.group.nunique()}  "
          f"tiles in multi-tile groups: {(gsize > 1).sum()}  largest group: {gsize.max()}")

    # ---- frozen ImageNet embeddings: leakage check + reused by the feature analysis ----
    net = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
    E = []
    for k in range(0, len(paths), 256):
        X = np.stack([np.asarray(Image.open(p).resize((224, 224), Image.BILINEAR)) for p in paths[k:k + 256]])
        E.append(net.predict(X.astype("float32"), batch_size=64, verbose=0))
    E = np.concatenate(E)
    feat_dir = C.RESULTS_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feat_dir / "imagenet_effb0_bci.npz", emb=E, path=df.path.values, label=df.label.values)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)

    # ---- validation split from the official train split (groups kept whole) ----
    off_tr = (df.official_split == "train").values
    sub = df[off_tr]
    best = None
    for seed in range(50):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED + seed)
        fold = np.zeros(len(sub), int)
        for k, (_, idx) in enumerate(sgkf.split(sub, sub.label, sub.group)):
            fold[idx] = k
        is_val = fold < 3
        prop_all = np.bincount(sub.label, minlength=4) / len(sub)
        prop_val = np.bincount(sub.label[is_val], minlength=4) / max(is_val.sum(), 1)
        dev = np.abs(prop_val - prop_all).sum() + abs(is_val.mean() - C.VAL_FRAC)
        if best is None or dev < best[0]:
            best = (dev, is_val, seed)
    split = np.array(["test"] * len(df), dtype=object)
    split[np.where(off_tr)[0]] = np.where(best[1], "val", "train")
    df["split"] = split

    # groups that straddle the OFFICIAL train/test boundary (cannot be fixed; flagged)
    tr_groups = set(df.group[df.split != "test"])
    df["test_near_dup_of_train"] = (df.split == "test") & df.group.isin(tr_groups)

    # ---- verification ----
    dev_groups = df[df.split != "test"].groupby("group").split.nunique()
    assert dev_groups.max() == 1, "a near-duplicate group spans train and val"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(set(df.md5[df.split == a]) & set(df.md5[df.split == b]))
        print(f"identical files {a}/{b}: {n}")
    sp = df.split.values
    tr = sp == "train"
    lines = ["Leakage check: nearest-neighbour similarity of each eval tile to the TRAIN set",
             "(frozen ImageNet EfficientNetB0 cosine; pHash Hamming distance out of 64)"]
    report = {"n_images": len(df), "phash_threshold": PHASH_T, "linked_pairs": int(len(I)),
              "n_groups": int(df.group.nunique()), "tiles_in_multi_tile_groups": int((gsize > 1).sum()),
              "split_seed_offset": best[2],
              "test_tiles_near_dup_of_train": int(df.test_near_dup_of_train.sum())}
    for name in ["val", "test"]:
        m = sp == name
        nn_cos = (En[m] @ En[tr].T).max(1)
        nn_ham = H[np.ix_(m, tr)].min(1)
        lines.append(f"  {name:5s}->train  cos max={nn_cos.max():.3f} p99={np.percentile(nn_cos, 99):.3f} "
                     f"median={np.median(nn_cos):.3f} | pHash min={nn_ham.min()} median={np.median(nn_ham):.0f} "
                     f"| tiles with pHash<={PHASH_T}: {(nn_ham <= PHASH_T).sum()}")
        report[f"{name}_to_train_median_cos"] = float(np.median(nn_cos))
        report[f"{name}_to_train_min_phash"] = int(nn_ham.min())
    S_tt = En[tr] @ En[tr].T
    np.fill_diagonal(S_tt, -1)
    lines.append(f"  train->train cos max={S_tt.max():.3f} median={np.median(S_tt.max(1)):.3f}")
    for ln in lines:
        print(ln)

    ref = df.dropna(subset=["refined_label"])
    agree = (ref.refined_label == ref.label).mean()
    lines += ["", f"Refined (patch-level) vs filename (slide-level) label agreement: {agree:.3f} "
              f"on {len(ref)} tiles with a valid refined label", pd.crosstab(ref.label, ref.refined_label,
                                                                           rownames=["filename"],
                                                                           colnames=["refined"]).to_string()]
    report["refined_agreement"] = float(agree)

    df.to_csv(C.OFFICIAL_MANIFEST, index=False)
    counts = pd.crosstab(df.split, df.label, margins=True)
    counts.columns = C.CLASS_NAMES + ["total"]
    with open(C.SPLIT_DIR / "split_report_official.txt", "w") as f:
        f.write(__doc__ + "\n" + counts.to_string() + "\n\n" + "\n".join(lines) + "\n\n")
        f.write(json.dumps(report, indent=2, default=str))
    print(counts)
    print("\n".join(lines[-7:]))
    print(f"saved {C.OFFICIAL_MANIFEST}")


if __name__ == "__main__":
    main()
