"""External test: trained networks applied, unchanged, to HER2-IHC-40x (University of Malaya, 40x scans).
(TensorFlow; run alone.)

No training, tuning, threshold or model choice ever uses these images: they are report-only, like a test set.
Use the WSI-based TEST split only. In the patch-based release, 1,748 of 2,197 test files are named
'..._score_train_...', i.e. they come from the WSI-based training pool, so that split is not slide-independent.

Labels: the class folder (class_0, class_1+, class_2+, class_3+). The filename also carries a score
('her2-<score>-score_...'); files where the two disagree are counted and reported separately.

Preprocessing is the one each network was trained with: 1024 px -> Lanczos to the run's source size
(256, or 512 for hires) -> PIL bilinear to its input size (224 or 448). The physical scale therefore
differs from BCI (0.46 um/px, 20x): a 40x tile covers a smaller field. This is reported, not corrected.
The release has no slide IDs, so the bootstrap CIs resample tiles and are optimistic.

Usage (from src/):
  python external_eval.py --data-dir <dir containing class_0 ... class_3+> \
      --runs DenseNet121__ce EfficientNetV2B0__ce EfficientNetV2B0__ce__hires MobileNetV3Large__corn \
      --fyp2-runs EfficientNetV2B0__ce [--cpu]
Outputs: results/external/her2_40x/<source>__<run>.json and .csv (per-tile predictions), summary.csv
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

import config as C

LEVELS = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
OUT = C.RESULTS_DIR / "external" / "her2_40x"
N_BOOT = 2000


def list_tiles(data_dir):
    rows = []
    for d in sorted(data_dir.glob("class_*")):
        for p in sorted(d.glob("*.png")):
            m = re.search(r"her2-(0|1\+|2\+|3\+)-score", p.name)
            rows.append(dict(path=str(p.relative_to(data_dir)), label=LEVELS[d.name[len("class_"):]],
                             filename_label=LEVELS[m.group(1)] if m else -1))
    df = pd.DataFrame(rows)
    assert len(df), f"no class_*/*.png under {data_dir}"
    return df


def cache(df, data_dir, source, size):
    """Same two-step resize as the BCI pipeline, cached under FYP3/data (the raw files are only read)."""
    out = C.ROOT / "data" / f"her2_40x_{source}_eval{size}"
    for p in df.path:
        dst = out / p
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(data_dir / p).convert("RGB").resize((source, source), Image.LANCZOS)
            im.resize((size, size), Image.BILINEAR).save(dst)
    return out


def metrics(y, prob, rng=None):
    p = prob.argmax(1)
    out = dict(n=int(len(y)), accuracy=float((p == y).mean()),
               balanced_accuracy=float(recall_score(y, p, average="macro", labels=range(4), zero_division=0)),
               macro_f1=float(f1_score(y, p, average="macro", labels=range(4), zero_division=0)),
               qwk=float(cohen_kappa_score(y, p, weights="quadratic")),
               recall_per_class=recall_score(y, p, average=None, labels=range(4), zero_division=0).round(4).tolist(),
               predicted_share=(np.bincount(p, minlength=4) / len(p)).round(4).tolist(),
               confusion=confusion_matrix(y, p, labels=range(4)).tolist())
    if rng is not None:                       # tile bootstrap (no slide IDs: optimistic)
        acc, qwk = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            acc.append((p[i] == y[i]).mean())
            qwk.append(cohen_kappa_score(y[i], p[i], weights="quadratic"))
        out["accuracy_ci95"] = np.percentile(acc, [2.5, 97.5]).round(4).tolist()
        out["qwk_ci95"] = np.percentile(qwk, [2.5, 97.5]).round(4).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="FYP3 (BCI-trained) run folders, grouped split")
    ap.add_argument("--fyp2-runs", nargs="*", default=[], help="FYP2 (GAN-trained) run folders")
    ap.add_argument("--cpu", action="store_true", help="hide the GPU (needed on macOS for 448 px networks)")
    args = ap.parse_args()
    if args.cpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    import common as U

    from pathlib import Path
    data_dir = Path(args.data_dir).resolve()
    df = list_tiles(data_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} tiles, per class {np.bincount(df.label, minlength=4).tolist()}, "
          f"folder/filename label mismatches {int((df.label != df.filename_label).sum())}", flush=True)
    y = df.label.values.astype(int)
    agree = (df.label == df.filename_label).values
    rows = []
    for source_name, root, runs in [("BCI", C.RUNS_DIR, args.runs),
                                    ("FYP2", C.FYP2_ROOT / "results" / "runs", args.fyp2_runs)]:
        for run in runs:
            s = json.load(open(root / run / "summary.json"))
            size = s.get("image_size", 224)
            src = (s.get("variant_spec") or {}).get("source_size", 256)
            C.IMAGE_SIZE = size                     # load_trained may rebuild the architecture at this size
            model = U.load_trained(root / run)
            cdir = cache(df, data_dir, src, size)
            prob = np.concatenate([model.predict(np.stack([np.asarray(Image.open(cdir / p).convert("RGB"))
                                                           for p in df.path[k:k + 128]]).astype(np.float32),
                                                 batch_size=32, verbose=0)
                                   for k in range(0, len(df), 128)])
            res = dict(source=source_name, run=run, source_size=src, input_size=size,
                       all=metrics(y, prob, np.random.default_rng(C.SEED)),
                       labels_agree=metrics(y[agree], prob[agree]))
            json.dump(res, open(OUT / f"{source_name}__{run}.json", "w"), indent=1)
            df.assign(pred=prob.argmax(1), **{f"p{k}": prob[:, k] for k in range(4)}) \
              .to_csv(OUT / f"{source_name}__{run}.csv", index=False)
            a = res["all"]
            rows.append(dict(source=source_name, run=run, input=size, accuracy=a["accuracy"],
                             accuracy_ci95=a["accuracy_ci95"], balanced_accuracy=a["balanced_accuracy"],
                             macro_f1=a["macro_f1"], qwk=a["qwk"], qwk_ci95=a["qwk_ci95"],
                             recall_0_1_2_3=a["recall_per_class"], predicted_share=a["predicted_share"]))
            print(f"{source_name} {run}: acc {a['accuracy']:.3f} {a['accuracy_ci95']}  bal.acc "
                  f"{a['balanced_accuracy']:.3f}  QWK {a['qwk']:.3f}  recall {a['recall_per_class']}", flush=True)
            del model
            U.tf.keras.backend.clear_session()
    if rows:
        summ = OUT / "summary.csv"
        new = pd.DataFrame(rows)
        if summ.exists():                           # keep runs evaluated in earlier invocations
            old = pd.read_csv(summ)
            new = pd.concat([old[~(old.source + old.run).isin(new.source + new.run)], new])
        new.to_csv(summ, index=False)


if __name__ == "__main__":
    sys.exit(main())
