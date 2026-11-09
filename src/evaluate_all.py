"""Step 3b: full metrics and an overfitting check for every trained (FP32) run.

  python evaluate_all.py                 # every finished run (each in its own process), then aggregate
  python evaluate_all.py --run ResNet50__ce

For each run the restored best weights are evaluated on the *clean* (un-augmented) train
split as well as val and test, so the train-vs-held-out gap is measured like-for-like
(history.csv only has the augmented, dropout-on training loss).

Outputs
  results/metrics/<run>.json      every metric, per split, with bootstrap 95% CIs on val/test
  results/metrics_all.csv         one row per (run, split)
  results/overfitting.csv         one row per run: gaps, learning-curve checks, verdict

Overfitting verdict (all on the restored best weights), FYP2's rule unchanged:
  gap_acc   = train acc - test acc            (OK <= 0.02, i.e. ~7 of 360 test images)
  val_test  = |val acc - test acc|            (OK <= 0.02: selection did not overfit val)
  nll_gap   = test log-loss - train log-loss  (OK <= 0.10)
  "none" if all three are OK, "mild" if one fails, "overfit" otherwise.

BCI differs from FYP2 in one way that matters here: tiles cut from one slide region are correlated, so
CIs come from a cluster bootstrap over slide fragments (analysis_slides.py), not over tiles. A second
verdict, verdict_noise_aware, counts a criterion as failed only if it also exceeds that sampling noise:
  gap_acc   fails only if train acc is above the test CI
  val_test  fails only if val acc > test acc and the cluster CI of (val - test) excludes 0
            (selection overfitting makes val look better than test; a harder val set does not)
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C

METRICS_DIR = C.RESULTS_DIR / f"metrics{C.SUFFIX}"
GAP_T, VALTEST_T, NLL_T = 0.02, 0.02, 0.10
N_BOOT = 2000


def extended_metrics(y, probs):
    """compute_metrics plus precision/specificity/F1 per class, MCC, log-loss, Brier, ECE."""
    import common as U
    from sklearn.metrics import f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support

    m = U.compute_metrics(y, probs)
    pred = probs.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(C.NUM_CLASSES), zero_division=0)
    cm = np.array(m["confusion_matrix"])
    for k in range(C.NUM_CLASSES):
        tn = cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]
        m[f"precision_{k}"] = float(prec[k])
        m[f"f1_{k}"] = float(f1[k])
        m[f"specificity_{k}"] = float(tn / max(cm.sum() - cm[k].sum(), 1))
    m["weighted_f1"] = float(f1_score(y, pred, average="weighted"))
    m["mcc"] = float(matthews_corrcoef(y, pred))
    p = np.clip(probs, 1e-7, 1.0)
    p = p / p.sum(1, keepdims=True)
    m["log_loss"] = float(log_loss(y, p, labels=range(C.NUM_CLASSES)))
    m["brier"] = float(np.mean(np.sum((p - np.eye(C.NUM_CLASSES)[y]) ** 2, 1)))
    conf, correct = p.max(1), (pred == y)
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 14)
    m["ece"] = float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * np.mean(idx == b)
                         for b in range(15) if np.any(idx == b)))
    return m


def cluster_draws(groups, seed):
    """N_BOOT bootstrap index draws that resample whole slide fragments with replacement."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    _, start, count = np.unique(groups[order], return_index=True, return_counts=True)
    members = [order[s:s + c] for s, c in zip(start, count)]
    for _ in range(N_BOOT):
        yield np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def bootstrap_ci(y, probs, groups, seed=C.SEED):
    from sklearn.metrics import cohen_kappa_score, f1_score
    pred = probs.argmax(1)
    acc, qwk, f1 = [], [], []
    for i in cluster_draws(groups, seed):
        acc.append(np.mean(pred[i] == y[i]))
        qwk.append(cohen_kappa_score(y[i], pred[i], weights="quadratic") if len(set(y[i])) > 1 else 1.0)
        f1.append(f1_score(y[i], pred[i], average="macro"))
    return {"accuracy_ci95": ci95(acc), "qwk_ci95": ci95(qwk), "macro_f1_ci95": ci95(f1)}


def ci95(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def diff_ci(ok_a, groups_a, ok_b, groups_b, seed=C.SEED):
    """Cluster-bootstrap 95% CI of accuracy(a) - accuracy(b) for two independent held-out sets."""
    return ci95([ok_a[i].mean() - ok_b[j].mean()
                 for i, j in zip(cluster_draws(groups_a, seed), cluster_draws(groups_b, seed + 1))])


def learning_curve_checks(run_dir, best_epoch):
    """Did validation loss climb after the chosen epoch (the classic overfitting signature)?"""
    h = pd.read_csv(run_dir / "history.csv")
    best = h[h.epoch == best_epoch].iloc[0]
    after = h[h.epoch > best_epoch]
    return dict(min_val_loss=float(h.val_loss.min()), min_val_loss_epoch=int(h.loc[h.val_loss.idxmin(), "epoch"]),
                val_loss_at_best=float(best.val_loss), train_loss_at_best=float(best.train_loss),
                val_loss_last=float(h.val_loss.iloc[-1]), train_loss_last=float(h.train_loss.iloc[-1]),
                val_loss_rise_after_best=float(after.val_loss.max() - best.val_loss) if len(after) else 0.0,
                val_qwk_drop_after_best=float(best.val_qwk - after.val_qwk.min()) if len(after) else 0.0)


def predict_split(model, split, source_dir, size, chunk=256):
    """Predict one split in chunks (PIL bilinear source -> network size, as common.load_split does),
    so a 3,311-image training split never sits in memory at once. Also returns each tile's slide fragment."""
    from PIL import Image
    df = pd.read_csv(C.MANIFEST)
    df = df[df.split == split].reset_index(drop=True)
    frag = pd.read_csv(C.SHARED_ANALYSIS / "slide_fragments.csv").set_index("path").fragment
    probs = []
    for k in range(0, len(df), chunk):
        x = np.stack([np.asarray(Image.open(source_dir / p).convert("RGB").resize((size, size), Image.BILINEAR))
                      for p in df.path[k:k + chunk]])
        probs.append(model.predict(x.astype(np.float32), batch_size=32, verbose=0))
    return df.label.values.astype(np.int32), np.concatenate(probs), df.path.map(frag).values


def needs_cpu(run):
    """448 px networks hang in inference on the Mac GPU (tensorflow-metal; 6.5 h stuck at 13 GB), so on macOS
    they run on CPU. Other platforms (CUDA) keep the GPU."""
    return sys.platform == "darwin" and \
        json.load(open(C.RUNS_DIR / run / "summary.json")).get("image_size", C.IMAGE_SIZE) > C.IMAGE_SIZE


def evaluate_run(run, cpu=False):
    if cpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    import common as U
    run_dir = C.RUNS_DIR / run
    s = json.load(open(run_dir / "summary.json"))
    size = s.get("image_size", C.IMAGE_SIZE)
    source_dir = C.ROOT / "data" / f"ihc_{(s.get('variant_spec') or {}).get('source_size', C.SOURCE_SIZE)}"
    C.IMAGE_SIZE = size                     # load_trained may rebuild the architecture at this size
    model = U.load_trained(run_dir)
    out = dict(run=run, model=s["model"], loss=s["loss"], variant=s.get("variant"), tier=s["tier"],
               params=s["params"], best_epoch=s["best_epoch"], epochs_run=s["epochs_run"],
               train_time_s=s["train_time_s"])
    held_out = {}
    for split in ["train", "val", "test"]:
        y, probs, groups = predict_split(model, split, source_dir, size)
        m = extended_metrics(y, probs)
        if split != "train":
            m.update(bootstrap_ci(y, probs, groups))
            held_out[split] = (probs.argmax(1) == y, groups)
        out[split] = m
    out["val_test_acc_diff_ci95"] = diff_ci(*held_out["val"], *held_out["test"])
    out["learning_curve"] = learning_curve_checks(run_dir, s["best_epoch"])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(METRICS_DIR / f"{run}.json", "w"), indent=2)
    print(f"{run}: train acc {out['train']['accuracy']:.4f}  val {out['val']['accuracy']:.4f}  "
          f"test {out['test']['accuracy']:.4f}", flush=True)


def verdict(gap_fail, valtest_fail, nll_fail):
    return ["none", "mild", "overfit", "overfit"][gap_fail + valtest_fail + nll_fail]


def aggregate():
    rows, ov = [], []
    for p in sorted(METRICS_DIR.glob("*.json")):
        d = json.load(open(p))
        for split in ["train", "val", "test"]:
            m = {k: v for k, v in d[split].items() if k != "confusion_matrix"}
            for k in [k for k in m if k.endswith("_ci95")]:
                m[k.replace("_ci95", "_ci_lo")], m[k.replace("_ci95", "_ci_hi")] = m.pop(k)
            rows.append(dict(run=d["run"], model=d["model"], loss=d["loss"], variant=d.get("variant"),
                             tier=d["tier"], split=split,
                             **m, confusion_matrix=json.dumps(d[split]["confusion_matrix"])))
        tr, va, te, lc = d["train"], d["val"], d["test"], d["learning_curve"]
        gap, valtest = tr["accuracy"] - te["accuracy"], abs(va["accuracy"] - te["accuracy"])
        nll_gap = te["log_loss"] - tr["log_loss"]
        (vt_lo, vt_hi), (te_lo, te_hi) = d["val_test_acc_diff_ci95"], te["accuracy_ci95"]
        fyp2_rule = (gap > GAP_T, valtest > VALTEST_T, nll_gap > NLL_T)
        noise_aware = (gap > GAP_T and tr["accuracy"] > te_hi,
                       va["accuracy"] - te["accuracy"] > VALTEST_T and vt_lo > 0, nll_gap > NLL_T)
        ov.append(dict(run=d["run"], model=d["model"], loss=d["loss"], variant=d.get("variant"), tier=d["tier"],
                       best_epoch=d["best_epoch"], epochs_run=d["epochs_run"],
                       train_acc=tr["accuracy"], val_acc=va["accuracy"], test_acc=te["accuracy"],
                       train_qwk=tr["qwk"], val_qwk=va["qwk"], test_qwk=te["qwk"],
                       gap_train_test_acc=gap, gap_train_val_acc=tr["accuracy"] - va["accuracy"],
                       val_test_acc_diff=valtest, train_log_loss=tr["log_loss"], val_log_loss=va["log_loss"],
                       test_log_loss=te["log_loss"], nll_gap=nll_gap, test_ece=te["ece"],
                       test_acc_ci_lo=te_lo, test_acc_ci_hi=te_hi,
                       val_minus_test_acc=va["accuracy"] - te["accuracy"],
                       val_minus_test_ci_lo=vt_lo, val_minus_test_ci_hi=vt_hi,
                       **lc, verdict=verdict(*fyp2_rule), verdict_noise_aware=verdict(*noise_aware)))
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"metrics_all{C.SUFFIX}.csv", index=False)
    o = pd.DataFrame(ov)
    o.to_csv(C.RESULTS_DIR / f"overfitting{C.SUFFIX}.csv", index=False)
    print(o[["run", "train_acc", "val_acc", "test_acc", "gap_train_test_acc", "val_minus_test_acc", "nll_gap",
             "verdict", "verdict_noise_aware"]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.run:
        evaluate_run(args.run, args.cpu)
        return
    runs = sorted(p.name for p in C.RUNS_DIR.iterdir() if (p / "summary.json").exists())
    for r in runs:
        if args.force or not (METRICS_DIR / f"{r}.json").exists():
            # one process per run: releases TF/Metal memory between models (8 GB machine)
            subprocess.run([sys.executable, __file__, "--run", r] + (["--cpu"] if needs_cpu(r) else []), check=False)
    aggregate()


if __name__ == "__main__":
    main()
