"""Experiment 6 (v2): few-shot target adaptation of EfficientNetV2B0__ce (224 px) to HER2-IHC-40x.

Question: how much does a handful of LABELLED target tiles help, compared with zero-shot, HARMONISE (stain_p99) and
AdaBN (unsupervised)?  Unlike those two, this uses target labels.

Data (no overlap between roles):
  support pool  HARMONISE-1 fit + selection pools (3,500 development tiles)
  tuning set    R_cal (2,191 development tiles; HARMONISE-2 split by index blocks) -- every hyperparameter choice
  dev report    R_eval (2,402 development tiles)
  test          1,847-tile external WSI test split: EXPLORATORY SECOND LOOK (examined before); one evaluation per
                final configuration, never used for any choice.
All development tiles come from the same 86 slides (no slide IDs), so R_cal / R_eval may share slides with the
support tiles and flatter the method. The test split is slide-independent (21 other slides).

Regimes (BatchNorm stays in inference mode, as in FYP3 training; the source head initialises the head):
  head     only the final Dense layer, trained on frozen pooled features (full batch, no augmentation)
  partial  'block6*' + 'top_*' layers + head, image-level, dihedral flips/rotations as augmentation
  full     every layer, image-level, same augmentation
Protocol:
  stage 'tune'  k = 10 per class, seeds 101-103, small grid per regime, mean balanced accuracy on R_cal picks
                the configuration (ties -> fewer steps);
  stage 'final' k in {1, 5, 10, 20}, seeds 1-5 (different draws from tuning), chosen configuration only.
Support sampling: k tiles per class without replacement, numpy default_rng(seed).
Runs on the Mac GPU (224 px training worked there for every FYP3 run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["tune", "final"], required=True)
ap.add_argument("--regimes", nargs="+", default=["head", "partial", "full"])
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
tf = L.tf_setup()
import common as U  # noqa: E402
import config as C  # noqa: E402
import train as T  # noqa: E402

OUT = L.out_dir("06_fewshot")
CACHE = L.V2 / "_cache"
RUN = "EfficientNetV2B0__ce"
GRID = {
    "head": [dict(lr=1e-3, steps=50), dict(lr=1e-3, steps=200), dict(lr=1e-2, steps=50), dict(lr=1e-2, steps=200)],
    "partial": [dict(lr=1e-4, steps=30), dict(lr=1e-4, steps=100), dict(lr=1e-3, steps=30), dict(lr=1e-3, steps=100)],
    "full": [dict(lr=1e-5, steps=30), dict(lr=1e-5, steps=100), dict(lr=1e-4, steps=30), dict(lr=1e-4, steps=100)],
}
TUNE_SEEDS, FINAL_SEEDS = [101, 102, 103], [1, 2, 3, 4, 5]
KS = [1, 5, 10, 20]


def load():
    sup = [dict(np.load(CACHE / f"dev224_{p}.npz", allow_pickle=True)) for p in ["fit", "sel"]]
    xs, ys = np.concatenate([s["x"] for s in sup]), np.concatenate([s["y"] for s in sup]).astype(int)
    R = dict(np.load(CACHE / "dev224_R.npz", allow_pickle=True))
    ri = pd.read_csv(L.H2OUT / "R_index.csv")
    assert (ri.name.values == R["names"]).all()
    cal, ev = np.where(ri.subset == "R_cal")[0], np.where(ri.subset == "R_eval")[0]
    names = sorted(str(p.relative_to(L.EXT_TEST_DIR)) for p in L.EXT_TEST_DIR.glob("class_*/*.png"))
    tx = np.stack([np.asarray(Image.open(L.ROOT / "data" / "her2_40x_256_eval224" / n).convert("RGB"), np.uint8) for n in names])
    ty = np.array([L.ext_label(n) for n in names])
    return dict(xs=xs, ys=ys, xcal=R["x"][cal], ycal=R["y"][cal].astype(int), xev=R["x"][ev], yev=R["y"][ev].astype(int),
                xt=tx, yt=ty, tnames=np.array(names))


def support(ys, k, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.where(ys == c)[0], k, replace=False) for c in range(4)])


def build():
    C.IMAGE_SIZE = 224
    train_model, infer, bb = U.build_model("EfficientNetV2B0", "ce")
    infer.load_weights(L.RUNS / RUN / "model.weights.h5")
    feat = tf.keras.Model(infer.input, infer.get_layer("dropout").input)   # pooled backbone features
    return train_model, infer, bb, feat


def predict(m, x, chunk=128):
    return np.concatenate([m.predict(x[k:k + chunk].astype(np.float32), batch_size=64, verbose=0) for k in range(0, len(x), chunk)])


def aug(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))


def head_only(W0, b0, f_tr, y_tr, cfg, seed):
    tf.keras.utils.set_random_seed(seed)
    W, b = tf.Variable(W0), tf.Variable(b0)
    opt = T.make_optimizer(cfg["lr"])
    f = tf.constant(f_tr, tf.float32); yy = tf.constant(y_tr)
    for _ in range(cfg["steps"]):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(yy, f @ W + b, from_logits=True))
        g = tape.gradient(loss, [W, b])
        opt.apply_gradients(zip(g, [W, b]))
    return W.numpy(), b.numpy()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def finetune(net, init_w, x_tr, y_tr, cfg, regime, seed):
    train_model, infer, bb, _ = net
    train_model.set_weights(init_w)
    tf.keras.utils.set_random_seed(seed)
    bb.trainable = True
    if regime == "partial":
        for l in bb.layers:
            l.trainable = l.name.startswith(("block6", "top"))
    else:
        for l in bb.layers:
            l.trainable = True
    train_model.compile(optimizer=T.make_optimizer(cfg["lr"]), loss=U.ce_loss)
    bsz = min(32, len(x_tr))
    ds = tf.data.Dataset.from_tensor_slices((x_tr.astype(np.float32), y_tr)).shuffle(len(x_tr), seed=seed).repeat() \
        .map(lambda a, b: (aug(a), b)).batch(bsz)
    train_model.fit(ds, steps_per_epoch=cfg["steps"], epochs=1, verbose=0)
    n_trainable = int(sum(np.prod(v.shape) for v in train_model.trainable_weights))
    return infer, n_trainable


def evaluate(tag, probs_by_set, sets, extra):
    rows = []
    for nm, (p, y) in probs_by_set.items():
        if nm in sets:
            rows.append(dict(extra, set=nm, **{k: v for k, v in L.class_metrics(y, p).items()}))
    return rows


def main():
    D = load()
    if args.smoke:
        D["xcal"], D["ycal"], D["xev"], D["yev"] = D["xcal"][:64], D["ycal"][:64], D["xev"][:64], D["yev"][:64]
        D["xt"], D["yt"] = D["xt"][:64], D["yt"][:64]
    net = build()
    train_model, infer, bb, feat = net
    init_w = [w.copy() for w in train_model.get_weights()]
    dense = infer.get_layer("logits")
    W0, b0 = [w.copy() for w in dense.get_weights()]
    print("support pool per class", np.bincount(D["ys"], minlength=4).tolist(), flush=True)
    # frozen features for the head-only regime (deterministic; dropout inactive at inference)
    F = {k: predict(feat, D[k]) for k in ["xs", "xcal", "xev", "xt"]}
    zero = {"R_cal": (predict(infer, D["xcal"]), D["ycal"]), "R_eval": (predict(infer, D["xev"]), D["yev"]),
            "ext_test": (predict(infer, D["xt"]), D["yt"])}
    rows = evaluate("zero", zero, ["R_cal", "R_eval"] + (["ext_test"] if args.stage == "final" else []),
                    dict(stage=args.stage, regime="zero_shot", k=0, seed=0, cfg="-"))
    sets = ["R_cal"] if args.stage == "tune" else ["R_eval", "ext_test"]
    if args.stage == "final":
        chosen = json.load(open(OUT / "tuning_choice.json"))
    plan = []
    for regime in args.regimes:
        if args.stage == "tune":
            plan += [(regime, 10, s, cfg) for cfg in GRID[regime] for s in TUNE_SEEDS]
        else:
            plan += [(regime, k, s, chosen[regime]) for k in KS for s in FINAL_SEEDS]
    if args.smoke:
        plan = plan[:2]
    test_store = {}
    for regime, k, seed, cfg in plan:
        t0 = time.time()
        idx = support(D["ys"], k, seed)
        if regime == "head":
            W, b = head_only(W0, b0, F["xs"][idx], D["ys"][idx], cfg, seed)
            probs = {"R_cal": (softmax(F["xcal"] @ W + b), D["ycal"]), "R_eval": (softmax(F["xev"] @ W + b), D["yev"]),
                     "ext_test": (softmax(F["xt"] @ W + b), D["yt"])}
            ntr = int(W.size + b.size)
        else:
            m, ntr = finetune(net, init_w, D["xs"][idx], D["ys"][idx], cfg, regime, seed)
            probs = {s: (predict(m, D[{"R_cal": "xcal", "R_eval": "xev", "ext_test": "xt"}[s]]), D[{"R_cal": "ycal", "R_eval": "yev", "ext_test": "yt"}[s]])
                     for s in sets}
        rows += evaluate(regime, probs, sets, dict(stage=args.stage, regime=regime, k=k, seed=seed, cfg=json.dumps(cfg),
                                                   trainable_params=ntr, seconds=round(time.time() - t0, 1)))
        if args.stage == "final":
            test_store[f"{regime}_k{k}_s{seed}"] = probs["ext_test"][0]
        r = rows[-1]
        print(f"{args.stage} {regime} k={k} seed={seed} {cfg}: {r['set']} bal {r['balanced_accuracy']:.3f} "
              f"f1 {r['macro_f1']:.3f} qwk {r['qwk']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        L.write_csv(OUT / f"{args.stage}_results{'_smoke' if args.smoke else ''}.csv", pd.DataFrame(rows))
    if args.smoke:
        return
    df = pd.DataFrame(rows)
    if args.stage == "tune":
        t = df[(df.regime != "zero_shot")].groupby(["regime", "cfg"]).balanced_accuracy.agg(["mean", "std"]).reset_index()
        t["steps"] = t.cfg.map(lambda c: json.loads(c)["steps"])
        choice = {r: json.loads(t[t.regime == r].sort_values(["mean", "steps"], ascending=[False, True]).cfg.iloc[0])
                  for r in t.regime.unique()}
        prev = json.load(open(OUT / "tuning_choice.json")) if (OUT / "tuning_choice.json").exists() else {}
        L.write_json(OUT / "tuning_choice.json", {**prev, **choice})
        L.write_csv(OUT / f"tuning_summary_{'_'.join(args.regimes)}.csv", t)
        print(t.to_string(index=False)); print("chosen:", choice)
    else:
        np.savez_compressed(L.guard(OUT / f"final_test_probs_{'_'.join(args.regimes)}.npz"), names=D["tnames"], y=D["yt"], **test_store)


if __name__ == "__main__":
    main()
