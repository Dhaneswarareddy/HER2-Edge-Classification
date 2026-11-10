"""Experiment 5 (v2): AdaBN (adaptive batch normalisation; Li et al., 2016) on HER2-IHC-40x. Unsupervised, no training.

What AdaBN does: every BatchNorm layer normalises with stored source-domain (BCI) means and variances. AdaBN replaces
them with the target domain's, estimated from UNLABELLED target tiles. Weights are unchanged. FYP3 trains with the
backbone called with training=False (common.build_model), so the stored statistics are the ImageNet-then-frozen ones
the network was fine-tuned with; AdaBN re-estimates exactly those.

Protocol (fixed before any result):
  * adaptation tiles: the 1,500-tile HARMONISE-1 fit pool (unlabelled use; labels never read here) -- the same tiles
    HARMONISE fitted its stain gains on, so both methods use identical unlabelled information;
  * estimation: one pass, fixed order (seed), batch 32, every BN layer in training mode with momentum t/(t+1) so the
    stored statistics become the cumulative average over all batches; Dropout inside the backbone set to rate 0 during
    the pass (stochastic depth would otherwise perturb the statistics); no hyperparameter is tuned;
  * arms: adabn (raw inputs) and adabn_p99 (stain_p99-calibrated inputs, both for estimation and evaluation);
  * development report: R (4,593 labelled development tiles; R_eval subset reported too). Nothing is selected on it;
  * test: the 1,847-tile external test split, ONE evaluation per network and arm. EXPLORATORY SECOND LOOK: the split
    was examined before. Comparators (raw, stain_p99) are the stored predictions, re-checked against fresh inference.
Run one network per process:  python run_adabn.py --run EfficientNetV2B0__ce [--cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
tf = L.tf_setup(cpu=args.cpu)
sys.path.insert(0, str(L.NEXT))
import harmonise_lib as H  # noqa: E402
import common as U  # noqa: E402
import config as C  # noqa: E402

OUT = L.out_dir("05_adabn")
CACHE = L.V2 / "_cache"
P99 = json.load(open(L.NEXT / "outputs" / "harmonisation_params.json"))["gains"]["99"]
BATCH = 32


def build(run):
    s = json.load(open(L.RUNS / run / "summary.json"))
    C.IMAGE_SIZE = s.get("image_size", 224)
    _, infer, _ = U.build_model(s["model"], s["loss"])
    infer.load_weights(L.RUNS / run / "model.weights.h5")
    return infer, C.IMAGE_SIZE, (s.get("variant_spec") or {}).get("source_size", 256)


def backbone(infer):
    return [l for l in infer.layers if isinstance(l, tf.keras.Model)][0]


def prep_layer(infer):
    names = [l.name for l in infer.layers]
    return infer.get_layer("preprocess") if "preprocess" in names else (lambda x: x)


def bn_layers(bb):
    return [l for l in bb.submodules if isinstance(l, tf.keras.layers.BatchNormalization)]


def adapt(infer, batches):
    bb, pre = backbone(infer), prep_layer(infer)
    bns = bn_layers(bb)
    drops = [l for l in bb.submodules if isinstance(l, tf.keras.layers.Dropout)]
    for d in drops:
        d.rate = 0.0
    # Keras runs a BN layer with trainable=False in inference mode even under training=True; no gradient is computed
    # here, so making them trainable only lets the moving statistics update.
    for l in bns:
        l.trainable = True
    before = [l.moving_mean.numpy().copy() for l in bns]
    for t, xb in enumerate(batches):
        for l in bns:
            l.momentum = t / (t + 1.0)
        bb(pre(tf.constant(xb, tf.float32)), training=True)
    changed = float(np.mean([np.abs(l.moving_mean.numpy() - b).mean() for l, b in zip(bns, before)]))
    return len(bns), len(drops), changed


def calib(x):
    return np.stack([H.apply_stain_gains(t, P99["g_h"], P99["g_d"]) for t in x])


def dev_arrays(size, source, names_needed=None):
    """224: from the shared cache. 448: decode from the development archive (same path as harmonise_lib.prepare)."""
    if size == 224:
        return {p: dict(np.load(CACHE / f"dev224_{p}.npz", allow_pickle=True)) for p in ["fit", "R"]}
    sys.path.insert(0, str(L.NEXT / "harmonise2"))
    import h2_lib as H2
    d, fit, _, rest = H2.h1_pools()
    import pandas as pd
    r = pd.read_csv(L.H2OUT / "R_index.csv")
    out = {}
    for p, names in [("fit", fit), ("R", list(r.name))]:
        # decoded lazily, chunk by chunk (4,593 x 448 px held at once would exhaust this 8 GB Mac)
        out[p] = dict(x=Lazy(d, names, source, size), y=np.array([H.label_from_name(n) for n in names]), names=np.array(names))
    return out


class Lazy:
    """Array-like over development tiles: x[idx] decodes 1024 px -> Lanczos source -> bilinear size, uint8 (same path
    as harmonise_lib.prepare with scale 1). Supports len(), slices and integer index arrays."""
    def __init__(self, d, names, source, size):
        self.d, self.names, self.source, self.size = d, list(names), source, size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return np.stack([self._one(n) for n in self.names[idx]])
        if np.ndim(idx) > 0:
            return np.stack([self._one(self.names[i]) for i in np.asarray(idx)])
        return self._one(self.names[int(idx)])

    def _one(self, n):
        im = self.d.read(n).resize((self.source, self.source), Image.LANCZOS).resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(im, np.uint8)


def test_arrays(size):
    names = sorted(str(p.relative_to(L.EXT_TEST_DIR)) for p in L.EXT_TEST_DIR.glob("class_*/*.png"))
    cache = L.ROOT / "data" / ("her2_40x_512_eval448" if size == 448 else "her2_40x_256_eval224")
    x = np.stack([np.asarray(Image.open(cache / n).convert("RGB"), np.uint8) for n in names])
    return np.array(names), np.array([L.ext_label(n) for n in names]), x


def predict(model, x, f=None, chunk=64):
    """f (e.g. calib) is applied per chunk, so no full float32 copy is held (448 px R would need ~11 GB)."""
    f = f or (lambda a: a)
    return np.concatenate([model.predict(np.asarray(f(x[k:k + chunk]), np.float32), batch_size=16, verbose=0)
                           for k in range(0, len(x), chunk)])


def main():
    run = args.run
    infer, size, source = build(run)
    dev = dev_arrays(size, source)
    tn, ty, tx = test_arrays(size)
    rng = np.random.default_rng(L.SEED)
    order = rng.permutation(len(dev["fit"]["x"]))
    res = dict(run=run, input_size=size, adaptation_pool="HARMONISE-1 fit pool", n_adapt=len(order), batch=BATCH,
               seed=L.SEED, stain_p99=P99)

    # comparators re-checked with fresh inference (unadapted network)
    p_raw = predict(infer, tx)
    p_p99 = predict(infer, tx, calib)
    import pandas as pd
    ref_raw = pd.read_csv(L.EXT / f"BCI__{run}.csv").set_index("path").loc[tn, [f"p{k}" for k in range(4)]].values
    ref_p99 = pd.read_csv(L.H1TEST / f"{run}.csv").set_index("path").loc[tn, [f"p{k}" for k in range(4)]].values
    res["check_raw_vs_stored"] = float(np.abs(p_raw - ref_raw).max())
    res["check_p99_vs_stored"] = float(np.abs(p_p99 - ref_p99).max())
    print(f"{run}: fresh vs stored  raw {res['check_raw_vs_stored']:.2e}  stain_p99 {res['check_p99_vs_stored']:.2e}", flush=True)
    preds = {"raw": p_raw, "stain_p99": p_p99}
    dev_preds = {}

    for arm, f in [("adabn", lambda a: a), ("adabn_p99", calib)]:
        net, _, _ = build(run)
        xs = dev["fit"]["x"]
        n_bn, n_drop, changed = adapt(net, (np.asarray(f(xs[order[k:k + BATCH]]), np.float32) for k in range(0, len(order), BATCH)))
        res[f"{arm}_bn_layers"], res[f"{arm}_dropouts_disabled"], res[f"{arm}_mean_abs_change_moving_mean"] = n_bn, n_drop, changed
        print(f"{run} {arm}: {n_bn} BN layers re-estimated (mean |change| {changed:.4f}); {n_drop} dropout layers off", flush=True)
        dev_preds[arm] = predict(net, dev["R"]["x"], f)
        preds[arm] = predict(net, tx, f)
        del net
        tf.keras.backend.clear_session()

    # development report (no selection is made on it)
    r = pd.read_csv(L.H2OUT / "R_index.csv")
    assert (r.name.values == dev["R"]["names"]).all()
    yR = dev["R"]["y"].astype(int)
    for nm in ["none", "frozen"]:
        dev_preds["raw" if nm == "none" else "stain_p99"] = np.load(L.H2OUT / "preds" / "C" / f"{run}__{nm}.npz")["prob"]
    dev_rows = []
    for arm, p in dev_preds.items():
        for subset, idx in [("R_all", np.arange(len(yR))), ("R_eval", np.where(r.subset == "R_eval")[0])]:
            dev_rows.append(dict(run=run, arm=arm, subset=subset, **{k: v for k, v in L.class_metrics(yR[idx], p[idx]).items()}))
    test_rows = [dict(run=run, arm=arm, **L.class_metrics(ty, p)) for arm, p in preds.items()]

    # paired differences on the test split, filename-run bootstrap (optimistic: no slide IDs)
    g = L.filename_groups(list(tn))
    from sklearn.metrics import recall_score, f1_score
    pairs = []
    for a, b in [("adabn", "raw"), ("adabn", "stain_p99"), ("adabn_p99", "stain_p99"), ("adabn_p99", "raw")]:
        pa, pb = preds[a].argmax(1), preds[b].argmax(1)
        ba = lambda yy, pp: recall_score(yy, pp, average="macro", labels=range(4), zero_division=0)
        f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=range(4), zero_division=0)
        d_b, d_f = [], []
        for i in L.cluster_draws(g, 2000, L.SEED):
            d_b.append(ba(ty[i], pa[i]) - ba(ty[i], pb[i])); d_f.append(f1(ty[i], pa[i]) - f1(ty[i], pb[i]))
        pairs.append(dict(run=run, a=a, b=b, delta_balanced_accuracy=ba(ty, pa) - ba(ty, pb), ci95_bal=L.ci95(d_b),
                          delta_macro_f1=f1(ty, pa) - f1(ty, pb), ci95_f1=L.ci95(d_f)))
    res.update(dev=dev_rows, test=test_rows, paired_test=pairs)
    L.write_json(OUT / f"adabn_{run}.json", res)
    np.savez_compressed(L.guard(OUT / f"preds_{run}.npz"), names=tn, y=ty, **{f"test_{k}": v for k, v in preds.items()},
                        **{f"devR_{k}": v for k, v in dev_preds.items()})
    for t in test_rows:
        print(f"TEST {run} {t['arm']:10s} acc {t['accuracy']:.3f} bal {t['balanced_accuracy']:.3f} f1 {t['macro_f1']:.3f} "
              f"qwk {t['qwk']:.3f} recall {[round(t['recall_' + l], 3) for l in L.LEVELS]}", flush=True)
    for t in dev_rows:
        if t["subset"] == "R_eval":
            print(f"DEV  {run} {t['arm']:10s} bal {t['balanced_accuracy']:.3f} f1 {t['macro_f1']:.3f} qwk {t['qwk']:.3f}", flush=True)


if __name__ == "__main__":
    main()
