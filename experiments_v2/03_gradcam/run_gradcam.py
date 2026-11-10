"""Experiment 3 (v2): Grad-CAM for the two deployment models -- the gaps left by src/analysis_models.py.

Already done (reused, not repeated): Grad-CAM attention statistics for the nine 224 px baseline networks on BCI and
FYP2 test images, with figures of two CORRECT tiles per class (results/analysis/models/*.json, results/figures/figD6_*).

New here:
  * models: EfficientNetV2B0__ce__hires (448 px, deployed primary) and EfficientNetV2B0__ce (224 px, external/HARMONISE);
  * images: BCI grouped test (758), HER2-IHC-40x test (1,847; raw, and stain_p99 for the 224 px model) -- the external
    split was examined before, and nothing here is fitted or selected on it;
  * correct AND incorrect predictions, a 2+ focus figure, and the highest-background-attention tiles;
  * three panels per example: original, heatmap, overlay.

Method (same as analysis_models.gradcam): gradient of the predicted-class probability with respect to the last 4-D
backbone activation ('top_activation'); channel weights = spatially averaged gradients; ReLU; bilinear upsampling;
each map normalised to sum 1. The map is 7x7 (224 px) or 14x14 (448 px) before upsampling, so it localises regions
of roughly 32 px, not membranes. Grad-CAM shows where the score is sensitive, not what the network 'recognises'.

Per-tile statistics, as in analysis_models.cam_stats:
  dab_ratio   = sum(cam * DAB) / mean(DAB): > 1 means attention sits on above-average DAB (brown) pixels;
  bg_ratio    = (cam mass on background) / (background area share), background = HSV saturation < 8, tiles with >= 5%
                background only: < 1 means background is under-attended;
  top_dab_mass = cam mass on the 10% most DAB-dense pixels (1.0 would be all attention there; 0.10 = uniform).
DAB is measured relative to each tile's own illumination (Ruifrok-Johnston vectors), as everywhere in FYP3.
Run on CPU (tensorflow-metal hangs on 448 px inference on this Mac).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

tf = L.tf_setup(cpu=True)
sys.path.insert(0, str(L.NEXT))
import harmonise_lib as H  # noqa: E402
import common as U  # noqa: E402
import config as C  # noqa: E402

OUT = L.out_dir("03_gradcam")
RUN448, RUN224 = "EfficientNetV2B0__ce__hires", "EfficientNetV2B0__ce"
P99 = json.load(open(L.NEXT / "outputs" / "harmonisation_params.json"))


def build(run):
    s = json.load(open(L.RUNS / run / "summary.json"))
    C.IMAGE_SIZE = s.get("image_size", 224)
    _, infer, _ = U.build_model(s["model"], s["loss"])
    infer.load_weights(L.RUNS / run / "model.weights.h5")
    bb = [l for l in infer.layers if isinstance(l, tf.keras.Model)][0]
    inner = tf.keras.Model(bb.input, [bb.get_layer("top_activation").get_output_at(0), bb.output])
    return infer, inner, infer.get_layer("logits"), infer.get_layer("probs")


def gradcam(net, x, bs=16):
    _, inner, dense, head = net
    cams, probs = [], []
    for k in range(0, len(x), bs):
        xb = tf.constant(x[k:k + bs], tf.float32)
        with tf.GradientTape() as tape:
            conv, pooled = inner(xb, training=False)
            tape.watch(conv)
            p = head(dense(pooled, training=False))
            cls = tf.argmax(p, axis=1)
            score = tf.gather(p, cls, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        w = tf.reduce_mean(g, axis=[1, 2], keepdims=True)
        cams.append(tf.nn.relu(tf.reduce_sum(w * conv, axis=-1)).numpy())
        probs.append(p.numpy())
    return np.concatenate(cams), np.concatenate(probs)


def upsample(cam, size):
    c = tf.image.resize(cam[None, ..., None], [size, size], method="bilinear")[0, ..., 0].numpy()
    s = c.sum()
    return c / s if s > 0 else c


def tile_stats(img, cam_up):
    I0, bg = H.illumination(img.astype(np.uint8))
    od = -np.log(np.maximum(img.astype(np.float32) / I0, 1e-6))
    dab = np.clip((od.reshape(-1, 3) @ H.HED_FROM_RGB)[:, 2], 0, None).reshape(img.shape[:2])
    uniform = cam_up.sum() == 0
    top = dab >= np.quantile(dab, 0.9)
    return dict(dab_ratio=np.nan if uniform else float((cam_up * dab).sum() / max(dab.mean(), 1e-6)),
                bg_ratio=float((cam_up * bg).sum() / bg.mean()) if bg.mean() >= 0.05 and not uniform else np.nan,
                bg_area=float(bg.mean()), top_dab_mass=np.nan if uniform else float(cam_up[top].sum()))


# ------------------------------------------------------------------ image sources (same preprocessing as training/eval)
def bci_test(size):
    m = L.manifest(); m = m[m.split == "test"].reset_index(drop=True)
    root = L.ROOT / "data" / ("ihc_512_eval448" if size == 448 else "ihc_256")
    def load(p):
        im = Image.open(root / p).convert("RGB")
        return np.asarray(im if im.size == (size, size) else im.resize((size, size), Image.BILINEAR), np.uint8)
    return list(m.path), m.label.values.astype(int), load


def ext_test(size, calibrated=False):
    names = H.test_tiles()
    y = np.array([H.label_from_name(n) for n in names])
    cache = L.ROOT / "data" / ("her2_40x_512_eval448" if size == 448 else "her2_40x_256_eval224")
    if not calibrated:
        return names, y, lambda n: np.asarray(Image.open(cache / n).convert("RGB"), np.uint8)
    info = H.run_info(RUN224)
    arm = dict(scale=1.0, g_h=P99["gains"]["99"]["g_h"], g_d=P99["gains"]["99"]["g_d"])
    # float32 exactly as harmonise_lib.prepare returns it (HARMONISE-1 fed floats; a uint8 cast changes 1.4% of predictions)
    return names, y, lambda n: H.prepare(H.read_test(n), info, arm)


def run_set(tag, net, size, names, y, load, ref_probs=None, chunk=64):
    rows, cams_lo, probs = [], [], []
    for k in range(0, len(names), chunk):
        x = np.stack([load(n) for n in names[k:k + chunk]])
        cam, p = gradcam(net, x.astype(np.float32))
        for j in range(len(x)):
            st = tile_stats(x[j], upsample(cam[j], size))
            rows.append(dict(set=tag, name=names[k + j], label=int(y[k + j]), pred=int(p[j].argmax()),
                             confidence=float(p[j].max()), **st))
        cams_lo.append(cam); probs.append(p)
        print(f"  {tag}: {min(k + chunk, len(names))}/{len(names)}", flush=True)
    probs = np.concatenate(probs)
    check = None
    if ref_probs is not None:
        check = float(np.abs(probs - ref_probs).max())
        print(f"  {tag}: max |p - stored p| = {check:.2e}", flush=True)
    np.savez_compressed(L.guard(OUT / f"cams_{tag}.npz"), cam=np.concatenate(cams_lo), prob=probs, names=np.array(names))
    return pd.DataFrame(rows), check


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-p99", action="store_true", help="recompute only the stain_p99 224 px set and merge it")
    a = ap.parse_args()
    if a.only_p99:
        net = build(RUN224)
        names, y, load = ext_test(224, calibrated=True)
        ref = pd.read_csv(L.H1TEST / f"{RUN224}.csv").set_index("path").loc[names, [f"p{k}" for k in range(4)]].values
        df, chk = run_set("ext_test_p99_224", net, 224, names, y, load, ref)
        old = pd.read_csv(OUT / "cam_stats_per_tile.csv")
        stats = pd.concat([old[old.set != "ext_test_p99_224"], df.assign(model=RUN224)])
        checks = json.load(open(OUT / "reproduction_check.json")); checks["ext_p99_224"] = chk
        checks["ext_p99_224_uint8_cast_first_run"] = 0.1525286947597504
        L.write_json(OUT / "reproduction_check.json", checks)
        return summarise(stats)
    L.write_json(OUT / "config.json", dict(experiment="03_gradcam", layer="top_activation", target="predicted class",
                                           models=[RUN448, RUN224], bg_rule="HSV S < 8", bg_min_area=0.05,
                                           top_dab_quantile=0.9, device="CPU", stain_p99=P99["gains"]["99"],
                                           external_status="exploratory second look; nothing fitted on test"))
    checks, frames = {}, []
    for run, size in [(RUN448, 448), (RUN224, 224)]:
        net = build(run)
        names, y, load = bci_test(size)
        ref = np.load(L.RUNS / run / "predictions.npz")["test"]
        df, checks[f"BCI_{size}"] = run_set(f"BCI_test_{size}", net, size, names, y, load, ref)
        frames.append(df.assign(model=run))
        names, y, load = ext_test(size)
        ref = pd.read_csv(L.EXT / f"BCI__{run}.csv").set_index("path").loc[names, [f"p{k}" for k in range(4)]].values
        df, checks[f"ext_raw_{size}"] = run_set(f"ext_test_raw_{size}", net, size, names, y, load, ref)
        frames.append(df.assign(model=run))
        if size == 224:
            names, y, load = ext_test(224, calibrated=True)
            ref = pd.read_csv(L.H1TEST / f"{run}.csv").set_index("path").loc[names, [f"p{k}" for k in range(4)]].values
            df, checks["ext_p99_224"] = run_set("ext_test_p99_224", net, 224, names, y, load, ref)
            frames.append(df.assign(model=run))
        del net
        tf.keras.backend.clear_session()
    stats = pd.concat(frames)
    L.write_json(OUT / "reproduction_check.json", checks)
    summarise(stats)


def summarise(stats):
    L.write_csv(OUT / "cam_stats_per_tile.csv", stats)
    stats["correct"] = stats.label == stats.pred
    agg = stats.groupby(["set", "label", "correct"])[["dab_ratio", "bg_ratio", "top_dab_mass", "confidence"]] \
               .median().round(4).reset_index().merge(
        stats.groupby(["set", "label", "correct"]).size().rename("n").reset_index())
    L.write_csv(OUT / "cam_stats_summary.csv", agg)
    print(L.md_table(stats.groupby(["set", "correct"])[["dab_ratio", "bg_ratio", "top_dab_mass"]].median().round(3).reset_index()))


if __name__ == "__main__":
    main()
