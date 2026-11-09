"""Diagnosis C: what did the networks learn on FYP2 (GAN) vs BCI (real)?  (TensorFlow; run alone)

For each architecture trained with CE on BOTH datasets (FYP2 weights are read-only; the network is
rebuilt from the identical FYP2 code and its saved weights are loaded and verified against FYP2's
saved test predictions):

  1. Frozen ImageNet linear probe (EfficientNetB0, no fine-tuning): logistic regression fitted on each
     dataset's train split, scored on its test split. Measures how separable each dataset is in generic
     visual features before any task-specific learning.
  2. Embedding separability of the trained model (pooled backbone features, own test split):
     kNN accuracy (k=10, cosine; fitted on own train split), silhouette (cosine), and the ratio of
     between-class to within-class scatter.  Plus t-SNE maps.
  3. Cross-domain transfer: FYP2-trained model on BCI test, BCI-trained model on FYP2 test.
  4. Linear CKA (Kornblith et al. 2019) between the FYP2-trained and BCI-trained networks at five
     depths (global-average-pooled activations), and between each trained network and its ImageNet
     initialisation: where do the two learned representations diverge, and how far did each move?
  5. Grad-CAM (predicted class, last convolutional block), quantified per image on the own test split:
       DAB ratio  = attention-weighted mean DAB optical density / plain mean DAB  (>1: attends to stain)
       background ratio = attention share on background pixels / background area share
                          (<1: avoids background; only tiles with >= 5% background)
     BCI tiles also split by scan type (bright vs grey background, threshold from analysis_labels.py).
  6. Stain-perturbation sensitivity (counterfactual, own test split): stain concentrations are scaled in
     Ruifrok-Johnston optical-density space, relative to each image's own background illumination (DAB
     only, haematoxylin only, or all stains together, which mimics a thicker/thinner section), or the
     illumination itself is scaled (a brighter/darker scan). The shift of the expected score
     sum_k k*p_k is measured. A network that reads HER2 status from DAB should respond to DAB and hardly
     to haematoxylin, overall darkness or scan brightness.

Outputs: results/analysis/models/<arch>.json, results/analysis/model_summary.csv, cka.csv,
         stain_sensitivity.csv, figures/figD4_*.png ... figD7_*.png, figD10_stain_sensitivity.png
Usage:   python analysis_models.py [--archs EfficientNetB0 MobileNetV3Small ...] [--force]
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import cohen_kappa_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as U
import config as C
from analysis_data import _HED_FROM_RGB, _RGB_FROM_HED, rgb2hed

OUT = C.ANALYSIS_DIR
MOUT = OUT / "models"
FIG = C.RESULTS_DIR / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DS = ["FYP2", "BCI"]
DS_LABEL = {"FYP2": "FYP2 (GAN)", "BCI": "BCI (real)"}
CLASS_COLORS = ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]     # sequential: score 0 -> 3+
FYP2_DATA = C.FYP2_ROOT / "HER2 SCORES DATA" / "Generated_HER2_images"
STAINS = {"DAB": [0, 0, 1], "haematoxylin": [1, 0, 0], "all stains": [1, 1, 1]}
SCALES = [0.6, 0.8, 1.25, 1.5]           # stain concentration multipliers
ILLUM = [0.85, 0.92, 1.08, 1.15]         # illumination (scan brightness) multipliers


# ----------------------------------------------------------------------------- data

SMOKE = 0          # >0: only the first SMOKE images per split (pipeline test)


def load_images(ds, split, size=C.IMAGE_SIZE):
    if ds == "FYP2":
        m = pd.read_csv(C.FYP2_ROOT / "splits" / "split_manifest.csv")
        root = FYP2_DATA
    else:
        m = pd.read_csv(C.MANIFEST)
        root = C.DATA_DIR
    m = m[m.split == split].reset_index(drop=True)
    m["pos"] = np.arange(len(m))          # row in the full split (= row in the saved predictions)
    if SMOKE:
        m = m.groupby("label").head(SMOKE // 4).reset_index(drop=True)
    x = np.stack([np.asarray(Image.open(root / p).convert("RGB").resize((size, size), Image.BILINEAR))
                  for p in m.path])
    return x, m.label.values.astype(int), m


def predict(model, x, bs=32, chunk=512):
    """Chunked so a whole train split is never converted to float32 at once (3,388 tiles = 2 GB)."""
    return np.concatenate([model.predict(x[k:k + chunk].astype(np.float32), batch_size=bs, verbose=0)
                           for k in range(0, len(x), chunk)])


def metrics(y, prob):
    p = prob.argmax(1)
    return dict(accuracy=float((p == y).mean()), macro_f1=float(f1_score(y, p, average="macro")),
                qwk=float(cohen_kappa_score(y, p, weights="quadratic")),
                confusion=pd.crosstab(pd.Series(y), pd.Series(p)).reindex(index=range(4), columns=range(4),
                                                                           fill_value=0).values.tolist())


# ----------------------------------------------------------------------------- models

def run_dir(ds, arch):
    root = C.FYP2_ROOT / "results" / "runs" if ds == "FYP2" else C.RUNS_DIR
    return root / f"{arch}__ce"


def has_weights(ds, arch):
    return any((run_dir(ds, arch) / f).exists() for f in ["model.weights.h5", "model.keras"])


def load_model(ds, arch):
    """Rebuilt from code, so both datasets' networks have identical layer names. Most FYP2 runs saved
    only model.keras: its weights are copied into the rebuilt network (checked by the reload test)."""
    _, infer, _ = U.build_model(arch, "ce")
    d = run_dir(ds, arch)
    if (d / "model.weights.h5").exists():
        infer.load_weights(d / "model.weights.h5")
    else:
        infer.set_weights(tf.keras.models.load_model(d / "model.keras", compile=False).get_weights())
    return infer


def backbone_of(infer):
    return [l for l in infer.layers if isinstance(l, tf.keras.Model)][0]


def feature_model(infer):
    """Pooled backbone features. The nested backbone's own .output belongs to its inner graph, so the
    shared preprocess layer and backbone are called again on a new input (same weights)."""
    inp = tf.keras.Input(infer.input_shape[1:])
    return tf.keras.Model(inp, backbone_of(infer)(preprocess(infer, inp)))


def preprocess(infer, x):
    names = [l.name for l in infer.layers]
    return infer.get_layer("preprocess")(x) if "preprocess" in names else x


def depth_layers(backbone, n=5):
    """Positions of n layers with 4-D outputs at evenly spaced depths (last = final conv block).
    Positions, not names: Keras numbers unnamed layers per instance (re_lu_25 in one copy, re_lu_32 in
    the next), while every copy built from the same code has the same layer order."""
    cands = [i for i, l in enumerate(backbone.layers) if len(getattr(l, "output_shape", ()) or ()) == 4
             and not isinstance(l, tf.keras.layers.InputLayer)]
    idx = np.linspace(len(cands) * 0.15, len(cands) - 1, n).round().astype(int)
    return [cands[i] for i in idx]


def pooled_activations(infer, layers, x, bs=32):
    bb = backbone_of(infer)
    sub = tf.keras.Model(bb.input, [bb.layers[i].get_output_at(0) for i in layers])   # node 0 = inner graph
    outs = [[] for _ in layers]
    for k in range(0, len(x), bs):
        acts = sub(preprocess(infer, tf.constant(x[k:k + bs], tf.float32)), training=False)
        acts = acts if isinstance(acts, (list, tuple)) else [acts]
        for i, a in enumerate(acts):
            outs[i].append(tf.reduce_mean(a, axis=[1, 2]).numpy())
    return [np.concatenate(o) for o in outs]


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def scatter_ratio(E, y):
    E = StandardScaler().fit_transform(E)
    mu = E.mean(0)
    sb = sum((y == k).sum() * np.sum((E[y == k].mean(0) - mu) ** 2) for k in np.unique(y))
    sw = sum(np.sum((E[y == k] - E[y == k].mean(0)) ** 2) for k in np.unique(y))
    return float(sb / sw)


def gradcam(infer, x, bs=16):
    """Grad-CAM for the predicted class at the backbone's last 4-D layer; returns (cams, preds)."""
    bb = backbone_of(infer)
    last = [l.name for l in bb.layers if len(getattr(l, "output_shape", ()) or ()) == 4][-1]
    inner = tf.keras.Model(bb.input, [bb.get_layer(last).get_output_at(0), bb.output])
    dense, head = infer.get_layer("logits"), infer.get_layer("probs")
    cams, preds = [], []
    for k in range(0, len(x), bs):
        xb = tf.constant(x[k:k + bs], tf.float32)
        with tf.GradientTape() as tape:
            conv, pooled = inner(preprocess(infer, xb), training=False)
            tape.watch(conv)
            probs = head(dense(pooled))
            cls = tf.argmax(probs, axis=1)
            score = tf.gather(probs, cls, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        w = tf.reduce_mean(g, axis=[1, 2], keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[..., None]
        cam = tf.image.resize(cam, [x.shape[1], x.shape[2]], method="bilinear")[..., 0].numpy()
        cams.append(cam)
        preds.append(cls.numpy())
    cams = np.concatenate(cams)
    s = cams.reshape(len(cams), -1).sum(1)
    cams = cams / np.where(s > 0, s, 1)[:, None, None]
    return cams, np.concatenate(preds)


def cam_stats(x, cams):
    """DAB relative to each tile's own illumination: against 255, BCI's grey glass would count as DAB."""
    rows = []
    for img, cam, I0 in zip(x, cams, illumination(x)):
        dab = np.clip(rgb2hed(img, I0)[:, 2], 0, None).reshape(img.shape[:2])
        sat = np.asarray(Image.fromarray(img).convert("HSV"))[..., 1]
        bg = sat < 8
        uniform = cam.sum() == 0
        rows.append(dict(dab_ratio=np.nan if uniform else float((cam * dab).sum() / max(dab.mean(), 1e-6)),
                         bg_ratio=float((cam * bg).sum() / bg.mean()) if bg.mean() >= 0.05 and not uniform else np.nan,
                         bg_area=float(bg.mean())))
    return pd.DataFrame(rows)


def illumination(x):
    """Per-image illumination I0 (RGB): mean colour of background pixels (HSV saturation < 8); for
    tiles with < 2% background, the 99th percentile of each channel."""
    I0 = np.empty((len(x), 3), np.float32)
    for i, img in enumerate(x):
        bg = np.asarray(Image.fromarray(img).convert("HSV"))[..., 1] < 8
        I0[i] = img[bg].mean(0) if bg.mean() >= 0.02 else np.percentile(img.reshape(-1, 3), 99, axis=0)
    return np.maximum(I0, 1.0)


def stain_scaled(x, I0, channels, s, illum=1.0):
    """Beer-Lambert relative to each image's own illumination I0, not to 255: BCI's glass background is
    grey, and measuring against 255 would count that cast as stain. The chosen stain concentrations are
    multiplied by s, so background pixels (I = I0) keep their colour; illum scales I0 itself (a brighter
    or darker scan). s = illum = 1 returns the input exactly."""
    I0 = I0[:, None, None, :]
    od = -np.log(np.maximum(x.astype(np.float32) / I0, 1e-6))
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB * np.where(np.array(channels, bool), s, 1.0)
    rgb = np.minimum(I0 * illum, 255.0) * np.exp(-(hed @ _RGB_FROM_HED)).reshape(x.shape)
    return np.clip(rgb, 0, 255).astype(np.float32)


def stain_sensitivity(model, x, y, chunk=128):
    """Expected-score shift and class changes under each perturbation (vs the unperturbed round trip)."""
    I0 = illumination(x)

    def expected(ch, s, illum=1.0):
        p = np.concatenate([predict(model, stain_scaled(x[k:k + chunk], I0[k:k + chunk], ch, s, illum))
                            for k in range(0, len(x), chunk)])
        return p @ np.arange(C.NUM_CLASSES), p.argmax(1)

    e0, p0 = expected([1, 1, 1], 1.0)
    res = {"accuracy_roundtrip": float((p0 == y).mean())}
    perturbations = [(name, ch, s, 1.0) for name, ch in STAINS.items() for s in SCALES] + \
                    [("illumination", [1, 1, 1], 1.0, b) for b in ILLUM]
    for name, ch, s, b in perturbations:
        e, p = expected(ch, s, b)
        res[f"{name}|{s if name != 'illumination' else b}"] = dict(
            mean_delta_score=float((e - e0).mean()), frac_up=float((p > p0).mean()),
            frac_down=float((p < p0).mean()), accuracy=float((p == y).mean()))
    return res


# ----------------------------------------------------------------------------- analysis

def frozen_probe(data):
    """Linear probe on frozen ImageNet EfficientNetB0 features for both datasets."""
    net = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
    res = {}
    for ds in DS:
        (xtr, ytr, _), (xte, yte, _) = data[ds]["train"], data[ds]["test"]
        Etr, Ete = predict(net, xtr, 64), predict(net, xte, 64)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)).fit(Etr, ytr)
        res[ds] = metrics(yte, clf.predict_proba(Ete))
        res[ds]["silhouette"] = float(silhouette_score(Ete, yte, metric="cosine"))
        res[ds]["scatter_ratio"] = scatter_ratio(Ete, yte)
        res[ds]["tsne"] = TSNE(2, random_state=C.SEED, init="pca", perplexity=30).fit_transform(Ete).tolist()
        res[ds]["y_test"] = yte.tolist()
        print(f"frozen ImageNet probe {ds}: acc {res[ds]['accuracy']:.3f} f1 {res[ds]['macro_f1']:.3f} "
              f"qwk {res[ds]['qwk']:.3f}", flush=True)
    del net
    tf.keras.backend.clear_session()
    return res


def analyse_arch(arch, data, scan_of_bci_test):
    out = {"arch": arch}
    models = {ds: load_model(ds, arch) for ds in DS}
    # sanity: rebuilt models reproduce the saved test predictions of their own runs
    for ds in DS:
        saved = np.load(run_dir(ds, arch) / "predictions.npz")["test"][data[ds]["test"][2].pos.values]
        now = predict(models[ds], data[ds]["test"][0])
        out[f"{ds}_reload_max_abs_diff"] = float(np.abs(saved - now).max())
    # 2 + 3: own-domain embedding separability, cross-domain transfer
    for ds in DS:
        fm = feature_model(models[ds])
        (xtr, ytr, _), (xte, yte, _) = data[ds]["train"], data[ds]["test"]
        Etr, Ete = predict(fm, xtr), predict(fm, xte)
        knn = KNeighborsClassifier(10, metric="cosine").fit(Etr, ytr)
        out[f"{ds}_model"] = dict(
            own_test=metrics(yte, predict(models[ds], xte)),
            knn_accuracy=float((knn.predict(Ete) == yte).mean()),
            silhouette=float(silhouette_score(Ete, yte, metric="cosine")),
            scatter_ratio=scatter_ratio(Ete, yte),
            tsne=TSNE(2, random_state=C.SEED, init="pca", perplexity=30).fit_transform(Ete).tolist())
        other = "BCI" if ds == "FYP2" else "FYP2"
        out[f"{ds}_model"]["cross_test"] = metrics(data[other]["test"][1], predict(models[ds], data[other]["test"][0]))
    # 4: CKA at five depths, on each dataset's test images
    _, init_model, _ = U.build_model(arch, "ce")            # ImageNet initialisation (head unused)
    layers = depth_layers(backbone_of(models["FYP2"]))
    out["cka_layer_index"] = layers
    out["cka_layers"] = [backbone_of(models["FYP2"]).layers[i].name for i in layers]     # as named in the FYP2 copy
    for ds in DS:
        x = data[ds]["test"][0]
        A = {k: pooled_activations(m, layers, x) for k, m in [("FYP2", models["FYP2"]), ("BCI", models["BCI"]),
                                                              ("init", init_model)]}
        out[f"cka_on_{ds}"] = dict(fyp2_vs_bci=[linear_cka(a, b) for a, b in zip(A["FYP2"], A["BCI"])],
                                   fyp2_vs_init=[linear_cka(a, b) for a, b in zip(A["FYP2"], A["init"])],
                                   bci_vs_init=[linear_cka(a, b) for a, b in zip(A["BCI"], A["init"])])
    # 5: Grad-CAM on own test split
    for ds in DS:
        x, y, _ = data[ds]["test"]
        cams, pred = gradcam(models[ds], x)
        st = cam_stats(x, cams)
        st["label"], st["pred"] = y, pred
        if ds == "BCI":
            st["scan"] = scan_of_bci_test
        out[f"{ds}_cam"] = dict(
            dab_ratio_median=float(st.dab_ratio.median()), bg_ratio_median=float(st.bg_ratio.median()),
            n_with_background=int(st.bg_ratio.notna().sum()),
            by_class=st.groupby("label")[["dab_ratio", "bg_ratio"]].median().round(4).to_dict(),
            correct_vs_wrong=st.assign(correct=st.label == st.pred).groupby("correct")[["dab_ratio", "bg_ratio"]]
            .median().round(4).to_dict())
        if ds == "BCI":
            out["BCI_cam"]["by_scan"] = st.groupby("scan")[["dab_ratio", "bg_ratio"]].median().round(4).to_dict()
        # example overlays (first two correctly classified tiles per class)
        ex = [int(i) for k in range(4) for i in np.where((y == k) & (pred == k))[0][:2]]
        np.savez_compressed(MOUT / f"{arch}_{ds}_cam_examples.npz", x=x[ex], cam=cams[ex], y=y[ex])
    # 6: stain-perturbation sensitivity on own test split
    for ds in DS:
        x, y, _ = data[ds]["test"]
        out[f"{ds}_stain"] = stain_sensitivity(models[ds], x, y)
    del models, init_model
    tf.keras.backend.clear_session()
    return out


# ----------------------------------------------------------------------------- figures

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_tsne(probe, res, arch):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    panels = [(probe["FYP2"]["tsne"], probe["FYP2"]["y_test"], "FYP2 (GAN): frozen ImageNet features"),
              (probe["BCI"]["tsne"], probe["BCI"]["y_test"], "BCI (real): frozen ImageNet features"),
              (res["FYP2_model"]["tsne"], probe["FYP2"]["y_test"], f"FYP2 (GAN): fine-tuned {arch}"),
              (res["BCI_model"]["tsne"], probe["BCI"]["y_test"], f"BCI (real): fine-tuned {arch}")]
    for ax, (t, y, title) in zip(axes.flat, panels):
        t, y = np.array(t), np.array(y)
        for k in range(4):
            ax.scatter(t[y == k, 0], t[y == k, 1], s=7, color=CLASS_COLORS[k], label=C.BCI_LEVELS[k],
                       edgecolor="none" if k else INK2, linewidth=0.2)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 1].legend(title="HER2 score", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
                      markerscale=2.5)
    fig.suptitle("t-SNE of test-set embeddings (colour = filename label)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"figD4_tsne_{arch}.png", dpi=150)
    plt.close(fig)


def fig_cka(all_res):
    """One small panel per architecture (nine colours on one axis would be unreadable). Colour follows the
    network as in the other figures: blue = FYP2-trained, orange = BCI-trained, dark = the two compared."""
    archs = list(all_res)
    ncol = 3
    nrow = int(np.ceil(len(archs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.5 * nrow + 0.9), sharex=True, sharey=True, squeeze=False)
    spec = [("fyp2_vs_bci", "FYP2-trained vs BCI-trained network", INK, "-"),
            ("fyp2_vs_init", "FYP2-trained vs its ImageNet start", "#2a78d6", "--"),
            ("bci_vs_init", "BCI-trained vs its ImageNet start", "#eb6834", "--")]
    for ax, a in zip(axes.flat, archs):
        depth = np.linspace(0.15, 1.0, len(all_res[a]["cka_layers"]))
        for key, lab, c, ls in spec:
            ax.plot(depth, all_res[a]["cka_on_BCI"][key], color=c, lw=2, ls=ls, marker="o", ms=3.5, label=lab)
        ax.set_title(a, fontsize=9.5, loc="left", fontweight="bold")
        ax.set_ylim(0, 1.02)
        style(ax)
    for ax in axes.flat[len(archs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Relative depth in backbone")
    for ax in axes[:, 0]:
        ax.set_ylabel("Linear CKA")
    fig.suptitle("Representation similarity on BCI test images (1 = identical up to rotation and scale)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3,
               frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))     # the suptitle's height is already reserved by tight_layout
    fig.savefig(FIG / "figD5_cka.png", dpi=160)
    plt.close(fig)


def fig_cam_examples(arch):
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.2))
    for r, ds in enumerate(DS):
        p = MOUT / f"{arch}_{ds}_cam_examples.npz"
        if not p.exists():
            continue
        d = np.load(p)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c >= len(d["x"]):
                continue
            ax.imshow(d["x"][c])
            ax.imshow(d["cam"][c], cmap="magma", alpha=0.45)
            ax.set_title(f"{DS_LABEL[ds].split()[0]} {C.BCI_LEVELS[d['y'][c]]}", fontsize=8.5)
    fig.suptitle(f"Grad-CAM, {arch} (correct predictions, two per class; bright = high attention)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / f"figD6_gradcam_{arch}.png", dpi=140)
    plt.close(fig)


def balanced_accuracy(confusion):
    """Mean per-class recall (rows = true class). Chance is 0.25 on both test sets, whereas plain accuracy
    on BCI's imbalanced test set is 0.43 for a model that always answers 2+."""
    cm = np.asarray(confusion, float)
    n = cm.sum(1)
    return float((np.diag(cm)[n > 0] / n[n > 0]).mean())


def fig_transfer(all_res):
    rows = []
    for a, r in all_res.items():
        ba = {k: balanced_accuracy(r[m][t]["confusion"]) for k, m, t in
              [("fyp2_own", "FYP2_model", "own_test"), ("fyp2_on_bci", "FYP2_model", "cross_test"),
               ("bci_own", "BCI_model", "own_test"), ("bci_on_fyp2", "BCI_model", "cross_test")]}
        rows.append(dict(arch=a, **ba))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 1.6))
    y = np.arange(len(t))[::-1]
    spec = [("fyp2_own", "FYP2 model on FYP2 test", "#2a78d6", "o"),
            ("fyp2_on_bci", "FYP2 model on BCI test", "#2a78d6", "x"),
            ("bci_own", "BCI model on BCI test", "#eb6834", "o"),
            ("bci_on_fyp2", "BCI model on FYP2 test", "#eb6834", "x")]
    for col, lab, c, mk in spec:
        ax.scatter(t[col], y, s=55 if mk == "o" else 45, marker=mk, color=c, label=lab, zorder=3,
                   linewidths=2 if mk == "x" else 0.5, edgecolor=SURF if mk == "o" else None)
    ax.set_yticks(y, t.arch)
    ax.axvline(0.25, color=INK2, lw=0.8, ls=":")
    ax.text(0.255, y.max() + 0.45, "chance", fontsize=8, color=INK2, va="bottom")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-0.6, y.max() + 1.0)
    ax.set_xlabel("Balanced test accuracy (mean recall over the four scores)")
    ax.set_title("Cross-domain transfer (CE models)", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "figD7_cross_domain.png", dpi=160)
    plt.close(fig)


def stain_table(all_res):
    return pd.DataFrame([dict(arch=a, dataset=ds, stain=k.split("|")[0], scale=float(k.split("|")[1]), **v)
                         for a, r in all_res.items() for ds in DS if f"{ds}_stain" in r
                         for k, v in r[f"{ds}_stain"].items() if "|" in k])


def fig_stain(all_res):
    t = stain_table(all_res)
    if t.empty:
        return
    panels = list(STAINS) + ["illumination"]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.0), sharey=True)
    for ax, stain in zip(axes, panels):
        for ds, c in [("FYP2", "#2a78d6"), ("BCI", "#eb6834")]:
            d = t[(t.stain == stain) & (t.dataset == ds)]
            for _, g in d.groupby("arch"):                      # one thin line per architecture
                g = pd.concat([g, pd.DataFrame([dict(scale=1.0, mean_delta_score=0.0)])]).sort_values("scale")
                ax.plot(g.scale, g.mean_delta_score, color=c, lw=0.8, alpha=0.35)
            med = d.groupby("scale").mean_delta_score.median()
            med = pd.concat([med, pd.Series({1.0: 0.0})]).sort_index()
            ax.plot(med.index, med.values, color=c, lw=2.2, marker="o", ms=5, mec=SURF,
                    label=f"{ds}-trained networks on {ds} test tiles")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ticks = [*ILLUM[:2], 1.0, *ILLUM[2:]] if stain == "illumination" else [*SCALES[:2], 1.0, *SCALES[2:]]
        ax.set_xticks(ticks, [f"×{v:g}" for v in ticks], fontsize=8)
        ax.minorticks_off()
        ax.set_title("Scan brightness scaled" if stain == "illumination" else f"{stain} scaled",
                     loc="left", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Illumination I0" if stain == "illumination" else "Stain concentration (optical density)")
        style(ax)
    axes[0].set_ylabel("Mean shift of expected score\n(score units, 0 to 3)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2,
               frameon=False, fontsize=8.5)
    fig.text(0.99, 0.965, "thick line: median of the networks; thin lines: one network each", ha="right",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "figD10_stain_sensitivity.png", dpi=160)
    plt.close(fig)


def summary_table(probe, all_res):
    rows = [dict(arch="frozen ImageNet EfficientNetB0 (linear probe)", dataset=ds,
                 test_accuracy=probe[ds]["accuracy"], test_macro_f1=probe[ds]["macro_f1"], test_qwk=probe[ds]["qwk"],
                 silhouette=probe[ds]["silhouette"], scatter_ratio=probe[ds]["scatter_ratio"]) for ds in DS]
    for a, r in all_res.items():
        for ds in DS:
            m = r[f"{ds}_model"]
            st = r.get(f"{ds}_stain", {})
            rows.append(dict(arch=a, dataset=ds, test_accuracy=m["own_test"]["accuracy"],
                             test_balanced_accuracy=balanced_accuracy(m["own_test"]["confusion"]),
                             test_macro_f1=m["own_test"]["macro_f1"], test_qwk=m["own_test"]["qwk"],
                             knn_accuracy=m["knn_accuracy"], silhouette=m["silhouette"],
                             scatter_ratio=m["scatter_ratio"], cross_domain_accuracy=m["cross_test"]["accuracy"],
                             cross_domain_balanced_accuracy=balanced_accuracy(m["cross_test"]["confusion"]),
                             cross_domain_qwk=m["cross_test"]["qwk"],
                             cam_dab_ratio=r[f"{ds}_cam"]["dab_ratio_median"],
                             cam_bg_ratio=r[f"{ds}_cam"]["bg_ratio_median"],
                             **{f"shift_{k.replace(' ', '_')}_x1.5": st.get(f"{k}|1.5", {}).get("mean_delta_score")
                                for k in STAINS},
                             shift_illumination_x0_85=st.get("illumination|0.85", {}).get("mean_delta_score"),
                             reload_max_abs_diff=r[f"{ds}_reload_max_abs_diff"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "model_summary.csv", index=False)
    stain_table(all_res).to_csv(OUT / "stain_sensitivity.csv", index=False)
    cka = [dict(arch=a, on=on, layer=l, depth_index=i, fyp2_vs_bci=r[f"cka_on_{on}"]["fyp2_vs_bci"][i],
                fyp2_vs_init=r[f"cka_on_{on}"]["fyp2_vs_init"][i], bci_vs_init=r[f"cka_on_{on}"]["bci_vs_init"][i])
           for a, r in all_res.items() for on in DS for i, l in enumerate(r["cka_layers"])]
    pd.DataFrame(cka).to_csv(OUT / "cka.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU only, 200 images per split, results to analysis/smoke")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU only. ConvNeXtTiny's Grad-CAM gradient deadlocks on the Mac GPU (tensorflow-metal)")
    args = ap.parse_args()
    global SMOKE, OUT, MOUT, FIG
    if args.cpu or args.smoke:
        tf.config.set_visible_devices([], "GPU")
    if args.smoke:
        SMOKE = 200
        src = OUT
        OUT = MOUT = FIG = C.RESULTS_DIR / "analysis_smoke"
        OUT.mkdir(parents=True, exist_ok=True)
        for f in ["scan_type_threshold.json", "data_features.csv"]:
            (OUT / f).write_bytes((src / f).read_bytes())
        args.force = True
    MOUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    archs = args.archs or [a for a in C.MODELS if has_weights("FYP2", a) and has_weights("BCI", a)]
    print("architectures:", archs, flush=True)
    probe_path = OUT / "frozen_probe.json"
    if args.force or not probe_path.exists() or any(not (MOUT / f"{a}.json").exists() for a in archs):
        data = {ds: {s: load_images(ds, s) for s in ["train", "test"]} for ds in DS}
        thr = json.load(open(OUT / "scan_type_threshold.json"))["otsu_threshold_train"]
        feat = pd.read_csv(C.SHARED_ANALYSIS / "data_features.csv")
        bg = feat[feat.dataset == "BCI (real)"].set_index("path").bg_brightness
        scan = np.where(data["BCI"]["test"][2].path.map(bg) >= thr, "bright", "grey")
    # else: everything is cached and only the tables and figures are redrawn (no images needed)

    if probe_path.exists() and not args.force:
        probe = json.load(open(probe_path))
    else:
        probe = frozen_probe(data)
        json.dump(probe, open(probe_path, "w"))
    all_res = {}
    for a in archs:
        p = MOUT / f"{a}.json"
        if p.exists() and not args.force:
            all_res[a] = json.load(open(p))
            continue
        print(f"--- {a}", flush=True)
        all_res[a] = analyse_arch(a, data, scan)
        json.dump(all_res[a], open(p, "w"))
        r = all_res[a]
        print(f"  reload diff FYP2 {r['FYP2_reload_max_abs_diff']:.2e} BCI {r['BCI_reload_max_abs_diff']:.2e} | "
              f"FYP2->BCI acc {r['FYP2_model']['cross_test']['accuracy']:.3f} "
              f"BCI->FYP2 acc {r['BCI_model']['cross_test']['accuracy']:.3f} | "
              f"CKA last layer {r['cka_on_BCI']['fyp2_vs_bci'][-1]:.3f}", flush=True)
    # also pick up earlier per-arch results not requested this time
    for p in MOUT.glob("*.json"):
        all_res.setdefault(p.stem, json.load(open(p)))
    order = [a for a in C.MODELS if a in all_res]
    all_res = {a: all_res[a] for a in order}
    if all_res:
        summary_table(probe, all_res)
        fig_cka(all_res)
        fig_transfer(all_res)
        fig_stain(all_res)
        for a in all_res:
            fig_cam_examples(a)
        best = "EfficientNetB0" if "EfficientNetB0" in all_res else order[0]
        fig_tsne(probe, all_res[best], best)


if __name__ == "__main__":
    main()
