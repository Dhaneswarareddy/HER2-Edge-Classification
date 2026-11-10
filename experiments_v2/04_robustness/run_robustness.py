"""Experiment 4 (v2): image-quality and staining robustness of the two deployment models on BCI test.

Gap filled: src/analysis_models.py already scaled DAB / haematoxylin / brightness for the nine 224 px baselines,
reporting expected-score shifts only. Not done before: blur, JPEG, contrast and sensor noise; the deployed 448 px
model; macro-F1 / balanced accuracy / per-class recall under each condition.

Design (fixed before running):
  * images: BCI grouped TEST split (758 tiles), report-only; nothing is tuned on these results;
  * each perturbation is applied to the ORIGINAL 1024 px tile (0.46 um/px), then the model's own preprocessing runs
    unchanged (Lanczos -> 256 or 512 source, bilinear -> 224 or 448 input). The same physical change therefore reaches
    both models; originals are only read, never modified;
  * the 'clean' condition re-runs the same path and must reproduce the stored test predictions (checked);
  * deterministic: noise uses a per-tile seed; JPEG uses PIL at the given quality, 4:2:0 subsampling.
Conditions (plausible acquisition / staining variation, not adversarial):
  brightness  x0.85, 0.92, 1.08, 1.15          illumination gain (scan brightness); stain OD unchanged
  contrast    x0.8, 0.9, 1.1, 1.2               about each tile's mean intensity
  blur        Gaussian sigma 1, 2, 4 native px  (0.46, 0.92, 1.84 um): mild to clear defocus
  jpeg        quality 90, 70, 50, 30
  noise       Gaussian sigma 2, 5 grey levels
  dab         DAB optical density x0.8, x1.2   (Ruifrok-Johnston, relative to the tile's own illumination)
  haem        haematoxylin OD x0.8, x1.2
Metrics: accuracy, balanced accuracy, macro-F1, QWK, per-class recall; paired delta vs clean with a 95%
fragment-cluster bootstrap CI; prediction flip rate. Run on CPU.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

tf = L.tf_setup(cpu=True)
sys.path.insert(0, str(L.NEXT))
import harmonise_lib as H  # noqa: E402
import common as U  # noqa: E402
import config as C  # noqa: E402

OUT = L.out_dir("04_robustness")
RUNS = ["EfficientNetV2B0__ce__hires", "EfficientNetV2B0__ce"]
N_BOOT = 2000
CONDITIONS = [("clean", None)] + \
    [("brightness", v) for v in [0.85, 0.92, 1.08, 1.15]] + [("contrast", v) for v in [0.8, 0.9, 1.1, 1.2]] + \
    [("blur", v) for v in [1, 2, 4]] + [("jpeg", v) for v in [90, 70, 50, 30]] + [("noise", v) for v in [2, 5]] + \
    [("dab", v) for v in [0.8, 1.2]] + [("haem", v) for v in [0.8, 1.2]]


def perturb(im, kind, v, seed):
    if kind == "clean":
        return im
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(v))
    if kind == "jpeg":
        b = io.BytesIO(); im.save(b, "JPEG", quality=int(v)); b.seek(0)
        return Image.open(b).convert("RGB")
    x = np.asarray(im, np.float32)
    if kind == "brightness":
        x = x * v
    elif kind == "contrast":
        m = x.mean()
        x = (x - m) * v + m
    elif kind == "noise":
        x = x + np.random.default_rng(seed).normal(0, v, x.shape)
    elif kind in ("dab", "haem"):
        x = H.apply_stain_gains(x.astype(np.uint8), v if kind == "haem" else 1.0, v if kind == "dab" else 1.0)
    return Image.fromarray(np.clip(np.round(x), 0, 255).astype(np.uint8))


def pipeline(im, source, size):
    return np.asarray(im.resize((source, source), Image.LANCZOS).resize((size, size), Image.BILINEAR), np.float32)


def main():
    L.write_json(OUT / "config.json", dict(experiment="04_robustness", runs=RUNS, conditions=CONDITIONS, split="BCI test (report-only)",
                                           applied_at="native 1024 px before the model's own resize pipeline",
                                           jpeg="PIL default 4:2:0", noise_seed="tile index", n_boot=N_BOOT, device="CPU"))
    m = L.manifest(); m = m[m.split == "test"].reset_index(drop=True)
    y = m.label.values.astype(int); groups = L.fragments("test")
    raw_dir = C.BCI_IHC
    rows, checks = [], {}
    for run in RUNS:
        s = json.load(open(L.RUNS / run / "summary.json"))
        size = s.get("image_size", 224); source = (s.get("variant_spec") or {}).get("source_size", 256)
        C.IMAGE_SIZE = size
        _, net, _ = U.build_model(s["model"], s["loss"]); net.load_weights(L.RUNS / run / "model.weights.h5")
        preds = {}
        for kind, v in CONDITIONS:
            probs = []
            for k in range(0, len(m), 32):
                x = np.stack([pipeline(perturb(Image.open(raw_dir / p).convert("RGB"), kind, v, k + j), source, size)
                              for j, p in enumerate(m.path[k:k + 32])])
                probs.append(net.predict(x, batch_size=16, verbose=0))
            p = np.concatenate(probs); preds[(kind, v)] = p
            if kind == "clean":
                checks[run] = float(np.abs(p - np.load(L.RUNS / run / "predictions.npz")["test"]).max())
                print(f"{run}: clean reproduces stored test probabilities, max |dp| = {checks[run]:.2e}", flush=True)
            met = L.class_metrics(y, p)
            r = dict(run=run, input=size, condition=kind, value=v, **{k2: v2 for k2, v2 in met.items() if k2 != "confusion"},
                     flip_rate=float((p.argmax(1) != preds[("clean", None)].argmax(1)).mean()),
                     mean_expected_score_shift=float(((p - preds[("clean", None)]) @ np.arange(4)).mean()),
                     confusion=str(met["confusion"]))
            if kind != "clean":
                a, b = p.argmax(1), preds[("clean", None)].argmax(1)
                d_acc, d_f1, d_bal = [], [], []
                from sklearn.metrics import f1_score, recall_score
                for i in L.cluster_draws(groups, N_BOOT, L.SEED):
                    d_acc.append((a[i] == y[i]).mean() - (b[i] == y[i]).mean())
                    d_f1.append(f1_score(y[i], a[i], average="macro", labels=range(4), zero_division=0)
                                - f1_score(y[i], b[i], average="macro", labels=range(4), zero_division=0))
                    d_bal.append(recall_score(y[i], a[i], average="macro", labels=range(4), zero_division=0)
                                 - recall_score(y[i], b[i], average="macro", labels=range(4), zero_division=0))
                base = L.class_metrics(y, preds[("clean", None)])
                r.update(delta_accuracy=r["accuracy"] - base["accuracy"], delta_accuracy_ci95=str(np.round(L.ci95(d_acc), 4).tolist()),
                         delta_macro_f1=r["macro_f1"] - base["macro_f1"], delta_macro_f1_ci95=str(np.round(L.ci95(d_f1), 4).tolist()),
                         delta_balanced_accuracy=r["balanced_accuracy"] - base["balanced_accuracy"],
                         delta_balanced_accuracy_ci95=str(np.round(L.ci95(d_bal), 4).tolist()))
            rows.append(r)
            print(f"{run} {kind} {v}: acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} f1 {r['macro_f1']:.3f} "
                  f"flip {r['flip_rate']:.3f} recall {[round(r['recall_' + l], 3) for l in L.LEVELS]}", flush=True)
            L.write_csv(OUT / "robustness_results.csv", pd.DataFrame(rows))
        np.savez_compressed(L.guard(OUT / f"probs_{run}.npz"), **{f"{k}_{v}": p for (k, v), p in preds.items()})
        del net
        tf.keras.backend.clear_session()
    L.write_json(OUT / "reproduction_check.json", checks)
    figure(pd.DataFrame(rows))


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = ["brightness", "contrast", "blur", "jpeg", "noise", "dab", "haem"]
    fig, axes = plt.subplots(2, len(kinds), figsize=(3.0 * len(kinds), 6.2), sharey="row")
    for j, kind in enumerate(kinds):
        for run, c in [("EfficientNetV2B0__ce__hires", "#1f5fa8"), ("EfficientNetV2B0__ce", "#d9822b")]:
            d = df[(df.run == run) & (df.condition == kind)].sort_values("value")
            base = df[(df.run == run) & (df.condition == "clean")].iloc[0]
            axes[0, j].plot(d.value, d.macro_f1, "o-", c=c, label=f"{'448' if 'hires' in run else '224'} px")
            axes[0, j].axhline(base.macro_f1, c=c, ls=":", lw=1)
            for k, (lv, mk) in enumerate(zip(L.LEVELS, "os^D")):
                axes[1, j].plot(d.value, d[f"recall_{lv}"], mk + ("-" if "hires" in run else "--"), c=c, ms=3.5,
                                label=f"{'448' if 'hires' in run else '224'} px, recall {lv}" if j == 0 else None)
        axes[0, j].set_title(kind, fontsize=9.5); axes[1, j].set_xlabel("level", fontsize=8)
    axes[0, 0].set_ylabel("Macro-F1 (dotted = clean)"); axes[1, 0].set_ylabel("Per-class recall")
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Robustness on BCI test (758 tiles); perturbation applied at native 1024 px", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(L.guard(OUT / "fig_robustness.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
