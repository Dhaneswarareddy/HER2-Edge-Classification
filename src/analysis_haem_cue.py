"""Diagnosis C2: do FYP2-trained networks read the GAN's low haematoxylin as "1+"?  (TensorFlow; run alone)

Background. Illumination-corrected haematoxylin OD (data_features.csv, test medians): FYP2's 1+ images
0.037, its other classes 0.15-0.21, every BCI class 0.02-0.05. FYP2-trained networks call 71-81% of BCI
test tiles 1+ (8 of 9 networks; 23% of the tiles are 1+). analysis_models.py scaled haematoxylin only
x0.6-1.5, far short of this 4-5x gap, so it could not test the cue. Here haematoxylin alone is scaled
to the other group's level (Ruifrok-Johnston, relative to each tile's own illumination, exactly as in
analysis_models.py), and the FYP2-trained network is applied:

  A  FYP2 test tiles of score 0, 2+, 3+, haematoxylin x0.2 (down to the 1+ level): share predicted 1+
  B  FYP2 test tiles of score 1+,        haematoxylin x5   (up to the other classes' level): share predicted 1+
  C  BCI test tiles,                      haematoxylin x5   (up to FYP2 levels): share predicted 1+ and
     balanced accuracy

A and B control each other: a network that answers 1+ to any strongly perturbed image would rise in both.
Decision rule, written before the run: the haematoxylin cue counts as confirmed if the median over the
FYP2-trained networks moves the predicted-1+ share by >= 20 points in the expected direction in BOTH A
(up) and B (down). C is reported as an effect size on the cross-domain failure, with no threshold.

Outputs: results/analysis/haem_cue/<arch>.json, results/analysis/haem_cue.csv,
         figures/figD11_haem_examples.png
Usage:   python analysis_haem_cue.py --arch ResNet50    (one process per architecture)
         python analysis_haem_cue.py                    (table, decision and example figure)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_models as AM
import config as C

OUT = C.ANALYSIS_DIR / "haem_cue"
H = [1, 0, 0]                       # haematoxylin only
DOWN, UP = 0.2, 5.0
THRESHOLD = 0.20


def predict_scaled(model, x, s, chunk=128):
    I0 = AM.illumination(x)
    return np.concatenate([AM.predict(model, AM.stain_scaled(x[k:k + chunk], I0[k:k + chunk], H, s))
                           for k in range(0, len(x), chunk)])


def share_1plus(prob):
    return float((prob.argmax(1) == 1).mean())


def run_arch(arch):
    model = AM.load_model("FYP2", arch)
    xf, yf, _ = AM.load_images("FYP2", "test")
    xb, yb, _ = AM.load_images("BCI", "test")
    other, ones = yf != 1, yf == 1
    res = {"arch": arch}
    for key, x, s in [("A", xf[other], DOWN), ("B", xf[ones], UP)]:
        p0, p1 = predict_scaled(model, x, 1.0), predict_scaled(model, x, s)
        res[key] = dict(scale=s, n=len(x), share_1plus_before=share_1plus(p0), share_1plus_after=share_1plus(p1))
    p0, p1 = predict_scaled(model, xb, 1.0), predict_scaled(model, xb, UP)
    m0, m1 = AM.metrics(yb, p0), AM.metrics(yb, p1)
    res["C"] = dict(scale=UP, n=len(xb), share_1plus_before=share_1plus(p0), share_1plus_after=share_1plus(p1),
                    balanced_accuracy_before=AM.balanced_accuracy(m0["confusion"]),
                    balanced_accuracy_after=AM.balanced_accuracy(m1["confusion"]),
                    predicted_share_after=np.bincount(p1.argmax(1), minlength=C.NUM_CLASSES).tolist())
    json.dump(res, open(OUT / f"{arch}.json", "w"), indent=1)
    print(arch, {k: (round(res[k]["share_1plus_before"], 3), round(res[k]["share_1plus_after"], 3)) for k in "ABC"},
          f"BCI balanced acc {res['C']['balanced_accuracy_before']:.3f} -> {res['C']['balanced_accuracy_after']:.3f}",
          flush=True)


def fig_examples():
    """Two tiles per row, before and after: is the perturbed tissue still plausible?"""
    xf, yf, _ = AM.load_images("FYP2", "test")
    xb, yb, _ = AM.load_images("BCI", "test")
    rows = [("FYP2 score 0", xf[yf == 0][:2], DOWN), ("FYP2 score 3+", xf[yf == 3][:2], DOWN),
            ("FYP2 score 1+", xf[yf == 1][:2], UP), ("BCI score 2+", xb[yb == 2][:2], UP)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9, 2.3 * len(rows)))
    for r, (lab, x, s) in enumerate(rows):
        xs = AM.stain_scaled(x, AM.illumination(x), H, s).astype(np.uint8)
        for c, img in enumerate([x[0], xs[0], x[1], xs[1]]):
            axes[r, c].imshow(img)
            axes[r, c].set_axis_off()
            axes[r, c].set_title(f"{lab}" if c % 2 == 0 else f"haematoxylin x{s:g}", fontsize=8.5)
    fig.suptitle("Haematoxylin-only scaling used in the cue test (original, then scaled)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(AM.FIG / "figD11_haem_examples.png", dpi=130)
    plt.close(fig)


def summarise():
    rows = []
    for p in sorted(OUT.glob("*.json")):
        r = json.load(open(p))
        rows.append(dict(arch=r["arch"], **{f"{k}_{v}": r[k][f"share_1plus_{v}"] for k in "ABC" for v in ["before", "after"]},
                         C_balanced_accuracy_before=r["C"]["balanced_accuracy_before"],
                         C_balanced_accuracy_after=r["C"]["balanced_accuracy_after"]))
    t = pd.DataFrame(rows)
    order = [a for a in C.MODELS if a in set(t.arch)]
    t = t.set_index("arch").loc[order].reset_index()
    t.to_csv(C.ANALYSIS_DIR / "haem_cue.csv", index=False)
    dA = float((t.A_after - t.A_before).median())
    dB = float((t.B_after - t.B_before).median())
    confirmed = dA >= THRESHOLD and -dB >= THRESHOLD
    print(t.round(3).to_string(index=False))
    print(f"median change in predicted-1+ share: A {dA:+.3f} (expected up), B {dB:+.3f} (expected down); "
          f"C {float((t.C_after - t.C_before).median()):+.3f}, balanced accuracy "
          f"{float((t.C_balanced_accuracy_after - t.C_balanced_accuracy_before).median()):+.3f}")
    print(f"haematoxylin cue (>= {THRESHOLD:.2f} in both A and B, {len(t)} networks):",
          "CONFIRMED" if confirmed else "NOT confirmed")
    json.dump(dict(n_networks=len(t), median_change_A=dA, median_change_B=dB, threshold=THRESHOLD,
                   confirmed=confirmed), open(C.ANALYSIS_DIR / "haem_cue_decision.json", "w"), indent=1)
    fig_examples()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.arch:
        run_arch(args.arch)
    else:
        summarise()


if __name__ == "__main__":
    main()
