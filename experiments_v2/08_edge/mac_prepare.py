"""Experiment 5/edge (v2), Mac side. Two measurements that need no device, plus the reference files the device
scripts compare against.

A. Precision references for the deployed 448 px EfficientNetV2B0__ce__hires: TFLite FP32, FP16 and INT8 (the exact
   files in results/deploy/EfficientNetV2B0__ce__hires/, hashes recorded) run on the Mac on
   (i) the 120 balanced BCI-validation reference tiles already used for the Pi (deployment/reference/primary.npz), and
   (ii) all 727 BCI val + 758 test tiles (eval cache), to give accuracy, per-class recall and agreement with Keras
   for every precision. Output: refs_primary_precisions.npz (copied into the Pi bundle), precision_accuracy.csv.

B. Android preprocessing check. The app (android_app/.../HER2Classifier.kt) scales the tile straight to 448 px with
   Bitmap.createScaledBitmap(filter=true); training and the Pi use 1024 -> Lanczos 512 -> bilinear 448. Android's exact
   resampler cannot be run here, so two bounds are measured on BCI val+test with the FP16 model:
     direct_pil_bilinear   PIL bilinear 1024 -> 448 (area-aware, close to a filtered downscale)
     direct_cv_linear      OpenCV INTER_LINEAR 1024 -> 448 (plain bilinear sampling, no anti-aliasing)
   and compared with the training path. Output: android_preprocessing_check.csv.
CPU only (TFLite interpreter from TensorFlow 2.15).
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

tf = L.tf_setup(cpu=True)
OUT = L.out_dir("08_edge")
DEP = L.ROOT / "results" / "deploy" / "EfficientNetV2B0__ce__hires"
MODELS = {p: DEP / f"model_{p}.tflite" for p in ["fp32", "fp16", "int8"]}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Lite:
    def __init__(self, path, threads=4):
        self.i = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.i.allocate_tensors()
        self.a, self.b = self.i.get_input_details()[0], self.i.get_output_details()[0]

    def __call__(self, x):
        self.i.set_tensor(self.a["index"], x[None].astype(np.float32)); self.i.invoke()
        return self.i.get_tensor(self.b["index"])[0].astype(np.float64)


def bci_eval(split):
    m = L.manifest(); m = m[m.split == split].reset_index(drop=True)
    return m, [L.ROOT / "data" / "ihc_512_eval448" / p for p in m.path]


def main():
    import sys as _s
    sys.path.insert(0, str(L.SRC))
    import config as C
    ref = np.load(L.ROOT / "deployment" / "reference" / "primary.npz", allow_pickle=True)
    tiles, labels = ref["tiles_preprocessed"], ref["labels"].astype(int)
    keras_ref = ref["keras"]
    out = dict(labels=labels, paths=ref["paths"], keras=keras_ref)
    rows = []
    stored = {split: np.load(L.RUNS / "EfficientNetV2B0__ce__hires" / "predictions.npz")[split] for split in ["val", "test"]}
    for prec, path in MODELS.items():
        net = Lite(path)
        p120 = np.stack([net(t) for t in tiles]); out[f"mac_{prec}"] = p120
        row = dict(precision=prec, file=str(path.relative_to(L.ROOT)), sha256=sha(path), size_mb=path.stat().st_size / 2**20,
                   ref120_agreement_with_keras=float((p120.argmax(1) == keras_ref.argmax(1)).mean()),
                   ref120_accuracy=float((p120.argmax(1) == labels).mean()))
        for split in ["val", "test"]:
            m, files = bci_eval(split)
            p = np.stack([net(np.asarray(Image.open(f).convert("RGB"), np.float32)) for f in files])
            met = L.class_metrics(m.label.values.astype(int), p)
            row.update({f"{split}_{k}": met[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "qwk",
                                                         "recall_0", "recall_1+", "recall_2+", "recall_3+"]})
            row[f"{split}_agreement_with_keras"] = float((p.argmax(1) == stored[split].argmax(1)).mean())
            row[f"{split}_max_abs_dp_vs_keras"] = float(np.abs(p - stored[split]).max())
        rows.append(row)
        print(row, flush=True)
    np.savez_compressed(L.guard(OUT / "refs_primary_precisions.npz"), **out)
    L.write_csv(OUT / "precision_accuracy.csv", pd.DataFrame(rows))

    # ---- B. Android preprocessing check (FP16, the model the app ships)
    net = Lite(MODELS["fp16"])
    prow = []
    for split in ["val", "test"]:
        m, _ = bci_eval(split)
        y = m.label.values.astype(int)
        P = {"training_path": [], "direct_pil_bilinear": [], "direct_cv_linear": []}
        for p in m.path:
            im = Image.open(C.BCI_IHC / p).convert("RGB")
            P["training_path"].append(net(np.asarray(im.resize((512, 512), Image.LANCZOS).resize((448, 448), Image.BILINEAR), np.float32)))
            P["direct_pil_bilinear"].append(net(np.asarray(im.resize((448, 448), Image.BILINEAR), np.float32)))
            P["direct_cv_linear"].append(net(cv2.resize(np.asarray(im), (448, 448), interpolation=cv2.INTER_LINEAR).astype(np.float32)))
        base = np.stack(P["training_path"])
        for k, v in P.items():
            v = np.stack(v); met = L.class_metrics(y, v)
            prow.append(dict(split=split, preprocessing=k, agreement_with_training_path=float((v.argmax(1) == base.argmax(1)).mean()),
                             **{kk: met[kk] for kk in ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "recall_0", "recall_1+", "recall_2+", "recall_3+"]}))
            print(prow[-1], flush=True)
    L.write_csv(OUT / "android_preprocessing_check.csv", pd.DataFrame(prow))


if __name__ == "__main__":
    main()
