"""Edge deployment for FYP3: convert the selected model to TFLite (FP32, FP16, INT8).

  python deploy.py --run EfficientNetV2B0__ce__hires
  python deploy.py --run EfficientNetV2B0__ce__hires --variants fp32 fp16 int8

Outputs → results/deploy/<run>/
  model_fp32.tflite   (baseline, no quantisation)
  model_fp16.tflite   (float16 weights)
  model_int8.tflite   (full integer, float32 I/O, calibrated on training data)
  deploy_summary.json (size_mb, val/test metrics, agreement with FP32 Keras, notes)

Calibration uses N_CALIB class-balanced TRAIN images only (val/test never used here).
INT8 is reported as failed if predicted accuracy degrades >5 percentage points vs FP32 Keras,
or if fewer than 3 distinct classes are predicted (prediction collapse).

No physical Pi benchmarking is performed; mac latency is measured for reference.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Add FYP3/src to path so we can import config and common
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as U
import config as C

N_CALIB = 256           # class-balanced calibration images from TRAIN split
AGREE_THRESH = 0.95     # min agreement with FP32 Keras to call INT8 "ok"
ACC_DROP_THRESH = 0.05  # max allowed accuracy drop vs FP32 Keras val accuracy


def load_calib_data(image_size):
    """N_CALIB class-balanced images from the TRAIN split for INT8 calibration."""
    df = pd.read_csv(C.MANIFEST)
    df = df[df.split == "train"].reset_index(drop=True)
    rng = np.random.default_rng(C.SEED)
    per_class = N_CALIB // C.NUM_CLASSES
    idxs = []
    for k in range(C.NUM_CLASSES):
        cls_idx = np.where(df.label.values == k)[0]
        chosen = rng.choice(cls_idx, min(per_class, len(cls_idx)), replace=False)
        idxs.extend(chosen.tolist())
    idxs = rng.permutation(idxs)
    imgs = []
    for i in idxs:
        p = C.DATA_DIR / df.iloc[i].path
        im = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32))
    return np.stack(imgs)   # (N, H, W, 3) float32 0-255


def load_calib_hires(image_size, source_size):
    """For hires runs: use the cached eval files (source_size → image_size)."""
    df = pd.read_csv(C.MANIFEST)
    df = df[df.split == "train"].reset_index(drop=True)
    ev_dir = C.ROOT / "data" / f"ihc_{source_size}_eval{image_size}"
    rng = np.random.default_rng(C.SEED)
    per_class = N_CALIB // C.NUM_CLASSES
    idxs = []
    for k in range(C.NUM_CLASSES):
        cls_idx = np.where(df.label.values == k)[0]
        chosen = rng.choice(cls_idx, min(per_class, len(cls_idx)), replace=False)
        idxs.extend(chosen.tolist())
    idxs = rng.permutation(idxs)
    imgs = []
    for i in idxs:
        p = ev_dir / df.iloc[i].path
        if not p.exists():
            # load from source_size directory if present, else fallback to ihc_256
            src_p = C.ROOT / "data" / f"ihc_{source_size}" / df.iloc[i].path
            if src_p.exists():
                im = Image.open(src_p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            else:
                p2 = C.ROOT / "data" / "ihc_256" / df.iloc[i].path
                im = Image.open(p2).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        else:
            im = Image.open(p).convert("RGB")
        imgs.append(np.asarray(im, dtype=np.float32))
    return np.stack(imgs)


class TFLiteRunner:
    def __init__(self, path):
        self.interp = tf.lite.Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]
        self.inp_idx = inp["index"]
        self.out_idx = out["index"]
        self.inp_dtype = inp["dtype"]
        self.inp_shape = inp["shape"]   # [1, H, W, 3]

    def predict(self, x):
        """x: (N, H, W, 3) float32 0-255. Returns (N, num_classes) float32."""
        preds = []
        for img in x:
            inp = img[None].astype(self.inp_dtype)
            self.interp.set_tensor(self.inp_idx, inp)
            self.interp.invoke()
            preds.append(self.interp.get_tensor(self.out_idx)[0].astype(np.float32))
        return np.stack(preds)

    def latency_ms(self, x, n=50):
        """Median single-image inference time on the host CPU (ms)."""
        ts = []
        for i in range(n):
            img = x[i % len(x)][None].astype(self.inp_dtype)
            t0 = time.perf_counter()
            self.interp.set_tensor(self.inp_idx, img)
            self.interp.invoke()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))


def convert_fp32(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()


def convert_fp16(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def convert_int8(model, calib):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([img[None]] for img in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return conv.convert()


def eval_tflite(runner, x_va, y_va, x_te, y_te):
    p_va = runner.predict(x_va)
    p_te = runner.predict(x_te)
    m_va = U.compute_metrics(y_va, p_va)
    m_te = U.compute_metrics(y_te, p_te)
    return m_va, m_te, p_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder name under results/runs/")
    ap.add_argument("--variants", nargs="*", default=["fp32", "fp16", "int8"])
    ap.add_argument("--cpu", action="store_true", help="hide GPU (recommended for 448px on macOS)")
    args = ap.parse_args()

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    run_dir = C.RUNS_DIR / args.run
    summ = json.load(open(run_dir / "summary.json"))
    image_size = summ.get("image_size", C.IMAGE_SIZE)
    variant_spec = summ.get("variant_spec", {})
    source_size = variant_spec.get("source_size", 256)

    out_dir = C.RESULTS_DIR / "deploy" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Deploy: {args.run}  image_size={image_size}  source_size={source_size} ===")

    # Load val/test arrays (use the eval cache matching this run's input size)
    print("Loading val/test data...", flush=True)
    if image_size == 224:
        x_va, y_va = U.load_split("val", image_size)
        x_te, y_te = U.load_split("test", image_size)
    else:
        # hires: load from the pre-built eval cache
        ev_dir = C.ROOT / "data" / f"ihc_{source_size}_eval{image_size}"
        df = pd.read_csv(C.MANIFEST)
        def _load(split):
            sub = df[df.split == split].reset_index(drop=True)
            imgs = [np.asarray(Image.open(ev_dir / p).convert("RGB"), dtype=np.uint8)
                    for p in sub.path]
            return np.stack(imgs), sub.label.values.astype(np.int32)
        x_va, y_va = _load("val")
        x_te, y_te = _load("test")

    x_va_f = x_va.astype(np.float32)
    x_te_f = x_te.astype(np.float32)

    # Calibration data
    print("Loading calibration data...", flush=True)
    if image_size == 224:
        calib = load_calib_data(image_size)
    else:
        calib = load_calib_hires(image_size, source_size)

    # Load Keras model
    print("Loading Keras model...", flush=True)
    model = U.load_trained(run_dir)

    # Save original FP32 Keras model
    fp32_keras_path = out_dir / "model_fp32.keras"
    if not fp32_keras_path.exists():
        print(f"Saving original FP32 Keras model to {fp32_keras_path}...", flush=True)
        model.save(fp32_keras_path)

    # FP32 Keras baseline (reference)
    print("Computing FP32 Keras baseline...", flush=True)
    p_va_keras = model.predict(x_va_f, batch_size=16, verbose=0)
    p_te_keras = model.predict(x_te_f, batch_size=16, verbose=0)
    m_va_keras = U.compute_metrics(y_va, p_va_keras)
    m_te_keras = U.compute_metrics(y_te, p_te_keras)
    print(f"  FP32 Keras  val qwk={m_va_keras['qwk']:.4f} acc={m_va_keras['accuracy']:.4f} | "
          f"test qwk={m_te_keras['qwk']:.4f} acc={m_te_keras['accuracy']:.4f}", flush=True)

    results = []

    for variant in args.variants:
        path = out_dir / f"model_{variant}.tflite"
        print(f"\n--- {variant} ---", flush=True)
        t0 = time.time()
        note = ""
        status = "ok"

        # Convert
        if not path.exists():
            try:
                if variant == "fp32":
                    blob = convert_fp32(model)
                elif variant == "fp16":
                    blob = convert_fp16(model)
                elif variant == "int8":
                    blob = convert_int8(model, calib)
                else:
                    print(f"  Unknown variant {variant}, skipping.")
                    continue
                path.write_bytes(blob)
                print(f"  Converted in {time.time()-t0:.1f}s, size {path.stat().st_size/2**20:.2f} MB")
            except Exception as e:
                status = "convert_failed"
                note = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  CONVERSION FAILED: {note}")
                results.append(dict(run=args.run, variant=variant, status=status, note=note))
                continue
        else:
            print(f"  Already exists ({path.stat().st_size/2**20:.2f} MB), skipping conversion.")

        # Evaluate
        try:
            runner = TFLiteRunner(path)
            m_va, m_te, p_va_tfl = eval_tflite(runner, x_va_f, y_va, x_te_f, y_te)
            lat = runner.latency_ms(x_te_f)
            agreement = float(np.mean(p_va_tfl.argmax(1) == p_va_keras.argmax(1)))
            acc_drop = m_va_keras["accuracy"] - m_va["accuracy"]

            # INT8 sanity checks
            if variant == "int8":
                n_pred_classes = len(np.unique(p_va_tfl.argmax(1)))
                if n_pred_classes < 3:
                    status = "collapsed"
                    note = f"Prediction collapsed: only {n_pred_classes} classes predicted on val"
                elif acc_drop > ACC_DROP_THRESH:
                    status = "degraded"
                    note = f"Val accuracy drop {acc_drop:.3f} > threshold {ACC_DROP_THRESH}"
                elif agreement < AGREE_THRESH:
                    status = "degraded"
                    note = f"Agreement {agreement:.3f} < threshold {AGREE_THRESH}"

            size_mb = path.stat().st_size / 2**20
            print(f"  size={size_mb:.2f}MB  lat={lat:.1f}ms  agree={agreement:.3f}  "
                  f"val qwk={m_va['qwk']:.4f} acc={m_va['accuracy']:.4f} | "
                  f"test qwk={m_te['qwk']:.4f} acc={m_te['accuracy']:.4f}  status={status}  {note}")

            results.append(dict(
                run=args.run, variant=variant, status=status, note=note,
                size_mb=size_mb, mac_latency_ms=lat, agreement_with_fp32_keras=agreement,
                val_acc_drop=float(acc_drop),
                **{f"val_{k}": m_va[k] for k in ("accuracy", "macro_f1", "qwk") if k in m_va},
                **{f"test_{k}": m_te[k] for k in ("accuracy", "macro_f1", "qwk") if k in m_te},
                val_confusion=json.dumps(m_va.get("confusion_matrix", [])),
                test_confusion=json.dumps(m_te.get("confusion_matrix", [])),
            ))

        except Exception as e:
            status = "runtime_failed"
            note = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  RUNTIME FAILED: {note}")
            results.append(dict(run=args.run, variant=variant, status=status, note=note,
                                size_mb=path.stat().st_size/2**20 if path.exists() else None))

    # Save summary
    summary_path = out_dir / "deploy_summary.json"
    json.dump({
        "run": args.run,
        "model": summ["model"],
        "image_size": image_size,
        "params": summ["params"],
        "fp32_keras_val": {k: m_va_keras[k] for k in ("accuracy", "macro_f1", "qwk")},
        "fp32_keras_test": {k: m_te_keras[k] for k in ("accuracy", "macro_f1", "qwk")},
        "variants": results,
        "note": (
            "mac_latency_ms is measured on the host Mac CPU (TFLite XNNPACK). "
            "Raspberry Pi 4B inference was NOT performed — no physical hardware measurements. "
            "INT8 status='ok' requires agreement>=0.95 with FP32 Keras and val acc drop <0.05."
        )
    }, open(summary_path, "w"), indent=2)
    print(f"\nSaved {summary_path}")

    csv_path = out_dir / "deploy_summary.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

