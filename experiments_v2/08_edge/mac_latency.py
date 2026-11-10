"""Mac (M1, CPU) TFLite latency of the deployed models, one protocol for all: fresh interpreter per configuration,
10 warm-up + 200 timed invokes on real preprocessed tiles (cycled), threads 1/2/4, peak RSS of the process.
Run with nothing else heavy running (training or inference would distort timing). Network latency only; no decode/resize.

  python mac_latency.py            -> outputs/mac_latency.csv
"""
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = {
    "primary_448_fp32": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_fp32.tflite",
    "primary_448_fp16": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_fp16.tflite",
    "primary_448_int8_REJECTED": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_int8.tflite",
    "experimental_224_fp16": ROOT / "deployment/models/experimental_stain224_fp16/model.tflite",
}


def worker(name, threads):
    import tensorflow as tf
    path = MODELS[name]
    it = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    a, b = it.get_input_details()[0], it.get_output_details()[0]
    size = int(a["shape"][1])
    ref = np.load(HERE / "pi_bundle" / "tiles120.npz")["tiles"]          # 448 px uint8, BCI val reference tiles
    if size != ref.shape[1]:
        ref = np.stack([np.asarray(tf.image.resize(t, [size, size], method="bilinear")) for t in ref[:32]])
    x = ref.astype(np.float32)
    for k in range(10):
        it.set_tensor(a["index"], x[k % len(x)][None]); it.invoke()
    lat = []
    for k in range(200):
        it.set_tensor(a["index"], x[k % len(x)][None])
        s = time.perf_counter(); it.invoke(); lat.append((time.perf_counter() - s) * 1e3)
    lat = np.array(lat)
    print(json.dumps(dict(model=name, threads=threads, input_size=size, file_mib=path.stat().st_size / 2**20,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                          median_ms=float(np.median(lat)), mean_ms=float(lat.mean()), p95_ms=float(np.percentile(lat, 95)),
                          std_ms=float(lat.std()), n=len(lat),
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)))  # macOS: bytes


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], int(sys.argv[3]))
    import pandas as pd
    rows = []
    for name in MODELS:
        for t in [1, 2, 4]:
            out = subprocess.run([sys.executable, __file__, "--worker", name, str(t)], capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print("FAILED", name, t, out.stderr[-500:]); continue
            rows.append(json.loads(line[-1])); print(rows[-1], flush=True)
            time.sleep(10)
    (HERE / "outputs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "outputs" / "mac_latency.csv", index=False)


if __name__ == "__main__":
    main()
