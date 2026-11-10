#!/usr/bin/env python3
"""Raspberry Pi 4B: FP32 / FP16 / INT8 of the deployed 448 px EfficientNetV2B0 (results/deploy/...hires), same protocol
for every precision. Fills the gap in the 31 Oct Pi benchmark (FP16 only; the report's Pi INT8 figure had no measurement).

  python3 pi_precision_benchmark.py --threads 1 2 4 --out results_pi.csv

Each (precision, threads) runs in a fresh process. Measured: load time, 10 warm-up and 100 timed invokes (network only;
input = the 120 preprocessed BCI-validation reference tiles, cycled), peak RSS, CPU %, temperature, ARM clock and
throttle flags before/after, and agreement of every class prediction on the 120 tiles with the Mac TFLite output of the
SAME precision (and with Keras). Raw per-run JSON is kept next to the CSV. A row is labelled a Pi 4 result only if
/proc/device-tree/model says so. No power is measured.
"""
import argparse, csv, hashlib, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"fp32": "primary_hires_fp32.tflite", "fp16": "primary_hires_fp16.tflite", "int8": "primary_hires_int8.tflite"}


def device():
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def vc(*a):
    try:
        return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def interp(path, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter
        rt = "ai_edge_litert"
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        rt = "tflite_runtime"
    return Interpreter(model_path=str(path), num_threads=threads), rt


def worker(prec, threads):
    path = HERE / "models" / MODELS[prec]
    ref = np.load(HERE / "refs_primary_precisions.npz", allow_pickle=True)
    tiles = np.load(HERE / "tiles120.npz")["tiles"]
    r = dict(precision=prec, threads=threads, device=device(), is_raspberry_pi_4="Raspberry Pi 4" in device(),
             machine=platform.machine(), python=platform.python_version(), numpy=np.__version__,
             model_file=path.name, model_mb=path.stat().st_size / 2**20,
             model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             temp_start=vc("measure_temp"), clock_start=vc("measure_clock", "arm"), throttled_start=vc("get_throttled"))
    t0 = time.perf_counter()
    it, rt = interp(path, threads); it.allocate_tensors()
    r["runtime"], r["load_s"] = rt, time.perf_counter() - t0
    a, b = it.get_input_details()[0], it.get_output_details()[0]

    def run(x):
        it.set_tensor(a["index"], x[None].astype(np.float32)); it.invoke()
        return it.get_tensor(b["index"])[0].astype(np.float64)

    for k in range(10):
        run(tiles[k % len(tiles)])
    cpu0, w0 = time.process_time(), time.perf_counter()
    lat = []
    for k in range(100):
        x = tiles[k % len(tiles)]
        s = time.perf_counter(); run(x); lat.append((time.perf_counter() - s) * 1000)
    r["cpu_process_pct"] = 100 * (time.process_time() - cpu0) / (time.perf_counter() - w0)
    lat = np.array(lat)
    r.update(invoke_mean_ms=lat.mean(), invoke_median_ms=np.median(lat), invoke_p95_ms=np.percentile(lat, 95),
             invoke_p99_ms=np.percentile(lat, 99), invoke_std_ms=lat.std(), invoke_n=len(lat),
             throughput_fps=1000 / np.median(lat))
    p = np.stack([run(t) for t in tiles])
    r["agreement_with_mac_same_precision"] = float((p.argmax(1) == ref[f"mac_{prec}"].argmax(1)).mean())
    r["max_abs_dp_vs_mac_same_precision"] = float(np.abs(p - ref[f"mac_{prec}"]).max())
    r["agreement_with_keras"] = float((p.argmax(1) == ref["keras"].argmax(1)).mean())
    r["accuracy_120_reference_tiles"] = float((p.argmax(1) == ref["labels"]).mean())
    r["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    r.update(temp_end=vc("measure_temp"), clock_end=vc("measure_clock", "arm"), throttled_end=vc("get_throttled"))
    print(json.dumps(r, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"])
    ap.add_argument("--out", default="results_pi.csv")
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--cooldown", type=int, default=60, help="seconds between configurations")
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], int(a.worker[1]))
    rows = []
    raw = HERE / "raw"; raw.mkdir(exist_ok=True)
    for prec in a.precisions:
        for t in a.threads:
            out = subprocess.run([sys.executable, __file__, "--worker", prec, str(t)], capture_output=True, text=True)
            (raw / f"{prec}_t{t}.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(f"FAILED {prec} t{t}: see raw/{prec}_t{t}.log"); continue
            r = json.loads(line[-1]); rows.append(r)
            print(f"{prec} t{t}: median {r['invoke_median_ms']:.1f} ms, p95 {r['invoke_p95_ms']:.1f}, rss {r['peak_rss_mb']:.0f} MB, "
                  f"agree(mac same prec) {r['agreement_with_mac_same_precision']:.3f}, {r['temp_end']} {r['throttled_end']}", flush=True)
            time.sleep(a.cooldown)
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
