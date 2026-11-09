#!/usr/bin/env python3
"""Benchmark one mode on the current device. Run on a Raspberry Pi 4B for Pi figures.

  python3 benchmark.py --mode primary --threads 1 2 4 --out-dir results
  python3 benchmark.py --mode experimental --i-understand-experimental --threads 1 2 4 --out-dir results

Each thread count runs in a fresh process, so load time and peak memory are clean. Measured per configuration:
model load time; warm-up time; invoke latency mean, median, P95, P99; end-to-end latency including preprocessing;
peak RAM; CPU utilisation; agreement with Mac reference predictions and with Keras; temperature, clock and throttling.
Records the real device. A result is labelled as a Raspberry Pi 4 result only if the device reports that model.
"""
import argparse, csv, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import psutil
except ImportError:
    psutil = None
from her2_infer import Her2Engine
from her2_infer.engine import BASE


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


def governor():
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return ""


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def stats(v, prefix):
    v = np.array(v)
    return {f"{prefix}_mean_ms": float(v.mean()), f"{prefix}_median_ms": float(np.median(v)), f"{prefix}_p95_ms": float(np.percentile(v, 95)), f"{prefix}_p99_ms": float(np.percentile(v, 99)), f"{prefix}_std_ms": float(v.std()), f"{prefix}_n": len(v)}


def worker(a):
    threads = a.threads[0]; t0 = time.perf_counter(); e = Her2Engine(a.mode, threads); import_and_load_s = time.perf_counter() - t0
    ref = np.load(BASE / "reference" / f"{a.mode}.npz"); tiles = ref["tiles_preprocessed"]
    def inp(i):
        return e.from_resized(tiles[i])                        # called before each timer starts, so invoke latency is the network alone
    r = dict(mode=a.mode, threads=threads, device=device(), is_raspberry_pi_4="Raspberry Pi 4" in device(), machine=platform.machine(), os=platform.platform(), python=platform.python_version(), runtime=e.runtime, numpy=np.__version__,
             governor=governor(), input_size=e.cfg["input_size"], model_file_mb=os.path.getsize(BASE / "models" / e.cfg["model_dir"] / "model.tflite") / 2**20, load_s=e.load_s, import_and_load_s=import_and_load_s, temp_start=vc("measure_temp"))
    w = []
    for k in range(a.warmup):
        x = inp(k % len(tiles)); t1 = time.perf_counter(); e.invoke(x); w.append((time.perf_counter() - t1) * 1000)
    r.update(warmup_total_s=sum(w) / 1000, warmup_first_invoke_ms=w[0], warmup_mean_ms=float(np.mean(w)), warmup_last_ms=w[-1])
    p = psutil.Process() if psutil else None; c0 = p.cpu_times() if p else None
    if psutil:
        psutil.cpu_percent(None)
    lat = []; t = time.perf_counter()
    for k in range(a.runs):
        x = inp(k % len(tiles)); t1 = time.perf_counter(); e.invoke(x); lat.append((time.perf_counter() - t1) * 1000)
    wall = time.perf_counter() - t
    if p:
        c1 = p.cpu_times(); r.update(cpu_process_pct=100 * ((c1.user - c0.user) + (c1.system - c0.system)) / wall, cpu_system_pct=psutil.cpu_percent(None))
    r.update(stats(lat, "invoke")); r["throughput_fps"] = a.runs / wall
    pre, e2e = [], []
    for f in sorted((BASE / "reference" / f"raw_{a.mode}").glob("*.png")):
        for _ in range(a.e2e_repeats):
            t1 = time.perf_counter(); x = e.preprocess(f); t2 = time.perf_counter(); e.invoke(x); t3 = time.perf_counter(); pre.append((t2 - t1) * 1000); e2e.append((t3 - t1) * 1000)
    if e2e:
        r.update(stats(pre, "preprocess")); r.update(stats(e2e, "end_to_end"))
    q = np.stack([e.invoke(inp(i)) for i in range(len(tiles))]); a_t, a_k = ref["tflite_mac"].argmax(1), ref["keras"].argmax(1); y = ref["labels"]; pr = q.argmax(1)
    r.update(agreement_tiles=len(tiles), agreement_with_mac_tflite=float((pr == a_t).mean()), agreement_with_keras=float((pr == a_k).mean()), max_abs_dp_vs_mac_tflite=float(np.abs(q - ref["tflite_mac"]).max()), max_abs_dp_vs_keras=float(np.abs(q - ref["keras"]).max()), accuracy_on_reference_tiles=float((pr == y).mean()),
             peak_rss_benchmark_process_mb=peak_rss_mb(), temp_end=vc("measure_temp"), arm_clock=vc("measure_clock", "arm"), throttled=vc("get_throttled"))
    print(json.dumps(r))


def ram_probe(a):
    """Peak memory of real inference only: load the model, then score raw tiles. No reference data is loaded here."""
    e = Her2Engine(a.mode, a.threads[0]); after = peak_rss_mb(); fs = sorted((BASE / "reference" / f"raw_{a.mode}").glob("*.png"))
    for f in fs:
        e.predict(f)
    print(json.dumps(dict(rss_after_load_mb=after, peak_rss_mb=peak_rss_mb(), ram_probe_tiles=len(fs))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["primary", "experimental"]); ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4]); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--e2e-repeats", type=int, default=3); ap.add_argument("--out-dir", default="results"); ap.add_argument("--i-understand-experimental", action="store_true"); ap.add_argument("--worker", action="store_true"); ap.add_argument("--ram-probe", action="store_true"); a = ap.parse_args()
    if a.mode == "experimental" and not a.i_understand_experimental:
        sys.exit("Pass --i-understand-experimental to benchmark experimental mode.")
    if a.ram_probe:
        return ram_probe(a)
    if a.worker:
        return worker(a)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); rows = []
    for t in a.threads:
        cmd = [sys.executable, __file__, "--worker", "--mode", a.mode, "--threads", str(t), "--warmup", str(a.warmup), "--runs", str(a.runs), "--e2e-repeats", str(a.e2e_repeats)] + (["--i-understand-experimental"] if a.mode == "experimental" else [])
        s = subprocess.run(cmd, capture_output=True, text=True); m = subprocess.run(cmd[:2] + ["--ram-probe"] + cmd[3:], capture_output=True, text=True)
        if s.returncode or m.returncode:
            sys.exit((s.stderr + m.stderr)[-2000:])
        rows.append({**json.loads(s.stdout.strip().splitlines()[-1]), **json.loads(m.stdout.strip().splitlines()[-1])}); r = rows[-1]; print(f"{a.mode} threads={t}: load {r['load_s']:.2f}s warm-up {r['warmup_total_s']:.1f}s invoke median {r['invoke_median_ms']:.1f} p95 {r['invoke_p95_ms']:.1f} ms peak RAM {r['peak_rss_mb']:.0f} MB agreement {r['agreement_with_mac_tflite']:.3f}", flush=True)
    tag = "pi4" if rows[0]["is_raspberry_pi_4"] else "NOT_A_PI"; f = out / f"benchmark_{a.mode}_{tag}.csv"
    with open(f, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", f)
    if not rows[0]["is_raspberry_pi_4"]:
        print("This is not a Raspberry Pi 4. Do not report these numbers as Pi results.")


if __name__ == "__main__":
    main()
