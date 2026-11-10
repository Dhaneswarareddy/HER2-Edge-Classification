"""Android latency benchmark (Samsung Galaxy M35, Realme 6) with the official TFLite benchmark_model binary.

Differences from the earlier run_samsung_m35_benchmark.py / run_realme_6_benchmark.py (kept unchanged):
  * every raw benchmark_model output is saved (the earlier run kept only parsed numbers);
  * XNNPACK is requested explicitly (--use_xnnpack=true) and the flags are recorded;
  * 100 timed runs after 10 warm-up runs, 60 s cool-down between configurations;
  * temperature: CPU thermal zones are read if the OS exposes them (often not without root); battery temperature is
    recorded separately and labelled as such. No throttle flag exists on Android, so none is reported;
  * the INT8 model is benchmarked for speed only and is labelled 'rejected for deployment' (agreement 0.928 < 0.95).
benchmark_model feeds random input: these are network latencies, not accuracy. On-device accuracy is NOT measured by
this script (see README).

  python android_benchmark.py --device-name samsung_m35 [--serial <adb serial>]
Requires USB debugging and a connected phone; nothing is installed besides files under /data/local/tmp.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
BIN = ROOT / "deployment" / "android_aarch64_benchmark_model"
MODELS = {
    "primary_448_fp32": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_fp32.tflite",
    "primary_448_fp16": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_fp16.tflite",
    "primary_448_int8_REJECTED": ROOT / "results/deploy/EfficientNetV2B0__ce__hires/model_int8.tflite",
    "experimental_224_fp16": ROOT / "deployment/models/experimental_stain224_fp16/model.tflite",
}
THREADS = [1, 2, 4, 8]


def adb(serial, *a, check=True):
    cmd = [ADB] + (["-s", serial] if serial else []) + list(a)
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout


def temps(serial):
    out = {}
    z = adb(serial, "shell", "for z in /sys/class/thermal/thermal_zone*; do echo $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null); done", check=False)
    cpu = [l.split() for l in z.splitlines() if len(l.split()) == 2 and re.search(r"cpu|soc|big|little|mtktscpu", l, re.I)]
    out["cpu_thermal_zones_c"] = {t: int(v) / 1000 for t, v in cpu if v.lstrip("-").isdigit()} or "not readable"
    b = adb(serial, "shell", "dumpsys battery | grep -i temperature", check=False)
    m = re.search(r"^\s*temperature:\s*(\d+)", b, re.M)  # Samsung also prints a battery-event log with 'temperature:'
    out["battery_temp_c"] = int(m.group(1)) / 10 if m else None
    return out


def parse(raw):
    r = {}
    m = re.findall(r"count=(\d+) first=(\d+) curr=(\d+) min=(\d+) max=(\d+) avg=([\d.]+) std=(\d+) p5=(\d+) median=(\d+) p95=(\d+)", raw)
    if m:
        c = m[-1]
        r.update(n=int(c[0]), min_ms=int(c[3]) / 1e3, max_ms=int(c[4]) / 1e3, mean_ms=float(c[5]) / 1e3, std_ms=int(c[6]) / 1e3,
                 p5_ms=int(c[7]) / 1e3, median_ms=int(c[8]) / 1e3, p95_ms=int(c[9]) / 1e3)
    mm = re.search(r"Peak memory footprint \(MB\): init=([\d.]+) overall=([\d.]+)", raw) or \
        re.search(r"Memory footprint delta.*?init=([\d.]+)\s+overall=([\d.]+)", raw, re.S)
    if mm:
        r.update(mem_init_mb=float(mm.group(1)), mem_overall_mb=float(mm.group(2)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--cooldown", type=int, default=60)
    a = ap.parse_args()
    out = HERE / "outputs" / f"android_{a.device_name}"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    info = {k: adb(a.serial, "shell", "getprop", p).strip() for k, p in
            [("model", "ro.product.model"), ("board", "ro.product.board"), ("android", "ro.build.version.release"),
             ("sdk", "ro.build.version.sdk"), ("abi", "ro.product.cpu.abi")]}
    info["cpuinfo_max_freq_khz"] = adb(a.serial, "shell", "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq", check=False).split()
    json.dump(info, open(out / "device_info.json", "w"), indent=1)
    adb(a.serial, "push", str(BIN), "/data/local/tmp/benchmark_model")
    adb(a.serial, "shell", "chmod", "755", "/data/local/tmp/benchmark_model")
    rows = []
    for name, path in MODELS.items():
        adb(a.serial, "push", str(path), f"/data/local/tmp/{name}.tflite")
        for t in THREADS:
            t0 = temps(a.serial)
            flags = f"--graph=/data/local/tmp/{name}.tflite --num_threads={t} --use_xnnpack=true --warmup_runs=10 --num_runs=100 --report_peak_memory_footprint=true"
            raw = adb(a.serial, "shell", f"/data/local/tmp/benchmark_model {flags}", check=False)
            (out / "raw" / f"{name}_t{t}.txt").write_text(raw)
            r = dict(device=info["model"], model=name, threads=t, flags=flags, file_mb=path.stat().st_size / 2**20,
                     **parse(raw), temp_start=json.dumps(t0), temp_end=json.dumps(temps(a.serial)))
            rows.append(r)
            print(name, t, r.get("median_ms"), r.get("mem_overall_mb"), flush=True)
            pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
            time.sleep(a.cooldown)


if __name__ == "__main__":
    main()
