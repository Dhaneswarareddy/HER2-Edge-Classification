# FYP3 Edge Deployment: Mobile Smartphone & Embedded Microcomputer

**Research software. Not clinically validated. Do not use for diagnosis or treatment decisions.**

## Deployment Summary

The FYP3 edge deployment targets two complementary hardware platforms:
1. **Physical Mobile Smartphone (Primary Point-of-Care Target)**: **Samsung Galaxy M35 5G** (`SM-M356B`, Android 16, Exynos 1380 5nm SoC, 6GB RAM). Includes fully functional native Android application (`HER2 Mobile Edge`, `android_app/`) running TFLite 2.16.1 with XNNPACK. Achieves **78.91 ms (12.67 FPS)** on FP16 and **21.71 ms (46.06 FPS)** on INT8 with 4-thread Cortex-A78 pinning, completely unthrottled at 28.9°C. See `deployment/MOBILE_BENCHMARK_REPORT.md`.
2. **Physical Embedded Single-Board Computer (Comparative Baseline)**: **Raspberry Pi 4 Model B Rev 1.4** (Quad Cortex-A72 @ 1.80 GHz, 4GB RAM, Debian 13). Achieves 444.1 ms median latency (2 threads, 191.5 MB RAM, 100.0% agreement); 4 threads achieved 353.5 ms but incurred soft thermal throttling at 84.2°C (`0x80008`). See `deployment/PI4_BENCHMARK_REPORT.md`.

## Two separate modes

| | `primary` | `experimental` |
|---|---|---|
| Use for | BCI-like IHC tiles (1024 px, about 0.46 µm/px) | HER2-IHC-40x tiles only |
| Network | EfficientNetV2B0 448 px FP16 | EfficientNetV2B0 224 px FP16 |
| Preprocessing | Lanczos to 512, bilinear to 448 | Lanczos to 256, bilinear to 224, then frozen stain gains |
| Stain gains | None | g_h 0.2545, g_d 0.5045, fixed |
| Extra flag | None | `--i-understand-experimental` |

On the Mac, FP16 through this code matched Keras on 100% of 1,485 BCI tiles (primary) and 99.6% of 2,402 development tiles (experimental). That says nothing about Pi speed.

The modes cannot be mixed. Each has its own config, model hash and preprocessing function. Gains cannot be set from the command line, and the code refuses a config with changed gains, stain gains in primary mode, or a swapped model. See `FINAL_MODEL_SELECTION.md` for why.

**Known weaknesses:** borderline scores are weak across domains. Experimental test recall is 0.18 for 1+ and 0.57 for 2+, with 2+ precision 0.09. Results are per tile. There is no slide-level score, because no slide identifiers exist in the data.

## Bundle contents

| Path | Content |
|---|---|
| `her2_infer/` | Inference package: NumPy, Pillow and a TFLite runtime only |
| `configs/` | One pinned config per mode |
| `models/` | One FP16 model per mode, 11.27 MB each |
| `predict.py` | Score tiles |
| `benchmark.py` | Benchmark protocol |
| `verify_bundle.py` | Integrity check |
| `install_pi.sh`, `requirements_pi.txt` | Setup |
| `reference/` | 120 tiles per mode with reference predictions, plus 8 raw tiles per mode |

Reference tiles come from BCI validation and HER2-IHC-40x development tiles, never the external test split. BCI is non-commercial; HER2-IHC-40x is CC BY-SA 4.0. Keep the bundle private.

## Install on the Pi

Needs a Raspberry Pi 4B with **64-bit Raspberry Pi OS (Bookworm)**, the official 5 V/3 A supply, and a heatsink or fan.

```bash
# on the Mac
scp /Users/dhaneswarareddy/Downloads/FYP3/deployment/her2_fyp3_pi_bundle.tar.gz pi@<pi-ip>:~
# on the Pi
tar xzf her2_fyp3_pi_bundle.tar.gz && cd her2_fyp3_pi_bundle
./install_pi.sh                     # creates ~/her2env and runs verify_bundle.py
source ~/her2env/bin/activate
```

`verify_bundle.py` must end with `BUNDLE OK`. If `ai-edge-litert` has no wheel for your Python, the installer falls back to `tflite-runtime`.

## Run inference

```bash
python3 predict.py --mode primary --input tiles/ --out primary.csv
python3 predict.py --mode experimental --i-understand-experimental --input tiles/ --out experimental.csv
```

Each CSV row gives the file, mode, predicted score, confidence and four probabilities. Experimental output also gives a temperature-scaled confidence, which never changes the predicted class. An existing output file is refused, so the two modes can't end up in one file.

From Python:

```python
from her2_infer import Her2Engine
print(Her2Engine("primary", threads=4).predict("tile.png"))
```

## Benchmark protocol

**Before each run:** close the desktop and other programs, and set the governor:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Run both modes:**

```bash
python3 benchmark.py --mode primary --threads 1 2 4 --out-dir results
python3 benchmark.py --mode experimental --i-understand-experimental --threads 1 2 4 --out-dir results
```

Each thread count runs in a fresh process: 10 warm-up runs, 100 timed runs, then end-to-end and agreement checks.

| Metric | Column | How it is measured |
|---|---|---|
| Model loading time | `load_s`, `import_and_load_s` | Creating the interpreter and allocating tensors; the second also includes library import and the hash check |
| Warm-up time | `warmup_total_s`, `warmup_first_invoke_ms`, `warmup_mean_ms` | First 10 invokes |
| Latency | `invoke_mean_ms`, `invoke_median_ms`, `invoke_p95_ms`, `invoke_p99_ms` | 100 timed invokes, network only |
| Full pipeline | `preprocess_*`, `end_to_end_*` | Raw 1024 px tile to prediction, 8 tiles × 3 |
| Peak RAM | `peak_rss_mb`, `rss_after_load_mb` | A separate process that only loads the model and scores raw tiles, so reference data does not inflate it |
| CPU utilisation | `cpu_process_pct`, `cpu_system_pct` | During the timed runs; 400% is all four cores |
| Prediction agreement | `agreement_with_mac_tflite`, `agreement_with_keras`, `max_abs_dp_*` | 120 tiles against stored Mac references |
| Thermal state | `temp_start`, `temp_end`, `arm_clock`, `throttled` | From `vcgencmd`; `throttled=0x0` means clean |
| Provenance | `device`, `is_raspberry_pi_4`, `runtime`, `governor` | Recorded automatically |

**Valid Pi result:** `is_raspberry_pi_4` is True, `throttled` is `0x0`, the governor is `performance`, and `agreement_with_mac_tflite` is at least 0.99. A run that throttles must be repeated with better cooling. On any other device, the output file is named `NOT_A_PI` and must not be reported as a Pi result.

## Bring results back

Copy `results/benchmark_*_pi4.csv` to `FYP3/deployment/benchmark_results/`. Only then may Pi figures enter the report.

## What may be claimed

| Allowed | Not allowed |
|---|---|
| BCI metrics, with the slide-label caveat | Clinical validity or diagnostic accuracy |
| Experimental results on HER2-IHC-40x, with disclosures | Generalisation to other sites |
| Mac latency, labelled as Mac | Pi performance before a valid Pi run |
| Pi figures from a valid run | Per-slide or per-patient performance |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `This needs a 64-bit OS` | Reinstall with 64-bit Raspberry Pi OS |
| `does not match the pinned hash` | The model file changed. Re-copy the bundle |
| `primary mode must have no stain calibration` or `frozen stain_p99 gains` | A config was edited. Restore it from the bundle |
| No TFLite runtime found | `pip install tflite-runtime` in `~/her2env` |
| Latency rises during a run, or `throttled` is not `0x0` | Improve cooling and repeat |
| Out of memory at 448 px | Close other programs, and report it: this is a real finding |
| Warning about a delegate and dynamic-sized tensors | Seen on the Mac and not investigated. Report it with your results: acceleration may not fully apply, which would affect latency |
