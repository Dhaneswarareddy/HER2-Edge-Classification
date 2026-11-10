# Experiment 8: mobile and edge benchmarking (complete, 10 Nov 2026)

**What existed before** (`deployment/`):
- Pi 4B FP16 latency;
- Samsung M35 and Realme 6 latency;
- Android app v1.0.

**Problems found** (`../REPORT_CORRECTIONS_v2.md` items 6–11, 31):
- no Pi INT8 measurement, although a value was reported;
- phone "throttle" flags that Android does not have;
- battery temperature reported as if it were CPU temperature;
- power and energy claims with no measurement;
- INT8 presented as deployable;
- an app preprocessing path different from the validated one;
- on-device accuracy never measured.

**Done here.** Every device was measured with one protocol, the app was fixed (v1.1), and on-device accuracy was
measured for the first time.

## A. Accuracy of every exported precision (Mac; `mac_prepare.py`, `outputs/precision_accuracy.csv`)

| Precision | MiB | Val acc | Test acc | Test QWK | Test 1+ recall | Agreement with Keras (val / test) |
|---|---|---|---|---|---|---|
| FP32 | 22.4 | 0.950 | 0.975 | 0.959 | 0.960 | 1.000 / 1.000 |
| FP16 (deployed) | 11.3 | 0.950 | 0.975 | 0.959 | 0.960 | 1.000 / 1.000 |
| INT8 | 6.8 | 0.919 | 0.933 | 0.907 | **0.855** | **0.928** / 0.943 |

INT8 fails the 0.95 agreement gate, so it is speed-only.

## B. The app's preprocessing (`outputs/android_preprocessing_check.csv`; `pil_resample_port.py`)

- **v1.0** resized 1024 → 448 px in one step. Simulated on the Mac, this changes 0.1–2.8% of predictions, with no
  consistent accuracy change.
- **v1.1** uses `PilResample.kt`, a port of Pillow's fixed-point resampler. Its Python twin is **bit-identical** to PIL
  on BCI tiles (max difference 0 over 20 tiles).

## C. Device latency (median ms, 10 warm-up + 100 timed runs, no core pinning; `outputs/edge_latency_table.csv`)

| Device | Model | 1T | 2T | 4T | 8T |
|---|---|---|---|---|---|
| Samsung M35 (Android 17) | 448 px FP16 (deployed) | 214.9 | 119.0 | **81.4** | 95.1 |
| Samsung M35 | 448 px INT8 (rejected) | 75.4 | 38.7 | 21.7 | 45.3 |
| Samsung M35 | 224 px FP16 | 53.5 | 28.6 | 16.8 | 20.3 |
| Realme 6 (Android 11) | 448 px FP16 (deployed) | 301.7 | 174.9 | 142.6 | **139.1** |
| Realme 6 | 448 px INT8 (rejected) | 98.8 | 53.6 | 48.4 | 34.3 |
| Realme 6 | 224 px FP16 | 73.4 | 40.1 | 32.2 | 26.3 |
| Raspberry Pi 4B | 448 px FP16 (deployed) | 761.8 | 437.0 | **336.5** | — |
| Raspberry Pi 4B | 448 px INT8 (rejected) | 361.3 | 187.9 | 104.1 | — |

FP32 rows are in the CSV; on CPU, FP32 latency equals FP16.

**Temperature and throttling:**
- Phones: battery 29.6–32.0 °C. CPU zones are not readable without root, and Android has no throttle flag.
- Pi: SoC at most 62.8 °C, `throttled=0x0` throughout, with an official-rating supply. Before the supply change the Pi
  showed `0x50000`, meaning under-voltage had occurred.

**Platform dependence of INT8:**
- Pi INT8 agrees with Keras on 109 of 120 tiles, and with Mac INT8 on 114 of 120.
- Pi FP32 and FP16 agree on 120/120.

## D. On-device accuracy of the app (v1.1; `ondevice_validation.py`; `outputs/ondevice_<device>/summary.json`)

Inputs: the full BCI grouped test split (758 original 1024 px tiles) and the 120 BCI-validation reference tiles. They
are processed through exactly the app's UI path: `PilResample` and TFLite FP16, 4 threads.

| | Test acc | Bal. acc | QWK | Agreement with Keras (758) | Max \|Δp\| | Reference 120: acc / agreement |
|---|---|---|---|---|---|---|
| Samsung M35 | 0.975 | 0.973 | 0.959 | 758/758 | 0.018 | 0.950 / 120/120 |
| Realme 6 | 0.975 | 0.973 | 0.959 | 758/758 | 0.068 | 0.950 / 120/120 |

In-app median time per 1024 px tile:
- Samsung: resize 211 ms, network 132 ms.
- Realme: resize 145 ms, network 227 ms.

These are single passes, not a controlled benchmark.

## E. App v1.1 changes (`android_app/`, APK `deployment/apk/HER2_Mobile_Edge.apk`)

- Exact preprocessing (`PilResample.kt`), with resize time shown separately.
- INT8 labelled "speed test only", with a warning on every INT8 result.
- ASCO/CAP texts corrected, including HER2-low for 1+.
- Removed "big.LITTLE affinity", "Neural Engine" and "FPS".
- A disclaimer stating the external-domain failure.
- A dark-mode fix: radio labels were invisible on the Realme. This bug was already present in v1.0's screenshots.
- A research validation mode.
- Screenshots re-captured on both phones (`deployment/app_screenshots/`), using `capture_screenshots_v11.py`, which
  locates controls through the UI hierarchy.

**Packaging note.** Android 17 warns that TFLite 2.16.1's `libtensorflowlite_jni.so` is not 16 KB page-aligned. Both
phones use 4 KB pages. A LiteRT upgrade would need re-validation.

## Not measured

- Power in this experiment. Phone power was measured afterwards on battery in `../10_power/`; Pi power was not
  measured.
- Accuracy on the external domain on device. The model is not fit for that domain anyway.

## Mac latency (`outputs/mac_latency_run{1,2}.csv`)

Not reproducible under desktop load: the two runs differ by up to 2.4×. Only the 224 vs 448 ordering is robust.

## Files

- **Scripts:**
  - `mac_prepare.py`, `mac_latency.py`, `pil_resample_port.py`;
  - `android_benchmark.py`, `ondevice_validation.py`, `capture_screenshots_v11.py`;
  - `pi_bundle/` (`pi_precision_benchmark.py`, models, reference tiles);
  - `make_edge_figures.py`.
- **Index:** `ondevice_tiles_index.csv`.
- **Outputs:**
  - `outputs/android_<device>/` (benchmark.csv, raw logs, device_info.json);
  - `outputs/ondevice_<device>/`;
  - `outputs/pi4/`;
  - `outputs/edge_latency_table.csv`, `outputs/fig_edge_benchmark.png` (copied to
    `final_report_figures/fig7_smartphone_edge_benchmark.png`).
