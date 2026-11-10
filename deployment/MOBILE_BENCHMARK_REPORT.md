# Mobile Benchmark Report: Samsung Galaxy M35 5G and Realme 6

**Revised 10 November 2026.** All numbers below were re-measured on 10 Nov 2026 with one protocol
(`experiments_v2/08_edge/android_benchmark.py`; raw `benchmark_model` output for every configuration is in
`experiments_v2/08_edge/outputs/android_<device>/raw/`). The 31 Oct version of this report contained unsupported
claims; see "What changed" at the end and `experiments_v2/REPORT_CORRECTIONS_v2.md` items 7–11.

**Research software. Not clinically validated.**

## 1. Devices and protocol

| | Samsung Galaxy M35 5G | Realme 6 |
|---|---|---|
| Model / board | SM-M356B / s5e8835 | RMX2001 / RM6785 |
| SoC | Exynos 1380: 4× Cortex-A78 2.4 GHz + 4× Cortex-A55 2.0 GHz | Helio G90T: 2× Cortex-A76 2.05 GHz + 6× Cortex-A55 2.0 GHz |
| Android | 17 (API 37) | 11 (API 30) |
| Memory page size | 4 KB | 4 KB |

**Tool.** Official TFLite `benchmark_model` (aarch64), pushed to `/data/local/tmp`.
- Flags: `--use_xnnpack=true --warmup_runs=10 --num_runs=100 --report_peak_memory_footprint=true`.
- Threads: 1, 2, 4 and 8.
- A 60 s cool-down between configurations.
- No CPU affinity is set: the scheduler chooses the cores.

**Input.** Random input, so these figures are network latency only. Accuracy on the phones is measured separately
(§4).

**Temperature.**
- CPU thermal zones are not readable without root ("not readable" on both phones).
- The values below are **battery** temperatures, read with `dumpsys battery` before and after each configuration.
- Android exposes no throttle flag, so throttling cannot be reported.

**Power.** Not measured.

## 2. Latency (median ms; p95 in brackets) and peak memory

### Samsung Galaxy M35 5G (battery 29.7 → 31.7 °C over all 16 configurations)

| Model | 1 thread | 2 threads | 4 threads | 8 threads | Peak memory |
|---|---|---|---|---|---|
| **448 px FP16 (deployed)** | 214.9 [232.2] | 119.0 [124.4] | **81.4** [107.5] | 95.1 [729.4] | 120 MB |
| 448 px FP32 | 214.0 [220.0] | 127.6 [178.2] | 80.7 [154.1] | 85.0 [153.9] | 90 MB |
| 448 px INT8 (**rejected**, speed only) | 75.4 [78.7] | 38.7 [40.5] | 21.7 [35.1] | 45.3 [167.3] | 40 MB |
| 224 px FP16 (experimental) | 53.5 [53.7] | 28.6 [30.6] | 16.8 [18.0] | 20.3 [42.1] | 92 MB |

### Realme 6 (battery 29.6 → 32.0 °C over all 16 configurations)

| Model | 1 thread | 2 threads | 4 threads | 8 threads | Peak memory |
|---|---|---|---|---|---|
| **448 px FP16 (deployed)** | 301.7 [304.4] | 174.9 [181.7] | 142.6 [163.4] | **139.1** [181.8] | 124 MB |
| 448 px FP32 | 300.5 [303.3] | 173.6 [179.6] | 142.0 [162.1] | 143.2 [191.3] | 95 MB |
| 448 px INT8 (**rejected**, speed only) | 98.8 [99.4] | 53.6 [58.9] | 48.4 [49.5] | 34.3 [49.6] | 42 MB |
| 224 px FP16 (experimental) | 73.4 [74.3] | 40.1 [44.5] | 32.2 [40.0] | 26.3 [48.7] | 94 MB |

Data: `deployment/benchmark_results/benchmark_samsung_m35_2026-11-10.csv` and `benchmark_realme_6_2026-11-10.csv`.

## 3. Interpretation

- **FP16 runs at FP32 speed on CPU.** XNNPACK expands the FP16 weights to FP32, so FP16 halves the file
  (11.3 vs 22.4 MiB) without changing latency.
- **Best configuration for the deployed model:**
  - Samsung: 4 threads (81 ms). 8 threads is slower and much more variable (p95 729 ms).
  - Realme: 8 threads (139 ms), only marginally better than 4 threads (143 ms).
  - Without core pinning, the scheduler decides which cores run the threads. Micro-architectural explanations
    (synchronisation stalls on little cores) were not tested and are not claimed.
- **INT8 is 3.8× (Samsung) and 4.1× (Realme) faster, but it is not deployable.**
  - It fails the project's agreement gate: 0.928 < 0.95 on 727 validation tiles.
  - Test 1+ recall falls from 0.960 to 0.855.
- **The 224 px model is about 5× faster** than the 448 px model, but it is 0.05 less accurate on BCI test, and it is
  the model used for HARMONISE experiments only.

## 4. Accuracy on the phones (app v1.1, `experiments_v2/08_edge/ondevice_validation.py`)

The app's validation mode classifies pushed tiles through exactly the user-interface path:
- decode;
- `PilResample.kt` (Lanczos 512 → bilinear 448, bit-exact with PIL);
- TFLite FP16, 4 threads.

Tiles: the full BCI grouped test split (758 original 1024 px tiles) and the 120 BCI-validation reference tiles.

| | Test acc | Test bal. acc | Test QWK | Agreement with Keras (test) | Max \|Δp\| | 120 reference tiles: acc / agreement |
|---|---|---|---|---|---|---|
| Samsung M35 | 0.975 | 0.973 | 0.959 | **758/758** | 0.018 | 0.950 / 120/120 |
| Realme 6 | 0.975 | 0.973 | 0.959 | **758/758** | 0.068 | 0.950 / 120/120 |

**In-app time per 1024 px tile** (median, one pass per tile, not a controlled benchmark):

| Phone | Resize | Network | Total |
|---|---|---|---|
| Samsung | 211 ms | 132 ms | 348 ms |
| Realme | 145 ms | 227 ms | 373 ms |

## 5. What changed from the 31 Oct report

| 31 Oct claim | Now |
|---|---|
| "throttled = 0x0", "zero thermal throttling", "rock-solid 28.9 °C" | Android has no throttle flag, and the temperatures were battery readings. Reported as battery temperatures |
| INT8 "46.06 FPS real-time", "whole-slide pan-and-scan with zero lag" | INT8 failed the accuracy gate; no slide viewer exists. Speed only |
| "Pinned to the 4 Cortex-A78 cores"; "8-thread A55 barrier stall (+97.6%)" | No affinity was set. Explanations withdrawn |
| "Per-watt efficiency" | No power measured. Removed |
| Samsung 224 px FP16, 1 thread: 176.8 ms (then 30.7 ms at 2 threads) | 53.5 ms → 28.6 ms. The old 1-thread value was a scheduling artefact |
| Realme 448 px FP16: 409.0 / 228.8 / 190.1 / 145.2 ms | 301.7 / 174.9 / 142.6 / 139.1 ms. Conditions for the old run are not documented |
| Samsung 448 px FP16, 4 threads: 78.9 ms; INT8 21.7 ms | 81.4 ms; 21.7 ms. Consistent |
| App "computes the same prediction as the validated model" | Not true for v1.0 (one-step 1024 → 448 resize). True for v1.1: 758/758 on both phones |
| "Android 16" (Samsung) | The phone now reports Android 17 after an OS update |

The 31 Oct CSVs are kept in `deployment/benchmark_results/benchmark_samsung_m35.csv` and `benchmark_realme_6.csv`.
