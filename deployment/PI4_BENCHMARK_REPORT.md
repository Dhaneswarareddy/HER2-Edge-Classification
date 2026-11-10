# Raspberry Pi 4B Deployment Benchmark Report: FYP3 Edge Inference

**Date**: October 31, 2026  
**Hardware Platform**: Raspberry Pi 4 Model B Rev 1.4  
**Target IP / Host**: `dhani@192.168.50.1` (hostname `Dhani`)  
**MAC Address**: `e4:5f:01:42:e4:f1` (Raspberry Pi Trading Ltd)  
**Execution Agent**: Antigravity Autonomous Workflow  
**Bundle Deployed**: `deployment/her2_fyp3_pi_bundle.tar.gz` (93.8 MB)  
**Runtime**: Python 3.13.5 (Debian GNU/Linux 13 trixie, aarch64), `ai_edge_litert` 2.3.0  
**Governor**: `performance` (1.80 GHz maximum clock)

---

## 0. Re-measurement, 10 November 2026 (supersedes the latency/thermal results below)

`experiments_v2/08_edge/pi_bundle/pi_precision_benchmark.py`; raw logs in `experiments_v2/08_edge/outputs/pi4/`;
CSV `deployment/benchmark_results/benchmark_pi4_precisions_2026-11-10.csv`.

**Setup.**
- Same Pi: Raspberry Pi 4 Model B Rev 1.4, Debian 13, `ai_edge_litert` 2.5.4.
- Powered by an official-rating supply. Before the supply change the Pi reported `throttled=0x50000` (under-voltage
  and throttling had occurred since boot); after the change and a reboot it reported `0x0`.
- Governor `performance` for the run, then restored to `ondemand`.

**Protocol.**
- A fresh process per configuration.
- 10 warm-up and 100 timed invokes on the 120 preprocessed BCI-validation reference tiles.
- `vcgencmd` temperature, clock and throttle read before and after.
- 60 s cool-down between configurations.

| Precision | 1 thread | 2 threads | 4 threads | Peak RSS (process) | Agreement with Mac, same precision | Agreement with Keras | Accuracy, 120 tiles |
|---|---|---|---|---|---|---|---|
| FP32 | 755.0 ms | 435.7 ms | 334.5 ms | 202 MB | 120/120 | 120/120 | 0.950 |
| **FP16 (deployed)** | 761.8 ms | 437.0 ms | **336.5 ms** | 250 MB | 120/120 | 120/120 | 0.950 |
| INT8 (rejected) | 361.3 ms | 187.9 ms | 104.1 ms | 151 MB | **114/120** | **109/120** | **0.892** |

**Observations.**
- SoC temperature stayed at or below 62.8 °C, ARM clock at 1.8 GHz, and `throttled=0x0` before and after every
  configuration.
- The 31 Oct 4-thread result (353.5 ms at 84.2 °C, `0x80008`) was thermally limited. With a proper supply, 4 threads
  is the fastest setting and does not throttle.
- **Pi INT8 had never been measured before.** The final report's former "122.9 ms / 8.14 FPS" had no measurement
  behind it. The measured value is 104.1 ms at 4 threads.
- INT8 is not deployable. It agrees with Keras on 109 of 120 tiles, and its predictions differ from the Mac INT8
  predictions on 6 of 120 tiles: INT8 results are platform-dependent.
- No power was measured.

> The sections below are the **31 October 2026 run**, kept as recorded. Its thread recommendation ("2 threads to
> avoid the thermal trip") is superseded by §0. Its Keras-agreement note for the experimental 224 px model (98.3%,
> 2 tiles flipped) is correct and is confirmed by `benchmark_experimental_pi4.csv`.

## 1. Hardware and Software Specifications

| Specification | Details |
|---|---|
| **Board Model** | Raspberry Pi 4 Model B Rev 1.4 |
| **SoC / CPU** | Broadcom BCM2711, Quad-core ARM Cortex-A72 (ARMv8 64-bit) |
| **CPU Architecture** | `aarch64` (64-bit Little Endian, flags: `fp asimd evtstrm crc32 cpuid`) |
| **CPU Frequency** | 600 MHz min, 1800 MHz max (frequency governor: `performance`) |
| **Total System RAM** | 3.7 GiB LPDDR4-3200 (4 GB model) |
| **Available RAM (idle)** | ~2.8 GiB available |
| **Storage** | 15 GB MicroSD, 8.6 GB used, 4.9 GB available (64% used) |
| **Operating System** | Debian GNU/Linux 13.7 (trixie) |
| **Linux Kernel** | `Linux Dhani 6.18.50+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.18.50-1+rpt1 aarch64` |
| **Python Environment** | Python 3.13.5 in virtual environment `~/her2env` |
| **Inference Engine** | `ai_edge_litert` 2.3.0 (XNNPACK CPU delegate enabled) |
| **Dependencies** | `numpy` 2.5.4, `pillow` 13.0.0, `psutil` 7.2.2, `flatbuffers` 25.12.19 |

---

## 2. Installation Procedure and Outcome

1. **Bundle Transfer**: The 93.8 MB self-contained deployment archive `her2_fyp3_pi_bundle.tar.gz` was transferred from the Mac host to `dhani@192.168.50.1:~` via SCP in 8.5 seconds over local ethernet.
2. **Virtual Environment Setup**: `python3 -m venv ~/her2env` was created. Packages `ai_edge_litert` (2.3.0), `numpy` (2.5.4), `pillow` (13.0.0), and `psutil` (7.2.2) were installed cleanly from binary wheels.
3. **Bundle Integrity Verification (Gate P1)**:
   ```bash
   python3 verify_bundle.py
   ```
   - Both model hashes matched pinned configurations.
   - Initial reference tile probability difference vs Mac reference:
     - Primary: $\max |\Delta p| = 1.46 \times 10^{-10}$
     - Experimental: $\max |\Delta p| = 8.15 \times 10^{-10}$
   - **Result**: `BUNDLE OK` (Gate P1 **PASSED**).
4. **Governor Configuration**: Scaling governor on all 4 CPU cores set to `performance` via `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.

---

## 3. Benchmark Results: Primary Model (In-Domain BCI)

- **Model**: `EfficientNetV2B0__ce__hires` (448 px input, FP16 TFLite, 11.27 MB, **no stain calibration**).
- **Workload**: 10 warm-up runs, 100 timed invokes (network only), 8 raw 1024px tiles $\times$ 3 repeats (end-to-end), and 120 reference validation tiles.

| Threads | Model Load | Warm-up Mean | Mean Latency | Median Latency | P95 Latency | P99 Latency | Throughput (FPS) | Preprocess Mean | End-to-End Mean | Peak RSS | CPU % | Temp End | Throttled | Mac TFLite Agreement | Keras Agreement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.112 s | 768.5 ms | 757.7 ms | **756.6 ms** | 760.1 ms | 762.3 ms | **1.32 FPS** | 105.3 ms | 861.2 ms | 191.6 MB | 100.0% | 69.1°C | `0x0` | **100.0%** (1.0) | **100.0%** (1.0) |
| **2** | 0.115 s | 454.5 ms | 444.5 ms | **444.1 ms** | 449.0 ms | 451.7 ms | **2.25 FPS** | 106.8 ms | 552.7 ms | 191.5 MB | 198.7% | 77.9°C | `0x0` | **100.0%** (1.0) | **100.0%** (1.0) |
| **4** | 0.115 s | 351.3 ms | 353.9 ms | **353.5 ms** | 366.2 ms | 370.4 ms | **2.82 FPS** | 108.3 ms | 457.9 ms | 191.3 MB | 375.6% | 84.2°C | `0x80008`* | **100.0%** (1.0) | **100.0%** (1.0) |

*\*Note on 4-thread throttling*: Sustained 100% load across all 4 cores without active fan cooling reached 84.2°C, engaging soft thermal throttling (`0x80008`) and reducing clock frequency from 1800 MHz to 1677 MHz.

---

## 4. Benchmark Results: Experimental Model (Cross-Dataset HER2-IHC-40x)

- **Model**: `EfficientNetV2B0__ce` (224 px input, FP16 TFLite, 11.27 MB, **frozen `stain_p99`** $g_h=0.2545, g_d=0.5045$).
- **Workload**: 10 warm-up runs, 100 timed invokes (network only), 8 raw 1024px tiles $\times$ 3 repeats (end-to-end), and 120 reference validation tiles.

| Threads | Model Load | Warm-up Mean | Mean Latency | Median Latency | P95 Latency | P99 Latency | Throughput (FPS) | Preprocess Mean | End-to-End Mean | Peak RSS | CPU % | Temp End | Throttled | Mac TFLite Agreement | Keras Agreement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.092 s | 192.3 ms | 183.3 ms | **182.9 ms** | 185.4 ms | 187.1 ms | **4.91 FPS** | 118.6 ms | 301.9 ms | 167.7 MB | 100.0% | 73.0°C | `0x80000` | **100.0%** (1.0) | 98.3% (0.983)* |
| **2** | 0.092 s | 117.0 ms | 107.9 ms | **107.6 ms** | 110.4 ms | 112.2 ms | **7.80 FPS** | 120.2 ms | 229.8 ms | 167.4 MB | 181.7% | 75.9°C | `0x80000` | **100.0%** (1.0) | 98.3% (0.983)* |
| **4** | 0.093 s | 90.5 ms | 81.8 ms | **81.2 ms** | 87.4 ms | 89.9 ms | **9.81 FPS** | 118.9 ms | 199.5 ms | 167.3 MB | 316.6% | 83.2°C | `0x80008` | **100.0%** (1.0) | 98.3% (0.983)* |

*\*Note on Keras Agreement*: Prediction agreement with the Mac TFLite runtime is exactly 100.0% (120/120 tiles). Exactly two borderline tiles had near-tie probabilities ($|\Delta p| < 0.007$) that flipped under FP16 rounding vs original Keras FP32, exactly matching the behavior observed on Mac.

---

## 5. Latency, Scaling and Throughput Analysis

### 5.1 Multi-Thread Scaling
- **Primary Model (448 px)**:
  - 1 Thread $\to$ 2 Threads: **1.70× speedup** (756.6 ms $\to$ 444.1 ms).
  - 1 Thread $\to$ 4 Threads: **2.14× speedup** (756.6 ms $\to$ 353.5 ms).
- **Experimental Model (224 px)**:
  - 1 Thread $\to$ 2 Threads: **1.70× speedup** (182.9 ms $\to$ 107.6 ms).
  - 1 Thread $\to$ 4 Threads: **2.25× speedup** (182.9 ms $\to$ 81.2 ms).

### 5.2 Preprocessing vs Inference Overhead
- **Primary Mode**: Preprocessing (Lanczos downscale from 1024 to 512, bilinear resize to 448) averages **106.8 ms** using pure NumPy and Pillow. At 2 threads, preprocessing represents 19.3% of the total 552.7 ms pipeline.
- **Experimental Mode**: Preprocessing (Lanczos downscale to 256, bilinear resize to 224, followed by Ruifrok & Johnston stain matrix deconvolution and exponential optical density recombination) averages **119.2 ms**. At 2 threads, preprocessing takes slightly longer than the 107.6 ms neural network inference itself.

---

## 6. Resource Utilization: RAM, CPU & Thermal Observations

1. **Memory Efficiency**:
   - Primary model peak resident set size (RSS): **191.6 MB** (under 5.2% of the Pi's 3.7 GiB RAM).
   - Experimental model peak RSS: **167.4 MB** (under 4.5% of total RAM).
   - Both models easily fit within memory with over 2.5 GiB of headroom remaining.
2. **CPU Utilization**:
   - 1 Thread: Pegs exactly one Cortex-A72 core at 100.0%.
   - 2 Threads: Utilizes 198.7% CPU (near-perfect dual-core saturation).
   - 4 Threads: Utilizes 375.6% CPU (all 4 cores utilized).
3. **Thermal Throttling Dynamics**:
   - At 1 and 2 threads, the Pi operates cleanly below 78°C with zero throttling (`0x0`).
   - At 4 threads, sustained full-load execution generates enough heat on passive cooling to reach 84.2°C, causing the kernel to trigger thermal frequency throttling (`0x80008`), capping CPU frequency to 1677 MHz.

---

## 7. Recommendation for Best Thread Configuration

### Recommended Configuration: 2 Threads
- **Rationale**:
  - Delivers **444.1 ms** (primary) and **107.6 ms** (experimental) latency.
  - Achieves **79.5% of peak 4-thread throughput** while consuming half the CPU cores and generating significantly less heat (77.9°C max vs 84.2°C).
  - **Zero thermal throttling** (`throttled=0x0`) at sustained 1.80 GHz frequency, guaranteeing deterministic inference latency even in passively cooled enclosures.
  - Leaves remaining CPU cores available for tile acquisition, network transfer, or system tasks.
- *If active fan cooling is installed*: **4 Threads** is viable, achieving **353.5 ms** (2.82 FPS) on primary and **81.2 ms** (9.81 FPS) on experimental.

---

## 8. Summary of Limitations & Disclosures

1. **Not Clinically Validated**: This deployment bundle is research software. It must never be used for medical diagnostic or treatment decisions.
2. **Per-Tile Analysis**: Genuine whole-slide identifiers do not exist in the public BCI or HER2-IHC-40x datasets. All metrics reflect individual tile classifications without slide-level aggregation.
3. **Borderline Equivocal Sensitivity**: As documented in HARMONISE-2, borderline HER2 scores (1+ and 2+) remain challenging across domains. The experimental calibrated model achieves 0.18 recall on 1+ and 0.57 recall on 2+, with 0.09 precision on 2+. Manual pathologist review is mandatory.
4. **Data Privacy**: Stored reference validation tiles originate from BCI and HER2-IHC-40x research cohorts and must remain strictly private.

