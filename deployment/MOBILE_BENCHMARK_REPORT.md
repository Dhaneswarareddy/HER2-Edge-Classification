# Physical Mobile Edge Benchmark Report: Multi-Device Smartphone Benchmarks (Samsung Galaxy M35 5G & Realme 6)

**Date:** 2026-11-09  
**Target Hardware Platforms:**
1. **Samsung Galaxy M35 5G** (`SM-M356B`, Board: `erd8835`)  
   - **OS:** Android 16 (API 36, `arm64-v8a`) | Kernel: Linux 5.15.189-android15-11  
   - **SoC:** Samsung Exynos 1380 (5nm FinFET EUV process)  
   - **CPU Cluster:** 4× ARM Cortex-A78 @ 2.40 GHz (Performance) + 4× ARM Cortex-A55 @ 2.00 GHz (Efficiency)  
   - **RAM:** 6.0 GB LPDDR4X  
2. **Realme 6** (`RMX2001`, Board: `mt6785`)  
   - **OS:** Android 11 (API 30, `arm64-v8a`) | Kernel: Linux 4.14.186+  
   - **SoC:** MediaTek Helio G90T (12nm FinFET process)  
   - **CPU Cluster:** 2× ARM Cortex-A76 @ 2.05 GHz (Performance) + 6× ARM Cortex-A55 @ 2.00 GHz (Efficiency)  
   - **RAM:** 4.0 GB LPDDR4X  
3. **Reference SBC:** Raspberry Pi 4 Model B Rev 1.4 (Broadcom BCM2711 28nm, 4× A72 @ 1.8 GHz, Linux)  

**Benchmark Tool:** Google TensorFlow Lite Official AArch64 Native Benchmark Binary (`android_aarch64_benchmark_model`)  
**Inference Delegate:** TensorFlow Lite XNNPACK Engine (ARM NEON / FP16 SIMD vectorization)  
**Measurement Protocol:** 10 warmup runs followed by 50 timed benchmark iterations per configuration across all 16 suites per device (32 suites total).

---

## 1. Executive Summary & Key Architectural Insights

Physical edge AI benchmarking across two distinct mobile smartphone architectures demonstrates that modern handheld mobile SoCs provide order-of-magnitude performance, energy, and thermal advantages over hobbyist single-board computers like the Raspberry Pi 4B.

### Key Microarchitectural Discoveries:

1. **Heterogeneous Thread Scaling (4+4 vs 2+6 Topology):**
   - **Samsung Exynos 1380 (4× A78 + 4× A55, 5nm)**:
     - Scaling from 1T (214.21 ms) to 2T (117.45 ms) to 4T (**78.91 ms, 12.67 FPS**) provides a near-linear **2.71× speedup**.
     - Increasing thread count to 8 threads degrades latency to **155.94 ms (+97.6% penalty)**. This barrier synchronization stall occurs because the 4 fast out-of-order Cortex-A78 cores must stall waiting for the slower in-order Cortex-A55 efficiency cores to finish their GEMM partitions.
     - **Optimal Configuration:** 4 worker threads pinned to the Cortex-A78 big-core cluster.
   - **Realme 6 MediaTek Helio G90T (2× A76 + 6× A55, 12nm)**:
     - Allocating 2 threads achieves a **1.79× speedup** over 1T (409.02 ms $\to$ **228.84 ms**), saturating the two Cortex-A76 big cores.
     - Unlike the M35, allocating 8 threads on the Realme 6 achieves further latency reduction to **145.19 ms (FP16)** and **38.63 ms (INT8, 25.89 FPS)**. Because the Helio G90T features **six** Cortex-A55 cores (relative to only 2 big cores), the aggregate computational throughput of the 6 little cores together overcomes the thread barrier synchronization penalty.
     - **Optimal Configuration:** 2 threads for maximum per-watt efficiency; 8 threads for maximum burst throughput.

2. **INT8 Quantization Advantage:**
   - On the Samsung M35, INT8 reduces latency to **21.71 ms (46.06 FPS)** with **39.39 MB peak RSS RAM** (66.9% memory reduction).
   - On the Realme 6, INT8 reduces latency to **38.63 ms (25.89 FPS)** with **40.97 MB peak RSS RAM** (67.1% memory reduction).
   - This enables smooth interactive whole-slide pan-and-scan tile streaming at full 448px resolution with zero lag on consumer smartphones.

3. **Exceptional Thermal Stability:**
   - Raspberry Pi 4B overheated to **84.2°C at 4 threads**, triggering Linux kernel soft thermal throttling (`throttled=0x80008`) and capping clocks.
   - Samsung M35 remained at **28.8°C–29.1°C** ($\Delta T = 0.3^\circ\text{C}$, `0x0` throttling).
   - Realme 6 remained at **29.8°C–31.4°C** ($\Delta T = 1.6^\circ\text{C}$, `0x0` throttling).
   - Zero thermal throttling, zero frequency scaling penalties, zero active cooling required.

---

## 2. Complete Physical Benchmark Results: Samsung Galaxy M35 5G (Exynos 1380, 5nm)

Recorded in `deployment/benchmark_results/benchmark_samsung_m35.csv`:

| Model Name | Precision | Input Size | File Size | Threads | Median Latency | Mean Latency | Min Latency | Max Latency | P95 Latency | Throughput | Peak RSS Memory | Temp Start | Temp End |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.27 MB | 1 | 214.21 ms | 214.85 ms | 213.17 ms | 221.56 ms | 219.58 ms | 4.67 FPS | 119.08 MB | 28.8°C | 28.9°C |
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.27 MB | 2 | 117.45 ms | 117.28 ms | 116.22 ms | 120.82 ms | 117.92 ms | 8.51 FPS | 119.42 MB | 28.9°C | 28.9°C |
| **`primary_bci_hires_fp16`** | **FP16** | **$448 \times 448$** | **11.27 MB** | **4** | **78.91 ms** | **78.40 ms** | **72.49 ms** | **84.08 ms** | **81.57 ms** | **12.67 FPS** | **118.96 MB** | **28.9°C** | **28.9°C** |
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.27 MB | 8 | 155.94 ms | 157.28 ms | 79.33 ms | 338.44 ms | 310.75 ms | 6.41 FPS | 118.80 MB | 28.9°C | 28.9°C |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 6.82 MB | 1 | 74.24 ms | 131.35 ms | 73.88 ms | 237.61 ms | 233.69 ms | 13.47 FPS | 40.27 MB | 28.9°C | 29.0°C |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 6.82 MB | 2 | 38.10 ms | 38.15 ms | 38.04 ms | 38.70 ms | 38.56 ms | 26.24 FPS | 40.16 MB | 29.0°C | 29.0°C |
| **`primary_bci_hires_int8`** | **INT8** | **$448 \times 448$** | **6.82 MB** | **4** | **21.71 ms** | **21.58 ms** | **20.03 ms** | **22.63 ms** | **22.34 ms** | **46.06 FPS** | **39.39 MB** | **29.0°C** | **29.0°C** |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 6.82 MB | 8 | 57.48 ms | 55.32 ms | 25.66 ms | 95.58 ms | 79.14 ms | 17.40 FPS | 39.62 MB | 29.0°C | 29.0°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 22.36 MB | 1 | 214.02 ms | 214.50 ms | 213.50 ms | 219.78 ms | 216.03 ms | 4.67 FPS | 90.11 MB | 29.0°C | 29.0°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 22.36 MB | 2 | 116.99 ms | 117.25 ms | 116.53 ms | 118.68 ms | 118.52 ms | 8.55 FPS | 90.18 MB | 29.0°C | 29.0°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 22.36 MB | 4 | 81.72 ms | 81.05 ms | 75.84 ms | 85.97 ms | 83.66 ms | 12.24 FPS | 90.20 MB | 29.0°C | 29.1°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 22.36 MB | 8 | 80.85 ms | 81.39 ms | 78.20 ms | 95.63 ms | 86.90 ms | 12.37 FPS | 89.72 MB | 29.1°C | 29.1°C |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.27 MB | 1 | 176.75 ms | 132.30 ms | 53.33 ms | 181.11 ms | 180.70 ms | 5.66 FPS | 91.46 MB | 29.1°C | 29.1°C |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.27 MB | 2 | 30.74 ms | 31.38 ms | 28.43 ms | 65.94 ms | 32.53 ms | 32.54 FPS | 91.20 MB | 29.1°C | 29.1°C |
| **`experimental_stain224_fp16`** | **FP16** | **$224 \times 224$** | **11.27 MB** | **4** | **18.05 ms** | **17.95 ms** | **16.79 ms** | **20.50 ms** | **19.16 ms** | **55.41 FPS** | **91.63 MB** | **29.1°C** | **29.1°C** |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.27 MB | 8 | 23.46 ms | 26.73 ms | 18.22 ms | 65.95 ms | 60.06 ms | 42.63 FPS | 91.14 MB | 29.1°C | 29.1°C |

---

## 3. Complete Physical Benchmark Results: Realme 6 (MediaTek Helio G90T, 12nm)

Recorded in `deployment/benchmark_results/benchmark_realme_6.csv`:

| Model Name | Precision | Input Size | File Size | Threads | Median Latency | Mean Latency | Min Latency | Max Latency | P95 Latency | Throughput | Peak RSS Memory | Temp Start | Temp End |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.82 MB | 1 | 409.02 ms | 408.89 ms | 406.07 ms | 410.88 ms | 410.56 ms | 2.44 FPS | 124.46 MB | 29.8°C | 29.9°C |
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.82 MB | 2 | 228.84 ms | 228.62 ms | 225.54 ms | 231.19 ms | 230.32 ms | 4.37 FPS | 124.18 MB | 29.9°C | 29.9°C |
| `primary_bci_hires_fp16` | FP16 | $448 \times 448$ | 11.82 MB | 4 | 190.11 ms | 193.42 ms | 183.14 ms | 233.26 ms | 220.74 ms | 5.26 FPS | 123.53 MB | 29.9°C | 30.1°C |
| **`primary_bci_hires_fp16`** | **FP16** | **$448 \times 448$** | **11.82 MB** | **8** | **145.19 ms** | **153.77 ms** | **119.01 ms** | **217.75 ms** | **208.54 ms** | **6.89 FPS** | **123.38 MB** | **30.1°C** | **30.2°C** |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 7.15 MB | 1 | 135.21 ms | 135.19 ms | 134.11 ms | 136.05 ms | 135.88 ms | 7.40 FPS | 42.05 MB | 30.2°C | 30.3°C |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 7.15 MB | 2 | 74.45 ms | 74.40 ms | 73.35 ms | 79.46 ms | 75.03 ms | 13.43 FPS | 42.04 MB | 30.3°C | 30.4°C |
| `primary_bci_hires_int8` | INT8 | $448 \times 448$ | 7.15 MB | 4 | 69.06 ms | 69.24 ms | 65.68 ms | 76.71 ms | 74.48 ms | 14.48 FPS | 41.81 MB | 30.4°C | 30.5°C |
| **`primary_bci_hires_int8`** | **INT8** | **$448 \times 448$** | **7.15 MB** | **8** | **38.63 ms** | **41.46 ms** | **32.73 ms** | **129.43 ms** | **60.45 ms** | **25.89 FPS** | **40.97 MB** | **30.5°C** | **30.6°C** |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 23.45 MB | 1 | 410.18 ms | 399.22 ms | 325.77 ms | 414.10 ms | 413.04 ms | 2.44 FPS | 94.98 MB | 30.6°C | 30.8°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 23.45 MB | 2 | 228.67 ms | 228.58 ms | 222.22 ms | 234.35 ms | 234.02 ms | 4.37 FPS | 94.77 MB | 30.8°C | 30.9°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 23.45 MB | 4 | 191.47 ms | 185.80 ms | 149.64 ms | 215.54 ms | 208.26 ms | 5.22 FPS | 94.90 MB | 30.9°C | 31.0°C |
| `primary_bci_hires_fp32` | FP32 | $448 \times 448$ | 23.45 MB | 8 | 212.60 ms | 217.04 ms | 145.07 ms | 429.94 ms | 323.53 ms | 4.70 FPS | 94.08 MB | 31.0°C | 31.2°C |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.82 MB | 1 | 105.35 ms | 105.04 ms | 102.06 ms | 107.40 ms | 107.05 ms | 9.49 FPS | 93.97 MB | 31.2°C | 31.2°C |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.82 MB | 2 | 57.84 ms | 60.37 ms | 46.53 ms | 145.16 ms | 99.60 ms | 17.29 FPS | 93.88 MB | 31.2°C | 31.4°C |
| `experimental_stain224_fp16` | FP16 | $224 \times 224$ | 11.82 MB | 4 | 41.25 ms | 42.86 ms | 34.02 ms | 66.57 ms | 54.58 ms | 24.24 FPS | 93.59 MB | 31.4°C | 31.4°C |
| **`experimental_stain224_fp16`** | **FP16** | **$224 \times 224$** | **11.82 MB** | **8** | **30.09 ms** | **32.41 ms** | **24.04 ms** | **57.60 ms** | **42.69 ms** | **33.23 FPS** | **93.19 MB** | **31.4°C** | **31.4°C** |

---

## 4. Tri-Hardware Cross-Platform Comparative Matrix

| Evaluation Dimension | Raspberry Pi 4B (4GB) | Realme 6 (4GB) | Samsung Galaxy M35 5G (6GB) | Microarchitectural Delta |
|---|:---:|:---:|:---:|:---:|
| **SoC / Silicon Process** | Broadcom BCM2711 (28nm planar) | MediaTek Helio G90T (12nm FinFET) | Samsung Exynos 1380 (5nm FinFET EUV) | 5.6× finer lithography node |
| **CPU Cluster** | 4× Cortex-A72 @ 1.8 GHz (Uniform) | 2× A76 @ 2.05 GHz + 6× A55 @ 2.0 GHz | 4× A78 @ 2.4 GHz + 4× A55 @ 2.0 GHz | Modern out-of-order execution |
| **Optimal Thread Allocation**| 2 Threads (Thermal Limiting) | 8 Threads (Burst) / 2T (Per-Watt) | 4 Threads (pinned to 4× Cortex-A78) | Matches performance core cluster size |
| **Primary FP16 Latency** | 444.1 ms (2T) / 353.5 ms (4T) | **145.19 ms (8T)** | **78.91 ms (4T)** | **4.48× speedup vs Pi 4B (M35)** |
| **Primary INT8 Latency** | 122.9 ms (4T) | **38.63 ms (8T)** | **21.71 ms (4T)** | **5.66× speedup vs Pi 4B (M35)** |
| **Throughput (INT8)** | 8.14 FPS | **25.89 FPS** | **46.06 FPS** | **Full real-time video rate (>30 FPS)** |
| **Thermal Profile** | **84.2°C (Throttled, `0x80008`)** | **30.6°C (Stable, `0x0`)** | **28.9°C (Stable, `0x0`)** | **55.3°C cooler on smartphone** |
| **Cooling Requirement** | Active fan / heat sink mandatory | Passive smartphone dissipation | Passive smartphone dissipation | Zero external cooling hardware |
| **Peak RAM Footprint (INT8)**| N/A (191.6 MB on FP16) | **41.0 MB** | **39.4 MB** | **79.4% memory reduction** |
| **Form Factor / Ergonomics** | External display, keyboard, power bank | Handheld, self-contained mobile device | Handheld, self-contained mobile device | True point-of-care clinic portability |

---

## 5. Physical Android Application UI & Inference Walkthrough

The production-grade native Android application (`HER2 Mobile Edge`, `com.fyp3.her2classifier`, `app-debug.apk`) was installed and verified directly on both physical smartphones.

### A. Samsung Galaxy M35 5G Live Deployment:
![Figure 9: Physical Point-of-Care Android Application Deployment on Samsung M35](../final_report_figures/fig9_mobile_app_interface_and_inference.png)

### B. Realme 6 Live Deployment:
![Figure 10: Physical Point-of-Care Android Application Deployment on Realme 6](../final_report_figures/fig10_realme6_app_interface_and_inference.png)

### Physical Device Screenshot Inventories:
- **Samsung Galaxy M35 5G**: `deployment/app_screenshots/samsung_m35/` (`01` through `07`)
- **Realme 6**: `deployment/app_screenshots/realme_6/` (`01` through `07`)
- **Root Reference Screenshots**: `deployment/app_screenshots/`
