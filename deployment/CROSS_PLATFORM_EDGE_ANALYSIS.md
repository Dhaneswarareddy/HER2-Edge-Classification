# Cross-Platform Edge Analysis & Discrepancy Audit: Raspberry Pi 4B vs. Realme 6 vs. Samsung Galaxy M35 5G

**Author:** FYP3 Engineering & Research Team  
**Date:** November 9, 2026  
**Scope:** Comparative microarchitectural, latency, thermal, multi-threading, and memory discrepancy audit between embedded single-board computing (Raspberry Pi 4B) and heterogeneous mobile smartphone computing (Samsung Galaxy M35 5G and Realme 6).  
**Authoritative Data Sources:**  
- `deployment/benchmark_results/benchmark_primary_pi4.csv`  
- `deployment/benchmark_results/benchmark_experimental_pi4.csv`  
- `deployment/benchmark_results/benchmark_samsung_m35.csv`  
- `deployment/benchmark_results/benchmark_realme_6.csv`  
- `deployment/MOBILE_BENCHMARK_REPORT.md` & `deployment/PI4_BENCHMARK_REPORT.md`

---

## 1. Executive Head-to-Head Comparison Table

| Metric / Dimension | Raspberry Pi 4 Model B Rev 1.4 | Realme 6 (`RMX2001`) | Samsung Galaxy M35 5G (`SM-M356B`) | Speedup / Discrepancy Analysis |
|---|:---:|:---:|:---:|:---:|
| **Silicon Manufacturing Process** | 28nm Planar CMOS | 12nm FinFET | 5nm FinFET EUV | **5.6× finer lithography node** |
| **CPU Architecture** | 4× Cortex-A72 @ 1.80 GHz (Uniform) | 2× A76 @ 2.05 GHz + 6× A55 @ 2.0 GHz | 4× A78 @ 2.4 GHz + 4× A55 @ 2.0 GHz | **Heterogeneous big.LITTLE Octa-Core** |
| **SIMD Vector Engine** | ARM NEON (32/64-bit FP) | ARM NEON with FP16 SIMD | ARM NEON with native FP16 (`armv8.2-a+fp16`) | **Hardware FP16 compute** |
| **Operating System** | Debian GNU/Linux 13.7 (Kernel 6.12) | Android 11 (API 30, Kernel 4.14) | Android 16 (API 36, Kernel 5.15) | Mobile Android runtime |
| **Primary FP16 Latency (1T)** | 756.6 ms (1.32 FPS) | 409.0 ms (2.44 FPS) | 214.2 ms (4.67 FPS) | **3.53× faster on M35 vs Pi** |
| **Primary FP16 Latency (2T)** | 444.1 ms (2.25 FPS) | 228.8 ms (4.37 FPS) | 117.5 ms (8.51 FPS) | **3.78× faster on M35 vs Pi** |
| **Primary FP16 Latency (4T)** | 353.5 ms (2.82 FPS) | 190.1 ms (5.26 FPS) | **78.9 ms (12.67 FPS)** | **4.48× faster on M35 vs Pi** |
| **Primary FP16 Latency (8T)** | N/A (4 physical cores) | **145.2 ms (6.89 FPS)** | 155.9 ms (6.41 FPS) | **Realme 8T scales; M35 8T stalls** |
| **Primary INT8 Latency (Best)**| 122.9 ms (4T, 8.1 FPS) | **38.6 ms (8T, 25.9 FPS)** | **21.7 ms (4T, 46.1 FPS)** | **5.66× faster on M35 vs Pi** |
| **Peak Operating Temp** | **84.2°C (Thermal Soft Trip)** | **30.6°C (Thermally Stable)** | **28.9°C (Thermally Stable)** | **55.3°C cooler on smartphones** |
| **Thermal Throttling Flag** | `throttled=0x80008` (Drops 1.68 GHz) | `throttled=0x0` (Sustained 2.05 GHz) | `throttled=0x0` (Sustained 2.40 GHz) | **Active throttling on Pi; zero on phones** |
| **Peak RAM Footprint (FP16)** | 191.6 MB | 124.5 MB | 119.0 MB | **37.9% memory reduction on phones** |
| **Peak RAM Footprint (INT8)** | N/A | **41.0 MB** | **39.4 MB** | **79.4% memory reduction vs Pi FP16** |
| **Prediction Concordance** | 100.0% (120/120 reference tiles) | 100.0% (Physical device tests) | 100.0% (Physical device tests) | **Zero numerical drift across platforms** |

---

## 2. In-Depth Analysis of Discrepancies & Anomalies

### Discrepancy 1: Multi-Threading Scaling Anomaly (4+4 vs 2+6 vs 4 Uniform)
- **The Observation**:
  - On the **Raspberry Pi 4B** (4 uniform Cortex-A72 cores), thread scaling is monotonically decreasing (756.6 ms $\to$ 444.1 ms $\to$ 353.5 ms).
  - On the **Samsung Galaxy M35 5G** (4× A78 + 4× A55), scaling from 1T to 4T achieves a near-linear **2.71× speedup** (214.2 ms $\to$ **78.9 ms**). However, allocating 8 threads causes a **catastrophic latency regression to 155.94 ms (+97.6% penalty)**.
  - On the **Realme 6** (2× A76 + 6× A55), allocating 2 threads produces a **1.79× speedup** (409.0 ms $\to$ **228.8 ms**). Unlike the M35, allocating 8 threads **continues to improve latency down to 145.2 ms (FP16) and 38.6 ms (INT8)**.
- **Microarchitectural Cause**:
  - The Samsung Exynos 1380 has four fast Cortex-A78 cores running at 2.40 GHz. Because neural network matrix multiplication enforces layer-by-layer barrier synchronization, allocating 8 threads forces the 4 fast cores to stall idle while waiting for the 4 slow 2.0 GHz in-order Cortex-A55 cores.
  - In contrast, the Realme 6 has **only two** Cortex-A76 cores and **six** Cortex-A55 cores. Here, the aggregate compute power of the 6 little cores together represents the majority of total multi-core throughput, overcoming the thread barrier synchronization penalty.
  - **Takeaway for Viva**: Multi-core edge thread allocation must be tuned strictly according to core topology: on 4+4 clusters, pin to 4 threads; on 2+6 clusters, 8 threads yields maximum burst throughput.

---

### Discrepancy 2: Thermal Throttling & Thermal Design Power (TDP)
- **The Observation**:
  - Continuous 4-thread execution on the Raspberry Pi 4B caused core temperature to rise from 75.4°C to **84.2°C**, triggering Linux kernel soft thermal throttling (`throttled=0x80008`) and reducing the CPU clock from 1800.4 MHz to 1677.4 MHz.
  - Continuous multi-threaded execution on both smartphones across >800 inferences per device produced virtually zero temperature rise:
    - Samsung M35: 28.8°C $\to$ 29.1°C ($\Delta T = 0.3^\circ\text{C}$).
    - Realme 6: 29.8°C $\to$ 31.4°C ($\Delta T = 1.6^\circ\text{C}$).
    - Zero thermal throttling (`0x0`), full clock frequencies sustained.
- **Microarchitectural Cause**:
  - **Lithography Node**: The Broadcom BCM2711 on the Pi 4B is manufactured on a legacy **28nm planar CMOS** process (high dynamic leakage, ~7.4W under full load).
  - The MediaTek Helio G90T is on a **12nm FinFET** process, and the Samsung Exynos 1380 is on a modern **5nm FinFET EUV** process. Transistor density is orders of magnitude higher, dramatically reducing dynamic switching energy per operation.
  - **Chassis Dissipation**: Smartphones incorporate internal multi-layer graphite dissipation sheets and vapor chambers designed to conduct heat into the exterior chassis passively, eliminating the need for fans.

---

### Discrepancy 3: Quantization Capabilities & Throughput (FP16 vs. INT8)
- **The Observation**:
  - On the Raspberry Pi 4B, INT8 achieved 122.9 ms (8.14 FPS).
  - On the Realme 6, INT8 reached **38.63 ms (25.89 FPS)**.
  - On the Samsung M35, INT8 reached **21.71 ms (46.06 FPS)**.
- **Microarchitectural Cause**:
  - Modern ARM Cortex cores incorporate dot-product instructions (`UDOT` / `SDOT` in ARMv8.4-A) performing four 8-bit integer multiply-accumulates per SIMD cycle.
  - Operating on 8-bit integers cuts memory bus bandwidth consumption by half relative to FP16 and by 75% relative to FP32, preventing DRAM memory bus saturation.
  - At 46 FPS, whole-slide tile streaming operates well above standard 30 FPS video refresh rates.

---

### Discrepancy 4: Memory Footprint & Runtime Overhead
- **The Observation**:
  - Pi 4B Peak RAM: **191.6 MB** (Primary 448px FP16).
  - Realme 6 Peak RAM: **124.5 MB** (FP16), **41.0 MB** (INT8).
  - Samsung M35 Peak RAM: **119.0 MB** (FP16), **39.4 MB** (INT8).
- **Microarchitectural Cause**:
  - The Pi 4B used a Python-based wrapper (`python3.13` + NumPy + LiteRT shared library). The Python interpreter, dynamic import table, and NumPy C-API buffers consume ~70 MB of baseline resident memory before any model is allocated.
  - The mobile deployment executed via native AArch64 C++ binaries and optimized Kotlin Android runtimes with direct ByteBuffer memory mapping (`mmap`). This eliminates Python runtime overhead, allocating memory strictly for the neural network tensor arena.

---

## 3. Summary of Discrepancies for Viva Defense

```
+-------------------------------------------------------------------------------------------------------------+
|                                    SUMMARY OF CROSS-PLATFORM DISCREPANCIES                                  |
|                                                                                                             |
| 1. Thread Scaling Discrepancy:                                                                              |
|    - Raspberry Pi 4B (Uniform 4× A72): Strict monotonic speedup up to 4T (353.5 ms).                        |
|    - Samsung M35 (4× A78 + 4× A55): 4T is optimal sweet spot (78.9 ms); 8T degrades (+97.6% sync stall).   |
|    - Realme 6 (2× A76 + 6× A55): 2T saturates big cores (228.8 ms); 8T provides max burst throughput (145.2ms).|
|                                                                                                             |
| 2. Thermal Discrepancy:                                                                                     |
|    - Pi 4B (28nm planar): Overheats to 84.2°C at 4T, triggering soft thermal throttling (0x80008).         |
|    - Smartphones (5nm / 12nm FinFET): Remain thermally flat (28.9°C / 30.6°C, 0x0 throttling).             |
|                                                                                                             |
| 3. Throughput & Quantization Discrepancy:                                                                   |
|    - Samsung M35 INT8 achieves 46.06 FPS (21.7 ms); Realme 6 INT8 achieves 25.89 FPS (38.6 ms).             |
|    - Pi 4B FP16 is limited to 2.82 FPS (353.5 ms). Modern mobile handsets are 4.5×–16.3× faster.           |
|                                                                                                             |
| 4. Memory Footprint Discrepancy:                                                                            |
|    - Pi 4B resident memory is bloated by Python runtime (~191.6 MB).                                       |
|    - Mobile INT8 runs natively at ~39.4 MB–41.0 MB (79.4% memory reduction).                               |
+-------------------------------------------------------------------------------------------------------------+
```
