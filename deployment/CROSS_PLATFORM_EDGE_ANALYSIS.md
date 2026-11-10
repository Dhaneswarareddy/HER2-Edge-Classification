# Cross-Platform Edge Analysis: Samsung Galaxy M35 5G, Realme 6, Raspberry Pi 4B

**Revised 10 November 2026.** This replaces the 31 Oct version. That version compared devices using an unmeasured Pi
INT8 latency (122.9 ms), phone "throttle flags" that Android does not have, power/energy claims with no measurement,
and INT8 figures from a model that failed its accuracy gate (`experiments_v2/REPORT_CORRECTIONS_v2.md` items 6–10).

All values below were measured on 10 Nov 2026 with one protocol (`experiments_v2/08_edge/`):
- 10 warm-up runs and 100 timed invokes per configuration;
- network latency only;
- no core pinning.

## Deployed model (EfficientNetV2B0, 448 px, FP16): median network latency per tile

| Device | 1 thread | 2 threads | 4 threads | 8 threads | Fastest | Temperature during run | Throttling |
|---|---|---|---|---|---|---|---|
| Samsung Galaxy M35 5G (Exynos 1380) | 214.9 ms | 119.0 ms | **81.4 ms** | 95.1 ms | 81.4 ms (4T) | battery 29.7–31.7 °C | not observable (Android) |
| Realme 6 (Helio G90T) | 301.7 ms | 174.9 ms | 142.6 ms | **139.1 ms** | 139.1 ms (8T) | battery 29.6–32.0 °C | not observable (Android) |
| Raspberry Pi 4B (BCM2711) | 761.8 ms | 437.0 ms | **336.5 ms** | — | 336.5 ms (4T) | SoC ≤ 62.8 °C | `0x0` throughout |

**Speed relative to the Pi** (fastest configuration of each device):

| Device | Fastest | vs Pi |
|---|---|---|
| Samsung | 81 ms | 4.1× faster |
| Realme | 139 ms | 2.4× faster |
| Pi | 337 ms | — |

## Other precisions (fastest configuration, median)

| Device | FP32 | INT8 (rejected: speed only) | 224 px FP16 |
|---|---|---|---|
| Samsung M35 | 80.7 ms | 21.7 ms | 16.8 ms |
| Realme 6 | 142.0 ms | 34.3 ms | 26.3 ms |
| Raspberry Pi 4B | 334.5 ms | 104.1 ms | (31 Oct run: 81.2 ms) |

**INT8 must not be used for scoring.**
- Mac: agreement with Keras 0.928 on 727 validation tiles, and test 1+ recall 0.855 vs 0.960.
- Pi: agreement 0.908 on 120 tiles, and only 114 of 120 agree with the Mac INT8 output. INT8 results are
  platform-dependent.

## Accuracy delivered on the device

| | BCI test accuracy (758 tiles) | Agreement with Keras |
|---|---|---|
| Samsung M35, app v1.1 | 0.975 | 758/758 |
| Realme 6, app v1.1 | 0.975 | 758/758 |
| Raspberry Pi 4B, FP16 (120 reference tiles) | 0.950 (reference tiles) | 120/120 |

## Practical conclusions

1. **Phones are fast enough for interactive single-tile scoring.**
   - Network time is 81–139 ms per tile.
   - In the app, the exact resize adds 145–211 ms per 1024 px tile.
   - The Pi is 2.4–4.1× slower, but usable for batch work.
2. **There is no measured basis for thermal or energy advantages.**
   - Phone temperatures are battery temperatures.
   - The Pi stayed below 63 °C without throttling once powered by an official-rating supply.
   - Power was not measured on any device.
3. **FP16 is the right deployment format:** lossless, and half the file size. INT8 is not, until it passes the
   accuracy gate.
4. **Before clinical-style use**, any device needs:
   - a focus and brightness quality check (the 448 px model is defocus-sensitive: `experiments_v2/04_robustness`);
   - validation on the target laboratory's staining. The model does not transfer to HER2-IHC-40x.

Data: `deployment/benchmark_results/edge_latency_table_2026-11-10.csv`; figure:
`final_report_figures/fig7_smartphone_edge_benchmark.png`.
