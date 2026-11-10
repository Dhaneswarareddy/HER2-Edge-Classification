# Raspberry Pi 4B Benchmark Results

Actual benchmark results from physical Raspberry Pi 4 Model B Rev 1.4 (Quad-core Cortex-A72 @ 1.80 GHz, 4 GB RAM, Debian GNU/Linux 13.7 aarch64, LiteRT 2.3.0, governor `performance`).

## Files
- `benchmark_pi4_precisions_2026-11-10.csv`: **10 Nov 2026 re-measurement**, FP32/FP16/INT8 at 1/2/4 threads, official-rating supply, no throttling (see `PI4_BENCHMARK_REPORT.md` §0).
- `benchmark_samsung_m35_2026-11-10.csv`, `benchmark_realme_6_2026-11-10.csv`: **10 Nov 2026 phone re-measurement** (see `MOBILE_BENCHMARK_REPORT.md`).
- `edge_latency_table_2026-11-10.csv`: all three devices in one table.
- `benchmark_samsung_m35.csv`, `benchmark_realme_6.csv`: 31 Oct phone run (superseded; its INT8 rows are speed-only, its temperatures are battery temperatures).
- `benchmark_primary_pi4.csv`: Primary model (`EfficientNetV2B0__ce__hires`, 448 px FP16) across 1, 2, and 4 threads.
- `benchmark_experimental_pi4.csv`: Experimental calibrated model (`EfficientNetV2B0__ce`, 224 px FP16 with frozen `stain_p99`) across 1, 2, and 4 threads.

## Summary (31 Oct 2026 run; the 4-thread rows hit the soft temperature limit, `0x80008`)

| Mode | Threads | Median Latency | P95 Latency | Throughput | Peak RAM | Agreement |
|---|---|---|---|---|---|---|
| Primary (448 px) | 1 | 756.6 ms | 760.1 ms | 1.32 FPS | 191.6 MB | 1.000 |
| Primary (448 px) | 2 | 444.1 ms | 449.0 ms | 2.25 FPS | 191.5 MB | 1.000 |
| Primary (448 px) | 4 | 353.5 ms | 366.2 ms | 2.82 FPS | 191.3 MB | 1.000 |
| Experimental (224 px) | 1 | 182.9 ms | 185.4 ms | 4.91 FPS | 167.7 MB | 1.000 |
| Experimental (224 px) | 2 | 107.6 ms | 110.4 ms | 7.80 FPS | 167.4 MB | 1.000 |
| Experimental (224 px) | 4 | 81.2 ms | 87.4 ms | 9.81 FPS | 167.3 MB | 1.000 |

Detailed analysis in `deployment/PI4_BENCHMARK_REPORT.md`.
