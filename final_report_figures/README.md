# FYP3 Final Report Figures

This directory contains the publication-quality, 300 DPI figures generated for `FYP3_FINAL_REPORT.md` and `FYP3_PRESENTATION_OUTLINE.md`.

All figures are programmatically generated from authoritative saved experiment records, recomputed evaluation tables, and physical Raspberry Pi 4B hardware benchmark logs. No synthetic or extrapolated numbers are used.

---

## Figure Index and Authoritative Data Sources

| Figure Filename | Title / Description | Authoritative Data Sources |
|---|---|---|
| `fig1_bci_model_comparison.png` | In-Domain Performance on BCI Benchmark (Phase A & B): Validation vs Test QWK and Test Acc vs BalAcc across all architectures | `final_reports/tables/recomputed_metrics_all_runs.csv` |
| `fig2_paired_effects_forest.png` | Paired Effects Forest Plot: Bootstrap 95% CIs resampled over slide fragments for Resolution, Batch Size, Pretraining, and Mixed Training | `final_reports/tables/paired_effects.csv` |
| `fig3_stain_optical_density_distributions.png` | Optical Density Profiles Across Datasets: Background-subtracted DAB and Haematoxylin distributions across FYP2 GAN, BCI, and HER2-IHC-40x | `final_reports/tables/cross_dataset_features.csv` |
| `fig4_cross_dataset_collapse_and_harmonise.png` | External Generalization on HER2-IHC-40x Test Set: Raw CNN collapse to 3+ vs HARMONISE-1 stain calibration restoration | `final_reports/tables/h2_dev_vs_h1_test.csv`, `final_reports/tables/external_recomputed.csv` |
| `fig5_classwise_performance_and_qwk_paradox.png` | Class-Wise Performance & The Clinical QWK Paradox: High QWK (>0.80) masking catastrophic failure on equivocal 1+ and 2+ classes | `final_reports/tables/h2_dev_vs_h1_test.csv`, `next_experiment/harmonise2/outputs/C_classwise.csv` |
| `fig6_harmonise2_bootstrap_and_flagging.png` | HARMONISE-2 Statistical Reliability: Bootstrap CV vs sample size ($n \ge 1,000$ tiles needed) and Selective Review 2+ case loss | `next_experiment/harmonise2/outputs/A_params_summary.csv`, `next_experiment/harmonise2/outputs/E_flagging.csv` |
| `fig7_smartphone_edge_benchmark.png` | Physical Smartphone Edge AI Benchmark: Multi-thread latency scaling on Samsung Galaxy M35 5G (Exynos 1380), Realme 6 (Helio G90T), 4-thread Cortex-A78 sweet spot, 46.1 FPS INT8 throughput, 67% RAM reduction, and Pi 4B comparison | `deployment/benchmark_results/benchmark_samsung_m35.csv`, `deployment/benchmark_results/benchmark_realme_6.csv`, `deployment/benchmark_results/benchmark_primary_pi4.csv` |
| `fig7_raspberry_pi4_hardware_benchmark.png` | Physical Raspberry Pi 4B Edge Benchmark: Median latency, throughput (FPS), preprocessing breakdown, and thermal throttling at 4T (84.2°C) | `deployment/benchmark_results/benchmark_primary_pi4.csv`, `deployment/benchmark_results/benchmark_experimental_pi4.csv` |
| `fig8_quantization_and_agreement.png` | Quantization Efficiency & Cross-Platform Parity: FP32 vs FP16 vs INT8 model sizes and 100.0% prediction agreement on Pi 4B | `final_reports/tables/deploy_recomputed.csv`, `deployment/benchmark_results/benchmark_primary_pi4.csv` |
| `fig9_mobile_app_interface_and_inference.png` | Physical Point-of-Care Android Application Deployment (Samsung M35): 4-panel composite of live device screenshots from Samsung Galaxy M35 5G (Hardware controls, Score 0, Score 2+ FISH reflex, Score 3+ targeted therapy) | `deployment/app_screenshots/samsung_m35/*.png` |
| `fig10_realme6_app_interface_and_inference.png` | Physical Point-of-Care Android Application Deployment (Realme 6): 4-panel composite of live device screenshots from Realme 6 (Helio G90T, Android 11) | `deployment/app_screenshots/realme_6/*.png` |

---

## Reproduction Commands

To reproduce the figures:

```bash
# Generate core research figures 1-6 & 8, and Pi 4B benchmark figure:
python3 generate_final_report_figures.py

# Generate physical smartphone edge benchmark figure (Figure 7 - M35, Realme 6, Pi 4B):
python3 generate_smartphone_figures.py

# Generate physical Android app composite figure for Samsung Galaxy M35 5G (Figure 9):
python3 create_app_composite_figure.py

# Generate physical Android app composite figure for Realme 6 (Figure 10):
python3 create_realme_6_composite_figure.py
```

