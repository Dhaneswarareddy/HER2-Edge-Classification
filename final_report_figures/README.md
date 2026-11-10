# FYP3 Final Report Figures

This directory contains the publication-quality, 300 DPI figures generated for `FYP3_FINAL_REPORT.md` and `FYP3_PRESENTATION_OUTLINE.md`.

All figures are generated from saved experiment records and benchmark logs. **Correction (10 Nov 2026):** the earlier
Figure 7 and Figure 8 contained hard-coded values with no measurement behind them (Pi INT8 122.9 ms, 81.5 °C, agreement
bars 100/100/98.3). Both figures are now rebuilt from measured data only, and Figures 9 and 10 come from the corrected
app v1.1 (`experiments_v2/REPORT_CORRECTIONS_v2.md`, item 41).

---

## Figure Index and Authoritative Data Sources

| Figure Filename | Title / Description | Authoritative Data Sources |
|---|---|---|
| `fig1_bci_model_comparison.png` | In-Domain Performance on BCI Benchmark (Phase A & B): Validation vs Test QWK and Test Acc vs BalAcc across all architectures | `final_reports/tables/recomputed_metrics_all_runs.csv` |
| `fig2_paired_effects_forest.png` | Paired Effects Forest Plot: Bootstrap 95% CIs resampled over slide fragments for Resolution, Batch Size, Pretraining, and Mixed Training | `final_reports/tables/paired_effects.csv` |
| `fig3_stain_optical_density_distributions.png` | Optical Density Profiles Across Datasets: Background-subtracted DAB and Haematoxylin distributions across FYP2 GAN, BCI, and HER2-IHC-40x | `final_reports/tables/cross_dataset_features.csv` |
| `fig4_cross_dataset_collapse_and_harmonise.png` | External Generalization on HER2-IHC-40x Test Set: Raw CNN collapse to 3+ vs HARMONISE-1 stain calibration restoration | `final_reports/tables/h2_dev_vs_h1_test.csv`, `final_reports/tables/external_recomputed.csv` |
| `fig5_classwise_performance_and_qwk_paradox.png` | Class-Wise Performance & The Clinical QWK Paradox: High QWK (>0.80) masking catastrophic failure on equivocal 1+ and 2+ classes | `final_reports/tables/h2_dev_vs_h1_test.csv`, `next_experiment/harmonise2/outputs/C_classwise.csv` |
| `fig6_harmonise2_bootstrap_and_flagging.png` | HARMONISE-2 reliability on development tiles: bootstrap CV of the gains vs sample size (n ≥ 1,000 tiles needed), and share of 2+ tiles referred vs realised coverage | `next_experiment/harmonise2/outputs/A_params_summary.csv`, `next_experiment/harmonise2/outputs/E_flagging.csv` |
| `fig7_smartphone_edge_benchmark.png` | Edge deployment measured 10 Nov 2026: deployed 448 px FP16 latency vs threads on Samsung M35, Realme 6 and Pi 4B; fastest configuration per model (INT8 speed-only); on-device BCI test accuracy and agreement (758/758) | `experiments_v2/08_edge/outputs/edge_latency_table.csv`, `.../ondevice_*/summary.json`, `.../precision_accuracy.csv` (script `experiments_v2/08_edge/make_edge_figures.py`) |
| `fig7_raspberry_pi4_hardware_benchmark.png` | Raspberry Pi 4B, **31 Oct 2026 run** (superseded by the 10 Nov re-measurement; this run hit the soft temperature limit at 4 threads, 84.2 °C, `0x80008`) | `deployment/benchmark_results/benchmark_primary_pi4.csv`, `deployment/benchmark_results/benchmark_experimental_pi4.csv` |
| `fig8_quantization_and_agreement.png` | FP32 / FP16 / INT8 file sizes (MiB) and agreement with Keras on all 727 BCI validation tiles; INT8 below the 95% gate | `final_reports/tables/deploy_recomputed.csv`, `experiments_v2/08_edge/outputs/precision_accuracy.csv` |
| `fig9_mobile_app_interface_and_inference.png` | Android app v1.1 on Samsung Galaxy M35 5G (Android 17): configuration, 2+ and 3+ reference tiles, INT8 warning | `deployment/app_screenshots/samsung_m35/*.png` |
| `fig10_realme6_app_interface_and_inference.png` | Android app v1.1 on Realme 6 (Android 11): same four panels | `deployment/app_screenshots/realme_6/*.png` |

---

## Reproduction Commands

To reproduce the figures:

```bash
# Generate core research figures 1-6 & 8, and Pi 4B benchmark figure:
python3 generate_final_report_figures.py

# Figure 7 (edge, 10 Nov 2026 measurements; generate_smartphone_figures.py is superseded):
python3 experiments_v2/08_edge/make_edge_figures.py

# App screenshots (needs a connected phone with app v1.1):
python3 experiments_v2/08_edge/capture_screenshots_v11.py --serial <adb serial> --out deployment/app_screenshots/<device>

# Generate physical Android app composite figure for Samsung Galaxy M35 5G (Figure 9):
python3 create_app_composite_figure.py

# Generate physical Android app composite figure for Realme 6 (Figure 10):
python3 create_realme_6_composite_figure.py
```

