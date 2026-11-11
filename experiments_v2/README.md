# experiments_v2: completion audit and new experiments (10 Nov 2026)

This folder completes the 8 requested experiments for the HER2 project. Each experiment was audited first, and only
the missing or incomplete parts were implemented. Each experiment folder has a `README.md` with its method, full
results and limitations. Corrections to the existing reports are in `REPORT_CORRECTIONS_v2.md`.

## Integrity rules followed

- **Phase 1 changed nothing outside `experiments_v2/`.** Phase 1 covers experiments 1–7 and the Mac part of
  experiment 8. Every write went through `v2lib.guard()`, which refuses paths outside this folder. Checkpoints,
  manifests, label maps, results and earlier scripts were only read.
- **Phase 2 corrected existing files in place, on the owner's explicit instruction (10 Nov 2026).** This covered
  reports, figures, figure scripts, the Android app and screenshots. The previous versions remain in git history.
  - Every correction traces to an item in `REPORT_CORRECTIONS_v2.md`.
  - Historical logs (`ANTIGRAVITY_*`, `CLAUDE_*`, stage reports) were not rewritten: they received a correction or
    staleness banner.
  - No checkpoint, dataset, manifest or stored result file was modified or deleted. New measurements were added
    next to the old ones.
- **Re-computation was checked against stored values before any new result was used.**
  - Fresh inference matches stored predictions to ≤ 6.5e-6 for 5 networks (experiments 3, 4 and 5).
  - The TFLite FP32 file matches Keras to 5e-6 (experiment 8).
  - The HARMONISE-2 G3 gate class counts were reproduced.
  - The HARMONISE-1 fit/selection sampling was reproduced (asserted).
- **Locked data were never used for fitting.** No training, adaptation, threshold, temperature, hyperparameter or
  model choice used the BCI test split or the external test split.
  - BCI choices: validation only.
  - External choices: development tiles only (R_cal for tuning; R_eval for development reports).
- **The external test split (1,847 tiles) is not untouched.** It was examined in earlier diagnostics and evaluated
  by HARMONISE-1. Every external number here is an **exploratory second look**, with all choices fixed on
  development data first.
- **Seeds.** v2 analyses use 2027; training uses seed 42 as in FYP3; few-shot tuning seeds 101–103, final seeds 1–5.
  Each experiment folder has a `config.json` or a documented protocol.
- **Negative and contradictory results are kept.** Examples:
  - few-shot partial is non-monotone;
  - AdaBN loses to stain_p99 on 2 networks;
  - temperature scaling is not reliably useful in-domain;
  - Mac latency is not reproducible on this machine.
  Two errors in my own earlier statements were corrected and are marked as such (items 3 and 22).

## A. Audit table

| # | Experiment | Before this work | Done here | Status |
|---|---|---|---|---|
| 1 | Clinical cost and risk | Not done (only QWK and accuracy) | Error taxonomy following ASCO/CAP categories; 4 cost scenarios fixed in advance; ±50% sensitivity; paired cost CIs; all 20 BCI runs + external raw / stain_p99 / FYP2 | **Complete** |
| 2 | Uncertainty and selective prediction | Partly (HARMONISE-2 E, development tiles only) | BCI calibration and referral for 20 runs; external test with R_cal-fitted T and thresholds | **Complete** |
| 3 | Grad-CAM | Partly (nine 224 px baselines, correct tiles only) | Deployed 448 px and 224 px models; BCI + external (raw, stain_p99); correct and wrong; 2+ focus; background attention; 7,057 tiles with per-tile statistics | **Complete** |
| 4 | Robustness | Partly (DAB / H / brightness, expected-score shift only, 224 px) | 22 conditions × 2 deployment models; class metrics with paired CIs and flip rate | **Complete** |
| 5 | AdaBN | Not done | 5 networks × 2 arms; development + exploratory test; clinical error profile | **Complete** |
| 6 | Few-shot adaptation | Not done | 3 regimes × k ∈ {1, 5, 10, 20} × 5 draws, tuned on R_cal | **Complete** |
| 7 | Ablation | Resolution / batch existed (E1, E2) | Frozen linear, frozen MLP and partial unfreezing trained; paired analysis incl. E1/E2 re-analysis | **Complete** |
| 8 | Mobile / edge | Pi FP16 and phone latency existed, with unsupported claims | One protocol on Samsung M35, Realme 6 and Pi 4B (all precisions); app fixed (v1.1, bit-exact preprocessing); first on-device accuracy (758/758 agreement on both phones); Mac precision accuracy; screenshots recaptured | **Complete** (phone power added afterwards in `10_power/`; Pi power not measured) |

## B. Main results (actual values; details and CIs in each README)

**1. Clinical risk** (`01_clinical_risk/`)
- BCI test expected workflow cost (S2): 448 px 0.12 [0.06, 0.22] vs 224 px 0.31 [0.20, 0.48]. Paired Δ test
  −0.19 [−0.33, −0.09]. In the cost-weight sensitivity analysis (BCI validation), 448 px is cheapest in
  100% of draws.
- External, 224 px EfficientNetV2B0:
  - stain_p99 cuts negatives-called-3+ from 758 to 64;
  - but sends 595 of 974 negatives (61%) to 2+ and misses 36 2+ tiles.
- The FYP2 GAN-trained ResNet50 has the lowest external cost (0.54) in 100% of draws.

**2. Uncertainty** (`02_uncertainty/`)
- In-domain, referral works. 448 px at 93.7% coverage: accuracy 0.975 → 0.989 (+0.014 [+0.005, +0.026]); 58% of
  errors referred, only 3.7% of 2+.
- External, referral targets 2+: at 70% coverage, 50.5% of 2+ tiles are referred.
- T fitted on R_cal cuts external ECE from 0.205 to 0.099.

**3. Grad-CAM** (`03_gradcam/`)
- Correct 3+ attention follows strong DAB and avoids glass in both domains (DAB ratio 1.23–1.29 on BCI, 1.22–1.42
  externally).
- Correct external 1+ calls attend to glass and pale areas (bg ratio 1.07–1.23): a likely shortcut.
- Errors cluster on sparse tiles (> 50% background in 55–63% of BCI errors vs 32% of correct tiles).
- Side finding: a uint8 round trip of stain_p99 inputs changes 1.4% of predictions (max |Δp| 0.15).

**4. Robustness** (`04_robustness/`)
- Brightness ±15% costs 12–19 points of balanced accuracy for both models.
- The 448 px model is defocus-sensitive: blur σ = 2 px costs 0.17 vs 0.03 at 224 px, and σ = 4 px costs 0.42.
- JPEG ≥ 70, sensor noise, ±20% DAB and haematoxylin ×1.2 have a negligible effect; haematoxylin ×0.8 costs about
  3 points of balanced accuracy (CIs exclude 0).

**5. AdaBN** (`05_adabn/`)
- Better than raw on all 5 networks: test balanced accuracy +0.06 to +0.23, every CI above 0.
- Against stain_p99 it depends on the network: from −0.13 (MobileNetV3Small) to +0.33 (448 px, where it rescues the
  model from 0.186 to 0.516).
- Best unsupervised external balanced accuracy on any network is about 0.52–0.53.

**6. Few-shot** (`06_fewshot/`)
- With 5 labelled tiles per class, full fine-tuning reaches external test balanced accuracy 0.782 ± 0.053 (QWK 0.918).
  Zero-shot is 0.301, stain_p99 0.503 and AdaBN 0.522.
- With 20 per class: 0.845 ± 0.048 (QWK 0.950).
- Partial fine-tuning is non-monotone; its 3-draw tuning was optimistic.

**7. Ablation** (`07_ablation/`)
- Frozen backbone: −0.19 accuracy / −0.26 QWK vs full fine-tuning. An MLP head does not help.
- Partial unfreezing (76% of weights): −0.09 val QWK.
- 448 vs 224 px at matched batch 8 (EfficientNetV2B0): +0.105 val QWK [+0.059, +0.166] and +0.063 test.
- MobileNetV3Small: no confirmed resolution effect.

**8. Edge** (`08_edge/`, measured 10 Nov 2026)
- Deployed 448 px FP16, median network latency:

  | Device | Fastest configuration |
  |---|---|
  | Samsung M35 | 81.4 ms (4 threads) |
  | Realme 6 | 139.1 ms (8 threads) |
  | Pi 4B | 336.5 ms (4 threads, no throttling, ≤ 62.8 °C) |

- **App v1.1 on both phones:** BCI test accuracy 0.975, **758/758** agreement with Keras.
- **INT8:** 3–4× faster but rejected. Agreement 0.928 on the Mac and 0.908 on the Pi, so it is platform-dependent.
- **The Pi INT8 figure was never measured before.** The former 122.9 ms had no measurement behind it; the measured
  value is 104.1 ms.
- **Mac latency is not reproducible** under desktop load.

## C. Figures and tables

| Experiment | Figures (300 dpi) | Main tables |
|---|---|---|
| 1 | `01_clinical_risk/outputs/fig_risk_heatmaps.png`, `fig_error_taxonomy.png` | `risk_table_all.csv`, `risk_table_key.csv`, `paired_cost_differences.csv`, `cost_sensitivity.json` |
| 2 | `02_uncertainty/outputs/fig_uncertainty.png` | `bci_calibration_all_runs.csv`, `bci_selective_prediction_all_runs.csv`, `external_calibration_test.csv`, `external_selective_prediction_test.csv` |
| 3 | `03_gradcam/outputs/fig_gradcam_*.png` (9), `fig_cam_statistics.png` | `cam_stats_per_tile.csv`, `cam_stats_summary.csv`, `figure_examples.csv`, `reproduction_check.json` |
| 4 | `04_robustness/outputs/fig_robustness.png` | `robustness_results.csv`, `reproduction_check.json` |
| 5 | `05_adabn/outputs/fig_adabn.png` | `summary_test.csv`, `summary_dev.csv`, `summary_paired_test.csv`, `checks.csv`, `adabn_<run>.json` |
| 6 | `06_fewshot/outputs/fig_fewshot_data_efficiency.png` | `final_results.csv`, `final_summary_by_k.csv`, `reference_methods.csv`, `gain_over_zero_shot_test.csv`, `tuning_*.csv/json` |
| 7 | `07_ablation/outputs/fig_ablation.png` | `ablation_table.csv`, `ablation_paired.csv` |
| 8 | `08_edge/outputs/fig_edge_benchmark.png` (= report Fig. 7) | `08_edge/outputs/precision_accuracy.csv`, `edge_latency_table.csv`, `ondevice_<device>/summary.json`, `android_preprocessing_check.csv`, `mac_latency_run{1,2}.csv` |

## D. Implementation files

| File | Role |
|---|---|
| `v2lib.py` | Shared paths, write guard, data loaders for stored predictions, metrics, cluster bootstrap |
| `_shared/build_dev_cache.py` | 224 px development-tile cache (`_cache/`, 1.1 GB; regenerable) |
| `01_clinical_risk/run_clinical_risk.py` | Experiment 1 |
| `02_uncertainty/run_uncertainty.py` | Experiment 2 |
| `03_gradcam/run_gradcam.py`, `make_figures.py` | Experiment 3 |
| `04_robustness/run_robustness.py` | Experiment 4 |
| `05_adabn/run_adabn.py`, `summarise_adabn.py` | Experiment 5 |
| `06_fewshot/run_fewshot.py`, `analyze_fewshot.py` | Experiment 6 |
| `07_ablation/train_ablation.py`, `analyze_ablation.py` | Experiment 7 (new checkpoints in `07_ablation/runs/`) |
| `08_edge/mac_prepare.py`, `mac_latency.py`, `android_benchmark.py`, `pi_bundle/pi_precision_benchmark.py` | Experiment 8 |
| `REPORT_CORRECTIONS_v2.md` | 50 numbered items (item 30 withdrawn): errors, unsupported measurements, methodological corrections, suggested wording |

All scripts run with the project interpreter (`Final_Year_Project/Codes/Project/bin/python`, TF 2.15). 448 px and
Grad-CAM work runs on CPU, because tensorflow-metal hangs at 448 px on this Mac. Training uses the Mac GPU.

## E. Limitations (cross-cutting)

1. **Exploratory external test.** The external test split was examined before. Its CIs use filename-run clusters
   because the release has no slide IDs, so they are optimistic.
2. **Possible development-set leakage.** R_cal, R_eval and the few-shot / AdaBN pools come from the same 86
   development slides. Development numbers can be optimistic.
3. **One training seed per arm** for every trained model. The CIs cover tile and fragment sampling only.
4. **Approximate clustering.** BCI fragments are reconstructed by edge matching, not verified slides.
5. **Tile-level analysis only.** No slide or patient aggregation, and no pathologist review of Grad-CAM maps or of
   the error taxonomy.
6. **Illustrative costs.** Clinical cost weights are not validated utilities; only counts and rates are objective.
7. **Edge limits.** Phone power comes from the battery fuel gauge (`10_power/`, one run per configuration); Pi power
   was not measured. Phone temperatures are battery temperatures. On-device accuracy
   covers BCI tiles only.
8. **Runs at the epoch cap.** Three ablation / E1 runs ended at the 45-epoch cap.

## F. Overall research summary

- **In-domain (BCI).**
  - The deployed 448 px EfficientNetV2B0 is strong: test accuracy 0.975, QWK 0.959.
  - It also has the lowest clinical cost under every cost weighting tested.
  - Its gain over 224 px is a real resolution effect for this architecture, not a batch-size artefact.
  - Its confidence supports a sensible referral workflow.
  - It needs a fully fine-tuned backbone; frozen ImageNet features are about 19 points worse.
  - Its weaknesses: brightness changes, and especially defocus. The second point matters for phone-camera use.
- **Cross-domain (HER2-IHC-40x).** No unsupervised method makes the BCI networks usable.
  - Raw: about 0.30 balanced accuracy.
  - stain_p99: about 0.50, but it moves errors to 2+ over-calls and breaks the 448 px model.
  - AdaBN: about 0.52, robust for the 448 px model but network-dependent.
  - Labelled target data is what works: 20 labelled tiles (5 per class) give about 0.78, and 80 give about 0.85.
  - The practical route to a new site is a small labelled calibration set and fine-tuning, plus referral of 2+ cases.
    Stain-only normalisation is not enough.
- **Deployment.** On both phones, app v1.1 reproduces the validated model exactly (758/758 BCI test tiles).
  Latency per tile is 81–139 ms on the phones and 337 ms on the Pi. INT8 is not deployable.
- **Reporting.** Several published numbers in the earlier reports were wrong or unsupported: Table 1/2 cells, Pi INT8,
  phone throttling, energy claims, the DAB mechanism, and the dataset source. They are corrected in the revised reports
  (10 Nov), and every correction is itemised in `REPORT_CORRECTIONS_v2.md`.

## Still open

1. **Power.** Phone power was measured afterwards from the battery fuel gauge (`10_power/`, next-semester
   material). Pi power and an external-meter check of the phone values need a USB-C power meter.
2. **App packaging.** The app's TFLite 2.16.1 library is not 16 KB page-aligned, which Android 15+ devices with 16 KB
   pages will need. A LiteRT upgrade would need re-validation with `08_edge/ondevice_validation.py`.
3. **E6 and E7** from `NEXT_EXPERIMENTS.md` were run on 10–11 Nov 2026 (`09_resolution_e6_e7/`, next-semester material).
   Neither met its pre-registered rule: Δ val QWK +0.038 [−0.008, +0.078] (E6) and +0.038 [−0.005, +0.090] (E7).
   **Not attempted:** slide-level aggregation.
