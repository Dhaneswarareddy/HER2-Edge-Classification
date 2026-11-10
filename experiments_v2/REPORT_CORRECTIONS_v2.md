# FYP3: factual errors, unsupported measurements and recommended revisions

**Scope.** This document checks `FYP3_FINAL_REPORT.md` (9 Nov 2026 version), `FYP3_PRESENTATION_OUTLINE.md`,
`deployment/MOBILE_BENCHMARK_REPORT.md`, `deployment/CROSS_PLATFORM_EDGE_ANALYSIS.md`, `deployment/PI4_BENCHMARK_REPORT.md`
and `FYP3_FINAL_AUDIT.md` against the stored artifacts. **Status (10 Nov 2026, evening):** on the owner's instruction all
items below have been applied to the documents themselves (revised `FYP3_FINAL_REPORT.md`, deployment reports, README,
presentation outline, figures, Android app); historical logs received correction banners. Earlier versions are in git. Line numbers refer to
`FYP3_FINAL_REPORT.md` unless stated otherwise.

Severity:
- **A** = factually wrong or unsupported by any artifact; must change.
- **B** = overstated, or mixes data sets; should change.
- **C** = wording, citation or presentation.

## 1. Wrong values and wrong statements

| # | Sev. | Where | Current claim | What the artifacts show | Correction |
|---|---|---|---|---|---|
| 1 | A | Table 1, line 200 | MobileNetV3Large CORN test: acc 0.908, bal. acc 0.884, F1 0.879, QWK 0.878 | These are the **CE** row's test values, copied. `results/runs/MobileNetV3Large__corn/summary.json`: test acc 0.937, bal. acc 0.919, macro-F1 0.922, QWK 0.923 | Replace with the CORN values |
| 2 | A | §12.2, line 316 | External tiles have "median background-subtracted DAB 0.48 vs 0.22 on BCI" and "depressed haematoxylin counterstaining" | No artifact contains 0.48 / 0.22. `final_reports/tables/cross_dataset_features_by_class.csv`: external haematoxylin is **3–7× stronger** than BCI (0.09–0.29 vs 0.02–0.08). External DAB is higher in 1+, 2+ and 3+ (0.16 / 0.31 / 1.29 vs 0.09 / 0.12 / 0.49) and lower in 0 (0.05 vs 0.09) | Use the per-class table. Haematoxylin is elevated, not depressed |
| 3 | B | §13.4, line 373, and `ANTIGRAVITY_RELIABILITY_REPORT.md` / `HARMONISE2_EXECUTION_REPORT.md` §1.4 and §6.2 | Final report: the 448 px collapse is caused by "dimming haematoxylin ... destroyed nuclear context". HARMONISE-2 execution report: "suppressing DAB optical density dims faint membrane signals ... dropping 3+ recall (0.849 → 0.307)" | HARMONISE-2 D dose-response on 1,200 development tiles (`next_experiment/harmonise2/outputs/D_external_EfficientNetV2B0__ce__hires.csv`). 3+ recall: none 0.849; **DAB gain only (g_d 0.50) 0.852**; **H gain only (g_h 0.25) 0.266**; both 0.307. The collapse follows the **haematoxylin** gain; DAB dimming alone leaves 3+ recall unchanged | Final report: the direction (haematoxylin) is supported, but "destroyed nuclear context" is an untested interpretation. HARMONISE-2 execution report: its DAB mechanism is **contradicted by its own data** and must be withdrawn. (The 30 Oct project audit repeated the DAB explanation; it was wrong.) |
| 4 | B | §13.4 | "scale2 restored 448px balanced accuracy from 0.186 back to 0.322" | 0.186 is the **external test** result (HARMONISE-1). 0.322 is on **1,200 development R_eval tiles** (HARMONISE-2 D). Different data sets. On the same development tiles: none 0.279, frozen 0.228, frozen_scale2 0.322 (+0.043), which **fails** the pre-set rule because 3+ recall dropped by more than 0.10 | Compare within one data set and state that the rule failed |
| 5 | B | §13.5, line 379 | "flagging the top 40% most uncertain tiles discarded 74% to 79% of all true 2+ cases" | `E_flagging.csv` reports coverages 0.5, 0.7 and 0.9 only; no 0.6 (40% flagged) setting exists. At coverage 0.5: 2+ referred 75.8% (EfficientNetV2B0) to 93.9% (MobileNetV3Small) | Quote measured coverages, e.g. "at 50% coverage, 76–94% of 2+ tiles were referred" |
| 6 | A | Line 515–516 and Fig. 7/8 generator | Pi 4B INT8 latency 122.9 ms (4 threads), 8.14 FPS | **No Pi INT8 measurement exists.** Both Pi CSVs contain FP16 only. 122.9 is hard-coded in `generate_smartphone_figures.py:173` | Remove, or replace with a measured value (script ready: `experiments_v2/08_edge/pi_bundle/`) |
| 7 | A | Lines 437–473, 500, 518; `MOBILE_BENCHMARK_REPORT.md` | Phones "throttled=0x0", "zero thermal throttling" | `0x0` is a Raspberry Pi `vcgencmd` flag; Android has no such flag, and none was read (`run_samsung_m35_benchmark.py` reads only `dumpsys battery`). The temperatures are **battery** temperatures, not CPU | Report "battery temperature X °C; CPU throttling not measured" |
| 8 | A | Lines 488, 513; `CROSS_PLATFORM_EDGE_ANALYSIS.md`; `MOBILE_BENCHMARK_REPORT.md` §1 | "per-watt efficiency", "energy ... advantages" | No power or energy was measured on any device | Remove every power or energy claim |
| 9 | A | Line 441 ff., 494, 515 | INT8 448 px phone latency presented as a deployment result ("real-time", "zero lag") | The INT8 model failed the project's own gate: agreement with FP32 0.928 < 0.95; val accuracy 0.919 vs 0.950; 1+ test recall 0.96 → 0.85 (`results/deploy/EfficientNetV2B0__ce__hires/deploy_summary.csv`). The app ships it as a selectable model | Label INT8 "speed only, rejected for deployment", or remove it from the app |
| 10 | B | Lines 437–473 | Thread-scaling interpretations ("near-linear", "barrier synchronisation stall", "Sync Stall") | Samsung 224 px FP16: 176.8 ms at 1 thread, 30.7 ms at 2. A 5.7× speed-up from doubling threads is super-linear and impossible from parallelism alone. INT8 at 1 thread: median 74 ms, mean 131 ms. Single-thread runs were likely scheduled on the A55 little cores (no CPU affinity was set) | Treat as uncontrolled scheduling. Remove the micro-architectural explanations or mark them as hypotheses |
| 11 | B | Line 416–418; Android app | The app computes the same prediction as the validated model | `HER2Classifier.kt:83` scales 1024 → 448 px directly (`createScaledBitmap`). Validated path: 1024 → Lanczos 512 → bilinear 448. On-device accuracy was never measured | See `experiments_v2/08_edge/` (measured effect of this mismatch on the Mac) |
| 12 | B | Line 21–29 (Abstract), §14 | "100.0% prediction agreement across the 120 reference clinical tiles" | Primary: 120 BCI **validation** tiles. Experimental: 120 HER2-IHC-40x **development** tiles (R_eval), not test tiles (`deployment/reference/*.npz`) | Name the tile sets |
| 13 | B | Lines 95, 100–101, 117, 297 | "Qaiser et al." centre; "86 Dev WSIs / 21 Test WSIs"; "publisher-defined slide/patient blocks" | The dataset is the University of Malaya HER2-IHC-40x release (Data in Brief 2025; Zenodo 10.5281/zenodo.15179607). It reports 86/21 WSIs, but the release has **no slide IDs** (HARMONISE-2 B). The tiles cannot be grouped by slide | Correct the source. Say "from 21 slides (no slide identifiers released)" |
| 14 | B | Line 132 | BCI slides reconstructed "by parsing tile index coordinates and spatial clustering" | Fragments were rebuilt by matching continuous tile **edges** (`src/analysis_slides.py`): 2,763 accepted links, 691 fragments | Describe edge matching |
| 15 | B | Lines 73, 321 | HARMONISE is "reference-free" | Gains are fitted by matching percentiles to **BCI training tiles** (`next_experiment/run_harmonise.py`, stage fit). It needs a reference domain | Say "unsupervised (no target labels), reference = BCI training tiles" |
| 16 | C | Ref. 1, line 580 | BCI cited as *IEEE TMI* | BCI: Liu et al., CVPR Workshops 2022 (arXiv:2204.11425), as in `DATASET.md` | Fix the citation |
| 17 | C | Ref. 2, line 581 | Qaiser et al. cited as the external dataset | Qaiser et al. is the Warwick HER2 contest (FYP2's GAN source), not HER2-IHC-40x | Cite the Data in Brief 2025 article for HER2-IHC-40x |
| 18 | C | Units throughout §14 | File sizes 11.27 MB and 11.82 MB, 6.82 MB and 7.15 MB for the same files | Same files: 11.27 MiB = 11.82 MB; 6.82 MiB = 7.15 MB | Use one unit |
| 19 | B | `ANTIGRAVITY_EXECUTION_REPORT.md` | Frozen-parameter SHA-256 `90b14643…` | Actual file and `.sha256`: `89a0a487…` | Correct the hash |
| 28 | A | Table 1 (lines 188–203) and Table 2 (lines 225–234): further wrong cells beyond item 1 | Each run's `results/runs/<run>/summary.json` and `history.csv` (also `final_reports/tables/recomputed_metrics_all_runs.csv`). See the detail table after this one | Regenerate both tables from the stored summaries. Key-finding 1's validation range becomes 85.4%–90.8% |
| 29 | B | HARMONISE stain_p99 deployment (§13, `deployment/` experimental mode) | Implies the calibrated image can be handled like any image | `experiments_v2/03_gradcam`: casting the calibrated tile to uint8 before inference changed probabilities by up to 0.15 and 1.4% of predicted classes. HARMONISE-1 fed float32 | State that the reported stain_p99 metrics need float32 input. An 8-bit round trip (PNG, JPEG, camera buffer) changes predictions |

| 30 | — | **Withdrawn (10 Nov 2026).** Earlier draft: "Pi Keras agreement 98.3% contradicted by its own CSV" | That check read the **primary** model's CSV (`benchmark_primary_pi4.csv`, agreement 1.0). The 98.3% in `PI4_BENCHMARK_REPORT.md` §4 belongs to the **experimental 224 px** model, and `benchmark_experimental_pi4.csv` confirms it: 0.983 (2 of 120 tiles, max \|Δp\| 0.007). The Pi report is correct on this point | No change needed |
| 31 | B | Mac latency figures used as a baseline in §14 and `CROSS_PLATFORM_EDGE_ANALYSIS.md` | Single Mac latency values (e.g. FP32 71 ms, FP16 75.5 ms in `deploy_summary.csv`) | Two repeated runs in `experiments_v2/08_edge` (same protocol, 200 invokes each) under ordinary desktop load gave medians differing by up to 2.4× (448 px FP16, 2 threads: 352 vs 148 ms) | Treat Mac latencies as indicative only, or measure on an idle machine with repeats |

**Item 28 in detail: cells that differ from the stored runs** (report value → stored value). Cells not listed match.

| Run | Wrong cells |
|---|---|
| MobileNetV3Small CE (Table 1, and "standard" in Table 2) | best epoch 19 → **34**; val acc 0.898 → **0.854**; val bal. acc 0.874 → **0.822**; val F1 0.894 → **0.833**; val QWK 0.865 → **0.763**; test bal. acc 0.895 → 0.886; test F1 0.906 → 0.895; train time 16 → 23 min. The validation cells nearly duplicate the MobileNetV3Small **hires** run (0.898 / 0.897 / 0.893 / 0.865), so the E2 comparison in the report is wrong |
| NASNetMobile CE | best epoch 31 → 28; val 0.861 / 0.840 / 0.847 / 0.772 → **0.884 / 0.851 / 0.865 / 0.805**; test 0.926 / 0.908 / 0.918 / 0.902 → 0.921 / 0.895 / 0.912 / 0.904 |
| ResNet50 CE | val bal. acc 0.869 → 0.881; val F1 0.889 → 0.892; test F1 0.941 → 0.936 |
| MobileNetV3Large CE | best epoch 18 → 22 |
| MobileNetV3Large CORN | best epoch 15 → **45** (ended at the cap, item 23); test row see item 1 |
| MobileNetV2 CE | best epoch 13 → 33 |
| MobileNetV3Small lowbatch (Table 2) | val bal. acc 0.844 → 0.831; val F1 0.837 → 0.844; val QWK 0.774 → 0.775; test bal. acc 0.888 → 0.875; test F1 0.875 → 0.881; train time 15 → 28 min |
| MobileNetV3Small hires (Table 2) | val bal. acc 0.894 → 0.897; val F1 0.887 → 0.893; test F1 0.887 → 0.893 |

The validation ranking changes: MobileNetV3Small CE drops from 2nd (0.865) to last (0.763) on val QWK. NASNetMobile
rises (0.772 → 0.805).

## 2. Omitted results that change the interpretation

| # | Sev. | Omission | Evidence | Recommended text |
|---|---|---|---|---|
| 20 | A | FYP2 (GAN-trained) networks score **better** on HER2-IHC-40x than every BCI network, with or without HARMONISE | `final_reports/cross_dataset_audit.md`: balanced accuracy 0.61–0.73 uncalibrated, vs 0.41–0.53 for calibrated BCI networks. A 6-feature stain logistic rule fitted on FYP2 reaches acc 0.89 / QWK 0.96. The external set was hue-histogram filtered per class | Add to §12 and to the discussion. §15.1 ("Myth of synthetic generalisation") must be qualified: synthetic training fails on BCI, but the external test set is unusually separable by colour |
| 21 | A | Clinical form of the HARMONISE trade-off | `experiments_v2/01_clinical_risk`: on 224 px EfficientNetV2B0, calibration cuts negatives-called-3+ from 758 to 64 but raises missed 2+ from 6 to 36 and negatives-called-2+ from 80 to 595 (61% of negatives) | Add an error-type table; QWK and accuracy alone hide this |
| 22 | B | Resolution effect confirmed on 1 of 2 architectures | E2 (MobileNetV3Small), recomputed in `experiments_v2/07_ablation/outputs/ablation_paired.csv`. 448 px vs batch-matched 224 px (batch 8): Δ val QWK +0.090 [−0.006, +0.205], Δ test QWK +0.017 [−0.035, +0.078]. 448 px vs standard 224 px (batch 32): val QWK +0.102 [+0.027, +0.172], but test −0.014 [−0.064, +0.040]. *(An earlier draft of this item said "hires val acc 0.898 = baseline 0.898". That repeated the report's wrong baseline row; see item 28.)* | Replace "the single most critical driver" with "a confirmed improvement for EfficientNetV2B0 (448 vs batch-matched 224: val QWK +0.105 [+0.059, +0.166], test +0.063 [+0.025, +0.116]). For MobileNetV3Small it was not confirmed against the batch-matched control and not reproduced on test. One seed per arm" |
| 23 | B | Three runs ended at the 45-epoch cap, not by early stopping | `history.csv`: ResNet50 CE (best 43/45), MobileNetV3Large CORN (best 45/45), EfficientNetV2B0 hires (best 38/45) | State it. The 448 px model may be under-trained relative to its comparator |
| 24 | B | Failed and retried runs | DenseNet121 CORN segfaulted once (`train_DenseNet121__corn_segfault1.log`); the retry ended with rc 133 after writing results. EfficientNetV2B0 lowbatch deadlocked once. `evaluate_all` hung once | Mention in the methods (reproducibility) |
| 25 | B | `results/REPORT.md`, `selection.json`, `metrics_all.csv`, `overfitting.csv` predate E1–E5 | 16 of 20 grouped runs. The four newest runs have no overfitting verdict | State that these files are stale, or regenerate them into a new location |
| 26 | B | E6 (DenseNet121 at 448 px) and E7 (1024 px) planned but not run | `NEXT_EXPERIMENTS.md` | List as not done |
| 27 | B | External test status | Examined in diagnostics, then evaluated by HARMONISE-1, and again in `experiments_v2` | Every external result must say "previously examined split", never "untouched" |

## 2b. Further items found while applying the corrections (10 Nov 2026)

| # | Sev. | Where | Claim | What the artifacts show | Applied correction |
|---|---|---|---|---|---|
| 32 | A | Report Abstract, §12.1 | Zero-shot accuracy 41.9%–50.8%; "93.1% to 99.1% of all external tiles were predicted as 3+" | Over the 10 BCI-trained networks: accuracy 0.419–0.586, and 42.7%–93.8% of tiles are predicted 3+ (EfficientNetV2B0 224 px: 87.9%; 448 px: 42.7%) | Ranges corrected; per-network table added |
| 33 | B | Report §9, key finding 3 | CORN is lower than CE "across all architectures" | MobileNetV3Large: CORN val QWK 0.825 > CE 0.800 | "No consistent gain"; directions per architecture |
| 34 | B | Report §5, item 1 | BCI labels agree with "independent third-party patch-level annotations" on 69.7% / 73.6% of 2+ tiles | The numbers reproduce exactly, but the refined labels (Zenodo 10.5281/zenodo.17200604) have an undocumented labelling method | Described as third-party refined labels of undocumented method |
| 35 | B | Report §2; app v1.0 texts | ASCO/CAP 2+ includes "intense complete staining in ≤ 10%" (2013 wording); app said 1+ "targeted therapy not indicated" | ASCO/CAP 2018 2+ = weak-to-moderate complete in > 10%. Since DESTINY-Breast04 (2022) and the 2023 update, 1+ (HER2-low) matters for trastuzumab deruxtecan | Definitions updated in the report and app |
| 36 | B | Report Table 3 | E2 batch effect described as null | Test ΔQWK −0.030 [−0.067, −0.003] (CI excludes 0) | Row added |
| 37 | A | Android app v1.0 | INT8 offered as a normal option; "big.LITTLE affinity"; "Edge AI Neural Engine"; "Real-Time … Screening"; FPS from one tile | No affinity set; CPU-only inference; INT8 failed the gate | App v1.1: INT8 labelled speed-only with warnings; texts corrected; single-tile ms shown |
| 38 | B | Android app v1.0 (both phones' screenshots) | Precision and thread radio buttons unlabeled on the Realme 6 | Default text colour in system dark mode was white on white cards | Fixed (explicit colours); screenshots recaptured |
| 39 | B | Report §14 / app | On-device accuracy implied | Never measured before | Measured: 758/758 agreement with Keras, test accuracy 0.975 on both phones (`08_edge/ondevice_validation.py`) |
| 40 | B | Report §14 | Samsung M35 "Android 16" | The phone now reports Android 17 (OS updated after the 31 Oct run) | Stated as such |
| 41 | B | `generate_final_report_figures.py`, `generate_smartphone_figures.py` | Fig. 8 agreement bars hard-coded [100, 100, 98.3]; Fig. 7 hard-coded Pi INT8 122.9 ms and 81.5 °C | No measurement existed | Fig. 8 now reads `08_edge/outputs/precision_accuracy.csv`; Fig. 7 rebuilt from 10 Nov data (`08_edge/make_edge_figures.py`) |
| 42 | C | Fig. 6 annotation | "At 50% coverage: 79% of 2+ discarded" | Target 0.5 realised 0.44 coverage (margin signal) on development tiles | Annotation states the realised coverage |
| 43 | C | Android app | TFLite 2.16.1 JNI library not 16 KB page-aligned (Android 15+ warning) | Both test phones use 4 KB pages, so it runs | Disclosed; a LiteRT upgrade would need re-validation |

## 3. Wording to soften

| Line | Current | Suggested |
|---|---|---|
| 258 | "+10.5 points ... (p < 0.001)" | No p-value was computed. Use the CI: "+0.105 [0.056, 0.168]" |
| 451, 494 | "Real-Time", "zero lag", "whole-slide pan-and-scan" | Latency per 448 px tile on random input; no WSI viewer was built or tested |
| 541 | "The Myth of Synthetic Generalisation" | "Synthetic training did not transfer to BCI" (see item 20) |
| 566 | "single most critical performance driver" | See item 22 |

## 4. New results available for the report

See `experiments_v2/README.md` for every experiment added on 10 Nov 2026, with numbers, figures and limitations.
