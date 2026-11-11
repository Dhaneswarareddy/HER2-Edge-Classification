# Project Status and Roadmap: every experiment in pipeline order, with the semester split

**Snapshot: 10 November 2026; E6/E7 results added 11 November 2026.** This is the single place that lists every experiment done in the HER2 project, in the
order a reader should meet them (data → preprocessing → training → evaluation → cross-laboratory generalisation →
deployment). It records what each step found, where the evidence is, and which steps belong to this semester's
submission and which to next semester's.

**For anyone writing a report from this repository:**
- Use only numbers that appear in this file or in the files it points to. Every number here was copied from a stored
  result file; none is estimated.
- The full technical write-up of the completed work is `FYP3_FINAL_REPORT.md` (corrected 10 Nov 2026). Per-experiment
  details are in `experiments_v2/<nn>_<name>/README.md`.
- `experiments_v2/REPORT_CORRECTIONS_v2.md` lists 50 numbered corrections to earlier reports (item 30 withdrawn; items 44–50 come from an independent re-check of this semester's numbers on 10 Nov 2026).
  Older documents (`ANTIGRAVITY_*`, `CLAUDE_*`, `results/REPORT*.md`) are historical logs and must not be quoted for
  numbers.
- The "must disclose" list in §5 is not optional: those points have to appear in any report.

---

## 1. Project in one paragraph

FYP2 trained HER2 IHC scoring networks (classes 0, 1+, 2+, 3+) on synthetic conditional-GAN tiles and reached about
100% accuracy on them. This project moves to real tissue. It trains and selects networks on the public BCI dataset
with a leakage-controlled split, tests them on a second laboratory's data (HER2-IHC-40x, University of Malaya), studies
why they fail there and what fixes it, and deploys the selected model on two Android phones and a Raspberry Pi 4B
with the on-device predictions verified tile by tile. The selected model is EfficientNetV2B0 at 448 px input
(`EfficientNetV2B0__ce__hires`): BCI test accuracy 0.975, QWK 0.959. It does not transfer to the second laboratory
without a few labelled tiles from that laboratory.

---

## 2. The semester split

All work is ordered along the pipeline in §3. **This semester covers Steps 1–14 (Stages A–C): building and validating
the in-domain classifier.** **Next semester covers Steps 15–33 (Stages D–E plus the new work): making it work in
another laboratory and on devices.** The cut follows the pipeline order: everything in this semester's part is
needed before anything in the next semester's part makes sense, and each part tells a complete story.

| | This semester (Steps 1–14) | Next semester (Steps 15–33) |
|---|---|---|
| Story | From synthetic to real tissue: a validated, leakage-controlled BCI classifier, what drives its accuracy, and how trustworthy it is in-domain | Does it work elsewhere? Cross-laboratory failure, its causes and fixes; then deployment, on-device verification, speed and energy |
| Stages | A Data, B Model development, C In-domain evaluation | D Cross-laboratory generalisation, E Deployment and edge, F New experiments |
| Status | All 14 steps complete | Steps 15–32 complete (Step 30 phone power, 10 Nov; Steps 31–32 E6/E7, 11 Nov); Step 33 onward planned |
| Headline result | 448 px EfficientNetV2B0: BCI test acc 0.975, QWK 0.959; resolution effect confirmed against a batch-matched control | Zero-shot collapse (bal. acc 0.26–0.38); 5 labelled tiles per class recover 0.78; app reproduces the model on 758/758 tiles; 81–337 ms per tile |

E6 and E7 belong to Stage B by topic (they extend the resolution study), but they were started after the snapshot, so
they are next-semester work by the owner's decision (§4, Steps 31–32). Neither met its pre-registered rule.

---

## 3. All steps, in pipeline order

Status: ✅ complete · 🔄 running · ⏳ planned. "Where" points to the evidence.

### Stage A — Data foundation (this semester)

**Step 1. Datasets and provenance audit** ✅
- FYP2 synthetic: 2,400 cGAN tiles (600 per class), 256 px, labels = generator condition.
- BCI (internal): 4,873 IHC tiles, 1024×1024, 0.46 µm/px; Liu et al., CVPR Workshops 2022; non-commercial licence.
- HER2-IHC-40x (external): 9,940 tiles, 1024×1024; University of Malaya, Data in Brief 2025
  (doi 10.1016/j.dib.2025.111922), Zenodo 10.5281/zenodo.15179607, CC BY-SA 4.0; 107 WSIs at 40×.
- Label audit:
  - BCI labels come from the parent whole-slide image, not the tile. A third-party refined patch label (3,513 tiles)
    agrees with the BCI label on 82% of tiles, but only on 69.7% of validation 2+ and 73.6% of test 2+ tiles.
  - BCI scan type is associated with class: grey scans are 25/44/52/68% of tiles in classes 0/1+/2+/3+ (possible
    shortcut).
  - HER2-IHC-40x tiles were selected per class by hue histogram, which makes that set unusually colour-separable.
- Where: `DATASET.md`, `FYP3_FINAL_REPORT.md` §5.

**Step 2. Leakage-controlled BCI split** ✅
- BCI has no slide IDs and its official split leaks (neighbouring tiles share continuous borders).
- Tiles were linked by edge continuity (`src/analysis_slides.py`): 2,763 accepted links, 691 multi-tile fragments; with the
  1,645 tiles that link to nothing, the split has 2,336 groups.
- Whole fragments assigned to train / val / test = 3,388 / 727 / 758 tiles. All CIs resample fragments.
- Where: `splits/`, `FYP3_FINAL_REPORT.md` §6.1.

**Step 3. External-data protocol** ✅
- HER2-IHC-40x development archive 8,093 tiles; WSI-based test split 1,847 tiles (0/1+/2+/3+ = 658/316/111/762).
- Development tiles split further: 1,500-tile fit pool; R_cal 2,191 (tuning) and R_eval 2,402 (development reports).
- No slide IDs: external CIs resample filename runs and are optimistic.
- Where: `FYP3_FINAL_REPORT.md` §6.2.

**Step 4. Preprocessing pipeline** ✅
- 1024 px → Lanczos to a 256 px (or 512 px) cache → bilinear to the 224 px (or 448 px) network input.
- The same two-step resize is reproduced bit-exactly in the Android app v1.1 (Step 26).
- Where: `src/config.py`, `src/train_variant.py` (`ensure_cache`).

**Step 5. Cross-dataset stain and image characterisation** ✅
- Optical densities per class (Ruifrok–Johnston, relative to each tile's illumination):
  - per-class median (over tiles) of the tile-mean DAB: BCI 0.09/0.09/0.12/0.49 vs HER2-IHC-40x 0.05/0.16/0.31/1.29 (classes 0/1+/2+/3+);
  - same statistic for haematoxylin: BCI 0.08/0.02/0.03/0.03 vs HER2-IHC-40x 0.29/0.11/0.17/0.09 — 3.4–7× stronger externally.
- External tiles are blurrier (sharpness 68–766 vs 554–2,279) with larger stain blobs (5–14 px vs 2–3 px), consistent
  with a different magnification.
- Where: `final_reports/tables/cross_dataset_features_by_class.csv`, Figure 3, report §12.2.

### Stage B — Model development (this semester)

**Step 6. Baseline architecture comparison (Phase A)** ✅
- 9 ImageNet backbones, 224 px, two-stage FYP2 recipe (head 5 epochs; fine-tune ≤ 40 epochs; early stopping on val
  QWK); cross-entropy for all, plus CORN ordinal loss for 3. 12 runs. One seed (42) per run.
- Validation accuracy 0.854–0.908; best validation QWK: EfficientNetV2B0 CE 0.869 (selected for the resolution study).
- CORN gives no consistent gain (lower than CE for DenseNet121 and EfficientNetV2B0, higher for MobileNetV3Large).
- Test scores exceed validation for every network: the test split happens to be easier.
- Where: Table 1 / Figure 1 in the report (§9); `results/runs/*/summary.json`.

**Step 7. Training-variant screening (MobileNetV3Small)** ✅
- Class weights: val QWK +0.023 [−0.040, +0.088]; label smoothing +0.004 [−0.045, +0.054]: neither meets the
  pre-set rule (CI must exclude 0).
- 448 px input: +0.102 [+0.031, +0.173] val QWK, but test QWK −0.014 — which triggered E1/E2.
- Where: `src/variants.py` (header), `results/runs/MobileNetV3Small__ce__{cw,ls}/`.

**Step 8. E1/E2 — resolution vs batch size** ✅
- E1 EfficientNetV2B0, 448 px vs 224 px at the same batch (8): **+0.105 val QWK [0.056, 0.168]**, +0.063 test QWK
  [0.026, 0.116]. Batch effect alone (b8 vs b32 at 224 px): −0.040 [−0.109, 0.027], i.e. none.
- Primary model `EfficientNetV2B0__ce__hires`: val acc 0.950, QWK 0.935; **test acc 0.975, bal. acc 0.973, QWK 0.959**.
  Training took 252 min (5.2× the 224 px run).
- E2 MobileNetV3Small: resolution effect not confirmed (val QWK +0.090 [−0.007, 0.207]; test +0.017).
- Where: Tables 2–3, Figure 2 (report §10); `final_reports/tables/paired_effects.csv`.

**Step 9. E4/E5 — synthetic pretraining and mixing** ✅
- E4: initialise from the FYP2 synthetic model → val QWK −0.015 [−0.075, 0.042].
- E5: FYP2's 1,680 synthetic training tiles added to the 3,388 BCI training tiles and shuffled together (≈33% synthetic on average, not a fixed ratio per batch) → val QWK −0.023 [−0.073, 0.020].
- Neither helps on BCI (all CIs span 0).
- Where: Tables 4–5 (report §11).

**Step 10. Backbone ablation (Exp 7)** ✅
- Frozen ImageNet backbone + linear head: −0.19 accuracy / −0.26 QWK vs full fine-tuning; an MLP head does not improve accuracy or QWK (validation balanced
  accuracy rises +0.052 [+0.011, +0.084], still far below full fine-tuning).
- Partial unfreezing (only `block6*` and the top layers; these still hold 76% of the weights): −0.09 val QWK.
- Re-analysis confirms E1's resolution effect (+0.105 val QWK at matched batch).
- Where: `experiments_v2/07_ablation/`.

### Stage C — In-domain evaluation and trust (this semester)

**Step 11. Clinical error taxonomy and cost (Exp 1)** ✅
- Errors mapped to ASCO/CAP-relevant categories (e.g. negative called 3+, 2+ missed); 4 cost scenarios fixed in
  advance, ±50% sensitivity.
- BCI test expected workflow cost (scenario S2): 448 px 0.12 [0.06, 0.22] vs 224 px 0.31 [0.20, 0.48]; paired Δ −0.19
  [−0.33, −0.09]; 448 px cheapest in 100% of cost draws (sensitivity analysis on BCI validation, all 20 runs).
- Where: `experiments_v2/01_clinical_risk/`.

**Step 12. Calibration and selective prediction (Exp 2)** ✅
- In-domain referral works: 448 px at 93.7% coverage, accuracy 0.975 → 0.989 (+0.014 [+0.005, +0.026]); 58% of errors
  referred, only 3.7% of 2+ tiles referred.
- Temperature scaling is not reliably useful in-domain.
- Where: `experiments_v2/02_uncertainty/`.

**Step 13. Grad-CAM explainability (Exp 3)** ✅
- 7,057 tiles with per-tile attention statistics; deployed 448 px and 224 px models; correct and wrong tiles.
- Correct 3+ attention follows strong DAB and avoids glass: on BCI, DAB ratio 1.23 (224 px) and 1.29 (448 px),
  background ratio 0.75 and 0.66.
- Errors cluster on sparse tiles (> 50% background in 55–63% of BCI errors vs 32% of correct tiles).
- Where: `experiments_v2/03_gradcam/`.

**Step 14. Robustness to image perturbations (Exp 4)** ✅
- 22 conditions × 2 models, class metrics with paired CIs and flip rates.
- Brightness ±15% costs 12–19 points of balanced accuracy; the 448 px model is defocus-sensitive (blur σ = 2 px:
  −0.17 vs −0.03 at 224 px; σ = 4 px: −0.42). JPEG ≥ 70 and sensor noise: negligible;
  ±20% DAB and haematoxylin ×1.2 have a negligible effect; haematoxylin ×0.8 costs about 3 points of balanced accuracy (−0.031 at 448 px, −0.030 at 224 px; both CIs exclude 0).
- Where: `experiments_v2/04_robustness/`.

> **End of this semester's part.** Steps 1–14 deliver a validated in-domain classifier, the evidence for why it is
> accurate (resolution, full fine-tuning), what does not help (synthetic data, class weights, CORN), and how it behaves
> (clinical error profile, referral, attention, robustness).

### Stage D — Cross-laboratory generalisation (next semester)

**Step 15. E3 — zero-shot external test** ✅
- 10 BCI networks on the 1,847 external test tiles: accuracy 0.419–0.586, balanced accuracy 0.260–0.381; 43–94% of
  all tiles predicted 3+.
- Where: report §12.1, Figure 4.

**Step 16. HARMONISE-1 — unsupervised stain calibration** ✅
- Scale haematoxylin and DAB optical densities so the target's 99th percentile matches BCI training tiles
  (g_h 0.2545, g_d 0.5045, fitted on development tiles only).
- 224 px EfficientNetV2B0: balanced accuracy 0.301 → 0.503; but false 3+ calls become false 2+ calls (595 of 974
  negatives called 2+; 2+ precision 0.093).
- It breaks the 448 px model (balanced accuracy 0.186).
- Where: Table 6 (report §12.3).

**Step 17. HARMONISE-2 — reliability analysis (A–E)** ✅
- A: about 1,000 unlabelled tiles needed for stable gains (CV ≈ 0.05).
- B: neither dataset allows slide-level grouping.
- C: the "QWK paradox": QWK 0.802 with 1+ recall 0.139 and 2+ recall 0.090.
- D: the 448 px collapse follows the **haematoxylin** gain (3+ recall 0.849 → 0.266), not DAB.
- E: out-of-domain referral concentrates on 2+ (74–94% of 2+ referred at 50% coverage).
- Where: report §13, Figures 5–6; `next_experiment/harmonise2/` (full repo only).

**Step 18. FYP2 synthetic networks on the external set** ✅
- Trained only on GAN tiles, they reach balanced accuracy 0.61–0.73 externally — better than any BCI network.
- Explanation: both the GAN classes and the hue-filtered external set are separable by DAB colour; a 6-feature stain
  rule fitted on FYP2 reaches 0.89 accuracy on the external set. A colour shortcut, not better tissue understanding.
- Where: report §12.4.

**Step 19. External calibration and referral (Exp 2, external part)** ✅
- Temperature fitted on R_cal cuts external ECE from 0.205 to 0.099; at 70% coverage, 50.5% of 2+ tiles are referred.
- Where: `experiments_v2/02_uncertainty/`.

**Step 20. External attention and clinical cost (Exps 1 and 3, external part)** ✅
- Correct external 1+ calls attend to glass and pale areas (background ratio 1.07–1.23): a likely shortcut.
- stain_p99 on 224 px EfficientNetV2B0 cuts negatives-called-3+ from 758 to 64 but sends 61% of negatives to 2+ and
  misses 36 2+ tiles; the FYP2 ResNet50 has the lowest external cost (0.54) in 100% of draws.
- Side finding: an 8-bit round trip of calibrated images changes 1.4% of predictions; calibrated images must stay
  float32.

**Step 21. AdaBN — unsupervised BatchNorm re-estimation (Exp 5)** ✅
- Better than raw on all 5 networks (test balanced accuracy +0.06 to +0.23, every CI above 0).
- Against HARMONISE it is network-dependent (−0.13 for MobileNetV3Small to +0.33 for the 448 px model, which it rescues
  from 0.186 to 0.516). Best unsupervised external balanced accuracy on any network ≈ 0.52–0.53.
- Where: `experiments_v2/05_adabn/`.

**Step 22. Few-shot supervised adaptation (Exp 6)** ✅
- 5 labelled tiles per class, full fine-tuning: external test balanced accuracy **0.782 ± 0.053** (QWK 0.918); 20 per
  class: **0.845 ± 0.048** (QWK 0.950). Zero-shot 0.301, HARMONISE 0.503, AdaBN 0.522.
- Partial fine-tuning is non-monotone in k.
- Where: `experiments_v2/06_fewshot/`.

### Stage E — Deployment and edge (next semester)

**Step 23. Model export and precision accuracy gate** ✅
- FP32 22.4 MiB, FP16 11.3 MiB, INT8 6.8 MiB.
- FP16 = FP32 = Keras (agreement 1.000; test acc 0.975). INT8 agreement 0.928 on 727 val tiles (gate 0.95) and test 1+
  recall 0.855 vs 0.960 → **INT8 rejected** (speed reference only).
- Where: `experiments_v2/08_edge/outputs/precision_accuracy.csv`, Figure 8.

**Step 24. Raspberry Pi 4B deployment** ✅
- Runtime `ai_edge_litert`; FP16 448 px median 761.8 / 437.0 / **336.5 ms** at 1 / 2 / 4 threads; INT8 104.1 ms at
  4 threads (first real measurement; the earlier 122.9 ms had no measurement behind it).
- SoC ≤ 62.8 °C, `throttled=0x0` with an official-rating supply (before the supply change: under-voltage flagged).
- FP32/FP16 agree with Keras on 120/120 reference tiles; INT8 109/120 (and only 114/120 with Mac INT8: platform-
  dependent).
- Where: `deployment/PI4_BENCHMARK_REPORT.md` §0, `experiments_v2/08_edge/outputs/pi4/`.

**Step 25. Android app v1.0 → v1.1** ✅
- v1.0 resized 1024 → 448 in one step (differs from training); v1.1 ports Pillow's fixed-point resampler
  (`PilResample.kt`), bit-identical to PIL.
- v1.1 also: INT8 labelled "speed test only" with a warning; ASCO/CAP texts incl. HER2-low; removed unsupported
  "big.LITTLE affinity", "Neural Engine" and "FPS" claims; disclaimer about the external-domain failure; dark-mode fix
  (option labels were invisible on the Realme); research validation mode.
- Where: `android_app/`, `deployment/apk/HER2_Mobile_Edge.apk`, Figures 9–10.

**Step 26. On-device accuracy verification** ✅
- The full BCI test split (758 original 1024 px tiles) run through the app's own path on both phones:
  **758/758 agreement with Keras**, accuracy 0.975, bal. acc 0.973, QWK 0.959 (max |Δp| 0.018 Samsung, 0.068 Realme).
- Where: `experiments_v2/08_edge/outputs/ondevice_<device>/summary.json`.

**Step 27. Smartphone latency benchmarks** ✅
- Official `benchmark_model`, 10 warm-up + 100 timed runs, 1/2/4/8 threads, no core pinning.
- Deployed FP16 448 px: Samsung Galaxy M35 5G **81.4 ms** (4 threads); Realme 6 **139.1 ms** (8 threads).
- In the app, the exact resize adds 145–211 ms per 1024 px tile.
- Where: `deployment/MOBILE_BENCHMARK_REPORT.md`, Figure 7.

**Step 28. Thermal behaviour** ✅
- Phones: battery 29.6–32.0 °C during all runs (CPU zones need root; Android has no throttle flag, so throttling is
  not observable). Pi: see Step 24.

**Step 29. Cross-platform comparison** ✅
- Phones are 2.4–4.1× faster than the Pi; FP16 is the right format; INT8 needs quantisation-aware training first.
- Where: `deployment/CROSS_PLATFORM_EDGE_ANALYSIS.md`.

**Step 30. Smartphone power and energy per inference** ✅ (10 Nov 2026) — see §4.

### Stage F — New experiments and future work (next semester)

**Step 31. E6 — resolution effect on a heavier network (DenseNet121)** ✅ (11 Nov 2026; rule not met) — see §4.
**Step 32. E7 — native-resolution input (MobileNetV3Small, 896 px)** ✅ (11 Nov 2026; rule not met) — see §4.
**Step 33 onward** — the planned work in §6.

---

## 4. Work started on 10 November 2026 (after the main report)

### Step 30 — smartphone power (`experiments_v2/10_power/`)

- **What.** Whole-phone battery power on the Samsung Galaxy M35 5G and Realme 6 while the deployed 448 px FP16 model runs
  continuously (4 threads and 1 thread, 300 s each), against a screen-on idle baseline before and after. Read from
  each phone's own battery fuel gauge (no root, no external meter); USB unplugged; screen on at fixed brightness.
- **Status: complete** (10 Nov 2026, 22:28–22:55; one run per configuration). Next-semester material (Stage E).
- **Results** (deployed 448 px FP16; whole phone, screen on):

  | | Samsung Galaxy M35 5G | Realme 6 |
  |---|---|---|
  | Idle, screen on | 0.74 W | 0.75 W |
  | 4 threads: power (± SE) / added by inference | 4.75 W (± 0.41) / 4.01 W | 4.64 W (± 0.05) / 3.89 W |
  | 4 threads: energy per tile, total / added | 0.47 J / 0.39 J | 0.73 J / 0.61 J |
  | 1 thread: energy per tile, total / added | 0.69 J / 0.49 J | 0.84 J / 0.62 J |
  | Sustained vs short-benchmark median latency, 4 threads | 99.1 vs 81.4 ms (+22%) | 151.6 vs 142.6 ms (+6%) |

- **Findings.** Inference adds about 4 W at 4 threads; 4 threads costs no more energy per tile than 1 thread and
  less in total. Under 5 minutes of continuous load on battery the Samsung is about 20% slower than in the short
  benchmark (cause not established).
- **Limits.** Whole-device power including the screen, from the phone's own fuel gauge, not an external meter. The
  Realme gauge updates about once a second (SE 1–2%); the Samsung's only about every 30 s (SE about 9%). The charge
  counter was too coarse to use. Details: `experiments_v2/10_power/README.md`.

### Steps 31–32 — E6 and E7 (`experiments_v2/09_resolution_e6_e7/`)

- **Question.** E1 showed that 448 px input is a real improvement for EfficientNetV2B0, and E2 did not confirm it for
  MobileNetV3Small. E6 asks whether it holds for a heavier network; E7 whether going closer to the native 1024 px helps
  further. Both were pre-planned in `NEXT_EXPERIMENTS.md` (full FYP3 repository) with their decision rule fixed in advance: an effect counts
  only if the paired 95% fragment-bootstrap CI of Δ val QWK excludes 0.
- **E6:** DenseNet121 CE at 448 px batch 4 (`hires`) vs 224 px batch 4 (`lowbatch`); existing reference
  `DenseNet121__ce` (224 px, batch 16: val QWK 0.854, test acc 0.942).
- **E7:** MobileNetV3Small CE at 896 px from the native 1024 px tiles, batch 2 (`hires1024`), vs `__ce__hires` (448 px)
  and `__ce__lowbatch`.
- **Method.** Unchanged FYP3 training code (`src/train_variant.py`), seed 42, validation-only selection; outputs
  isolated in `experiments_v2/09_resolution_e6_e7/runs/`.
- **Status: complete** (10 Nov 22:25 → 11 Nov 16:12, M1 Mac; one seed per arm). Next-semester material.
- **Results** (BCI grouped split; paired 95% fragment-cluster bootstrap CI; source
  `experiments_v2/09_resolution_e6_e7/outputs/e6e7_table.csv`, `e6e7_paired.csv`):

  | | Run | Val acc / QWK | Test acc / QWK |
  |---|---|---|---|
  | E6 | DenseNet121 448 px, batch 4 | 0.919 / 0.861 | 0.958 / 0.949 |
  | E6 | DenseNet121 224 px, batch 4 (control) | 0.893 / 0.822 | 0.939 / 0.915 |
  | E7 | MobileNetV3Small 896 px, batch 2 | 0.941 / 0.903 | 0.962 / 0.945 |
  | E7 | MobileNetV3Small 448 px, batch 8 (E2 run, comparator) | 0.898 / 0.865 | 0.908 / 0.874 |

  | Pre-registered pair | Δ val QWK [95% CI] | Rule (CI excludes 0) | Report-only: Δ test QWK |
  |---|---|---|---|
  | E6: 448 vs 224 px, both batch 4 | +0.038 [−0.008, +0.078] | **not met** | +0.033 [+0.004, +0.078] |
  | E7: 896 vs 448 px | +0.038 [−0.005, +0.090] | **not met** | +0.071 [+0.033, +0.135] |

- **Findings.**
  - Higher resolution was ahead by about +0.04 val QWK in both comparisons, but neither CI excludes 0, so neither counts.
  - Val accuracy (E6, E7) and val balanced accuracy (E7) have CIs above 0, and so do the test QWKs. These are secondary
    or report-only metrics.
  - Correct wording: "a consistent, small, unconfirmed advantage". Not "the resolution effect generalises".
  - The primary model is unchanged. EfficientNetV2B0 448 px (val QWK 0.935) beats every E6/E7 run on validation.
- **Limits.**
  - One seed per arm; training variance is not covered.
  - DenseNet121 448 px (best epoch 41/45) and MobileNetV3Small 896 px (40/45) ended at the 45-epoch cap.
  - E7's 448 px comparator early-stopped at epoch 25 (best 15) and used batch 8 rather than 2.
  - BCI only; no external evaluation; the 896 px model was not timed on any device.
  - Comparisons against the 224 px MobileNetV3Small runs favour 896 px clearly, but they were not pre-registered.
    Treat them as a hypothesis for the multi-seed repeats (Step 34).
  - Details: `experiments_v2/09_resolution_e6_e7/README.md`.

---

## 5. Must-disclose list (any report)

1. **The external test split is not untouched.** It was examined in earlier diagnostics and by HARMONISE-1. Every
   external number is an exploratory second look, with all choices fixed on development data first.
2. **No locked data were used for fitting.** No training, adaptation, threshold, temperature or model choice used the
   BCI test split or the external test split.
3. **One training seed per configuration.** CIs cover tile/fragment sampling, not training variance. Three runs ended
   at the 45-epoch cap.
4. **BCI labels are slide-level** (2+ agreement with refined patch labels only 70–74%); BCI fragments are reconstructed,
   not verified slides; the external set has no slide IDs (optimistic CIs).
5. **Tile level only.** No slide or patient aggregation; no pathologist review of errors or Grad-CAM maps.
6. **Clinical costs are illustrative**, not validated utilities.
7. **Edge limits.** Phone temperatures are battery temperatures; no throttling flag exists on Android; phone power is
   whole-device battery power from each phone's fuel gauge, one run per configuration (§4); Pi power was not measured; on-device accuracy covers BCI tiles
   only. The app's TFLite 2.16.1 library is not 16 KB page-aligned (both test phones use 4 KB pages).
8. **Mac latency is not reproducible** under desktop load (two runs differ by up to 2.4×).
9. **Earlier reports contained errors** that are corrected and itemised in `experiments_v2/REPORT_CORRECTIONS_v2.md`
   (e.g. an unmeasured Pi INT8 latency, phone "throttle flags", power claims without measurement, the DAB-vs-
   haematoxylin mechanism, the dataset source).
10. **Not a medical device.** Research software; must not guide treatment.

---

## 6. Next semester: planned work in pipeline order (Step 33 onward)

| Order | Stage | Work | Why | Effort / needs |
|---|---|---|---|---|
| 33 | A Data | Obtain slide-identified data (or slide IDs for HER2-IHC-40x) and a **third laboratory** held out completely | Unbiased external test; slide-level CIs | Data access |
| 34 | B Training | Multi-seed repeats (≥ 3 seeds) of the primary model and key controls | Training variance is currently unmeasured | ~4 h per 448 px seed on the M1 |
| 35 | B Training | Multi-seed repeat of the E6 / E7 pairs (Steps 31–32), with matched epochs and batch | Both showed +0.04 val QWK, unconfirmed with one seed | ~10 h (DenseNet121 448 px) and ~5 h (896 px) per seed |
| 36 | B Training | Stain/brightness/blur augmentation aimed at the measured weaknesses | Brightness ±15% and defocus are the biggest in-domain risks | ~4 h per run |
| 37 | C Evaluation | Pathologist review of error cases and Grad-CAM maps | Clinical validity of the explanations | Pathologist time |
| 38 | C Evaluation | Slide-level aggregation of tile scores | Clinical scoring is per slide | Needs Step 33 |
| 39 | D Cross-lab | Site-acceptance protocol: small labelled calibration set + fine-tuning + 2+ referral, validated on the third laboratory | Few-shot is the only method that worked | Needs Step 33 |
| 40 | E Deployment | Image-quality gate in the app (focus and brightness) before inference | The 448 px model is defocus-sensitive | App work |
| 41 | E Deployment | Quantisation-aware training so INT8 passes the 0.95 gate | INT8 is 3–4× faster | Training + re-validation |
| 42 | E Deployment | GPU / NNAPI delegates on the phones, with the same accuracy gate | Further speed-up | Re-validation with `ondevice_validation.py` |
| 43 | E Deployment | LiteRT upgrade for 16 KB page alignment, then re-run on-device validation | Required on newer Android devices | App rebuild |
| 44 | E Deployment | Raspberry Pi 4B power measurement | Pi energy per tile is unknown | USB-C inline power meter (e.g. FNIRSI FNB58) |
| 45 | E Deployment | Phone power with an external meter or an in-app sampler, repeated runs, and the cause of the sustained-load slowdown | Validate the fuel-gauge estimates (§4) | Meter or app change |

---

## 7. This semester's presentation: slides and evidence (Steps 1–14)

Every slide below has a stored figure or table behind it. `FYP3_PRESENTATION_OUTLINE.md` (16 slides) covers both
semesters; its Slides 1–7 and the in-domain parts of Slide 15 correspond to this semester.

| # | Slide | Step | Figure / table to use |
|---|---|---|---|
| 1 | Title, scope, research questions | — | — |
| 2 | Clinical background: HER2 scoring (0/1+/2+/3+, ASCO/CAP), why 1+/2+ is hard | — | Report §2 |
| 3 | Motivation: FYP2 synthetic ~100% → does it hold on real tissue? | — | Report §2–3 |
| 4 | Datasets and label audit | 1 | Report §5 table; label-audit numbers in Step 1 |
| 5 | Leakage-controlled split (691 fragments; 3,388/727/758) | 2 | Mermaid diagram in report §5; §6.1 |
| 6 | Preprocessing and stain characterisation | 4–5 | `final_report_figures/fig3_stain_optical_density_distributions.png` |
| 7 | Baseline comparison of 9 backbones (+ CORN) | 6 | `final_report_figures/fig1_bci_model_comparison.png`, Table 1 |
| 8 | Resolution vs batch size (E1/E2): the main in-domain result | 7–8 | `final_report_figures/fig2_paired_effects_forest.png`, Tables 2–3 |
| 9 | What did not help: synthetic pretraining/mixing, class weights, label smoothing, CORN | 7, 9 | Tables 4–5; Fig. 2 (E4/E5 rows) |
| 10 | Why it works: backbone ablation | 10 | `experiments_v2/07_ablation/outputs/fig_ablation.png` |
| 11 | Clinical error profile and cost | 11 | `experiments_v2/01_clinical_risk/outputs/fig_risk_heatmaps.png`, `fig_error_taxonomy.png` (BCI panels) |
| 12 | Confidence and referral | 12 | `experiments_v2/02_uncertainty/outputs/fig_uncertainty.png` (show the BCI curves; the external curves are next semester) |
| 13 | Where the model looks | 13 | `experiments_v2/03_gradcam/outputs/fig_gradcam_bci448_per_class.png`, `fig_gradcam_bci448_2plus.png`, `fig_cam_statistics.png` |
| 14 | Robustness (brightness, defocus) | 14 | `experiments_v2/04_robustness/outputs/fig_robustness.png` (its y-axis is macro-F1; the numbers quoted in Step 14 are balanced accuracy from `robustness_results.csv`) |
| 15 | Conclusions, limitations (§5 items 3–6, 10), and next semester's plan (Stages D–F) | — | §2, §6 of this file |

---

## 8. Where everything is

| What | Location |
|---|---|
| Full technical report (all completed steps) | `FYP3_FINAL_REPORT.md` |
| Presentation outline | `FYP3_PRESENTATION_OUTLINE.md` |
| Extended experiments 1–8 (audit table, results, limitations) | `experiments_v2/README.md` and each `experiments_v2/0N_*/README.md` |
| Corrections register | `experiments_v2/REPORT_CORRECTIONS_v2.md` |
| Edge reports | `deployment/MOBILE_BENCHMARK_REPORT.md`, `deployment/CROSS_PLATFORM_EDGE_ANALYSIS.md`, `deployment/PI4_BENCHMARK_REPORT.md` |
| Figures 1–10 and their data sources | `final_report_figures/README.md` |
| Phone power (Step 30) | `experiments_v2/10_power/` |
| E6 / E7 (Steps 31–32) | `experiments_v2/09_resolution_e6_e7/` |
| Reproduction | `FYP3_REPRODUCTION_GUIDE.md` |
