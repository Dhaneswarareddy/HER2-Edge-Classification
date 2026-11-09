# FYP3 final model selection

30 Oct 2026. Evidence comes from stored run outputs, independently recomputed (`final_reports/`). Two configurations are selected for **different purposes** and must never be combined.

| | Primary | Experimental |
|---|---|---|
| Purpose | Scoring BCI-like tiles | Studying cross-domain calibration on HER2-IHC-40x |
| Model | EfficientNetV2B0, CE, 448 px (`EfficientNetV2B0__ce__hires`) | EfficientNetV2B0, CE, 224 px (`EfficientNetV2B0__ce`) |
| Format | FP16 TFLite, 11.27 MB | FP16 TFLite, 11.27 MB (new export) |
| Stain calibration | **None, and it must not be applied** | **Frozen `stain_p99`:** g_h 0.2545, g_d 0.5045 |
| Status | Selected model | Research configuration, not a product candidate |

Neither is clinically validated, and neither has been benchmarked on a Raspberry Pi.

## Disclosures

- **Genuine WSI identifiers are unavailable** in BCI and HER2-IHC-40x, so there are no per-slide or per-patient results.
- **External test labels were previously exposed** through diagnostics, and the split was evaluated once for HARMONISE-1. It is not a pristine test.
- **Borderline HER2 performance remains weak** across domains, and on BCI the 2+ result reflects slide-level labels.
- **All latency figures are Mac CPU.**

## Primary model

**Selection rule:** highest BCI validation QWK, decided on validation only, with paired CIs. The test split is reported, not used.

| Evidence | Value |
|---|---|
| BCI validation | Accuracy 0.950, QWK 0.935 |
| BCI test (report-only) | Accuracy 0.975, QWK 0.959, balanced accuracy 0.973 |
| vs 224 px baseline, val QWK | +0.065 [+0.026, +0.108] |
| vs 224 px at equal batch size, val QWK | +0.105 [+0.056, +0.168]; test +0.063 [+0.026, +0.116] |
| Overfitting verdict | Mild: train − test 2.4 points, log-loss gap 0.088 |
| FP16 vs Keras, through the deployment code | 100% class agreement on all 1,485 BCI val and test tiles; largest probability change 0.020 |

| Split | Score | Tiles | Recall | Precision | F1 |
|---|---|---|---|---|---|
| BCI val | 0 | 36 | 0.944 | 0.971 | 0.958 |
| BCI val | 1+ | 172 | 0.936 | 0.947 | 0.942 |
| BCI val | 2+ | 321 | 0.975 | 0.943 | 0.959 |
| BCI val | 3+ | 198 | 0.924 | 0.963 | 0.943 |
| BCI test | 0 | 36 | 0.972 | 0.946 | 0.959 |
| BCI test | 1+ | 173 | 0.960 | 0.982 | 0.971 |
| BCI test | 2+ | 323 | 0.981 | 0.981 | 0.981 |
| BCI test | 3+ | 226 | 0.978 | 0.965 | 0.971 |

**Why not the alternatives:**

| Tier | Run | Params (M) | Val QWK | Val acc | Test QWK | Test acc |
|---|---|---|---|---|---|---|
| heavy | DenseNet121 / ce | 7.04 | 0.854 | 0.908 | 0.910 | 0.942 |
| heavy | ConvNeXtTiny / ce | 27.82 | 0.844 | 0.902 | 0.898 | 0.926 |
| lite | MobileNetV3Small / ce / hires | 0.94 | 0.865 | 0.898 | 0.874 | 0.908 |
| lite | MobileNetV3Large / corn | 3.00 | 0.825 | 0.894 | 0.923 | 0.937 |
| mid | EfficientNetV2B0 / ce / hires | 5.92 | 0.935 | 0.950 | 0.959 | 0.975 |
| mid | EfficientNetV2B0 / ce | 5.92 | 0.869 | 0.895 | 0.915 | 0.926 |

- **Other 224 px baselines:** lower validation QWK, and tier winners are not separable from their runners-up.
- **CORN, class weights, label smoothing, GAN pretraining, mixed training:** no gain on validation.
- **INT8:** rejected. Agreement 0.928 is below the 0.95 threshold, and test accuracy falls 4.2 points.
- **FP32:** no accuracy benefit over FP16, at twice the size.

**Limits of this selection:**
- It is best **on BCI only**. On HER2-IHC-40x it reaches 0.435 accuracy with 1+ recall 0.11 and 2+ recall 0.01 (test), and it ranks tiles worse than the 224 px model.
- Only 2 of 9 architectures were tested at 448 px, so it is the best *tested*, not the best possible.
- One seed. MobileNetV3Small did not confirm the resolution effect.
- BCI's test split has unusually strong 3+ staining, so validation is the more cautious estimate.
- Mac latency is 75.5 ms. Pi feasibility at 448 px is unknown.

**Borderline evidence on BCI.** Against third-party patch labels (protocol undocumented):

| Split | Slide label | Tiles with patch label | Patch label agrees with slide label | Model agrees with slide label | Model agrees with patch label |
|---|---|---|---|---|---|
| BCI val | 0 | 14 | 0.643 | 0.929 | 0.714 |
| BCI val | 1+ | 120 | 0.950 | 0.975 | 0.925 |
| BCI val | 2+ | 234 | 0.697 | 0.991 | 0.697 |
| BCI val | 3+ | 121 | 0.959 | 0.983 | 0.959 |
| BCI test | 0 | 21 | 0.571 | 1.000 | 0.571 |
| BCI test | 1+ | 129 | 0.969 | 0.992 | 0.961 |
| BCI test | 2+ | 261 | 0.736 | 0.996 | 0.732 |
| BCI test | 3+ | 173 | 0.936 | 1.000 | 0.936 |

On slide-labelled 2+ tiles the model follows the slide label almost always, while the patch labels agree with the slide only 70–74% of the time. This does not show reliable tile-level reading of 2+.

## Experimental configuration

**Why this one:** `EfficientNetV2B0__ce` was HARMONISE-1's pre-registered primary model and is the only network that detects 2+ at all once calibrated. It was **not** chosen by test results.

| Model | Split | Arm | Acc | Bal. acc | Macro-F1 | QWK | Recall 0/1+/2+/3+ | Precision 2+ | F1 1+ | F1 2+ |
|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNetV2B0 224 | dev | raw | 0.367 | 0.295 | 0.204 | 0.082 | 0.01 / 0.17 / 0.00 / 1.00 | 0.00 | 0.26 | 0.00 |
| EfficientNetV2B0 224 | dev | stain_p99 | 0.497 | 0.517 | 0.429 | 0.525 | 0.26 / 0.16 / 0.67 / 0.97 | 0.11 | 0.23 | 0.18 |
| EfficientNetV2B0 224 | test | raw | 0.448 | 0.301 | 0.235 | 0.108 | 0.01 / 0.18 / 0.01 / 1.00 | 0.01 | 0.27 | 0.01 |
| EfficientNetV2B0 224 | test | stain_p99 | 0.571 | 0.503 | 0.448 | 0.593 | 0.30 / 0.18 / 0.57 / 0.97 | 0.09 | 0.24 | 0.16 |
| MobileNetV3Small 224 | dev | raw | 0.379 | 0.288 | 0.198 | 0.117 | 0.13 / 0.03 / 0.00 / 1.00 | 0.00 | 0.05 | 0.00 |
| MobileNetV3Small 224 | dev | stain_p99 | 0.717 | 0.539 | 0.507 | 0.799 | 0.93 / 0.14 / 0.09 / 1.00 | 0.25 | 0.23 | 0.14 |
| MobileNetV3Small 224 | test | raw | 0.508 | 0.337 | 0.301 | 0.210 | 0.19 / 0.15 / 0.00 / 1.00 | 0.00 | 0.24 | 0.00 |
| MobileNetV3Small 224 | test | stain_p99 | 0.755 | 0.528 | 0.516 | 0.802 | 0.89 / 0.14 / 0.09 / 0.99 | 0.29 | 0.22 | 0.14 |
| MobileNetV3Large CORN 224 | dev | raw | 0.329 | 0.254 | 0.147 | 0.109 | 0.01 / 0.00 / 0.00 / 1.00 | 0.00 | 0.00 | 0.00 |
| MobileNetV3Large CORN 224 | dev | stain_p99 | 0.543 | 0.429 | 0.404 | 0.409 | 0.44 / 0.22 / 0.07 / 0.99 | 0.07 | 0.31 | 0.07 |
| MobileNetV3Large CORN 224 | test | raw | 0.423 | 0.261 | 0.180 | 0.100 | 0.02 / 0.03 / 0.00 / 1.00 | 0.00 | 0.05 | 0.00 |
| MobileNetV3Large CORN 224 | test | stain_p99 | 0.600 | 0.431 | 0.424 | 0.405 | 0.39 / 0.35 / 0.01 / 0.97 | 0.01 | 0.43 | 0.01 |
| DenseNet121 224 | dev | raw | 0.332 | 0.255 | 0.136 | 0.025 | 0.02 / 0.00 / 0.00 / 1.00 | 0.00 | 0.00 | 0.00 |
| DenseNet121 224 | dev | stain_p99 | 0.627 | 0.447 | 0.404 | 0.758 | 0.96 / 0.01 / 0.04 / 0.78 | 0.08 | 0.02 | 0.05 |
| DenseNet121 224 | test | raw | 0.419 | 0.260 | 0.176 | 0.032 | 0.01 / 0.04 / 0.00 / 0.99 | 0.00 | 0.08 | 0.00 |
| DenseNet121 224 | test | stain_p99 | 0.624 | 0.410 | 0.382 | 0.765 | 0.94 / 0.00 / 0.00 / 0.70 | 0.00 | 0.00 | 0.00 |
| EfficientNetV2B0 448 | dev | raw | 0.344 | 0.266 | 0.259 | 0.350 | 0.20 / 0.02 / 0.04 / 0.80 | 0.01 | 0.04 | 0.01 |
| EfficientNetV2B0 448 | dev | stain_p99 | 0.149 | 0.246 | 0.153 | -0.027 | 0.04 / 0.09 / 0.63 / 0.22 | 0.06 | 0.11 | 0.11 |
| EfficientNetV2B0 448 | test | raw | 0.435 | 0.290 | 0.318 | 0.432 | 0.24 / 0.11 / 0.01 / 0.80 | 0.00 | 0.15 | 0.00 |
| EfficientNetV2B0 448 | test | stain_p99 | 0.104 | 0.186 | 0.108 | -0.155 | 0.05 / 0.15 / 0.47 / 0.08 | 0.04 | 0.13 | 0.08 |

| Finding | Evidence |
|---|---|
| Endpoint met | Balanced accuracy 0.301 → 0.503 on test (Δ +0.202; CI optimistic) |
| FP16 vs Keras | 99.6% class agreement on 2,402 development tiles; disagreements examined were near-ties |
| Stable to tile sampling | CV 0.042 (g_d), 0.030 (g_h) at n = 1,500; balanced accuracy range 0.005 |
| Depends on class mix | At 5% 3+, g_d rises to 0.75 and balanced accuracy falls to 0.41 |
| 1+ remains poor | Recall 0.18 on test |
| 2+ is imprecise | Recall 0.57, precision 0.09 on test |
| Confidence cannot flag safely | At 50% target coverage, 76% of 2+ tiles would be referred |
| Unsafe on BCI | Balanced accuracy 0.875 → 0.407 |

**MobileNetV3Small** scores higher overall (test accuracy 0.755, QWK 0.80) but finds only 9% of 2+, so its headline is misleading.

**Do not generalise this configuration.** The gains belong to one dataset. Another site would need its own unlabelled calibration of about 1,000 tiles with a sensible class mix, and that has not been tested.

## Rejected combinations

| Combination | Why rejected |
|---|---|
| 448 px + stain calibration | Collapses: test balanced accuracy 0.186, 3+ recall 0.08. The haematoxylin gain is the cause |
| 448 px + refitted gains | Same collapse: gains are nearly identical across resolutions |
| 448 px + calibration + scale | 3+ recall falls to 0.70; fails the pre-set rule |
| Stain calibration on BCI tiles | Balanced accuracy 0.875 → 0.407 |
| DenseNet121 + calibration | QWK 0.76 with 1+ and 2+ recall near zero |

## What would change this selection

- Slide identifiers, or a multi-site external set.
- A 448 px run of a heavy architecture, or a second seed.
- A Pi benchmark showing 448 px is too slow. FP16 at 224 px would then be the fallback to test, with lower BCI accuracy.
