# FYP3: the FYP2 pipeline reproduced on real HER2 IHC (BCI)

> **Stale (note added 10 Nov 2026).** This report predates the E1–E5 runs, HARMONISE and `experiments_v2`; it covers 16 of the 20 grouped runs. Use `FYP3_FINAL_REPORT.md` and `experiments_v2/README.md` for current results (`experiments_v2/REPORT_CORRECTIONS_v2.md`, item 25).


A separate experiment. FYP2 remains the main project and was not modified; its results and weights were only read.

## 0. Findings (final, 30 Oct 2026)

**Status.** All runs are finished and evaluated (grouped split):
- 9 CE baselines;
- CORN for the 3 tier winners;
- 3 improvement variants on MobileNetV3Small, plus a confirmation run on EfficientNetV2B0;
- the full network analysis and a haematoxylin-cue test;
- 2 reference runs on the official split.

Each statement is marked **confirmed** (measured, with the evidence section named) or **hypothesis** (consistent
with the data, not tested). Decisions used validation only, and every comparison between two runs is a paired
difference on the same tiles with a 95% fragment-cluster bootstrap CI.

### Performance gap

- **Confirmed.** The FYP2 pipeline, unchanged, reaches 0.99–1.00 test accuracy on FYP2's GAN images. On BCI's real IHC
  tiles it reaches 0.91–0.95 test and 0.85–0.91 validation accuracy (sections 3–4).
  - Validation QWK is 0.76–0.87 on BCI, against 0.997–1.000 on FYP2.
- **Confirmed.** No tier winner is separable from its runner-up on validation (section 3). Δ val QWK [95% CI]:
  - DenseNet121 vs ConvNeXtTiny: +0.011 [−0.026, +0.043];
  - EfficientNetV2B0 vs EfficientNetB0: +0.024 [−0.017, +0.059];
  - MobileNetV3Large vs MobileNetV2: +0.016 [−0.079, +0.100].

  The tier ranking on BCI is therefore not reliable.

### Why BCI is harder: the data (no trained CNN involved)

1. **Confirmed: BCI's classes are far less separable before any task training** (section 6f).
   - A linear probe on frozen ImageNet EfficientNetB0 features reaches 0.978 accuracy / QWK 0.991 on FYP2, against
     0.765 / 0.698 on BCI.
   - Class separation (silhouette) is 0.168 vs 0.006.

   Most of the gap is a property of the images, not of the training pipeline.
2. **Confirmed: real stain separates the classes far less than the GAN images do** (section 6a).
   - Median tile DAB against each tile's own background, for 0 / 1+ / 2+ / 3+:
     - BCI: 0.08 / 0.10 / 0.12 / 0.25;
     - FYP2: 0.09 / 0.09 / 0.46 / 0.94.
   - A random forest on stain alone reaches macro-F1 0.49 on BCI, against 0.82 on FYP2.
3. **Confirmed: BCI labels describe the slide, not the tile** (section 6b).
   - The patch-level reference labels disagree with the slide label on 16–18% of held-out tiles.
   - On tiles where the two agree, the CE baselines reach 0.94–0.98 accuracy on validation and 0.97–1.00 on test.
4. **Confirmed: the FYP2 pipeline throws away resolution that BCI needs** (section 7).
   - FYP2's images are natively 256 px, so its 256 → 224 px pipeline loses nothing there. BCI's tiles are 1024 px, so the
     same pipeline keeps 1 pixel in 16.
   - Doubling the resolution (512 px source, 448 px input) is the one improvement that passed its pre-set rule, and it
     was confirmed on a second architecture (finding 9).
5. **Confirmed: scan type is associated with the label** (section 6c). Grey-background scans make up 25% / 44% / 52% / 68%
   of class 0 / 1+ / 2+ / 3+ tiles. As a shortcut it is minor (finding 7).
6. **Confirmed (measurement): the standard stain measure is distorted on BCI.** Optical density against 255 counts the
   grey glass as stain. Every stain statement here uses the background-corrected measure.

### What the networks learned (sections 6e–6h)

7. **Confirmed: the BCI networks read the brown stain, and scan brightness matters little.**
   - Grad-CAM attention is concentrated on DAB-rich pixels (ratio 1.05–1.15 on BCI, 1.13–1.40 on FYP2) and away from
     glass (background ratio < 1).
   - Scaling DAB moves the expected score in the right direction for every network.
   - Within a class, the error (expected score − slide label) rises with the tile's own DAB in all 16 BCI runs. The
     coefficients are +0.04 to +0.14 per SD, and every CI excludes 0. Many "errors" are therefore tiles whose own
     stain disagrees with their slide's label.
   - A 15% change in scan brightness shifts the expected score by at most 0.075 (median 0.056), below the pre-set
     0.10 rule. The observational brightness coefficient is small (−0.003 to −0.089 per SD).
8. **Confirmed: neither dataset's networks transfer to the other.**
   - Balanced accuracy (chance 0.25) is 0.30–0.46 for FYP2 → BCI and 0.28–0.51 for BCI → FYP2.
   - 8 of 9 FYP2-trained networks label 71–81% of BCI tiles "1+"; only 23% are.
   - On BCI test images, CKA (representation similarity) between the FYP2-trained and BCI-trained networks falls to
     0.65–0.85 at the last layer (0.45–0.86 on FYP2 images).

   The two training sets teach different decision rules on top of similar early features.
   - **Not confirmed: "FYP2 networks read low haematoxylin as 1+"** (section 6h). Bringing haematoxylin to the other
     group's level moved the median network's 1+ share by +16 points (×0.2). The pre-set rule needed ≥ 20 in both
     directions. Adding haematoxylin to BCI tiles removed most 1+ answers but did not raise balanced accuracy (−0.04).
   - **Hypothesis (untested):** FYP2 networks fail on BCI mainly because their DAB thresholds are calibrated to the
     GAN's exaggerated stain. BCI's 2+/3+ tiles carry about as much DAB as the GAN's 0/1+ images (finding 2).

### Improvements (section 7)

9. **Confirmed: higher input resolution helps.**

   | Model | Δ val QWK [95% CI] | Δ test QWK [95% CI] |
   |---|---|---|
   | MobileNetV3Small (screen) | +0.102 [+0.031, +0.173] | −0.014 [−0.063, +0.038] |
   | EfficientNetV2B0 (pre-registered confirmation) | **+0.065 [+0.026, +0.108]** | +0.044 [+0.009, +0.089] |

   - EfficientNetV2B0 at 448 px reaches 0.950 val / 0.975 test accuracy and 0.935 / 0.959 QWK. This is the best BCI
     result; the 224 px baseline reaches 0.895 / 0.926 accuracy.
   - Caveat: the batch size dropped from 32 to 8 with the resolution (memory), so the two are not separated.
   - Training cost rises about 5× (252 vs 49 min).
10. **Confirmed: class weights and label smoothing do not help.**
    - Class weights: Δ val QWK +0.023 [−0.040, +0.088].
    - Label smoothing: +0.004 [−0.045, +0.054]. It also lowers validation recall for score 0 (−0.11, CI excludes 0).
11. **Confirmed: CORN is not better than CE on BCI.**
    - Δ val QWK: DenseNet121 −0.013 [−0.064, +0.032]; EfficientNetV2B0 −0.039 [−0.095, +0.012]; MobileNetV3Large
      +0.025 [−0.004, +0.056].
    - The selection rule still picks MobileNetV3Large-CORN for the lite tier (higher point estimate); that choice is within noise.
12. **Not run, by pre-set rule:** the illumination and HED-stain augmentations. Their perturbation gates were not met
    (0.056 and 0.010 against 0.10; section 6g).

### Overfitting and evaluation design

13. **Confirmed: on BCI the FYP2 recipe overfits by FYP2's own rule; on FYP2 it does not** (section 5b).
    - Train − test accuracy is 2.2–6.6 points on BCI, against ≤ 0.7 on FYP2.
    - The log-loss gap is 0.05–0.20, against ≤ 0.02.
    - Noise-aware verdict: 8 of 16 runs "overfit", 7 "mild", 1 "none".
    - Early stopping on validation QWK worked: validation is *below* test for every run, so model selection did not overfit validation.
    - With 16–18% of tiles carrying a slide label their own appearance contradicts, a network that fits its training
      tiles well cannot transfer that fit.
    - The 2× resolution runs have the smallest gaps: train − test 2.2–2.5 points, log-loss gap 0.05–0.09.
14. **Confirmed: BCI's official split leaks across slides; the effect on accuracy is a few points** (section 1b).
    - 63% of official test tiles share a slide fragment with official training tiles.
    - On identical tiles, the leaky model is better by +1.3 to +1.9 points on easy tiles (CIs include 0) and +5.5 on harder ones.
15. **Confirmed: one held-out split is imprecise** (section 5c).
    - Test accuracy is 2.4–7.3 points above validation for every one of the 9 baselines.
    - A third of validation errors come from three large slide fragments.
    - Hence the paired, fragment-clustered comparisons throughout.

### Answer to the brief, in one paragraph

The FYP2 pipeline drops from about 100% (GAN images) to 91–95% test accuracy on real BCI IHC tiles. The drop is
explained mainly by the data:
- the classes are barely separable in generic features (frozen-probe accuracy 0.77 vs 0.98);
- real DAB separates 0/1+/2+ weakly;
- 16–18% of tiles carry a slide-level label their own stain contradicts.

The networks trained on each dataset read stain in the right direction, but they learn different decision rules and do
not transfer. One pipeline choice that is harmless on FYP2 does hurt BCI: downsampling its 1024 px tiles to 256 px.
Doubling the resolution was the only justified improvement that held up. It lifts the mid-tier model to 0.975 test
accuracy, with a smaller overfitting gap, at about 5× the training cost.

## 1. Data

- **Dataset.** BCI (Breast Cancer Immunohistochemical image generation), Liu et al., 2022 (arXiv:2204.11425). It has 4,873 paired H&E/IHC 1024×1024 tiles, 3,896 official train and 977 official test. The licence is non-commercial research only; cite the BCI paper. The raw files in `FYP/data/raw/BCI_dataset` were only read.
- **Image type: IHC.** FYP2's images are HER2 IHC (DAB-stained membrane) patches. Its pipeline, augmentation and report are all built around DAB stain intensity. So FYP3 uses `BCI_dataset/IHC`; H&E is not used.
- **Labels: the BCI filename convention** `NNNNN_{train|test}_{0|1+|2+|3+}.png`. Per the BCI README, this is the score of the whole slide the tile was cut from, not of the tile itself. A second, patch-level label exists for 72% of tiles (`BCI_metadata_pairs_refined.csv`, method undocumented). It is used for analysis only and never for training or model selection.
- **Split (main): grouped, leakage-controlled.** This follows FYP2's recipe (`make_split.py`): StratifiedGroupKFold 70/15/15 over all 4,873 tiles. The groups are *slide fragments*: tiles cut side by side from one slide, reconstructed from the continuity of their borders (`analysis_slides.py`, section 1b). FYP2's pHash groups are not used for BCI because they link background-dominated tiles from different slides (section 1b).
- **Split (reference): the official BCI split.** This is the official train/test split, with validation drawn from 15% of official train. It is kept for comparison with published BCI results, but it leaks (section 1b). Two models were trained on it (`results/runs_official/`).
- **Resolution.** Tiles are downsampled once to 256×256 (Lanczos), so the networks see FYP2's geometry: 256 px source → 224 px input.

```
       Score_0  Score_1  Score_2  Score_3  total
split                                           
test        36      173      323      226    758
train      168      809     1499      912   3388
val         36      172      321      198    727
All        240     1154     2143     1336   4873

Leakage check: nearest-neighbour similarity of each eval tile to the TRAIN set
(frozen ImageNet EfficientNetB0 cosine) and accepted edge links across splits
  accepted edge links crossing any split boundary: 0 (of 2763)
  val  ->train  cos max=0.981 p99=0.971 median=0.901
  test ->train  cos max=0.984 p99=0.976 median=0.892
  train->train cos max=0.986 median=0.911
  official-test tiles per grouped split: train 676, val 145, test 156

Refined (patch-level) vs filename (slide-level) label agreement per split: train 0.817, val 0.822, test 0.841
```

## 1b. Slide leakage in the official BCI split

**Method.** BCI has no slide IDs, but tiles cut side by side from one slide share a continuous border. For every tile, the tile whose border best continues its right/bottom edge is found. A link is accepted only if it is mutual, unambiguous (ratio test) and as smooth as the tile's own adjacent pixel columns (`analysis_slides.py`).
- **Validation of the method.** 2,763 links were accepted. 99.96% of them join tiles with the same slide label, against 33% expected for random pairs. 99.9% of the 691 multi-tile fragments are label-pure, and the largest has 99 tiles. The links therefore join neighbouring tiles of the same slide.
- **Confirmed leakage.** 619 of 977 official test tiles belong to slide fragments that also contain official training tiles.
- **pHash grouping (FYP2's rule) does not transfer to BCI.** 39 of 53 pHash groups mix slide labels, even 6-bit pairs agree on label only 50% of the time, and linked tiles are mostly background. These are look-alike backgrounds, not duplicates.

![](figures/figD8_edge_matches.png)

**Effect on accuracy** (`analysis_leakage.py`). MobileNetV3Small and MobileNetV2 were trained on both splits. All 95% CIs come from a cluster bootstrap over slide fragments, because tiles from one fragment are correlated.

| Run | Split design | Held-out set | Tiles | Accuracy [95% CI] | Macro-F1 [95% CI] | QWK [95% CI] |
|---|---|---|---|---|---|---|
| MobileNetV2 / ce | official | val | 585 | 0.926 [0.904, 0.948] | 0.930 [0.905, 0.953] | 0.914 [0.873, 0.947] |
| MobileNetV2 / ce | official | test | 977 | 0.910 [0.890, 0.929] | 0.889 [0.856, 0.918] | 0.859 [0.813, 0.899] |
| MobileNetV2 / ce | grouped | val | 727 | 0.857 [0.823, 0.892] | 0.850 [0.812, 0.886] | 0.785 [0.699, 0.861] |
| MobileNetV2 / ce | grouped | test | 758 | 0.930 [0.897, 0.953] | 0.928 [0.896, 0.949] | 0.924 [0.874, 0.953] |
| MobileNetV3Small / ce | official | val | 585 | 0.899 [0.872, 0.924] | 0.870 [0.821, 0.911] | 0.836 [0.767, 0.896] |
| MobileNetV3Small / ce | official | test | 977 | 0.902 [0.881, 0.922] | 0.885 [0.853, 0.911] | 0.867 [0.826, 0.902] |
| MobileNetV3Small / ce | grouped | val | 727 | 0.854 [0.813, 0.894] | 0.833 [0.787, 0.873] | 0.763 [0.668, 0.842] |
| MobileNetV3Small / ce | grouped | test | 758 | 0.913 [0.873, 0.942] | 0.895 [0.851, 0.923] | 0.888 [0.810, 0.932] |

These headline numbers are not comparable across split designs: each scores different tiles, and difficulty varies strongly between slide fragments (section 5c). The fair test scores both models on the **same tiles**: official test tiles that the grouped split also holds out. Neither model trained on them; only the official-split model trained on their slide neighbours. For tiles in grouped *val*, the grouped model early-stopped on them, which favours it, so that difference is conservative.

| Run | Tiles | Subset | n (fragments) | Acc, official-split model | Acc, grouped-split model | Difference [95% CI] |
|---|---|---|---|---|---|---|
| MobileNetV2 / ce | official test ∩ grouped test | all | 156 (95) | 0.962 | 0.942 | +0.019 [-0.017, +0.062] |
| MobileNetV2 / ce | official test ∩ grouped test | seen fragment | 96 (39) | 0.979 | 0.969 | +0.010 [+0.000, +0.042] |
| MobileNetV2 / ce | official test ∩ grouped val | all | 145 (94) | 0.931 | 0.876 | +0.055 [+0.008, +0.099] |
| MobileNetV2 / ce | official test ∩ grouped val | seen fragment | 101 (51) | 0.970 | 0.901 | +0.069 [+0.029, +0.110] |
| MobileNetV3Small / ce | official test ∩ grouped test | all | 156 (95) | 0.942 | 0.929 | +0.013 [-0.024, +0.056] |
| MobileNetV3Small / ce | official test ∩ grouped test | seen fragment | 96 (39) | 0.969 | 0.990 | -0.021 [-0.056, +0.000] |
| MobileNetV3Small / ce | official test ∩ grouped val | all | 145 (94) | 0.903 | 0.848 | +0.055 [-0.008, +0.114] |
| MobileNetV3Small / ce | official test ∩ grouped val | seen fragment | 101 (51) | 0.941 | 0.881 | +0.059 [-0.011, +0.136] |

**The simple exposure comparison is confounded.** The official-split model scores much better on official test tiles whose fragment was in training:

| Run | Exposure | n | Accuracy | Macro-F1 |
|---|---|---|---|---|
| MobileNetV2 / ce | seen fragment | 603 | 0.942 | 0.922 |
| MobileNetV2 / ce | unseen fragment | 48 | 0.771 | 0.750 |
| MobileNetV2 / ce | unlinked | 326 | 0.871 | 0.854 |
| MobileNetV3Small / ce | seen fragment | 603 | 0.939 | 0.933 |
| MobileNetV3Small / ce | unseen fragment | 48 | 0.812 | 0.803 |
| MobileNetV3Small / ce | unlinked | 326 | 0.847 | 0.826 |

Only multi-tile fragments can be "seen", however, and large fragments are easier for every model. The grouped-split model, which saw none of these fragments, shows the same rise with fragment size:

| Run | Fragment size | % official test seen | official model, official test | grouped model, grouped val | grouped model, grouped test |
|---|---|---|---|---|---|
| MobileNetV2 / ce | 1 (unlinked) | 0% | 0.871 | 0.845 | 0.899 |
| MobileNetV2 / ce | 2-3 | 80% | 0.884 | 0.874 | 0.890 |
| MobileNetV2 / ce | 4-10 | 98% | 0.929 | 0.915 | 0.926 |
| MobileNetV2 / ce | >10 | 100% | 0.966 | 0.818 | 0.984 |
| MobileNetV3Small / ce | 1 (unlinked) | 0% | 0.847 | 0.857 | 0.848 |
| MobileNetV3Small / ce | 2-3 | 80% | 0.898 | 0.839 | 0.853 |
| MobileNetV3Small / ce | 4-10 | 98% | 0.923 | 0.922 | 0.949 |
| MobileNetV3Small / ce | >10 | 100% | 0.959 | 0.813 | 0.988 |

![](figures/figD9_leakage.png)

## 2. Training setup

**Identical to FYP2:** the 9 ImageNet backbones, 224×224 input with preprocessing in the graph, stage 1 frozen (5 epochs at 0.001), stage 2 full fine-tuning (≤ 40 epochs, Adam 0.0001, warm-up and cosine), BatchNorm frozen, dropout 0.3, early stopping on validation QWK (patience 10) with best weights restored, the same augmentation, seed 42, CE and CORN losses.

**Deliberate differences, all forced by the data or the scope:**

1. **Training images are streamed from disk** instead of being held in memory. The training split is twice FYP2's and the Mac has 8 GB. The pixels are identical.
2. **Leakage groups are slide fragments, not pHash twins.** The split recipe is FYP2's; only the grouping unit differs, because the leakage mechanism differs (section 1b).
3. **CE for all 9 models; CORN only for the best CE model per tier (by validation QWK).** In FYP2, CE vs CORN was a tie (3 vs 3 wins of 9), so repeating all 18 runs (~35 h here) was not justified. One paired comparison per tier still answers whether the ordinal loss helps on noisy real labels.
4. **No quantisation or deployment** (out of scope). Selection ties go to the smaller model (FYP2 used quantised latency).
5. **Improvement experiments** (section 6) run on top of the baseline, never instead of it.

## 3. Results on BCI

| Run | Tier | Params (M) | Input | Best ep. / run | Train (min) | Val acc | Val F1 | Val QWK | Test acc | Test F1 | Test QWK | Test bal. acc | Test recall 0/1+/2+/3+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50 / CE | heavy | 23.6 | 224 | 43 / 45 | 108 | 0.902 | 0.892 | 0.829 | 0.949 | 0.936 | 0.932 | 0.934 | 0.89 / 0.92 / 0.95 / 0.97 |
| DenseNet121 / CE | heavy | 7.04 | 224 | 32 / 42 | 131 | 0.908 | 0.901 | 0.854 | 0.942 | 0.927 | 0.910 | 0.922 | 0.86 / 0.93 / 0.96 / 0.94 |
| DenseNet121 / CORN | heavy | 7.04 | 224 | 30 / 40 | 143 | 0.880 | 0.884 | 0.841 | 0.914 | 0.901 | 0.874 | 0.893 | 0.86 / 0.84 / 0.96 / 0.91 |
| ConvNeXtTiny / CE | heavy | 27.82 | 224 | 13 / 23 | 124 | 0.902 | 0.892 | 0.844 | 0.926 | 0.912 | 0.898 | 0.891 | 0.78 / 0.89 / 0.94 / 0.95 |
| EfficientNetB0 / CE | mid | 4.05 | 224 | 31 / 41 | 111 | 0.891 | 0.895 | 0.846 | 0.933 | 0.928 | 0.902 | 0.914 | 0.86 / 0.90 / 0.95 / 0.94 |
| EfficientNetV2B0 / CE | mid | 5.92 | 224 | 31 / 41 | 49 | 0.895 | 0.893 | 0.869 | 0.926 | 0.922 | 0.915 | 0.917 | 0.89 / 0.93 / 0.94 / 0.91 |
| EfficientNetV2B0 / CE + hires | mid | 5.92 | 448 | 38 / 45 | 252 | 0.950 | 0.950 | 0.935 | 0.975 | 0.971 | 0.959 | 0.973 | 0.97 / 0.96 / 0.98 / 0.98 |
| EfficientNetV2B0 / CORN | mid | 5.92 | 224 | 16 / 26 | 30 | 0.862 | 0.860 | 0.831 | 0.906 | 0.894 | 0.892 | 0.900 | 0.89 / 0.87 / 0.92 / 0.92 |
| NASNetMobile / CE | mid | 4.27 | 224 | 28 / 38 | 100 | 0.884 | 0.865 | 0.805 | 0.921 | 0.912 | 0.904 | 0.895 | 0.83 / 0.86 / 0.95 / 0.93 |
| MobileNetV2 / CE | lite | 2.26 | 224 | 33 / 43 | 75 | 0.857 | 0.850 | 0.785 | 0.930 | 0.928 | 0.924 | 0.925 | 0.92 / 0.89 / 0.93 / 0.96 |
| MobileNetV3Large / CE | lite | 3.0 | 224 | 22 / 32 | 76 | 0.871 | 0.860 | 0.800 | 0.908 | 0.879 | 0.878 | 0.884 | 0.83 / 0.83 / 0.94 / 0.92 |
| MobileNetV3Large / CORN | lite | 3.0 | 224 | 45 / 45 | 47 | 0.894 | 0.886 | 0.825 | 0.937 | 0.922 | 0.923 | 0.919 | 0.86 / 0.91 / 0.94 / 0.96 |
| MobileNetV3Small / CE | lite | 0.94 | 224 | 34 / 44 | 23 | 0.854 | 0.833 | 0.763 | 0.913 | 0.895 | 0.888 | 0.886 | 0.81 / 0.88 / 0.93 / 0.93 |
| MobileNetV3Small / CE + cw | lite | 0.94 | 224 | 34 / 44 | 15 | 0.854 | 0.832 | 0.786 | 0.906 | 0.881 | 0.883 | 0.896 | 0.86 / 0.88 / 0.90 / 0.93 |
| MobileNetV3Small / CE + hires | lite | 0.94 | 448 | 15 / 25 | 45 | 0.898 | 0.893 | 0.865 | 0.908 | 0.893 | 0.874 | 0.898 | 0.89 / 0.85 / 0.93 / 0.92 |
| MobileNetV3Small / CE + ls | lite | 0.94 | 224 | 31 / 41 | 16 | 0.864 | 0.833 | 0.767 | 0.905 | 0.878 | 0.873 | 0.868 | 0.75 / 0.86 / 0.92 / 0.94 |

![](figures/figR2_learning_curves.png)

**CE vs CORN** (3 paired models): CORN higher validation QWK on 1, CE on 2. Loss chosen per model: ConvNeXtTiny: CE, DenseNet121: CE, EfficientNetB0: CE, EfficientNetV2B0: CE, MobileNetV2: CE, MobileNetV3Large: CORN, MobileNetV3Small: CE, NASNetMobile: CE, ResNet50: CE.

Paired difference CORN − CE on the same tiles (fragment-cluster CI):

| Model | Δ val QWK [95% CI] | Δ val macro-F1 [95% CI] | Δ test acc [95% CI] | Δ test QWK [95% CI] |
|---|---|---|---|---|
| DenseNet121 | -0.013 [-0.064, +0.032] | -0.017 [-0.052, +0.020] | -0.028 [-0.054, -0.008] | -0.036 [-0.073, -0.004] |
| EfficientNetV2B0 | -0.039 [-0.095, +0.012] | -0.034 [-0.072, +0.002] | -0.020 [-0.041, +0.002] | -0.023 [-0.061, +0.008] |
| MobileNetV3Large | +0.025 [-0.004, +0.056] | +0.025 [+0.000, +0.050] | +0.029 [+0.007, +0.060] | +0.045 [+0.009, +0.094] |

**Best model per tier** (chosen on validation QWK, see `select_models.py`; runs within 0.005 count as equal and the smaller one wins). The last column is the paired validation difference to the next-best run of the tier; a CI that includes 0 means validation cannot separate the two:

| Selection | Tier | Run | Val QWK | Val F1 | Test acc | Test F1 | Test QWK | Runner-up | Δ val QWK vs runner-up [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| FYP2 pipeline (baseline) | heavy | DenseNet121 / ce | 0.854 | 0.901 | 0.942 | 0.927 | 0.910 | ConvNeXtTiny / ce | +0.011 [-0.026, +0.043] |
| FYP2 pipeline (baseline) | mid | EfficientNetV2B0 / ce | 0.869 | 0.893 | 0.926 | 0.922 | 0.915 | EfficientNetB0 / ce | +0.024 [-0.017, +0.059] |
| FYP2 pipeline (baseline) | lite | MobileNetV3Large / corn | 0.825 | 0.886 | 0.937 | 0.922 | 0.923 | MobileNetV3Large / ce | +0.025 [-0.004, +0.056] |
| Best incl. improvements | heavy | DenseNet121 / ce | 0.854 | 0.901 | 0.942 | 0.927 | 0.910 | ConvNeXtTiny / ce | +0.011 [-0.026, +0.043] |
| Best incl. improvements | mid | EfficientNetV2B0 / ce / hires | 0.935 | 0.950 | 0.975 | 0.971 | 0.959 | EfficientNetV2B0 / ce | +0.065 [+0.026, +0.108] |
| Best incl. improvements | lite | MobileNetV3Small / ce / hires | 0.865 | 0.893 | 0.908 | 0.893 | 0.874 | MobileNetV3Large / corn | +0.040 [-0.059, +0.157] |

## 4. FYP2 vs BCI, same pipeline (CE)

FYP2's test set is 360 GAN images, which every model scores almost perfectly. For BCI, both held-out sets are shown, because they differ by several points for the same model (section 5c). The 95% CI is a fragment-cluster bootstrap.

| Model | FYP2 test acc | BCI val acc | BCI test acc [95% CI] | Δ test acc | FYP2 macro-F1 | BCI macro-F1 | FYP2 QWK | BCI QWK | BCI recall 0 | BCI recall 2+ |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 1.000 | 0.902 | 0.949 [0.921, 0.967] | -5.1 pts | 1.000 | 0.936 | 1.000 | 0.932 | 0.89 | 0.95 |
| DenseNet121 | 1.000 | 0.908 | 0.942 [0.908, 0.963] | -5.8 pts | 1.000 | 0.927 | 1.000 | 0.910 | 0.86 | 0.96 |
| ConvNeXtTiny | 1.000 | 0.902 | 0.926 [0.889, 0.951] | -7.4 pts | 1.000 | 0.912 | 1.000 | 0.898 | 0.78 | 0.94 |
| EfficientNetB0 | 1.000 | 0.891 | 0.933 [0.900, 0.956] | -6.7 pts | 1.000 | 0.928 | 1.000 | 0.902 | 0.86 | 0.95 |
| EfficientNetV2B0 | 0.992 | 0.895 | 0.926 [0.890, 0.950] | -6.6 pts | 0.992 | 0.922 | 0.997 | 0.915 | 0.89 | 0.94 |
| NASNetMobile | 1.000 | 0.884 | 0.921 [0.882, 0.947] | -7.9 pts | 1.000 | 0.912 | 1.000 | 0.904 | 0.83 | 0.95 |
| MobileNetV2 | 1.000 | 0.857 | 0.930 [0.897, 0.953] | -7.0 pts | 1.000 | 0.928 | 1.000 | 0.924 | 0.92 | 0.93 |
| MobileNetV3Large | 0.997 | 0.871 | 0.908 [0.864, 0.940] | -9.0 pts | 0.997 | 0.879 | 0.999 | 0.878 | 0.83 | 0.94 |
| MobileNetV3Small | 1.000 | 0.854 | 0.913 [0.873, 0.942] | -8.7 pts | 1.000 | 0.895 | 1.000 | 0.888 | 0.81 | 0.93 |

![](figures/figR1_fyp2_vs_bci.png)

## 5. Full metrics and overfitting check

These are the restored best weights. 95% confidence intervals come from 2,000 cluster-bootstrap resamples of whole slide fragments (FYP2 resampled images; BCI tiles of one fragment are correlated, so a tile-level bootstrap would be too narrow). Per-class precision, recall, F1 and specificity and the confusion matrices are in `results/metrics_all.csv` and `results/metrics/<run>.json`.

**Test split (758 tiles)**

| Run | Acc | Acc 95% CI | Bal acc | Macro-F1 | F1 95% CI | QWK | MCC | MAE | Off-by-2+ | Clinical acc | AUC | Log-loss | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50 / CE | 0.949 | [0.921, 0.967] | 0.934 | 0.936 | [0.902, 0.957] | 0.932 | 0.924 | 0.066 | 0.013 | 0.954 | 0.997 | 0.133 | 0.021 |
| DenseNet121 / CE | 0.942 | [0.908, 0.963] | 0.922 | 0.927 | [0.886, 0.951] | 0.910 | 0.914 | 0.080 | 0.021 | 0.946 | 0.995 | 0.144 | 0.010 |
| DenseNet121 / CORN | 0.914 | [0.871, 0.945] | 0.893 | 0.901 | [0.857, 0.929] | 0.874 | 0.873 | 0.113 | 0.024 | 0.918 | 0.993 | 0.187 | 0.026 |
| ConvNeXtTiny / CE | 0.926 | [0.889, 0.951] | 0.891 | 0.912 | [0.870, 0.940] | 0.898 | 0.890 | 0.094 | 0.016 | 0.929 | 0.993 | 0.185 | 0.029 |
| EfficientNetB0 / CE | 0.933 | [0.900, 0.956] | 0.914 | 0.928 | [0.893, 0.950] | 0.902 | 0.900 | 0.090 | 0.021 | 0.934 | 0.993 | 0.216 | 0.032 |
| EfficientNetV2B0 / CE | 0.926 | [0.890, 0.950] | 0.917 | 0.922 | [0.881, 0.945] | 0.915 | 0.891 | 0.088 | 0.013 | 0.929 | 0.994 | 0.213 | 0.045 |
| EfficientNetV2B0 / CE + hires | 0.975 | [0.959, 0.987] | 0.973 | 0.971 | [0.948, 0.985] | 0.959 | 0.963 | 0.036 | 0.009 | 0.976 | 0.999 | 0.091 | 0.017 |
| EfficientNetV2B0 / CORN | 0.906 | [0.870, 0.933] | 0.900 | 0.894 | [0.851, 0.919] | 0.892 | 0.862 | 0.113 | 0.018 | 0.916 | 0.988 | 0.233 | 0.021 |
| NASNetMobile / CE | 0.921 | [0.882, 0.947] | 0.895 | 0.912 | [0.870, 0.938] | 0.904 | 0.882 | 0.096 | 0.016 | 0.925 | 0.993 | 0.187 | 0.030 |
| MobileNetV2 / CE | 0.930 | [0.897, 0.953] | 0.925 | 0.928 | [0.896, 0.949] | 0.924 | 0.896 | 0.082 | 0.011 | 0.931 | 0.993 | 0.169 | 0.020 |
| MobileNetV3Large / CE | 0.908 | [0.864, 0.940] | 0.884 | 0.879 | [0.830, 0.910] | 0.878 | 0.863 | 0.119 | 0.025 | 0.917 | 0.992 | 0.200 | 0.027 |
| MobileNetV3Large / CORN | 0.937 | [0.909, 0.960] | 0.919 | 0.922 | [0.885, 0.950] | 0.923 | 0.906 | 0.078 | 0.013 | 0.941 | 0.995 | 0.184 | 0.030 |
| MobileNetV3Small / CE | 0.913 | [0.873, 0.942] | 0.886 | 0.895 | [0.851, 0.923] | 0.888 | 0.871 | 0.108 | 0.017 | 0.918 | 0.987 | 0.244 | 0.026 |
| MobileNetV3Small / CE + cw | 0.906 | [0.868, 0.934] | 0.896 | 0.881 | [0.840, 0.908] | 0.883 | 0.862 | 0.117 | 0.021 | 0.914 | 0.986 | 0.249 | 0.023 |
| MobileNetV3Small / CE + hires | 0.908 | [0.869, 0.935] | 0.898 | 0.893 | [0.849, 0.922] | 0.874 | 0.863 | 0.120 | 0.025 | 0.916 | 0.988 | 0.234 | 0.025 |
| MobileNetV3Small / CE + ls | 0.905 | [0.863, 0.935] | 0.868 | 0.878 | [0.826, 0.910] | 0.873 | 0.859 | 0.120 | 0.021 | 0.912 | 0.983 | 0.319 | 0.078 |

**Per-class test precision / recall (0, 1+, 2+, 3+)**

| Run | P 0 | P 1+ | P 2+ | P 3+ | R 0 | R 1+ | R 2+ | R 3+ |
|---|---|---|---|---|---|---|---|---|
| ResNet50 / CE | 0.91 | 0.92 | 0.96 | 0.96 | 0.89 | 0.92 | 0.95 | 0.97 |
| DenseNet121 / CE | 0.91 | 0.91 | 0.95 | 0.96 | 0.86 | 0.93 | 0.96 | 0.94 |
| DenseNet121 / CORN | 0.89 | 0.92 | 0.90 | 0.94 | 0.86 | 0.84 | 0.96 | 0.91 |
| ConvNeXtTiny / CE | 1.00 | 0.89 | 0.92 | 0.96 | 0.78 | 0.89 | 0.94 | 0.95 |
| EfficientNetB0 / CE | 1.00 | 0.89 | 0.93 | 0.96 | 0.86 | 0.90 | 0.95 | 0.94 |
| EfficientNetV2B0 / CE | 0.94 | 0.88 | 0.92 | 0.98 | 0.89 | 0.93 | 0.94 | 0.91 |
| EfficientNetV2B0 / CE + hires | 0.95 | 0.98 | 0.98 | 0.97 | 0.97 | 0.96 | 0.98 | 0.98 |
| EfficientNetV2B0 / CORN | 0.84 | 0.85 | 0.92 | 0.94 | 0.89 | 0.87 | 0.92 | 0.92 |
| NASNetMobile / CE | 0.97 | 0.89 | 0.90 | 0.96 | 0.83 | 0.86 | 0.95 | 0.93 |
| MobileNetV2 / CE | 0.94 | 0.91 | 0.92 | 0.96 | 0.92 | 0.89 | 0.93 | 0.96 |
| MobileNetV3Large / CE | 0.75 | 0.89 | 0.90 | 0.96 | 0.83 | 0.83 | 0.94 | 0.92 |
| MobileNetV3Large / CORN | 0.89 | 0.92 | 0.94 | 0.96 | 0.86 | 0.91 | 0.94 | 0.96 |
| MobileNetV3Small / CE | 0.88 | 0.87 | 0.91 | 0.96 | 0.81 | 0.88 | 0.93 | 0.93 |
| MobileNetV3Small / CE + cw | 0.74 | 0.85 | 0.92 | 0.97 | 0.86 | 0.88 | 0.90 | 0.93 |
| MobileNetV3Small / CE + hires | 0.84 | 0.84 | 0.92 | 0.95 | 0.89 | 0.85 | 0.93 | 0.92 |
| MobileNetV3Small / CE + ls | 0.84 | 0.86 | 0.90 | 0.95 | 0.75 | 0.86 | 0.92 | 0.94 |

![](figures/figR4_confusion_matrices_test.png)

### 5b. Overfitting check (same rules as FYP2)

Each run's best weights are re-evaluated on clean training tiles (no augmentation). A run counts as **not overfitted** only if all three hold: train − test accuracy ≤ 0.02; |validation − test accuracy| ≤ 0.02; test − train log-loss ≤ 0.10. Failing one check gives "mild"; failing two or more gives "overfit".

These thresholds were set for FYP2's near-perfect accuracy, and on BCI they sit below the sampling noise of one held-out set. The **noise-aware verdict** therefore counts the accuracy-gap check as failed only if train accuracy is also above the test CI. It counts the validation check as failed only if validation is *better* than test, with a CI excluding zero, since overfitting the selection makes validation look better, not worse. The log-loss check is unchanged.

**Result, FYP2 rule:** overfit: 15 run(s), mild: 1 run(s). **Noise-aware:** overfit: 8 run(s), mild: 7 run(s), none: 1 run(s).

| Run | Best ep. / run | Train acc | Val acc | Test acc | Test acc 95% CI | Train−test | Val−test [95% CI] | Train LL | Test LL | LL gap | Val-loss rise after best | Test ECE | Verdict (FYP2 rule) | Verdict (noise-aware) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50 / CE | 43 / 45 | 0.988 | 0.902 | 0.949 | [0.921, 0.967] | 0.040 | -0.046 [-0.080, -0.006] | 0.037 | 0.133 | 0.097 | 0.001 | 0.021 | overfit | mild |
| DenseNet121 / CE | 32 / 42 | 0.975 | 0.908 | 0.942 | [0.908, 0.963] | 0.033 | -0.034 [-0.068, +0.009] | 0.074 | 0.144 | 0.071 | 0.125 | 0.010 | overfit | mild |
| DenseNet121 / CORN | 30 / 40 | 0.957 | 0.880 | 0.914 | [0.871, 0.945] | 0.042 | -0.034 [-0.096, +0.036] | 0.109 | 0.187 | 0.078 | 0.009 | 0.026 | overfit | mild |
| ConvNeXtTiny / CE | 13 / 23 | 0.978 | 0.902 | 0.926 | [0.889, 0.951] | 0.052 | -0.024 [-0.063, +0.025] | 0.068 | 0.185 | 0.117 | 0.081 | 0.029 | overfit | overfit |
| EfficientNetB0 / CE | 31 / 41 | 0.994 | 0.891 | 0.933 | [0.900, 0.956] | 0.061 | -0.041 [-0.077, +0.003] | 0.020 | 0.216 | 0.196 | 0.051 | 0.032 | overfit | overfit |
| EfficientNetV2B0 / CE | 31 / 41 | 0.992 | 0.895 | 0.926 | [0.890, 0.950] | 0.066 | -0.031 [-0.067, +0.016] | 0.025 | 0.213 | 0.187 | 0.102 | 0.045 | overfit | overfit |
| EfficientNetV2B0 / CE + hires | 38 / 45 | 0.999 | 0.950 | 0.975 | [0.959, 0.987] | 0.024 | -0.024 [-0.047, +0.002] | 0.003 | 0.091 | 0.088 | 0.031 | 0.017 | overfit | mild |
| EfficientNetV2B0 / CORN | 16 / 26 | 0.955 | 0.862 | 0.906 | [0.870, 0.933] | 0.048 | -0.044 [-0.090, +0.010] | 0.124 | 0.233 | 0.109 | 0.036 | 0.021 | overfit | overfit |
| NASNetMobile / CE | 28 / 38 | 0.976 | 0.884 | 0.921 | [0.882, 0.947] | 0.056 | -0.036 [-0.076, +0.014] | 0.066 | 0.187 | 0.121 | 0.108 | 0.030 | overfit | overfit |
| MobileNetV2 / CE | 33 / 43 | 0.962 | 0.857 | 0.930 | [0.897, 0.953] | 0.032 | -0.073 [-0.113, -0.025] | 0.108 | 0.169 | 0.061 | 0.025 | 0.020 | overfit | mild |
| MobileNetV3Large / CE | 22 / 32 | 0.967 | 0.871 | 0.908 | [0.864, 0.940] | 0.059 | -0.037 [-0.109, +0.036] | 0.088 | 0.200 | 0.112 | 0.102 | 0.027 | overfit | overfit |
| MobileNetV3Large / CORN | 45 / 45 | 0.993 | 0.894 | 0.937 | [0.909, 0.960] | 0.057 | -0.043 [-0.097, +0.013] | 0.021 | 0.184 | 0.162 | 0.000 | 0.030 | overfit | overfit |
| MobileNetV3Small / CE | 34 / 44 | 0.958 | 0.854 | 0.913 | [0.873, 0.942] | 0.045 | -0.059 [-0.107, -0.003] | 0.114 | 0.244 | 0.130 | 0.024 | 0.026 | overfit | overfit |
| MobileNetV3Small / CE + cw | 34 / 44 | 0.942 | 0.854 | 0.906 | [0.868, 0.934] | 0.036 | -0.052 [-0.097, -0.000] | 0.154 | 0.249 | 0.095 | 0.013 | 0.023 | overfit | mild |
| MobileNetV3Small / CE + hires | 15 / 25 | 0.930 | 0.898 | 0.908 | [0.869, 0.935] | 0.022 | -0.009 [-0.049, +0.037] | 0.188 | 0.234 | 0.046 | 0.115 | 0.025 | mild | none |
| MobileNetV3Small / CE + ls | 31 / 41 | 0.953 | 0.864 | 0.905 | [0.863, 0.935] | 0.048 | -0.041 [-0.089, +0.019] | 0.220 | 0.319 | 0.099 | 0.030 | 0.078 | overfit | mild |

![](figures/figR3_generalisation_gap.png)

### 5c. How precise is one held-out split?

Errors cluster in a few large slide fragments, so one fragment can move a split's accuracy by several points. "Top-3" is the three fragments with the most errors in that split for that run (`analysis_leakage.py`, `results/analysis/heldout_*.csv`).

| Run | Split | Tiles | Accuracy | Errors | Top-3 fragments (tiles) | Share of errors in top-3 | Accuracy without top-3 |
|---|---|---|---|---|---|---|---|
| ResNet50 / ce | test | 758 | 0.949 | 39 | 160, 435, 1540 (7) | 15% | 0.956 |
| ResNet50 / ce | val | 727 | 0.902 | 71 | 25, 31, 140 (153) | 32% | 0.916 |
| DenseNet121 / ce | test | 758 | 0.942 | 44 | 352, 435, 16 (11) | 18% | 0.952 |
| DenseNet121 / ce | val | 727 | 0.908 | 67 | 25, 31, 304 (112) | 33% | 0.927 |
| DenseNet121 / corn | test | 758 | 0.914 | 65 | 170, 352, 435 (16) | 17% | 0.927 |
| DenseNet121 / corn | val | 727 | 0.880 | 87 | 25, 31, 126 (110) | 46% | 0.924 |
| ConvNeXtTiny / ce | test | 758 | 0.926 | 56 | 170, 368, 16 (12) | 16% | 0.937 |
| ConvNeXtTiny / ce | val | 727 | 0.902 | 71 | 25, 31, 304 (112) | 38% | 0.928 |
| EfficientNetB0 / ce | test | 758 | 0.933 | 51 | 352, 104, 122 (68) | 20% | 0.941 |
| EfficientNetB0 / ce | val | 727 | 0.891 | 79 | 31, 25, 140 (153) | 37% | 0.913 |
| EfficientNetV2B0 / ce | test | 758 | 0.926 | 56 | 352, 219, 368 (16) | 12% | 0.934 |
| EfficientNetV2B0 / ce | val | 727 | 0.895 | 76 | 25, 31, 304 (112) | 30% | 0.914 |
| EfficientNetV2B0 / ce / hires | test | 758 | 0.975 | 19 | 170, 221, 352 (90) | 32% | 0.981 |
| EfficientNetV2B0 / ce / hires | val | 727 | 0.950 | 36 | 31, 25, 456 (106) | 42% | 0.966 |
| EfficientNetV2B0 / corn | test | 758 | 0.906 | 71 | 221, 122, 219 (138) | 15% | 0.903 |
| EfficientNetV2B0 / corn | val | 727 | 0.862 | 100 | 25, 31, 126 (110) | 34% | 0.893 |
| NASNetMobile / ce | test | 758 | 0.921 | 60 | 352, 368, 160 (11) | 13% | 0.930 |
| NASNetMobile / ce | val | 727 | 0.884 | 84 | 25, 31, 140 (153) | 31% | 0.899 |
| MobileNetV2 / ce | test | 758 | 0.930 | 53 | 104, 170, 215 (17) | 13% | 0.938 |
| MobileNetV2 / ce | val | 727 | 0.857 | 104 | 25, 31, 140 (153) | 34% | 0.880 |
| MobileNetV3Large / ce | test | 758 | 0.908 | 70 | 170, 221, 352 (90) | 16% | 0.912 |
| MobileNetV3Large / ce | val | 727 | 0.871 | 94 | 25, 31, 776 (111) | 45% | 0.916 |
| MobileNetV3Large / corn | test | 758 | 0.937 | 48 | 221, 104, 160 (87) | 21% | 0.943 |
| MobileNetV3Large / corn | val | 727 | 0.894 | 77 | 25, 31, 140 (153) | 44% | 0.925 |
| MobileNetV3Small / ce | test | 758 | 0.913 | 66 | 160, 170, 221 (86) | 9% | 0.911 |
| MobileNetV3Small / ce | val | 727 | 0.854 | 106 | 25, 31, 140 (153) | 33% | 0.876 |
| MobileNetV3Small / ce / cw | test | 758 | 0.906 | 71 | 221, 160, 352 (85) | 11% | 0.906 |
| MobileNetV3Small / ce / cw | val | 727 | 0.854 | 106 | 31, 25, 126 (110) | 26% | 0.874 |
| MobileNetV3Small / ce / hires | test | 758 | 0.908 | 70 | 122, 170, 221 (138) | 16% | 0.905 |
| MobileNetV3Small / ce / hires | val | 727 | 0.898 | 74 | 31, 25, 126 (110) | 26% | 0.911 |
| MobileNetV3Small / ce / ls | test | 758 | 0.905 | 72 | 170, 219, 221 (91) | 8% | 0.901 |
| MobileNetV3Small / ce / ls | val | 727 | 0.864 | 99 | 25, 31, 1021 (108) | 31% | 0.890 |

**Hardest fragments** (≥ 5 held-out tiles, averaged over runs). Percentiles place the fragment's median tile within its own class in the grouped training split. *Raw* DAB is measured against 255, so a dark grey scan raises it; *corrected* DAB is measured against the tile's own background (section 6a).

| Split | Fragment | Slide label | Tiles | Runs | Mean accuracy | Errors per run | Patch labels agreeing | DAB pct, raw | DAB pct, corrected | Sharpness pct | Background pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 25 | 1+ | 52 | 16 | 0.707 | 15.2 | 100% | 87% | 74% | 21% | 22% |
| val | 31 | 3+ | 52 | 16 | 0.800 | 10.4 | 96% | 61% | 45% | 37% | 29% |
| val | 140 | 2+ | 49 | 16 | 0.953 | 2.3 | 92% | 87% | 89% | 54% | 17% |
| test | 221 | 2+ | 77 | 16 | 0.971 | 2.2 | 79% | 63% | 51% | 16% | 29% |
| val | 304 | 3+ | 8 | 16 | 0.742 | 2.1 | 100% | 48% | 26% | 15% | 22% |
| test | 170 | 1+ | 7 | 16 | 0.705 | 2.1 | - | 86% | 81% | 11% | 29% |
| test | 352 | 3+ | 6 | 16 | 0.656 | 2.1 | - | 47% | 28% | 15% | 37% |
| val | 900 | 3+ | 11 | 16 | 0.852 | 1.6 | 100% | 80% | 78% | 54% | 16% |

## 6. Diagnosis: evidence

The interpretation, with each finding marked confirmed or hypothesis, is in section 0. This section holds the evidence.

### 6a. Model-free: how separable are the classes from simple image statistics?

5-fold CV on the development splits only (FYP2 train+val, 2,040 images; BCI train+val, 4,115 tiles). Random forest on hand-crafted features (`analysis_data.py`); macro-F1, chance 0.25. *Grouped* folds keep leakage groups whole (FYP2 pHash twins, BCI slide fragments), as the main split does. *Image-level* folds (BCI only) ignore fragments, so the gap between the two measures how much slide-level cues inflate a model-free classifier.

| Feature set (macro-F1) | BCI (real), grouped CV | BCI (real), image-level CV | FYP2 (GAN), grouped CV |
|---|---|---|---|
| all | 0.772 | 0.812 | 0.96 |
| DAB, raw OD | 0.621 | 0.653 | 0.878 |
| DAB, illumination-corrected | 0.488 | 0.519 | 0.816 |
| mean colour only | 0.572 | 0.587 | 0.787 |
| background only | 0.437 | 0.456 | 0.61 |

![](figures/figD1_stain_distributions.png)

![](figures/figD2_handcrafted_separability.png)

### 6b. Label noise: slide-level vs patch-level labels

*Clean* tiles have refined = filename label. *Noisy* tiles disagree. "Errors matching refined" is the share of the model's errors (against the filename label) where the prediction equals the patch-level label.

| Run | Split | Acc (filename) | Noisy tiles | Acc clean | Acc noisy vs filename | Acc noisy vs refined | Errors matching refined | Errors on noisy tiles |
|---|---|---|---|---|---|---|---|---|
| ResNet50 / ce | test | 0.949 | 93 / 584 | 0.990 | 0.989 | 0.011 | 0.167 | 0.167 |
| DenseNet121 / ce | test | 0.942 | 93 / 584 | 0.996 | 0.935 | 0.054 | 0.625 | 0.750 |
| DenseNet121 / corn | test | 0.914 | 93 / 584 | 0.986 | 0.935 | 0.043 | 0.308 | 0.462 |
| ConvNeXtTiny / ce | test | 0.926 | 93 / 584 | 0.984 | 0.925 | 0.065 | 0.400 | 0.467 |
| EfficientNetB0 / ce | test | 0.933 | 93 / 584 | 0.976 | 0.925 | 0.043 | 0.211 | 0.368 |
| EfficientNetV2B0 / ce | test | 0.926 | 93 / 584 | 0.969 | 0.968 | 0.032 | 0.167 | 0.167 |
| EfficientNetV2B0 / ce / hires | test | 0.975 | 93 / 584 | 0.996 | 1.000 | 0.000 | 0.000 | 0.000 |
| EfficientNetV2B0 / corn | test | 0.906 | 93 / 584 | 0.957 | 0.957 | 0.032 | 0.120 | 0.160 |
| NASNetMobile / ce | test | 0.921 | 93 / 584 | 0.976 | 0.925 | 0.075 | 0.368 | 0.368 |
| MobileNetV2 / ce | test | 0.930 | 93 / 584 | 0.971 | 0.957 | 0.022 | 0.111 | 0.222 |
| MobileNetV3Large / ce | test | 0.908 | 93 / 584 | 0.971 | 0.957 | 0.032 | 0.167 | 0.222 |
| MobileNetV3Large / corn | test | 0.937 | 93 / 584 | 0.976 | 0.957 | 0.032 | 0.188 | 0.250 |
| MobileNetV3Small / ce | test | 0.913 | 93 / 584 | 0.965 | 0.935 | 0.054 | 0.217 | 0.261 |
| MobileNetV3Small / ce / cw | test | 0.906 | 93 / 584 | 0.955 | 0.914 | 0.086 | 0.267 | 0.267 |
| MobileNetV3Small / ce / hires | test | 0.908 | 93 / 584 | 0.967 | 0.935 | 0.054 | 0.227 | 0.273 |
| MobileNetV3Small / ce / ls | test | 0.905 | 93 / 584 | 0.957 | 0.935 | 0.054 | 0.185 | 0.222 |
| ResNet50 / ce | val | 0.902 | 87 / 489 | 0.960 | 0.943 | 0.057 | 0.238 | 0.238 |
| DenseNet121 / ce | val | 0.908 | 87 / 489 | 0.978 | 0.908 | 0.092 | 0.471 | 0.471 |
| DenseNet121 / corn | val | 0.880 | 87 / 489 | 0.968 | 0.908 | 0.092 | 0.381 | 0.381 |
| ConvNeXtTiny / ce | val | 0.902 | 87 / 489 | 0.953 | 0.966 | 0.034 | 0.136 | 0.136 |
| EfficientNetB0 / ce | val | 0.891 | 87 / 489 | 0.950 | 0.943 | 0.046 | 0.160 | 0.200 |
| EfficientNetV2B0 / ce | val | 0.895 | 87 / 489 | 0.953 | 0.954 | 0.046 | 0.174 | 0.174 |
| EfficientNetV2B0 / ce / hires | val | 0.950 | 87 / 489 | 0.988 | 0.966 | 0.034 | 0.375 | 0.375 |
| EfficientNetV2B0 / corn | val | 0.862 | 87 / 489 | 0.945 | 0.931 | 0.069 | 0.214 | 0.214 |
| NASNetMobile / ce | val | 0.884 | 87 / 489 | 0.955 | 0.920 | 0.069 | 0.240 | 0.280 |
| MobileNetV2 / ce | val | 0.857 | 87 / 489 | 0.953 | 0.770 | 0.184 | 0.410 | 0.513 |
| MobileNetV3Large / ce | val | 0.871 | 87 / 489 | 0.943 | 0.908 | 0.080 | 0.226 | 0.258 |
| MobileNetV3Large / corn | val | 0.894 | 87 / 489 | 0.955 | 0.920 | 0.069 | 0.240 | 0.280 |
| MobileNetV3Small / ce | val | 0.854 | 87 / 489 | 0.940 | 0.805 | 0.161 | 0.341 | 0.415 |
| MobileNetV3Small / ce / cw | val | 0.854 | 87 / 489 | 0.950 | 0.839 | 0.138 | 0.353 | 0.412 |
| MobileNetV3Small / ce / hires | val | 0.898 | 87 / 489 | 0.985 | 0.874 | 0.126 | 0.647 | 0.647 |
| MobileNetV3Small / ce / ls | val | 0.864 | 87 / 489 | 0.945 | 0.816 | 0.161 | 0.368 | 0.421 |

![](figures/figD3_label_noise.png)

### 6c. Scan type (background brightness ≥ 200 = bright)

| run | split | accuracy bright | accuracy grey | macro_f1 bright | macro_f1 grey | recall_3 bright | recall_3 grey |
|---|---|---|---|---|---|---|---|
| ConvNeXtTiny / ce | test | 0.897 | 0.948 | 0.884 | 0.95 | 0.951 | 0.951 |
| ConvNeXtTiny / ce | val | 0.922 | 0.888 | 0.911 | 0.889 | 0.943 | 0.859 |
| DenseNet121 / ce | test | 0.927 | 0.953 | 0.909 | 0.959 | 0.934 | 0.939 |
| DenseNet121 / ce | val | 0.932 | 0.89 | 0.925 | 0.888 | 1.0 | 0.883 |
| DenseNet121 / corn | test | 0.915 | 0.913 | 0.899 | 0.906 | 0.918 | 0.909 |
| DenseNet121 / corn | val | 0.932 | 0.843 | 0.925 | 0.848 | 0.971 | 0.834 |
| EfficientNetB0 / ce | test | 0.918 | 0.944 | 0.92 | 0.933 | 0.951 | 0.933 |
| EfficientNetB0 / ce | val | 0.919 | 0.871 | 0.914 | 0.88 | 0.943 | 0.834 |
| EfficientNetV2B0 / ce | test | 0.885 | 0.958 | 0.884 | 0.964 | 0.836 | 0.939 |
| EfficientNetV2B0 / ce | val | 0.909 | 0.886 | 0.899 | 0.893 | 0.886 | 0.859 |
| EfficientNetV2B0 / ce / hires | test | 0.979 | 0.972 | 0.971 | 0.976 | 0.984 | 0.976 |
| EfficientNetV2B0 / ce / hires | val | 0.971 | 0.936 | 0.974 | 0.931 | 1.0 | 0.908 |
| EfficientNetV2B0 / corn | test | 0.879 | 0.927 | 0.866 | 0.934 | 0.852 | 0.945 |
| EfficientNetV2B0 / corn | val | 0.886 | 0.845 | 0.877 | 0.859 | 0.943 | 0.798 |
| MobileNetV2 / ce | test | 0.894 | 0.958 | 0.899 | 0.961 | 0.918 | 0.976 |
| MobileNetV2 / ce | val | 0.863 | 0.852 | 0.845 | 0.859 | 0.829 | 0.828 |
| MobileNetV3Large / ce | test | 0.885 | 0.925 | 0.856 | 0.929 | 0.951 | 0.915 |
| MobileNetV3Large / ce | val | 0.896 | 0.852 | 0.89 | 0.848 | 0.943 | 0.896 |
| MobileNetV3Large / corn | test | 0.912 | 0.955 | 0.895 | 0.958 | 0.951 | 0.97 |
| MobileNetV3Large / corn | val | 0.919 | 0.876 | 0.908 | 0.875 | 0.914 | 0.89 |
| MobileNetV3Small / ce | test | 0.876 | 0.941 | 0.859 | 0.946 | 0.918 | 0.933 |
| MobileNetV3Small / ce | val | 0.853 | 0.855 | 0.831 | 0.854 | 0.886 | 0.847 |
| MobileNetV3Small / ce / cw | test | 0.855 | 0.946 | 0.836 | 0.948 | 0.885 | 0.951 |
| MobileNetV3Small / ce / cw | val | 0.857 | 0.852 | 0.834 | 0.838 | 0.886 | 0.822 |
| MobileNetV3Small / ce / hires | test | 0.894 | 0.918 | 0.874 | 0.913 | 0.803 | 0.963 |
| MobileNetV3Small / ce / hires | val | 0.909 | 0.89 | 0.905 | 0.892 | 0.914 | 0.853 |
| MobileNetV3Small / ce / ls | test | 0.858 | 0.941 | 0.832 | 0.945 | 0.902 | 0.951 |
| MobileNetV3Small / ce / ls | val | 0.866 | 0.862 | 0.83 | 0.842 | 0.914 | 0.859 |
| NASNetMobile / ce | test | 0.885 | 0.948 | 0.882 | 0.953 | 0.934 | 0.933 |
| NASNetMobile / ce | val | 0.906 | 0.869 | 0.882 | 0.874 | 0.943 | 0.847 |
| ResNet50 / ce | test | 0.921 | 0.969 | 0.91 | 0.972 | 0.967 | 0.97 |
| ResNet50 / ce | val | 0.925 | 0.886 | 0.909 | 0.885 | 0.943 | 0.89 |

**By class.** Grey scans are more common in higher classes, so background brightness could act as a shortcut. A shortcut would over-score grey tiles and under-score bright tiles of the same class (positive vs negative mean bias, expected score − label; val+test pooled). Corrected DAB differs between scan types within a class too, so this table alone cannot separate the two explanations (see 6e and 6f).

| Run | Class | Grey share (all tiles) | Corrected DAB, bright / grey | n bright / grey | Acc bright | Acc grey | Bias bright | Bias grey |
|---|---|---|---|---|---|---|---|---|
| ResNet50 / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.833 | 0.944 | +0.34 | +0.11 |
| ResNet50 / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.900 | 0.875 | +0.10 | +0.26 |
| ResNet50 / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.944 | 0.947 | -0.04 | +0.00 |
| ResNet50 / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.958 | 0.930 | -0.10 | -0.12 |
| DenseNet121 / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.833 | 0.944 | +0.36 | +0.19 |
| DenseNet121 / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.925 | 0.882 | +0.10 | +0.28 |
| DenseNet121 / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.941 | 0.947 | -0.05 | -0.01 |
| DenseNet121 / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.958 | 0.911 | -0.10 | -0.14 |
| DenseNet121 / corn | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.870 | 0.889 | +0.40 | +0.29 |
| DenseNet121 / corn | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.881 | 0.681 | +0.13 | +0.49 |
| DenseNet121 / corn | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.958 | 0.964 | -0.04 | +0.00 |
| DenseNet121 / corn | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.938 | 0.872 | -0.16 | -0.14 |
| ConvNeXtTiny / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.741 | 0.944 | +0.68 | +0.13 |
| ConvNeXtTiny / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.886 | 0.847 | +0.13 | +0.32 |
| ConvNeXtTiny / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.944 | 0.958 | -0.07 | -0.01 |
| ConvNeXtTiny / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.948 | 0.905 | -0.10 | -0.15 |
| EfficientNetB0 / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.870 | 0.889 | +0.32 | +0.26 |
| EfficientNetB0 / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.910 | 0.868 | +0.10 | +0.24 |
| EfficientNetB0 / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.923 | 0.947 | -0.04 | +0.01 |
| EfficientNetB0 / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.948 | 0.884 | -0.11 | -0.16 |
| EfficientNetV2B0 / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.833 | 0.944 | +0.35 | +0.09 |
| EfficientNetV2B0 / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.915 | 0.875 | +0.10 | +0.25 |
| EfficientNetV2B0 / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.909 | 0.961 | -0.09 | -0.00 |
| EfficientNetV2B0 / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.854 | 0.899 | -0.20 | -0.12 |
| EfficientNetV2B0 / ce / hires | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.963 | 0.944 | +0.11 | +0.13 |
| EfficientNetV2B0 / ce / hires | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.960 | 0.931 | +0.03 | +0.15 |
| EfficientNetV2B0 / ce / hires | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.983 | 0.975 | -0.03 | +0.01 |
| EfficientNetV2B0 / ce / hires | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.990 | 0.942 | -0.03 | -0.08 |
| EfficientNetV2B0 / corn | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.833 | 0.944 | +0.44 | +0.14 |
| EfficientNetV2B0 / corn | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.851 | 0.833 | +0.12 | +0.32 |
| EfficientNetV2B0 / corn | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.913 | 0.919 | -0.11 | -0.01 |
| EfficientNetV2B0 / corn | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.885 | 0.872 | -0.24 | -0.18 |
| NASNetMobile / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.759 | 0.944 | +0.52 | +0.11 |
| NASNetMobile / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.821 | 0.868 | +0.18 | +0.30 |
| NASNetMobile / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.958 | 0.941 | -0.05 | -0.02 |
| NASNetMobile / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.938 | 0.890 | -0.12 | -0.17 |
| MobileNetV2 / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.815 | 0.944 | +0.56 | +0.12 |
| MobileNetV2 / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.896 | 0.826 | +0.11 | +0.37 |
| MobileNetV2 / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.878 | 0.938 | -0.11 | -0.01 |
| MobileNetV2 / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.885 | 0.902 | -0.20 | -0.16 |
| MobileNetV3Large / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.852 | 0.944 | +0.42 | +0.19 |
| MobileNetV3Large / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.816 | 0.701 | +0.14 | +0.51 |
| MobileNetV3Large / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.930 | 0.947 | -0.09 | +0.01 |
| MobileNetV3Large / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.948 | 0.905 | -0.14 | -0.13 |
| MobileNetV3Large / corn | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.833 | 0.944 | +0.43 | +0.24 |
| MobileNetV3Large / corn | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.886 | 0.812 | +0.11 | +0.40 |
| MobileNetV3Large / corn | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.944 | 0.944 | -0.04 | -0.01 |
| MobileNetV3Large / corn | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.938 | 0.930 | -0.07 | -0.08 |
| MobileNetV3Small / ce | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.722 | 0.944 | +0.60 | +0.13 |
| MobileNetV3Small / ce | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.851 | 0.792 | +0.16 | +0.37 |
| MobileNetV3Small / ce | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.888 | 0.947 | -0.14 | -0.02 |
| MobileNetV3Small / ce | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.906 | 0.890 | -0.18 | -0.18 |
| MobileNetV3Small / ce / cw | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.815 | 0.889 | +0.40 | +0.16 |
| MobileNetV3Small / ce / cw | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.846 | 0.889 | +0.15 | +0.33 |
| MobileNetV3Small / ce / cw | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.860 | 0.916 | -0.18 | -0.04 |
| MobileNetV3Small / ce / cw | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.885 | 0.887 | -0.25 | -0.20 |
| MobileNetV3Small / ce / hires | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.889 | 0.944 | +0.41 | +0.17 |
| MobileNetV3Small / ce / hires | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.920 | 0.792 | +0.10 | +0.45 |
| MobileNetV3Small / ce / hires | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.909 | 0.944 | -0.16 | -0.02 |
| MobileNetV3Small / ce / hires | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.844 | 0.908 | -0.33 | -0.17 |
| MobileNetV3Small / ce / ls | 0 | 25% | 0.087 / 0.042 | 54 / 18 | 0.630 | 0.889 | +0.92 | +0.45 |
| MobileNetV3Small / ce / ls | 1+ | 44% | 0.092 / 0.100 | 201 / 144 | 0.856 | 0.792 | +0.21 | +0.45 |
| MobileNetV3Small / ce / ls | 2+ | 52% | 0.094 / 0.140 | 286 / 357 | 0.895 | 0.944 | -0.19 | -0.06 |
| MobileNetV3Small / ce / ls | 3+ | 68% | 0.262 / 0.289 | 96 / 327 | 0.906 | 0.905 | -0.35 | -0.32 |

### 6e. Which simple image cues do the predictions track?

This is the cross-validated R² of a linear model predicting each CE model's expected score on the test split from one group of hand-crafted statistics. The *true label* row is the reference: how much of the label those same statistics explain. A model row well above the label row means the model's output follows that cue more closely than the labels do.

| Dataset | Target | DAB, raw OD | DAB, corrected | haematoxylin | mean colour | background | texture | all | ρ(DAB raw) | ρ(DAB corrected) | ρ(background) | ρ(sharpness) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FYP2 | true label | 0.799 | 0.786 | 0.319 | 0.755 | 0.550 | 0.683 | 0.884 | 0.88 | 0.84 | 0.50 | 0.41 |
| FYP2 | ResNet50 | 0.799 | 0.785 | 0.319 | 0.756 | 0.551 | 0.683 | 0.885 | 0.84 | 0.80 | 0.55 | 0.35 |
| FYP2 | DenseNet121 | 0.802 | 0.787 | 0.320 | 0.760 | 0.556 | 0.685 | 0.886 | 0.87 | 0.83 | 0.55 | 0.37 |
| FYP2 | ConvNeXtTiny | 0.800 | 0.786 | 0.320 | 0.756 | 0.551 | 0.683 | 0.885 | 0.86 | 0.82 | 0.53 | 0.38 |
| FYP2 | EfficientNetB0 | 0.801 | 0.787 | 0.320 | 0.759 | 0.555 | 0.685 | 0.885 | 0.88 | 0.84 | 0.50 | 0.38 |
| FYP2 | EfficientNetV2B0 | 0.810 | 0.794 | 0.322 | 0.753 | 0.549 | 0.688 | 0.887 | 0.88 | 0.85 | 0.51 | 0.40 |
| FYP2 | NASNetMobile | 0.799 | 0.786 | 0.320 | 0.756 | 0.551 | 0.683 | 0.885 | 0.86 | 0.82 | 0.50 | 0.41 |
| FYP2 | MobileNetV2 | 0.800 | 0.787 | 0.320 | 0.755 | 0.548 | 0.683 | 0.889 | 0.86 | 0.82 | 0.52 | 0.38 |
| FYP2 | MobileNetV3Large | 0.800 | 0.785 | 0.317 | 0.756 | 0.549 | 0.683 | 0.886 | 0.86 | 0.83 | 0.54 | 0.37 |
| FYP2 | MobileNetV3Small | 0.802 | 0.788 | 0.320 | 0.755 | 0.550 | 0.684 | 0.885 | 0.87 | 0.83 | 0.55 | 0.36 |
| BCI | true label | 0.337 | 0.326 | 0.031 | 0.258 | 0.065 | 0.313 | 0.390 | 0.52 | 0.55 | -0.17 | 0.37 |
| BCI | ResNet50 | 0.357 | 0.345 | 0.031 | 0.268 | 0.068 | 0.327 | 0.415 | 0.57 | 0.57 | -0.23 | 0.35 |
| BCI | DenseNet121 | 0.378 | 0.365 | 0.032 | 0.286 | 0.070 | 0.348 | 0.437 | 0.60 | 0.60 | -0.24 | 0.40 |
| BCI | ConvNeXtTiny | 0.386 | 0.378 | 0.030 | 0.290 | 0.065 | 0.362 | 0.438 | 0.59 | 0.62 | -0.21 | 0.40 |
| BCI | EfficientNetB0 | 0.371 | 0.361 | 0.024 | 0.277 | 0.062 | 0.354 | 0.431 | 0.61 | 0.63 | -0.25 | 0.40 |
| BCI | EfficientNetV2B0 | 0.387 | 0.372 | 0.035 | 0.297 | 0.081 | 0.351 | 0.446 | 0.62 | 0.63 | -0.26 | 0.36 |
| BCI | NASNetMobile | 0.378 | 0.372 | 0.024 | 0.280 | 0.056 | 0.355 | 0.448 | 0.60 | 0.63 | -0.22 | 0.39 |
| BCI | MobileNetV2 | 0.391 | 0.374 | 0.038 | 0.301 | 0.084 | 0.351 | 0.456 | 0.63 | 0.61 | -0.28 | 0.33 |
| BCI | MobileNetV3Large | 0.398 | 0.378 | 0.037 | 0.292 | 0.082 | 0.360 | 0.452 | 0.64 | 0.64 | -0.29 | 0.38 |
| BCI | MobileNetV3Small | 0.406 | 0.390 | 0.029 | 0.295 | 0.080 | 0.362 | 0.440 | 0.65 | 0.66 | -0.28 | 0.39 |

**What drives the errors?** Linear regression of each run's error (expected score − label, val+test pooled) on standardised tile statistics with class fixed effects. Coefficients are in score units per standard deviation, with 95% fragment-cluster bootstrap CIs (`analysis_labels.py`). A positive DAB coefficient means the model scores a tile higher than its slide label when the tile itself carries more stain.

| Run | DAB (corrected) | Background brightness | Haematoxylin (corrected) | Tissue fraction | Sharpness (log) |
|---|---|---|---|---|---|
| ResNet50 / ce | +0.082 [+0.041, +0.126] | -0.013 [-0.032, +0.009] | +0.017 [-0.013, +0.052] | +0.011 [-0.058, +0.086] | -0.022 [-0.116, +0.061] |
| DenseNet121 / ce | +0.085 [+0.046, +0.130] | -0.021 [-0.038, -0.001] | +0.005 [-0.024, +0.035] | -0.031 [-0.097, +0.045] | +0.014 [-0.079, +0.095] |
| DenseNet121 / corn | +0.114 [+0.068, +0.166] | -0.035 [-0.058, -0.012] | -0.017 [-0.051, +0.018] | -0.018 [-0.090, +0.061] | -0.029 [-0.138, +0.066] |
| ConvNeXtTiny / ce | +0.109 [+0.062, +0.162] | -0.003 [-0.024, +0.020] | +0.016 [-0.024, +0.056] | -0.001 [-0.071, +0.077] | -0.026 [-0.131, +0.070] |
| EfficientNetB0 / ce | +0.097 [+0.053, +0.148] | -0.017 [-0.039, +0.007] | +0.023 [-0.014, +0.064] | -0.024 [-0.089, +0.054] | +0.016 [-0.081, +0.104] |
| EfficientNetV2B0 / ce | +0.082 [+0.045, +0.127] | -0.039 [-0.060, -0.015] | -0.024 [-0.062, +0.012] | -0.024 [-0.079, +0.038] | +0.012 [-0.060, +0.080] |
| EfficientNetV2B0 / ce / hires | +0.044 [+0.019, +0.073] | -0.017 [-0.031, -0.003] | -0.003 [-0.019, +0.015] | -0.015 [-0.051, +0.025] | +0.003 [-0.047, +0.053] |
| EfficientNetV2B0 / corn | +0.126 [+0.075, +0.183] | -0.040 [-0.066, -0.012] | -0.023 [-0.071, +0.021] | -0.044 [-0.107, +0.027] | +0.014 [-0.080, +0.102] |
| NASNetMobile / ce | +0.114 [+0.071, +0.166] | -0.005 [-0.026, +0.018] | -0.001 [-0.033, +0.032] | -0.031 [-0.096, +0.040] | +0.001 [-0.092, +0.084] |
| MobileNetV2 / ce | +0.135 [+0.080, +0.191] | -0.025 [-0.051, +0.005] | +0.009 [-0.033, +0.057] | +0.018 [-0.055, +0.102] | -0.056 [-0.168, +0.044] |
| MobileNetV3Large / ce | +0.096 [+0.050, +0.147] | -0.060 [-0.093, -0.025] | +0.007 [-0.039, +0.057] | -0.011 [-0.095, +0.083] | -0.015 [-0.125, +0.087] |
| MobileNetV3Large / corn | +0.082 [+0.038, +0.128] | -0.018 [-0.045, +0.011] | +0.026 [-0.011, +0.071] | +0.031 [-0.052, +0.130] | -0.062 [-0.185, +0.040] |
| MobileNetV3Small / ce | +0.117 [+0.066, +0.174] | -0.044 [-0.072, -0.013] | -0.016 [-0.074, +0.039] | -0.024 [-0.098, +0.059] | +0.005 [-0.096, +0.100] |
| MobileNetV3Small / ce / cw | +0.113 [+0.064, +0.170] | -0.055 [-0.084, -0.025] | +0.012 [-0.033, +0.060] | -0.018 [-0.085, +0.057] | +0.039 [-0.060, +0.124] |
| MobileNetV3Small / ce / hires | +0.115 [+0.075, +0.158] | -0.089 [-0.118, -0.062] | -0.021 [-0.050, +0.008] | -0.050 [-0.092, -0.004] | +0.026 [-0.038, +0.089] |
| MobileNetV3Small / ce / ls | +0.114 [+0.071, +0.163] | -0.062 [-0.085, -0.038] | -0.014 [-0.052, +0.023] | -0.026 [-0.079, +0.032] | -0.016 [-0.088, +0.054] |

### 6f. What the networks learned (`analysis_models.py`)

- **Frozen ImageNet probe:** a linear classifier on generic EfficientNetB0 features, measuring dataset separability with no training.
- **kNN / silhouette / scatter ratio:** class separation in each trained model's own embedding (test split).
- **Cross-domain:** the model trained on one dataset, scored on the other's test split. Balanced accuracy (mean recall over the four scores) is shown because BCI's test set is imbalanced: chance is 0.25 for both datasets, whereas plain accuracy on BCI is 0.43 for a model that always answers 2+.
- **Grad-CAM DAB ratio:** attention-weighted DAB divided by mean DAB, with DAB measured against each tile's own background. Above 1 means the model looks at stained regions.
- **Background ratio:** attention share on background divided by background area. Below 1 means it avoids background.

| Network | Trained/eval on | Test acc | Test bal. acc | Test QWK | kNN acc | Silhouette | Scatter ratio | Cross-domain bal. acc | Cross-domain QWK | CAM DAB ratio | CAM bg ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen ImageNet EfficientNetB0 (linear probe) | FYP2 | 0.978 | - | 0.991 | - | 0.168 | 0.174 | - | - | - | - |
| frozen ImageNet EfficientNetB0 (linear probe) | BCI | 0.765 | - | 0.698 | - | 0.006 | 0.053 | - | - | - | - |
| ResNet50 | FYP2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.779 | 0.342 | 0.404 | 0.467 | 1.255 | 0.873 |
| ResNet50 | BCI | 0.949 | 0.934 | 0.932 | 0.954 | 0.210 | 0.105 | 0.281 | 0.232 | 1.153 | 0.890 |
| DenseNet121 | FYP2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.711 | 0.941 | 0.432 | 0.475 | 1.196 | 0.902 |
| DenseNet121 | BCI | 0.942 | 0.922 | 0.910 | 0.949 | 0.230 | 0.205 | 0.328 | 0.242 | 1.070 | 0.939 |
| ConvNeXtTiny | FYP2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.677 | 1.435 | 0.422 | 0.456 | 1.399 | 0.754 |
| ConvNeXtTiny | BCI | 0.926 | 0.891 | 0.898 | 0.912 | 0.158 | 0.188 | 0.511 | 0.475 | 1.092 | 0.916 |
| EfficientNetB0 | FYP2 | 1.000 | 1.000 | 1.000 | 0.997 | 0.454 | 0.427 | 0.381 | 0.385 | 1.198 | 0.871 |
| EfficientNetB0 | BCI | 0.933 | 0.914 | 0.902 | 0.931 | 0.179 | 0.136 | 0.353 | 0.421 | 1.077 | 0.931 |
| EfficientNetV2B0 | FYP2 | 0.992 | 0.992 | 0.997 | 0.997 | 0.450 | 0.436 | 0.323 | 0.264 | 1.133 | 0.933 |
| EfficientNetV2B0 | BCI | 0.926 | 0.917 | 0.915 | 0.923 | 0.195 | 0.145 | 0.342 | 0.359 | 1.092 | 0.922 |
| NASNetMobile | FYP2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.673 | 0.725 | 0.455 | 0.472 | 1.154 | 0.905 |
| NASNetMobile | BCI | 0.921 | 0.895 | 0.904 | 0.921 | 0.228 | 0.179 | 0.342 | 0.279 | 1.066 | 0.949 |
| MobileNetV2 | FYP2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.620 | 0.400 | 0.393 | 0.413 | 1.134 | 0.935 |
| MobileNetV2 | BCI | 0.930 | 0.925 | 0.924 | 0.929 | 0.200 | 0.131 | 0.375 | 0.362 | 1.054 | 0.958 |
| MobileNetV3Large | FYP2 | 0.997 | 0.997 | 0.999 | 1.000 | 0.526 | 0.358 | 0.376 | 0.409 | 1.194 | 0.887 |
| MobileNetV3Large | BCI | 0.908 | 0.884 | 0.878 | 0.912 | 0.123 | 0.107 | 0.419 | 0.444 | 1.112 | 0.903 |
| MobileNetV3Small | FYP2 | 1.000 | 1.000 | 1.000 | 0.994 | 0.561 | 0.553 | 0.302 | 0.309 | 1.158 | 0.923 |
| MobileNetV3Small | BCI | 0.913 | 0.886 | 0.888 | 0.902 | 0.172 | 0.173 | 0.339 | 0.437 | 1.096 | 0.925 |

![](figures/figD7_cross_domain.png)

![](figures/figD5_cka.png)

![](figures/figD4_tsne_EfficientNetB0.png)

![](figures/figD6_gradcam_DenseNet121.png)

![](figures/figD6_gradcam_EfficientNetV2B0.png)

![](figures/figD6_gradcam_MobileNetV3Large.png)

Grad-CAM examples for the other six networks are in `results/figures/figD6_gradcam_<network>.png`.

**Linear CKA on BCI test images** at five evenly spaced depths (1 = early, 5 = last conv block); every value is in `results/analysis/cka.csv`:

| Network | FYP2-net vs BCI-net, depth 1 / 2 / 3 / 4 / 5 | FYP2-net vs ImageNet start, depth 5 | BCI-net vs ImageNet start, depth 5 |
|---|---|---|---|
| ResNet50 | 0.93 / 0.90 / 0.88 / 0.89 / 0.81 | 0.75 | 0.82 |
| DenseNet121 | 0.59 / 0.90 / 0.92 / 0.83 / 0.85 | 0.74 | 0.78 |
| ConvNeXtTiny | 0.97 / 0.95 / 0.85 / 0.87 / 0.69 | 0.69 | 0.63 |
| EfficientNetB0 | 1.00 / 0.97 / 0.85 / 0.70 / 0.78 | 0.81 | 0.67 |
| EfficientNetV2B0 | 0.96 / 0.72 / 0.92 / 0.84 / 0.70 | 0.85 | 0.68 |
| NASNetMobile | 1.00 / 0.72 / 0.89 / 0.73 / 0.68 | 0.56 | 0.49 |
| MobileNetV2 | 1.00 / 0.98 / 0.66 / 0.84 / 0.65 | 0.66 | 0.67 |
| MobileNetV3Large | 0.99 / 0.98 / 0.94 / 0.69 / 0.76 | 0.80 | 0.82 |
| MobileNetV3Small | 1.00 / 0.93 / 0.87 / 0.77 / 0.76 | 0.73 | 0.73 |

### 6g. Stain and scan-brightness perturbation (counterfactual)

Each test tile is altered in one controlled way and re-scored. Stain concentrations are scaled in Ruifrok–Johnston optical-density space, relative to the tile's own background, so the glass keeps its colour. Alternatively, the illumination itself is scaled, which mimics a brighter or darker scan. The table gives the mean shift of the expected score (Σ k·p_k, 0 to 3) per network: the median over networks, with the range in brackets. A network that reads HER2 from DAB should respond to DAB, and hardly at all to haematoxylin or scan brightness. The Ruifrok–Johnston stain vectors are generic, not calibrated for these scanners, so the stain channels are only approximately separated.

![](figures/figD10_stain_sensitivity.png)

| Perturbation | Scale | FYP2-trained on FYP2 test: median [range] | BCI-trained on BCI test: median [range] |
|---|---|---|---|
| DAB | ×0.6 | -0.073 [-0.183, -0.046] | -0.108 [-0.202, -0.051] |
| DAB | ×0.8 | -0.016 [-0.046, -0.005] | -0.020 [-0.040, -0.013] |
| DAB | ×1.25 | +0.017 [+0.005, +0.032] | +0.032 [+0.011, +0.048] |
| DAB | ×1.5 | +0.060 [+0.023, +0.096] | +0.088 [+0.036, +0.120] |
| haematoxylin | ×0.6 | +0.003 [-0.040, +0.028] | +0.049 [-0.013, +0.086] |
| haematoxylin | ×0.8 | +0.000 [-0.008, +0.004] | +0.010 [-0.014, +0.021] |
| haematoxylin | ×1.25 | +0.001 [-0.013, +0.005] | +0.003 [-0.009, +0.020] |
| haematoxylin | ×1.5 | +0.000 [-0.038, +0.009] | +0.033 [-0.009, +0.068] |
| all stains | ×0.6 | -0.076 [-0.198, -0.017] | +0.007 [-0.029, +0.058] |
| all stains | ×0.8 | -0.009 [-0.043, -0.004] | +0.003 [-0.015, +0.012] |
| all stains | ×1.25 | +0.008 [+0.002, +0.026] | +0.018 [+0.006, +0.037] |
| all stains | ×1.5 | +0.020 [+0.008, +0.075] | +0.072 [+0.022, +0.105] |
| scan brightness | ×0.85 | -0.003 [-0.011, +0.001] | +0.049 [+0.019, +0.075] |
| scan brightness | ×0.92 | -0.002 [-0.006, +0.001] | +0.014 [+0.006, +0.024] |
| scan brightness | ×1.08 | +0.003 [-0.000, +0.015] | +0.007 [-0.006, +0.020] |
| scan brightness | ×1.15 | +0.003 [-0.007, +0.038] | +0.009 [-0.017, +0.051] |

**Pre-registered decision rules** (`variants.py`, fixed before this analysis was run): a gated variant is trained only if the median over BCI-trained networks of |shift| reaches 0.10 in at least one of its two directions.

- `illum` (illumination, ×0.85 or ×1.15): 0.049 → **not met: not trained**
- `hed` (haematoxylin, ×0.8 or ×1.25): 0.011 → **not met: not trained**

### 6h. Do FYP2-trained networks read low haematoxylin as 1+? (`analysis_haem_cue.py`)

FYP2's 1+ images carry a fifth to a quarter of the haematoxylin of its other classes (median corrected OD 0.037 vs 0.15–0.21), and every BCI class is as low (0.02–0.05). FYP2-trained networks call most BCI tiles 1+ (section 6f). Section 6g scaled haematoxylin only ×0.6–1.5, too little to test this, so here it is scaled to the other group's level and the share of tiles predicted 1+ is compared before → after.

| FYP2-trained network | A: FYP2 0/2+/3+ tiles, H ×0.2 | B: FYP2 1+ tiles, H ×5 | C: BCI tiles, H ×5 | C: BCI balanced accuracy |
|---|---|---|---|---|
| ResNet50 | 0% → 11% | 100% → 0% | 77% → 18% | 0.404 → 0.348 |
| DenseNet121 | 0% → 14% | 100% → 1% | 76% → 13% | 0.432 → 0.316 |
| ConvNeXtTiny | 0% → 14% | 100% → 47% | 54% → 29% | 0.422 → 0.342 |
| EfficientNetB0 | 0% → 12% | 100% → 36% | 80% → 47% | 0.381 → 0.374 |
| EfficientNetV2B0 | 0% → 16% | 99% → 2% | 78% → 18% | 0.323 → 0.326 |
| NASNetMobile | 0% → 49% | 100% → 44% | 71% → 10% | 0.455 → 0.355 |
| MobileNetV2 | 0% → 25% | 100% → 0% | 80% → 18% | 0.393 → 0.355 |
| MobileNetV3Large | 0% → 18% | 99% → 8% | 76% → 21% | 0.376 → 0.358 |
| MobileNetV3Small | 0% → 25% | 100% → 0% | 81% → 17% | 0.302 → 0.337 |

**Decision rule (fixed before the run):** the cue counts as confirmed if the median network moves the predicted-1+ share by ≥ 20 percentage points in the expected direction in both A (up) and B (down). Median change: A +16 points, B -97 points → **not confirmed**.

- Removing haematoxylin (A, visually plausible: nuclei fade, DAB untouched) turns only a minority of tiles into 1+. Low haematoxylin is at most one of several 1+ cues.
- Adding haematoxylin (B) removes nearly all 1+ answers, but ×5 also creates visible artefacts (yellow speckle on the background, figure D11). That direction therefore cannot be attributed to haematoxylin alone.
- On BCI tiles (C), adding haematoxylin cuts the 1+ share by a median 60 points but does not make the FYP2 networks more accurate (balanced accuracy -0.038). The haematoxylin difference changes *which* wrong answer they give, not *whether* they fail.

![](figures/figD11_haem_examples.png)

## 7. Improvement experiments

Each variant is compared with its own baseline (same model and loss) on the same tiles: Δ = variant − baseline, with a 95% fragment-cluster bootstrap CI (`paired.py`). A variant counts as an improvement only if the CI of Δ validation QWK excludes 0. Decisions use validation; test is shown only to confirm. Variant definitions are in `variants.py`.

| Model / loss | Variant | Val QWK (base → var) | Δ val QWK [95% CI] | Δ val macro-F1 [95% CI] | Δ val recall 0 [95% CI] | Test acc (base → var) | Δ test acc [95% CI] | Δ test QWK [95% CI] | Train (min) |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNetV2B0 / CE | hires | 0.869 → 0.935 | +0.065 [+0.026, +0.108] | +0.057 [+0.030, +0.087] | +0.11 [+0.02, +0.23] | 0.926 → 0.975 | +0.049 [+0.028, +0.079] | +0.044 [+0.009, +0.089] | 252 |
| MobileNetV3Small / CE | cw | 0.763 → 0.786 | +0.023 [-0.040, +0.088] | -0.000 [-0.042, +0.036] | +0.06 [-0.10, +0.20] | 0.913 → 0.906 | -0.007 [-0.024, +0.013] | -0.004 [-0.039, +0.040] | 15 |
| MobileNetV3Small / CE | hires | 0.763 → 0.865 | +0.102 [+0.031, +0.173] | +0.061 [+0.019, +0.101] | +0.17 [+0.03, +0.31] | 0.913 → 0.908 | -0.005 [-0.031, +0.024] | -0.014 [-0.063, +0.038] | 45 |
| MobileNetV3Small / CE | ls | 0.763 → 0.767 | +0.004 [-0.045, +0.054] | -0.000 [-0.029, +0.027] | -0.11 [-0.22, -0.02] | 0.913 → 0.905 | -0.008 [-0.026, +0.009] | -0.014 [-0.042, +0.007] | 16 |

## 8. Limitations

- **No slide or patient IDs.** The grouped split keeps edge-connected tiles together, but 1,645 tiles have no edge link (the file numbering is shuffled, so it gives no slide order either). Tiles of one slide that are not edge-connected can still cross splits, so grouped-split numbers may remain slightly optimistic for unseen patients. For the same reason the fragment-cluster CIs are lower bounds on the true uncertainty.
- **One held-out split.** Grouped test accuracy is 2.4–7.3 points above validation accuracy for every one of the 9 baselines: the validation tiles are harder for every model (section 5c). Comparisons between models should use paired differences on the same tiles; absolute accuracies carry several points of split-to-split uncertainty.
- **Labels are slide-level.** The patch-level labels used for diagnosis have an undocumented origin; they are a second opinion, not ground truth.
- **Single seed per configuration.** Differences of about 1–2 points between runs are within run-to-run and bootstrap noise (see the CIs).
- **Unknown relative magnification.** FYP2 and BCI images are compared at the same pixel size (256 px), but the physical magnification of the GAN images is unknown, so cross-domain results mix stain, texture *and* scale differences.
- **Resolution and batch size change together.** The 448 px (hires) runs train with a quarter of the batch size (8 instead of 32) to fit in memory, so their gain cannot be attributed to resolution alone.
- **Some steps ran on the CPU** to avoid faults in the Mac GPU driver (tensorflow-metal): ConvNeXtTiny's network analysis (its Grad-CAM gradient deadlocks on the GPU) and the evaluation of the two 448 px runs (inference hangs). Rebuilt networks reproduce the saved GPU predictions to within about 1e-6.
