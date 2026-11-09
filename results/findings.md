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
