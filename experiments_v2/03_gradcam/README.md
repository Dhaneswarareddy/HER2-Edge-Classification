# Experiment 3: Grad-CAM for the deployment models (completes earlier work)

**Already done before this experiment (reused, not repeated).** `src/analysis_models.py` produced Grad-CAM statistics
for the nine 224 px baseline networks, with figures showing two **correct** tiles per class
(`results/figures/figD6_gradcam_*.png`). It did not cover:
- the deployed 448 px model;
- the external set;
- wrong predictions;
- a 2+ focus.

**New here.** Two models:
- EfficientNetV2B0__ce__hires (448 px, deployed primary);
- EfficientNetV2B0__ce (224 px, HARMONISE model).

Five tile sets:

| Set | Tiles |
|---|---|
| BCI grouped test, both models | 758 each |
| External test, raw, both models | 1,847 each |
| External test with stain_p99, 224 px | 1,847 |

Correct and wrong predictions are both included, with rule-based example selection (no hand-picking).

**Status.** Complete.

## Method

- **Grad-CAM.**
  - Gradient of the predicted-class probability with respect to `top_activation`.
  - Channel weights are spatially averaged gradients, followed by ReLU.
  - Bilinear upsampling; each map sums to 1.
  - The map is 7×7 at 224 px and 14×14 at 448 px, so it resolves regions of about 32 px. It **cannot** localise
    membranes. It shows where the score is sensitive, not what the network "recognises".
- **Per-tile statistics** (same definitions as `analysis_models.cam_stats`):
  - `dab_ratio`: > 1 means attention is on above-average brown.
  - `bg_ratio`: < 1 means blank glass is under-attended. Background is HSV S < 8; computed only when ≥ 5% of the
    tile is background.
  - `top_dab_mass`: share of the map on the 10% most DAB-dense pixels; 0.10 means uniform.
- **External split.** Previously examined; nothing is fitted or selected on it here.
- **Reproduction.** The predicted probabilities match the stored predictions to ≤ 4e-6 for all 5 sets
  (`outputs/reproduction_check.json`).

> **Finding on the way: uint8 sensitivity of HARMONISE.**
> - The first stain_p99 pass cast the calibrated image to uint8 before inference.
> - That changed predicted probabilities by up to **0.15**, and 1.4% of argmax labels.
> - HARMONISE-1 fed float32. With float32 input, the rerun reproduces its stored predictions to 7e-7.
> - Consequence: any deployment of stain_p99 that stores or transmits the calibrated tile as an 8-bit image (PNG,
>   JPEG or a camera buffer) will not reproduce the reported metrics exactly.

## Results (medians; `outputs/cam_stats_summary.csv`, `outputs/cam_stats_per_tile.csv`)

| Set | Correct n / wrong n | DAB ratio, correct / wrong | Bg ratio, correct / wrong |
|---|---|---|---|
| BCI test, 448 px | 739 / 19 | 1.15 / 1.30 | 0.86 / 0.91 |
| BCI test, 224 px | 702 / 56 | 1.09 / 1.08 | 0.91 / 0.97 |
| External raw, 448 px | 804 / 1,043 | 1.33 / 1.05 | 0.66 / 0.97 |
| External raw, 224 px | 828 / 1,019 | 1.20 / 1.09 | 0.71 / 0.88 |
| External stain_p99, 224 px | 1,054 / 793 | 1.22 / 1.11 | 0.69 / 0.95 |

Per class, correct tiles, DAB ratio / bg ratio:

| True class | BCI 448 | BCI 224 | External raw 448 | External raw 224 | External p99 224 |
|---|---|---|---|---|---|
| 0 | 0.99 / 1.00 | 1.04 / 0.98 | 1.15 / 0.90 | 1.07 / 1.10 (n=7) | 1.01 / 0.97 |
| 1+ | 1.11 / 0.89 | 1.06 / 0.91 | **0.85 / 1.20** | **0.80 / 1.23** | **0.89 / 1.07** |
| 2+ | 1.15 / 0.89 | 1.08 / 0.94 | n=1 | n=1 | 1.18 / 0.80 (n=63) |
| 3+ | 1.29 / 0.66 | 1.23 / 0.75 | 1.42 / 0.55 | 1.22 / 0.69 | 1.34 / 0.58 |

## Interpretation (descriptive, with caution)

1. **3+ attention follows strong DAB and avoids glass, in both domains.** For correct 3+ tiles, DAB ratio is
   1.22–1.42 and bg ratio is 0.55–0.75. That is consistent with the score depending on strongly brown regions; it
   does not show membrane-pattern use.
2. **On the external set, correct 1+ calls attend to pale areas and glass.** DAB ratio is 0.80–0.89 and bg ratio is
   1.07–1.23, for both models and with or without stain_p99. The 1+ decision there appears to rest on overall pale
   appearance or white space, not on faint membrane staining: a plausible shortcut.
3. **Errors are concentrated on tiles with a lot of blank glass.**
   - Wrong BCI tiles are > 50% background in 55% (224 px) and 63% (448 px) of cases, vs 32% of correct tiles.
   - On the external set the figures are 32–66% vs 4–30%.
   - In `fig_gradcam_bci448_per_class.png`, every one of the four most confident BCI 448 px errors is a sparse, mostly
     empty tile carrying its parent image's label (e.g. true 0 → 3+ at p = 0.91 with a tiny focus on a dark speck).
   - Label noise from sparse tiles is a likely contributor to the residual BCI errors.
4. **DAB attention separates correct from wrong external predictions, but not BCI ones.**
   - External: correct tiles have higher DAB ratio (Mann–Whitney p < 1e-18 in all three external sets).
   - BCI: no separation (p = 0.12 at 224 px; p = 0.44 at 448 px).
   - These p-values are descriptive. Tiles are not independent (same slides), so they overstate certainty.
5. **2+ remains the weak point.**
   - Raw external input yields only 1 correct 2+ tile at either resolution, so its 2+ attention cannot be
     characterised.
   - With stain_p99, 63 of 111 2+ tiles are correct. In `fig_gradcam_ext224_p99_2plus.png`, the missed 2+ tiles have
     clearly visible complete membranes, but attention sits on a corner or on glass.
6. **stain_p99 changes how the input looks.** Haematoxylin-stained nuclei become pale cyan in the p99-calibrated
   inputs (visible in every p99 panel). The network sees colours that never occur in BCI training data.

## Limitations

- Grad-CAM at 7×7 or 14×14 is coarse. The statistics are associations, not evidence of what the network uses.
  No deletion/insertion faithfulness test was run.
- Only one Grad-CAM variant and one layer. No pathologist annotation of membranes, so "on DAB" is a colour proxy.
- External 2+ under raw input has n = 1 correct tile per model, so no 2+ statement is possible there.

## Files

- **Scripts:** `run_gradcam.py` (main run, plus `--only-p99` for the float rerun); `make_figures.py`.
- **Statistics:**
  - `outputs/cam_stats_per_tile.csv` (7,057 rows = 2 × 758 + 3 × 1,847);
  - `outputs/cam_stats_summary.csv`;
  - `outputs/cams_<set>.npz` (low-resolution maps and probabilities).
- **Checks and settings:** `outputs/reproduction_check.json`, `outputs/config.json`.
- **Figures** (300 dpi; every example listed with name, label, prediction and statistics in
  `outputs/figure_examples.csv`):
  - `fig_gradcam_bci448_per_class.png`
  - `fig_gradcam_bci224_per_class.png`
  - `fig_gradcam_bci448_2plus.png`
  - `fig_gradcam_ext224_p99_2plus.png`
  - `fig_gradcam_ext224_raw_per_class.png`
  - `fig_gradcam_ext224_p99_per_class.png`
  - `fig_gradcam_ext448_raw_per_class.png`
  - `fig_gradcam_bci448_background.png`
  - `fig_gradcam_ext448_raw_background.png`
  - `fig_cam_statistics.png`
- **Logs:** `outputs/run.log`, `outputs/run_p99.log`, `outputs/figures.log`.
