# Experiment 4: image-quality and staining robustness (extends earlier work)

**Already done before.** `src/analysis_models.py` scaled DAB, haematoxylin and brightness for the nine 224 px
baselines, and reported only shifts in expected score.

**New here.**
- Blur, JPEG, contrast and sensor noise.
- The deployed 448 px model.
- Accuracy, balanced accuracy, macro-F1 and per-class recall under every condition.
- A paired 95% CI against clean images (fragment-cluster bootstrap, 2,000 draws).
- Prediction flip rate.

**Status.** Complete. 22 conditions × 2 models on the 758-tile BCI grouped **test** split, report only (nothing was
tuned on it).

## Method

- **Where the perturbation is applied.** On the original 1024 px tile (0.46 µm/px). Each model's own preprocessing then
  runs unchanged:
  - Lanczos to 256 or 512 px;
  - bilinear to 224 or 448 px.
  The same physical change therefore reaches both models.
- **Conditions:**

  | Perturbation | Levels |
  |---|---|
  | Brightness (illumination gain) | ×0.85, 0.92, 1.08, 1.15 |
  | Contrast | ×0.8, 0.9, 1.1, 1.2 |
  | Gaussian blur, σ in native px | 1, 2, 4 (= 0.46, 0.92, 1.84 µm) |
  | JPEG quality | 90, 70, 50, 30 |
  | Gaussian noise, σ in grey levels | 2, 5 |
  | DAB optical density | ×0.8, 1.2 |
  | Haematoxylin optical density | ×0.8, 1.2 |

  The two stain perturbations are Ruifrok–Johnston, relative to each tile's illumination.
- **Clean check.** The clean condition reproduces the stored test predictions: max |Δp| 1.5e-6 (448 px) and
  3.9e-6 (224 px), in `outputs/reproduction_check.json`.

## Results

Δ balanced accuracy vs clean, with 95% CI. Clean balanced accuracy: **448 px 0.973; 224 px 0.917.**

| Condition | 448 px Δ bal. acc [CI] | 448 px flip | 224 px Δ bal. acc [CI] | 224 px flip |
|---|---|---|---|---|
| Brightness ×0.85 | **−0.120** [−0.179, −0.077] | 6.2% | **−0.177** [−0.240, −0.125] | 14.5% |
| Brightness ×0.92 | −0.024 [−0.051, −0.007] | 1.7% | −0.045 [−0.080, −0.013] | 5.0% |
| Brightness ×1.08 | −0.055 [−0.087, −0.028] | 4.4% | −0.030 [−0.068, +0.005] | 5.9% |
| Brightness ×1.15 | **−0.187** [−0.249, −0.131] | 14.8% | **−0.154** [−0.217, −0.096] | 16.0% |
| Contrast ×0.8 | −0.019 [−0.049, −0.001] | 2.5% | −0.044 [−0.078, −0.023] | 5.0% |
| Contrast ×0.9 / ×1.1 | ≈ 0 (CIs include 0) | ≤ 0.7% | ≈ 0 (CIs include 0) | ≤ 1.8% |
| Contrast ×1.2 | −0.014 [−0.035, −0.001] | 1.6% | +0.002 [−0.016, +0.028] | 4.5% |
| Blur σ 1 | −0.053 [−0.095, −0.022] | 3.4% | −0.007 [−0.026, +0.004] | 0.8% |
| Blur σ 2 | **−0.166** [−0.231, −0.122] | 13.1% | −0.026 [−0.052, −0.008] | 3.2% |
| Blur σ 4 | **−0.419** [−0.505, −0.364] | 34.6% | **−0.168** [−0.233, −0.120] | 16.1% |
| JPEG 90 / 70 | ≈ 0 | ≤ 0.8% | ≈ 0 | ≤ 0.9% |
| JPEG 50 | −0.024 [−0.045, −0.011] | 2.6% | −0.014 [−0.037, −0.000] | 2.8% |
| JPEG 30 | −0.046 [−0.078, −0.025] | 5.7% | −0.035 [−0.076, −0.005] | 9.0% |
| Noise σ 2 / 5 | ≈ 0 | ≤ 0.7% | ≈ 0 | ≤ 1.1% |
| DAB ×0.8 | −0.010 [−0.019, −0.003] | 1.8% | −0.006 [−0.025, +0.012] | 4.2% |
| DAB ×1.2 | −0.013 [−0.040, +0.005] | 1.7% | −0.010 [−0.033, +0.010] | 3.7% |
| Haematoxylin ×0.8 | −0.031 [−0.063, −0.007] | 2.1% | −0.030 [−0.055, −0.011] | 5.0% |
| Haematoxylin ×1.2 | −0.012 [−0.033, +0.000] | 1.6% | +0.008 [−0.008, +0.029] | 2.4% |

Full table, including per-class precision and recall, QWK, confusion matrices, and CIs for accuracy and macro-F1:
`outputs/robustness_results.csv`.

## Interpretation

1. **Robust to compression, noise and moderate stain change.**
   - JPEG ≥ 70 and sensor noise: essentially no effect.
   - DAB or haematoxylin ±20%, and contrast ±10%: effects within about 3 points.
   - JPEG 50 is acceptable; JPEG 30 costs 3–5 points.
2. **Brightness is the main acquisition risk for both models.** A ±15% illumination change costs 12–19 points of
   balanced accuracy, and even ±8% costs 2–6 points. The errors are concentrated in class 0:
   - darker images: recall for class 0 falls 0.97 → 0.61 (448 px) and 0.89 → 0.44 (224 px);
   - brighter images: recall for class 1+ falls to about 0.6–0.7.
   Scanner or camera white balance and exposure must be controlled or normalised before deployment.
3. **The 448 px model is defocus-sensitive, and the resolution gain reverses under blur.**
   - At σ = 2 native px (0.9 µm; mild defocus), the 448 px model loses 0.17 and falls **below** the 224 px model
     (0.806 vs 0.891).
   - At σ = 4 it loses 0.42: 1+ recall 0.96 → 0.15, while 2+ recall stays at 0.97 with 2+ precision 0.57. Blurred
     tiles are pushed towards **2+**.
   - The 224 px model downsamples 4.6×, which hides fine blur.
   - Consequence: the higher-resolution model's advantage (`../REPORT_CORRECTIONS_v2.md`, item 22) depends on sharp
     focus. A focus or blur check is needed before using it on phone-camera or poorly focused scans.
4. **Comparison with the external domain.** Single-factor stain changes of ±20% are tolerated well. The external-set
   failure therefore cannot come from a modest DAB or haematoxylin scaling. That is consistent with the HARMONISE-2
   finding that the external shift is large (haematoxylin 3–7× stronger) and multi-factor.

## Limitations

- **Synthetic, one factor at a time.** Real defocus is spatially varying, and real scanner differences combine
  colour, brightness, sharpness and compression.
- **In-domain test only (BCI).** Robustness on the external set was not measured.
- **CI scope.** One training seed per model, so the CIs cover tile and fragment sampling, not training randomness.
- **Optimistic clustering.** 758 test tiles from 691 reconstructed fragments; fragment clusters are an approximation
  of slides.

## Files

- **Script:** `run_robustness.py`.
- **Results:**
  - `outputs/robustness_results.csv`;
  - `outputs/probs_<run>.npz` (probabilities per condition);
  - `outputs/reproduction_check.json`;
  - `outputs/config.json`.
- **Figure:** `outputs/fig_robustness.png`.
- **Log:** `outputs/run.log`.
