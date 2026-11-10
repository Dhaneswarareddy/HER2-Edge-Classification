# Experiment 2 (v2): confidence, calibration and selective prediction (gaps only)

**Status:** complete, validated. Uses stored predictions only, with no inference.
**Script:** `run_uncertainty.py`. **Outputs:** `outputs/`.

## What was already done (reused, not repeated)

HARMONISE-2 experiment E (`next_experiment/harmonise2/outputs/E_*.csv`) covers 5 networks on HER2-IHC-40x development
tiles: ECE, NLL, temperature scaling, reliability curves and confidence-based referral. Temperature T and referral
thresholds were fitted on R_cal (2,191 tiles) and reported on R_eval (2,402 tiles). Its conclusion: referral removes
up to 76–94% of 2+ tiles and is not usable.

## What is new here

| Part | Fit on | Reported on | Status of the report set |
|---|---|---|---|
| A. Calibration (ECE, NLL, Brier, T), all 20 BCI runs | BCI val | BCI val and test | Test is report-only |
| B. Selective prediction, all 20 BCI runs | BCI val (thresholds) | BCI test | Fragment-cluster CIs for the two deployment models |
| C. External calibration and selective prediction, 5 HARMONISE panel networks, raw and stain_p99 | R_cal development tiles | 1,847-tile external test split | **Exploratory second look**: the split was examined before. Nothing was fitted on it |

Settings:
- 15 equal-width bins. T comes from a 120-point log grid over [0.2, 20], minimising NLL.
- Three confidence signals: max softmax, top-2 margin, negative entropy.
- Target coverages: 0.5, 0.7, 0.8, 0.9, 0.95. Each threshold is a quantile of the fit set; the realised coverage on the
  report set is printed next to it.

Softmax confidence is a heuristic score, not statistically validated uncertainty, and none of this carries a coverage
guarantee.

## Results

**A. In-domain calibration is already good** (`outputs/bci_calibration_all_runs.csv`):
- BCI test ECE is 0.010–0.078 across the 20 runs. The 448 px model is 0.017, the 224 px EfficientNetV2B0 0.045.
- Temperature scaling fitted on val helps only some runs on test. It lowers ECE for 224 px EfficientNetV2B0
  (0.045 → 0.023) but raises it for DenseNet121 (0.010 → 0.026), MobileNetV2 and others.
- On BCI, temperature scaling is not reliably useful. That is a negative result.
- Label smoothing gives the worst ECE (0.078).

**B. In-domain referral works, and does not single out 2+** (`outputs/bci_selective_prediction_all_runs.csv`):

| Model (BCI test) | Target coverage | Realised coverage | Accuracy all → kept | Gain [95% CI] | 2+ referred | Errors referred | Error AUROC |
|---|---|---|---|---|---|---|---|
| 448 px | 0.90 | 0.937 | 0.975 → 0.989 | +0.014 [+0.005, +0.026] | 3.7% | 58% | 0.955 |
| 448 px | 0.80 | 0.851 | 0.975 → 0.998 | +0.023 [+0.012, +0.039] | 11.5% | 95% | 0.955 |
| 224 px | 0.90 | 0.925 | 0.926 → 0.959 | +0.033 [+0.020, +0.052] | 6.2% | 48% | 0.932 |
| 224 px | 0.80 | 0.859 | 0.926 → 0.980 | +0.054 [+0.035, +0.083] | 14.2% | 77% | 0.932 |

- Across all 20 runs, error-detection AUROC is 0.87–0.96. The three signals perform about the same (median 0.93).
- In-domain, 2+ tiles are referred at roughly the overall referral rate, so referral is not biased against 2+ here.

**C. External test (exploratory): the HARMONISE-2 conclusion holds on the slide-independent split.**

224 px EfficientNetV2B0 + stain_p99, thresholds from R_cal:

| Target coverage | Coverage | Accuracy all → kept | 2+ referred | Errors referred |
|---|---|---|---|---|
| 0.9 | 0.903 | 0.571 → 0.599 | 14.4% | 15.6% |
| 0.7 | 0.705 | 0.571 → 0.678 | **50.5%** | 47.2% |
| 0.5 | 0.530 | 0.571 → 0.800 | **74.8%** | 75.3% |

Temperature T fitted on R_cal and carried to the test split lowers ECE for every 224 px network. For example,
EfficientNetV2B0 + stain_p99 drops from 0.205 to 0.099, close to R_eval's 0.100, so the development-fitted T transfers.

Two fits stop at the grid edge (T = 20): DenseNet121 raw and the 448 px model with stain_p99. Their confidence is
essentially uninformative there.

## Interpretation

- In-domain, confidence is meaningful. Referring the least confident 6–15% of tiles to a pathologist catches roughly
  half to 95% of errors without disproportionately removing 2+ cases. This is the strongest positive uncertainty result
  in the project. It is tile-level only and comes from one split.
- Out of domain, referral concentrates on 2+ tiles. Half of the 2+ tiles are referred at 70% coverage, so a referral
  workflow would push the equivocal category back to manual review. The referral itself is not harmful, but it cannot
  substitute for human review of 2+. This now holds on the slide-independent test split as well as on development tiles.
- Calibration on new-domain data can be repaired with temperature scaling fitted on unlabelled-then-labelled
  development tiles, but this needs labelled target data (R_cal labels were used to fit T).

## Files

| File | Contents |
|---|---|
| `outputs/bci_calibration_all_runs.csv` | ECE, NLL, Brier, accuracy and mean confidence, raw and temperature-scaled, 20 runs × val/test |
| `outputs/bci_selective_prediction_all_runs.csv` | 20 runs × 3 signals × 5 coverages: thresholds from val, metrics on test, share of each class referred, CIs for the deployment models |
| `outputs/external_calibration_test.csv` | External test calibration, T from R_cal |
| `outputs/external_selective_prediction_test.csv` | External test selective prediction, thresholds from R_cal |
| `outputs/reliability_curves.csv` | Bin tables behind the figure |
| `outputs/fig_uncertainty.png` | Selective-prediction curves, reliability diagrams, confidence by true class (300 dpi) |
| `outputs/config.json`, `outputs/run.log` | Settings and console output |
