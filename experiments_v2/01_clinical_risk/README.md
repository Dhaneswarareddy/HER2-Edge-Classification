# Experiment 1 (v2): clinical error taxonomy and cost sensitivity

**Status:** complete, validated. No inference or training; it uses stored predictions only.
**Script:** `run_clinical_risk.py`. **Outputs:** `outputs/`. **Run:** `<project python> run_clinical_risk.py` (about 20 s).

## Inputs (read-only)

| Data | Source | n |
|---|---|---|
| BCI grouped val and test, all 20 runs | `results/runs/*/predictions.npz` | 727 / 758 tiles |
| External test, raw (10 BCI-trained, 3 FYP2-trained networks) | `results/external/her2_40x/*.csv` | 1,847 tiles |
| External test, HARMONISE-1 `stain_p99` (5 networks) | `next_experiment/outputs/test/*.csv` | 1,847 tiles |

The external test split was examined before (diagnostics, then one HARMONISE-1 evaluation). These numbers are recomputed
from those stored predictions, so **no new look at the test images was needed**.

## Method

**Objective part: error taxonomy.** Every confusion-matrix cell is assigned to a type that follows the ASCO/CAP IHC
categories: 0 and 1+ negative, 2+ equivocal (reflex in situ hybridisation, ISH), 3+ positive.

| Type | Cells (true → predicted) | Rate denominator |
|---|---|---|
| positive missed | 3+ → 0 or 1+ | true 3+ |
| equivocal missed (no reflex ISH) | 2+ → 0 or 1+ | true 2+ |
| ISH bypassed | 2+ → 3+ | true 2+ |
| false positive | 0 or 1+ → 3+ | true 0 + 1+ |
| unnecessary ISH | 0 or 1+ → 2+ | true 0 + 1+ |
| positive under-called | 3+ → 2+ | true 3+ |
| zero vs low | 0 ↔ 1+ | true 0 + 1+ |

Errors are also split into adjacent (|true − pred| = 1) and non-adjacent (≥ 2).

**Not objective: cost scenarios.** These are illustrative, were fixed before any result was computed, and are **not a
validated clinical utility**.

| Scenario | Definition |
|---|---|
| S0 | linear \|i − j\| |
| S1 | quadratic (i − j)², the QWK weighting |
| S2 workflow | 0↔1+ = 1; negative → 2+ (extra ISH) = 2; 3+ → 2+ (delay, ISH would follow) = 2; 2+ → 3+ (ISH bypassed) = 4; 2+ → negative (ISH missed) = 6; negative → 3+ = 8; 3+ → negative = 10 |
| S3 HER2-low-aware | S2 with 0↔1+ = 4. Since the ASCO/CAP 2023 update, separating 0 from 1+ matters for HER2-low therapy eligibility |

Reasoning behind S2's ordering:
- Missing a positive or an equivocal case removes a confirmatory step, so it costs most.
- Calling a negative positive risks over-treatment, so it costs nearly as much.
- An extra ISH test or a delay costs little.

**Sensitivity check.** Over 2,000 draws, each S2 cost is scaled independently by a factor uniform in [0.5, 1.5]. The
check records which model has the lowest expected cost in each draw.

**Uncertainty.** BCI uses a fragment-cluster bootstrap (2,000 draws). The external split uses a filename-run bootstrap.
The external release has no slide IDs, so those intervals are optimistic.

**Selection rule.** Any choice among BCI models uses validation only. BCI test and every external number are
report-only.

## Results

Expected S2 cost per tile, with 95% CI:

| Domain | Model | S2 cost | 3+ missed | 2+ missed | 2+ → 3+ | neg → 3+ | neg → 2+ | non-adjacent errors |
|---|---|---|---|---|---|---|---|---|
| BCI test | EffNetV2B0 448 px (primary) | **0.12** [0.06, 0.22] | 2 | 2 | 4 | 4 | 3 | 7 of 19 |
| BCI test | EffNetV2B0 224 px | 0.31 [0.20, 0.48] | 5 | 17 | 3 | 2 | 12 | 10 of 56 |
| BCI test | DenseNet121 224 | 0.28 | 7 | 9 | 4 | 5 | 9 | 16 of 44 |
| External | EffNetV2B0 224 raw | 3.65 [3.00, 4.29] | 0 | 6 | 104 | **758** | 80 | 815 of 1,019 |
| External | EffNetV2B0 224 + stain_p99 | 1.16 [0.95, 1.37] | 8 | **36** | 12 | 64 | **595** | 428 of 793 |
| External | MobileNetV3Small + stain_p99 | 1.26 [0.92, 1.62] | 4 | 35 | 66 | **205** | 24 | 226 of 452 |
| External | EffNetV2B0 448 + stain_p99 | 3.47 | **298** | 47 | 12 | 85 | 807 | 975 of 1,655 |
| External | FYP2 (GAN-trained) ResNet50, raw | **0.54** [0.38, 0.72] | 0 | 40 | 1 | 36 | 20 | 45 of 510 |

Paired differences (`outputs/paired_cost_differences.csv`). A CI excluding 0 is a real difference by the project's rule.

| Comparison | Split | Δ S2 cost [95% CI] |
|---|---|---|
| 448 px − 224 px, EffNetV2B0 | val | −0.19 [−0.31, −0.07] |
| 448 px − 224 px, EffNetV2B0 | test | −0.19 [−0.33, −0.09] |
| 448 px − 224 px at batch 8 | val | −0.44 [−0.61, −0.27] |
| CORN − CE, MobileNetV3Large | val | −0.09 [−0.17, −0.01] |
| CORN − CE, EfficientNetV2B0 | val | +0.15 [+0.01, +0.30] |
| CORN − CE, DenseNet121 | val | −0.05 [−0.18, +0.09] |
| stain_p99 − raw, external, 4 of the 224 px networks | ext. test | −1.2 to −2.6, all CIs exclude 0 |
| stain_p99 − raw, external, 448 px network | ext. test | **+1.49 [+1.03, +1.97]** (worse) |

Sensitivity (`outputs/cost_sensitivity.json`):
- **BCI validation:** the 448 px EfficientNetV2B0 has the lowest cost in **100%** of 2,000 cost draws.
- **External test:** the FYP2-trained ResNet50 without calibration has the lowest cost in **100%** of draws. Every
  HARMONISE-calibrated BCI network loses to it in every draw.

## What this adds

1. **In-domain.** The 448 px model's lower error rate is also lower clinical risk, and the ranking holds under every
   tested cost weighting. It makes fewer errors in total (19 vs 56 on test). A larger share of its few errors are
   non-adjacent (7 of 19), though the absolute number is the same as for the 224 px model.
2. **HARMONISE changes which errors happen; it does not remove the risk.** On the 224 px EfficientNetV2B0:
   - Negatives called 3+ fall from 758 to 64, and 2+ called 3+ from 104 to 12.
   - But 2+ tiles called negative (missed ISH) rise from 6 to 36, a rate of 0.32.
   - 595 of 974 negative tiles (61%) are now called 2+, which in a real workflow would trigger unnecessary ISH.
   - Calibration turns a false-positive problem into an over-referral problem.
3. **The QWK paradox has a clinical form.** MobileNetV3Small + stain_p99 reaches QWK 0.80, but it calls 47% of true 1+
   tiles 3+ (205 false positives in total).
4. **The 448 px model is clinically unsafe on the external data.** Raw, it misses 109 of 762 3+ tiles. With stain_p99
   it misses 298, which is the worst of any configuration.
5. **The GAN-trained FYP2 ResNet50 has the lowest risk on the external set.** This matches the earlier cross-dataset
   audit, which explained it by the stain distribution, not by better learning. This result should be in the report.
6. **CORN vs CE depends on the architecture.** Against CE, CORN lowers S2 cost for MobileNetV3Large, raises it for
   EfficientNetV2B0, and makes no clear difference for DenseNet121. This is the same "no consistent CORN benefit"
   finding seen with QWK.

## Limitations

- The cost values are illustrative. Only the error counts and rates are objective. The rankings above hold under the
  ±50% sensitivity range, but other weightings could exist.
- Tile-level only. There is no slide or patient aggregation, so these are not case-level clinical error rates.
- External CIs are optimistic because there are no slide IDs. BCI fragments are reconstructed, not verified slides.
- The external test split was examined previously, so its results are not untouched first-time test results.

## Files

| File | Contents |
|---|---|
| `outputs/risk_table_all.csv` | Every run and split: metrics, per-class P/R/F1, taxonomy counts and rates, S0–S3 costs with CIs, confusion matrix |
| `outputs/risk_table_key.csv` | Summary rows used above |
| `outputs/paired_cost_differences.csv` | Paired Δ cost with CIs |
| `outputs/cost_sensitivity.json` | Share of cost draws in which each model is lowest |
| `outputs/fig_risk_heatmaps.png` | Row-normalised confusions with high-cost cells outlined (300 dpi) |
| `outputs/fig_error_taxonomy.png` | Taxonomy rates for 6 key configurations |
| `outputs/config.json` | Taxonomy, cost parameters, seeds, bootstrap settings |
| `outputs/run.log` | Console output |
