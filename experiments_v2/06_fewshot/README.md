# Experiment 6: few-shot target adaptation (new)

**Question.** How much do a few *labelled* HER2-IHC-40x tiles help the 224 px EfficientNetV2B0 (CE) model, compared
with zero-shot, HARMONISE stain_p99 and AdaBN (both unsupervised)?

**Status.** Complete. 12 tuning configurations × 3 seeds, then 60 final runs (3 regimes × k ∈ {1, 5, 10, 20} × 5 support
draws). Every run evaluated on development R_eval and on the external test split.

## Protocol (fixed before any test evaluation)

| Role | Data |
|---|---|
| Support pool (labelled tiles drawn from here) | HARMONISE-1 fit + selection pools, 3,500 development tiles ([1366, 799, 222, 1113] per class) |
| Tuning (every hyperparameter choice) | R_cal, 2,191 development tiles |
| Development report | R_eval, 2,402 development tiles |
| Test | 1,847-tile external test split. **Previously examined; exploratory second look; never used for any choice** |

- **Regimes.** BatchNorm stays in inference mode in all three, as in FYP3 training.
  - `head`: final Dense layer only, trained on frozen pooled features.
  - `partial`: `block6*` + `top*` layers + head.
  - `full`: all layers.
  - `partial` and `full` use dihedral augmentation.
- **Tuning.**
  - k = 10, seeds 101–103.
  - Grid: 4 configurations per regime (`run_fewshot.py`, `GRID`).
  - The configuration with the highest mean R_cal balanced accuracy was chosen (`outputs/tuning_choice.json`):
    - head: lr 1e-2, 200 steps;
    - partial: lr 1e-3, 30 steps;
    - full: lr 1e-4, 100 steps.
- **Final runs.** Seeds 1–5, disjoint from the tuning draws; k tiles per class sampled without replacement.

## Results

Balanced accuracy, mean ± SD over 5 support draws. k = labelled tiles **per class** (total labels = 4k).

| Regime | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| **External test (exploratory)** | | | | |
| head | 0.469 ± 0.031 | 0.725 ± 0.090 | 0.758 ± 0.056 | 0.780 ± 0.058 |
| partial | 0.535 ± 0.079 | 0.792 ± 0.042 | 0.728 ± 0.069 | 0.755 ± 0.045 |
| full | 0.544 ± 0.024 | 0.782 ± 0.053 | 0.813 ± 0.043 | **0.845 ± 0.048** |
| **Development R_eval** | | | | |
| head | 0.406 ± 0.051 | 0.690 ± 0.045 | 0.759 ± 0.029 | 0.808 ± 0.019 |
| partial | 0.537 ± 0.083 | 0.706 ± 0.036 | 0.742 ± 0.049 | 0.786 ± 0.040 |
| full | 0.523 ± 0.061 | 0.736 ± 0.032 | 0.830 ± 0.020 | 0.855 ± 0.034 |

References on the same tiles (`outputs/reference_methods.csv`):

| Method | Labels used | R_eval bal. acc | Test bal. acc | Test QWK | Test 2+ recall | Test 2+ precision |
|---|---|---|---|---|---|---|
| Zero-shot (raw) | 0 | 0.298 | 0.301 | 0.108 | 0.009 | 0.012 |
| HARMONISE stain_p99 | 0 | 0.505 | 0.503 | 0.593 | 0.568 | 0.093 |
| AdaBN (raw inputs) | 0 | 0.432 | 0.522 | 0.704 | 0.198 | 0.182 |
| AdaBN + stain_p99 | 0 | 0.403 | 0.486 | 0.702 | 0.144 | 0.182 |
| Few-shot full, k=5 (20 labels) | 20 | 0.736 | 0.782 | 0.918 | 0.780 | 0.441 |
| Few-shot full, k=20 (80 labels) | 80 | 0.855 | 0.845 | 0.950 | 0.892 | 0.628 |

**Paired gain over zero-shot on the test tiles.** The gain in balanced accuracy is positive for every one of the
60 runs. Smallest per-seed gains (`outputs/gain_over_zero_shot_test.csv`):

| Setting | Smallest gain |
|---|---|
| head, k=1 | +0.11 |
| full, k=5 | +0.44 |
| full, k=20 | +0.46 |

## Interpretation

1. **Twenty labelled tiles beat both unsupervised methods by a wide margin.**
   - With 5 tiles per class, every regime is above HARMONISE stain_p99 on both sets: test balanced accuracy 0.73–0.79
     vs 0.50.
   - 2+ precision rises from 0.09 to 0.44.
   - At k = 1, only partial and full reach about the HARMONISE level, and their variance is large.
2. **Full fine-tuning is the best or tied-best regime from k = 10.** The head-only regime is weakest at small k.
3. **Partial is non-monotone** (test 0.792 at k=5, 0.728 at k=10). Its chosen setting (lr 1e-3, 30 steps) scored
   0.865 on R_cal during tuning but 0.742 on R_eval in the final runs. That is a 0.12 drop, larger than the drops
   for full (0.876 → 0.830) and head (0.737 → 0.759). The high-learning-rate setting is unstable, and choosing it from
   3 draws was optimistic (winner's curse). This negative result is kept as is; no setting was re-chosen after
   seeing it.
4. **2+ remains the hardest class.** Even at k = 20, 2+ precision is about 0.63, so roughly a third of tiles called
   2+ are not 2+.

## Limitations

- **Development results may be optimistic.** The development tiles have no slide IDs, so R_cal and R_eval can share
  slides with the support tiles. The test split comes from 21 different slides and is the cleaner estimate, but it is
  **not** an untouched test set.
- **The test set is easy to separate by colour.** The HER2-IHC-40x release was hue-histogram filtered per class
  (`../REPORT_CORRECTIONS_v2.md`, item 20). A real deployment site would probably need more labels.
- **One grid choice sits at the edge of the grid.** For full, lr 1e-4 and 100 steps are both the largest values
  tried, so a larger budget might do better. This was not explored, to avoid tuning more.
- **Scope is narrow.** One architecture (224 px EfficientNetV2B0 CE), 5 draws per k, and tile-level labels. No
  pathologist time per label was measured.

## Files

- **Scripts:**
  - `run_fewshot.py` (stages `tune` and `final`);
  - `analyze_fewshot.py`.
- **Tuning outputs:**
  - `outputs/tune_results.csv`, `outputs/tuning_summary_head_partial_full.csv`, `outputs/tuning_choice.json`.
- **Final outputs:**
  - `outputs/final_results.csv` (per run, per set);
  - `outputs/final_test_probs_head_partial_full.npz`;
  - `outputs/final_summary_by_k.csv`, `outputs/reference_methods.csv`, `outputs/gain_over_zero_shot_test.csv`.
- **Figure:** `outputs/fig_fewshot_data_efficiency.png`.
- **Logs:** `outputs/tune.log`, `outputs/final.log`.
- **Process note:** the final process hung in TensorFlow teardown after writing every output, as the tuning
  process did. It was stopped after all files were verified (60 + 60 + 3 rows; 60 probability arrays).
