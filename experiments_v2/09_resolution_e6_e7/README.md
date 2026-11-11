# E6 and E7: does the resolution effect generalise? (next-semester work; run 10–11 Nov 2026)

**Status: complete** (all three runs finished 11 Nov 2026, 16:12; analysed 11 Nov 2026). **Next-semester material** by
the owner's decision, although by topic it extends Stage B (model development).

**Result in one line.** Neither pre-registered comparison met the decision rule. Higher resolution helped by about
+0.04 val QWK in both, but neither CI excludes 0. The secondary metrics and the test split point the same way.

## Question and pre-registered rule

E1 showed that 448 px input is a real improvement for EfficientNetV2B0 against a batch-matched 224 px control (Δ val QWK
+0.105 [0.056, 0.168]). E2 did not confirm it for MobileNetV3Small. `NEXT_EXPERIMENTS.md` planned two follow-ups,
to run only if E1 showed a resolution effect (it did):

| | Run(s) | Compared with | Question |
|---|---|---|---|
| **E6** | `DenseNet121__ce__hires` (1024 → Lanczos 512 → bilinear 448 px, batch 4) and `DenseNet121__ce__lowbatch` (224 px, batch 4) | each other; reference `results/runs/DenseNet121__ce` (224 px, batch 16) | Does the resolution effect hold for a heavier, densely connected network? |
| **E7** | `MobileNetV3Small__ce__hires1024` (native 1024 px → 896 px input, batch 2) | `results/runs/MobileNetV3Small__ce__hires` (448 px, batch 8); context: `__ce__lowbatch`, `__ce` | Does going closer to native resolution help further? |

**Decision rule** (fixed in `NEXT_EXPERIMENTS.md` before the runs): an effect counts only if the paired 95%
fragment-cluster bootstrap CI of Δ **validation** QWK excludes 0. Test metrics are reported once and never used to
choose. The pre-registered pairs are E6 hires vs lowbatch and E7 hires1024 vs hires (448 px). All other pairs below
are context only. They were not part of the rule and must not be presented as confirmatory.

## Results

Source: `outputs/e6e7_table.csv` and `outputs/e6e7_paired.csv`, produced by `analyze_e6e7.py`. The method is the one
used in `07_ablation`: BCI grouped split, val 727 tiles and test 758, 2,000 fragment-cluster bootstrap draws, seed 2027.
Each arm is one training run (seed 42).

### Per-run metrics

| Run | Input | Batch | Best / run epochs | Train time | Val acc | Val bal. acc | Val macro-F1 | Val QWK | Test acc | Test bal. acc | Test macro-F1 | Test QWK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E6** DenseNet121 `hires` | 448 | 4 | 41 / 45 (cap) | 562 min | 0.919 | 0.901 | 0.909 | 0.861 | 0.958 | 0.950 | 0.942 | 0.949 |
| **E6** DenseNet121 `lowbatch` | 224 | 4 | 34 / 44 | 170 min | 0.893 | 0.887 | 0.890 | 0.822 | 0.939 | 0.925 | 0.913 | 0.915 |
| DenseNet121 original (reference) | 224 | 16 | 32 / 42 | 131 min | 0.908 | 0.894 | 0.901 | 0.854 | 0.942 | 0.922 | 0.927 | 0.910 |
| **E7** MobileNetV3Small `hires1024` | 896 | 2 | 40 / 45 (cap) | 323 min | 0.941 | 0.940 | 0.937 | 0.903 | 0.962 | 0.964 | 0.951 | 0.945 |
| MobileNetV3Small `hires` (E2) | 448 | 8 | 15 / 25 | 45 min | 0.898 | 0.897 | 0.893 | 0.865 | 0.908 | 0.898 | 0.893 | 0.874 |
| MobileNetV3Small `lowbatch` (E2) | 224 | 8 | 30 / 40 | 28 min | 0.857 | 0.831 | 0.844 | 0.775 | 0.905 | 0.875 | 0.881 | 0.857 |
| MobileNetV3Small original | 224 | 32 | 34 / 44 | 23 min | 0.854 | 0.822 | 0.833 | 0.763 | 0.913 | 0.886 | 0.895 | 0.888 |

Per-class recalls are in `outputs/e6e7_table.csv`.

### Paired differences (a − b, 95% fragment-cluster bootstrap CI)

| Exp. | a vs b | Role | Split | Δ acc | Δ bal. acc | Δ QWK |
|---|---|---|---|---|---|---|
| E6 | 448/b4 vs 224/b4 | **pre-registered** | val | +0.026 [+0.003, +0.048] | +0.014 [−0.015, +0.036] | **+0.038 [−0.008, +0.078]: rule not met** |
| E6 | 448/b4 vs 224/b4 | report only | test | +0.018 [+0.002, +0.039] | +0.026 [−0.003, +0.059] | +0.033 [+0.004, +0.078] |
| E6 | 448/b4 vs 224/b16 (original) | context | val | +0.011 [−0.003, +0.026] | +0.007 [−0.015, +0.031] | +0.007 [−0.024, +0.034] |
| E6 | 448/b4 vs 224/b16 (original) | report only | test | +0.016 [−0.001, +0.036] | +0.028 [−0.004, +0.064] | +0.039 [+0.009, +0.083] |
| E6 | 224/b4 vs 224/b16 (batch effect) | context | val | −0.015 [−0.037, +0.008] | −0.007 [−0.031, +0.025] | −0.032 [−0.064, +0.006] |
| E6 | 224/b4 vs 224/b16 (batch effect) | report only | test | −0.003 [−0.018, +0.012] | +0.002 [−0.024, +0.029] | +0.005 [−0.021, +0.036] |
| E7 | 896/b2 vs 448/b8 | **pre-registered** | val | +0.043 [+0.022, +0.065] | +0.042 [+0.014, +0.077] | **+0.038 [−0.005, +0.090]: rule not met** |
| E7 | 896/b2 vs 448/b8 | report only | test | +0.054 [+0.029, +0.086] | +0.066 [+0.038, +0.108] | +0.071 [+0.033, +0.135] |
| E7 | 896/b2 vs 224/b8 | context | val | +0.084 [+0.043, +0.129] | +0.109 [+0.063, +0.156] | +0.128 [+0.024, +0.255] |
| E7 | 896/b2 vs 224/b8 | report only | test | +0.057 [+0.030, +0.094] | +0.089 [+0.055, +0.136] | +0.088 [+0.033, +0.172] |
| E7 | 896/b2 vs 224/b32 (original) | context | val | +0.087 [+0.052, +0.122] | +0.118 [+0.067, +0.166] | +0.140 [+0.071, +0.221] |
| E7 | 896/b2 vs 224/b32 (original) | report only | test | +0.049 [+0.024, +0.085] | +0.078 [+0.044, +0.125] | +0.058 [+0.014, +0.125] |

Figure: `outputs/fig_e6e7_paired.png` (Δ QWK with CIs, val and test; the thick bars are the pre-registered pairs).

## Interpretation (with what it does and does not support)

1. **E6 (DenseNet121): resolution effect not confirmed by the pre-registered rule.**
   - 448 px beat the batch-matched 224 px control by +0.038 val QWK, but the CI includes 0.
   - Val accuracy (+0.026) and test QWK (+0.033) have CIs above 0 and point the same way. These are secondary or
     report-only metrics and do not count under the rule.
   - Against the original 224 px batch-16 run, 448 px is level on validation (+0.007).
   - Fair wording: "a consistent but small, unconfirmed advantage for 448 px on DenseNet121". Not "the resolution
     effect generalises".
2. **E7 (MobileNetV3Small): 896 px vs 448 px not confirmed by the pre-registered rule.** Δ val QWK is +0.038, with the
   CI including 0. Val accuracy and balanced accuracy, and every test metric, have CIs above 0. Two confounds make the
   comparison less clean than planned:
   - **Training length differed.** The 448 px comparator early-stopped at epoch 25 (best 15). The 896 px run trained
     all 45 epochs and its best epoch (40) was near the cap.
   - **Batch size differed.** 2 vs 8. Pixels per batch were matched; batch size was not.
3. **Context only (not pre-registered).**
   - At 896 px MobileNetV3Small is clearly better than both 224 px MobileNetV3Small runs on validation (e.g. +0.128
     [+0.024, +0.255] QWK vs `lowbatch`).
   - E2's 448 px vs 224 px comparison was not confirmed (+0.090 [−0.006, +0.205]).
   - This is a post-hoc comparison, so it is a hypothesis for a pre-registered multi-seed test, not a finding.
4. **Selection is unchanged.** The primary model stays `EfficientNetV2B0__ce__hires` (val QWK 0.935).
   - It is better on validation than every E6/E7 run (best: MobileNetV3Small 896 px, val QWK 0.903).
   - The 896 px MobileNetV3Small was not converted to TFLite or timed on any device. Its input has 4× the pixels of
     448 px, so its latency cannot be assumed.
5. **Batch effect on DenseNet121 (224 px, batch 4 vs 16).**
   - Val ΔQWK −0.032 [−0.064, +0.006]; the CI includes 0.
   - For comparison (`FYP3_FINAL_REPORT.md` Table 3): EfficientNetV2B0, batch 8 vs 32, val ΔQWK −0.040 [−0.109, 0.027].
     MobileNetV3Small, batch 8 vs 32, test ΔQWK −0.030 [−0.067, −0.003].

## Limits (must be stated with any E6/E7 number)

- One training seed per arm. The CIs cover tile and fragment sampling, not training variance; differences of about
  0.04 QWK are within what a seed change could produce. Multi-seed repeats are planned (roadmap §6, Step 34).
- Two of the three new runs ended at the 45-epoch cap (DenseNet121 hires, best 41/45; MobileNetV3Small hires1024,
  best 40/45), so they may be under-trained.
- BCI only (in-domain). No external (HER2-IHC-40x) evaluation of these runs was done.
- Fragments are reconstructed, not verified slides (see the roadmap, §5).

## Method

- Training code: `src/train_variant.py`, imported unchanged (two-stage schedule, seed 42, early stopping on val QWK,
  ≤ 45 epochs). `train_e6e7.py` only redirects the output folders:
  - run folders → `runs/<model>__ce__<variant>/`;
  - the new 1024/896 px cache for E7 → `experiments_v2/_cache/` (git-ignored, regenerable).
- The existing 512/448 px cache in `data/` is read only.
- Memory: batch 4 at 448 px and batch 2 at 896 px keep the pixels per batch equal to the original runs.
- Queue: `run_e6e7.sh` on the M1 Mac.
  - DenseNet121 hires: 10 Nov 22:25 → 11 Nov 07:49.
  - DenseNet121 lowbatch: → 10:40.
  - MobileNetV3Small hires1024: → 16:12.
  - All exited with rc 0. Validation was used only for early stopping (as in all FYP3 runs) and for the decision
    above. The test split was only evaluated once per run and reported.

## Files

- `train_e6e7.py`, `run_e6e7.sh`: training wrapper and queue. `analyze_e6e7.py`: paired analysis.
- `logs/train_<run>.log`, `queue.log`: training progress.
- `runs/<run>/`: `summary.json`, `history.csv`, `predictions.npz`. The checkpoints (`model.keras`,
  `model.weights.h5`) stay on the training machine and are git-ignored, like all other checkpoints in the repository.
- `outputs/`:
  - `e6e7_table.csv`: per-run metrics.
  - `e6e7_paired.csv`: all pairs.
  - `e6e7_decision.json`: the pre-registered verdicts.
  - `fig_e6e7_paired.png`, `analysis.log`.
