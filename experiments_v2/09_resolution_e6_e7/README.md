# E6 and E7: does the resolution effect generalise? (next-semester work; started 10 Nov 2026)

**Status: running.** No results yet. Nothing about the outcome of E6 or E7 may be reported until `runs/<name>/summary.json`
exists and the paired analysis below has been run.

## Question and pre-registered rule

E1 showed that 448 px input is a real improvement for EfficientNetV2B0 against a batch-matched 224 px control (Δ val QWK
+0.105 [0.056, 0.168]); E2 did not confirm it for MobileNetV3Small. `NEXT_EXPERIMENTS.md` planned two follow-ups, to run
only if E1 showed a resolution effect (it did):

| | Run(s) | Compared with | Question |
|---|---|---|---|
| **E6** | `DenseNet121__ce__hires` (1024 → Lanczos 512 → bilinear 448 px, batch 4) and `DenseNet121__ce__lowbatch` (224 px, batch 4) | each other; reference `results/runs/DenseNet121__ce` (224 px, batch 16) | Does the resolution effect hold for a heavier, densely connected network? |
| **E7** | `MobileNetV3Small__ce__hires1024` (native 1024 px → 896 px input, batch 2) | `results/runs/MobileNetV3Small__ce__hires` (448 px) and `__ce__lowbatch` | Does going closer to native resolution help further? |

Decision rule (fixed before the runs, as in `NEXT_EXPERIMENTS.md`): an effect counts only if the paired 95%
fragment-cluster bootstrap CI of Δ **validation** QWK excludes 0. Test metrics are reported once, never used to choose.

## Method

- Training code: `src/train_variant.py`, imported unchanged (two-stage schedule, seed 42, early stopping on val QWK,
  ≤ 45 epochs). `train_e6e7.py` only redirects the output folders:
  - run folders → `runs/<model>__ce__<variant>/`;
  - the new 1024/896 px cache for E7 → `experiments_v2/_cache/` (git-ignored, regenerable).
- The existing 512/448 px cache in `data/` is read only.
- Memory: batch 4 at 448 px and batch 2 at 896 px keep the pixels per batch equal to the original runs.
- Queue: `run_e6e7.sh` (DenseNet121 hires → DenseNet121 lowbatch → MobileNetV3Small hires1024), started 10 Nov 2026
  22:25 on the M1 Mac. Expected: E6 ≈ 14–16 h, E7 ≈ 3–5 h.

## After the runs finish

A paired analysis script (fragment-cluster bootstrap from `experiments_v2/v2lib.py`, as in
`07_ablation/analyze_ablation.py`) will be added here together with the results.

## Files

- `train_e6e7.py`, `run_e6e7.sh`: training wrapper and queue.
- `logs/train_<run>.log`, `queue.log`: progress.
- `runs/<run>/`: checkpoint, predictions, history and `summary.json` (written only at the end of each run).
