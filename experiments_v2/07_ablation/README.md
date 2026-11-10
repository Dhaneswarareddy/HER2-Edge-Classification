# Experiment 7: ablation (backbone freezing, head capacity, resolution and batch)

**Question.** Which design choices explain the EfficientNetV2B0 result on BCI?
- Fine-tuning the backbone vs frozen ImageNet features.
- Head capacity.
- Partial vs full unfreezing.
- Input resolution, separated from batch size.

**Status.**
- New, trained here: `frozen_linear`, `frozen_mlp`, `partial`.
- Existing, reused and not retrained: the full fine-tuning baseline, the E1/E2 resolution and batch runs, and the
  MobileNetV3Small arms.
- Paired analysis complete.

## Protocol

- **New arms** (`train_ablation.py`) import `src/train.py` and `src/common.py` unchanged. They match the baseline
  except for what is listed:
  - same data pipeline and augmentation;
  - seed 42, batch 32;
  - head stage 5 epochs at 1e-3;
  - fine-tune ≤ 40 epochs, warm-up 2 + cosine from 1e-4;
  - early stopping on val QWK, patience 10, best weights restored;
  - BN in inference mode.

  | Arm | Difference from baseline |
  |---|---|
  | `frozen_linear` | Backbone frozen throughout; linear head ≤ 45 epochs with the same early stopping |
  | `frozen_mlp` | As `frozen_linear`, head = Dense(256, ReLU) → Dropout(0.3) → Dense(4) |
  | `partial` | Stage 2 unfreezes only `block6*` + `top*`. These hold 4.49 M of 5.92 M parameters, so "partial" still trains 76% of the weights |

- **Decision set.** Validation (727 tiles). Test (758) is report only.
- **Paired Δ.** Same tiles; 95% fragment-cluster bootstrap, 2,000 draws.
- **Training seeds.** One per arm, as for every FYP3 run, so the CIs do not include training variance.

## Results (`outputs/ablation_table.csv`, `outputs/ablation_paired.csv`)

| Arm (EfficientNetV2B0 CE) | Input / batch | Trainable (stage 2) | Best / run epochs | Val acc | Val F1 | Val QWK | Test acc | Test bal. acc | Test QWK |
|---|---|---|---|---|---|---|---|---|---|
| Full fine-tuning (baseline) | 224 / 32 | all (5.9 M) | 31 / 41 | 0.896 | 0.893 | 0.869 | 0.926 | 0.917 | 0.915 |
| Frozen + linear head | 224 / 32 | 5 k | 39 / 45 (cap) | 0.703 | 0.635 | 0.603 | 0.732 | 0.636 | 0.654 |
| Frozen + MLP head | 224 / 32 | 329 k | 17 / 27 | 0.717 | 0.671 | 0.619 | 0.699 | 0.642 | 0.628 |
| Partial (block6 + top) | 224 / 32 | 4.49 M | 36 / 45 (cap) | 0.857 | 0.841 | 0.780 | 0.898 | 0.845 | 0.847 |
| Low-batch (E1) | 224 / 8 | all | 16 / 26 | 0.868 | 0.865 | 0.829 | 0.905 | 0.901 | 0.896 |
| **High-res (E1, deployed)** | **448 / 8** | all | 38 / 45 (cap) | **0.950** | **0.950** | **0.935** | **0.975** | **0.973** | **0.959** |

Paired differences (Δ, 95% CI):

| Comparison | Val Δ acc | Val Δ QWK | Test Δ acc | Test Δ QWK |
|---|---|---|---|---|
| Frozen linear − full FT | −0.193 [−0.230, −0.155] | −0.267 [−0.343, −0.202] | −0.194 [−0.256, −0.141] | −0.261 [−0.397, −0.178] |
| Frozen MLP − full FT | −0.179 [−0.233, −0.125] | −0.251 [−0.353, −0.156] | −0.227 [−0.295, −0.153] | −0.287 [−0.438, −0.196] |
| Frozen MLP − frozen linear | +0.014 [−0.033, +0.062] | +0.016 [−0.070, +0.102] | −0.033 [−0.079, +0.020] | −0.026 [−0.115, +0.057] |
| Partial − full FT | −0.039 [−0.067, −0.008] | −0.089 [−0.141, −0.039] | −0.028 [−0.049, −0.008] | −0.068 [−0.120, −0.035] |
| **448 px − 224 px, both batch 8** | **+0.083 [+0.051, +0.112]** | **+0.105 [+0.059, +0.166]** | **+0.070 [+0.042, +0.106]** | **+0.063 [+0.025, +0.116]** |
| 448 px (batch 8) − 224 px (batch 32) | +0.055 [+0.033, +0.076] | +0.065 [+0.029, +0.109] | +0.049 [+0.029, +0.080] | +0.044 [+0.009, +0.093] |
| MobileNetV3Small: 448 − 224, both batch 8 | +0.041 [0.000, +0.083] | +0.090 [−0.006, +0.205] | +0.003 [−0.021, +0.030] | +0.017 [−0.035, +0.078] |
| MobileNetV3Small: 448 (b8) − 224 (b32) | +0.044 [+0.011, +0.077] | +0.102 [+0.027, +0.172] | −0.005 [−0.032, +0.022] | −0.014 [−0.064, +0.040] |

Existing reference (not re-run; `outputs/reference_frozen_probe.json`): a frozen ImageNet **EfficientNetB0** (not V2)
with a logistic-regression probe on BCI reached accuracy 0.765 and QWK 0.698, comparable to the frozen arms here.

## Interpretation

1. **Fine-tuning the backbone is essential.** Frozen ImageNet features lose about 0.19 accuracy and 0.26 QWK, and the
   CIs are far from 0. A bigger head does not help: MLP vs linear is within ±0.03 with CIs spanning 0. The limitation
   is the features, not head capacity.
2. **Fine-tuning everything beats fine-tuning the top 76%.** Partial unfreezing is worse on both val and test
   (QWK −0.09 / −0.07). Updating the early blocks matters for IHC texture and colour.
3. **Resolution is the largest positive effect for EfficientNetV2B0, and it is not explained by batch size.**
   - Against the batch-matched 224 px control: +0.105 val QWK and +0.063 test QWK, both CIs above 0.
   - Batch 8 vs 32 at 224 px is, if anything, slightly worse (lowbatch val QWK 0.829 vs 0.869).
4. **Resolution does not generalise to MobileNetV3Small.** It is not significant against the batch-matched control,
   and against the standard run it improves val but not test. Resolution is "a confirmed improvement for
   EfficientNetV2B0", not a general rule.
5. **Costs of the higher resolution:**
   - 5.2× training time (252 vs 49 min);
   - 4× pixels per tile; Mac/edge latency in `../08_edge/outputs/mac_latency.csv`;
   - defocus sensitivity (`../04_robustness`: blur σ = 2 costs 0.17 balanced accuracy at 448 px vs 0.03 at 224 px).

## Limitations

- **One seed per arm.** Differences of a few points between arms can be within training noise; the paired CIs cover
  tile sampling only.
- **Three runs ended at the 45-epoch cap** (frozen_linear, partial, hires), so they may be slightly under-trained.
  frozen_linear was still improving slowly.
- **BCI only.** External-domain behaviour of the ablation arms was not evaluated, to keep the external split from
  further looks.
- **The frozen reference uses EfficientNetB0, not V2.** It comes from an earlier script with its own protocol.

## Files

- **Scripts:** `train_ablation.py --variant {frozen_linear, frozen_mlp, partial} [--smoke]`; `analyze_ablation.py`.
- **Runs:** `runs/<variant>/` with `model.keras`, `model.weights.h5`, `predictions.npz`, `history.csv` and
  `summary.json`, in the same format as `results/runs/`. Smoke runs are in `runs_smoke/`.
- **Results:** `outputs/ablation_table.csv`, `outputs/ablation_paired.csv`, `outputs/reference_frozen_probe.json`.
- **Figure:** `outputs/fig_ablation.png`.
- **Logs:** `logs/train_<variant>.log`, `logs/smoke_<variant>.log`.
