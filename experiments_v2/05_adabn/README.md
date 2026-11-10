# Experiment 5: AdaBN (adaptive batch normalisation), new

**Question.** Does re-estimating BatchNorm statistics on *unlabelled* HER2-IHC-40x tiles (AdaBN; Li et al., 2016)
improve the BCI-trained networks on the external domain? How does it compare with HARMONISE stain_p99, which uses the
same unlabelled tiles? Does it rescue the 448 px model that stain_p99 breaks?

**Status.** Complete for the 5-network panel: EfficientNetV2B0 CE at 224 px and 448 px, MobileNetV3Small CE,
MobileNetV3Large CORN and DenseNet121 CE.

## Protocol (fixed before any result; nothing tuned)

- **Adaptation tiles.** The 1,500-tile HARMONISE-1 fit pool, unlabelled; labels never read. These are the same tiles
  stain_p99 was fitted on.
- **One statistics pass.** Fixed seed order (2027), batch 32:
  - every BN layer in training mode, with momentum t/(t+1), so the stored statistics become the cumulative average over
    all batches;
  - backbone Dropout off;
  - BN layers set trainable for the pass, because Keras ignores `training=True` on frozen BN;
  - weights unchanged.
- **Arms:**
  - `adabn`: raw inputs;
  - `adabn_p99`: stain_p99 inputs, used for both the statistics and the evaluation;
  - comparators: `raw` and `stain_p99`, from stored predictions and re-checked by fresh inference.
- **Evaluation:**
  - development R (4,593 labelled development tiles; R_eval subset of 2,402 shown);
  - the external test split, **evaluated once per network and arm**. **Previously examined (exploratory second
    look), not an untouched test set.**
- **Paired test differences.** Filename-run cluster bootstrap, 2,000 draws. This is optimistic, because the release
  has no slide IDs.
- **Validation** (`outputs/checks.csv`):
  - fresh unadapted predictions match the stored raw and stain_p99 predictions to ≤ 6.5e-6 for all five networks;
  - BN layers re-estimated: 59 / 34 / 46 / 121 / 59;
  - moving means changed in every network.

## Results

Balanced accuracy (`outputs/summary_test.csv`, `outputs/summary_dev.csv`):

| Network | Dev R_eval: raw | stain_p99 | AdaBN | AdaBN+p99 | **Test: raw** | **stain_p99** | **AdaBN** | **AdaBN+p99** |
|---|---|---|---|---|---|---|---|---|
| EfficientNetV2B0 224 | 0.298 | **0.505** | 0.432 | 0.403 | 0.301 | 0.503 | **0.522** | 0.486 |
| EfficientNetV2B0 448 | 0.284 | 0.234 | **0.394** | 0.328 | 0.290 | 0.186 | **0.516** | 0.443 |
| MobileNetV3Small | 0.283 | **0.519** | 0.349 | 0.428 | 0.337 | **0.528** | 0.397 | 0.483 |
| MobileNetV3Large CORN | 0.253 | 0.399 | 0.364 | **0.459** | 0.261 | 0.431 | 0.407 | **0.519** |
| DenseNet121 | 0.252 | 0.439 | 0.377 | **0.449** | 0.260 | 0.410 | 0.361 | **0.490** |

Paired Δ balanced accuracy on the test split, 95% CI (`outputs/summary_paired_test.csv`):

| Network | AdaBN − raw | AdaBN − stain_p99 | AdaBN+p99 − stain_p99 |
|---|---|---|---|
| EfficientNetV2B0 224 | +0.221 [+0.177, +0.267] | +0.018 [−0.048, +0.080] | −0.017 [−0.083, +0.045] |
| EfficientNetV2B0 448 | +0.226 [+0.175, +0.279] | **+0.329** [+0.268, +0.386] | +0.257 [+0.184, +0.329] |
| MobileNetV3Small | +0.061 [+0.024, +0.091] | **−0.130** [−0.168, −0.098] | −0.045 [−0.085, −0.009] |
| MobileNetV3Large CORN | +0.146 [+0.115, +0.180] | −0.024 [−0.065, +0.016] | **+0.088** [+0.033, +0.149] |
| DenseNet121 | +0.101 [+0.066, +0.145] | −0.050 [−0.085, −0.006] | **+0.079** [+0.038, +0.124] |

Clinical errors on the test split (counts out of 1,847; experiment 1 definitions; `outputs/summary_test.csv`):

| Network / arm | 3+ → 0/1+ (positive missed) | 0/1+ → 3+ (false positive) | 0/1+ → 2+ (unnecessary ISH) | 2+ → 0/1+ | S2 workflow cost |
|---|---|---|---|---|---|
| EffV2B0 224, stain_p99 | 8 | 64 | 595 | 36 | **1.16** |
| EffV2B0 224, AdaBN | 60 | 121 | 85 | 55 | 1.33 |
| EffV2B0 448, stain_p99 | 298 | 85 | 807 | 47 | 3.47 |
| EffV2B0 448, AdaBN | 74 | 101 | 122 | 70 | **1.36** |
| DenseNet121, stain_p99 | 108 | 47 | 2 | 105 | 1.44 |
| DenseNet121, AdaBN+p99 | 26 | 108 | 215 | 45 | **1.22** |

## Interpretation

1. **AdaBN always helps over doing nothing.** Test balanced accuracy rises by +0.06 to +0.23 on all 5 networks, with
   every CI above 0. It needs no labels and no stain model.
2. **It is not uniformly better than stain_p99.** The winner depends on the network:
   - EfficientNetV2B0 224: tie.
   - 448 px: AdaBN much better.
   - MobileNetV3Small: stain_p99 better.
   - MobileNetV3Large and DenseNet121: AdaBN+p99 better.
   The best unsupervised result on any network is about 0.52–0.53 balanced accuracy, which is far from usable.
3. **AdaBN rescues the 448 px model.** stain_p99 drives it to 0.186 (below chance); AdaBN reaches 0.516, and 3+
   positives missed drop from 298 to 74. Normalising feature statistics avoids the haematoxylin-gain failure
   (`../REPORT_CORRECTIONS_v2.md`, item 3).
4. **The methods trade different clinical errors.** On EfficientNetV2B0 224, stain_p99 sends 595 negatives to ISH but
   misses only 8 positives; AdaBN sends 85 to ISH but misses 60 positives. QWK favours AdaBN (0.70 vs 0.59); the
   workflow cost favours stain_p99 (1.16 vs 1.33). Which is preferable depends on the cost of a missed positive, so it
   must be reported, not hidden in one score.
5. **Development data would have chosen well in 4 of 5 networks.** Picking, per network, the arm with the best R_eval
   balanced accuracy matches the best test arm in 4 of 5 cases. The miss is EfficientNetV2B0 224: R_eval picks stain_p99
   (0.503 on test) over AdaBN (0.522), a difference inside the CI. This rule was **not** pre-registered; it is a
   post-hoc observation.
6. **Labels beat all unsupervised fixes.** Few-shot full fine-tuning with 5 tiles per class reaches 0.78 test
   balanced accuracy (`../06_fewshot/README.md`).

## Limitations

- **One statistics pass, one seed, no tuning.** Batch size and pool size were not varied. 1,500 tiles is a modest
  sample for 121 DenseNet BN layers.
- **Pool overlap with development.** The adaptation pool comes from the same 86 development slides as R. Unlabelled
  use only, but the development numbers may be slightly optimistic. The test split comes from other slides.
- **Exploratory test numbers.** The test split was examined before. The filename-run bootstrap is optimistic (no slide
  IDs), so the CIs are too narrow.
- **No deployment test.** AdaBN changes the model weights file (BN statistics). It was not exported to TFLite or
  tested on devices.

## Files

- **Scripts:**
  - `run_adabn.py --run <network> [--cpu]` (one network per process; 448 px decodes the development archive lazily);
  - `summarise_adabn.py`.
- **Per network:** `outputs/adabn_<run>.json` (all metrics, checks and paired differences), `outputs/preds_<run>.npz`
  (test and R probabilities, every arm), `outputs/log_<run>.txt`.
- **Summaries:** `outputs/summary_test.csv`, `outputs/summary_dev.csv`, `outputs/summary_paired_test.csv`,
  `outputs/checks.csv`.
- **Figure:** `outputs/fig_adabn.png`.
