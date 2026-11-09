# BCI dataset (as used in FYP3)

| | |
|---|---|
| Name | **BCI**: Breast Cancer Immunohistochemical image generation dataset |
| Paper | S. Liu, C. Zhu, F. Xu, X. Jia, Z. Shi, M. Jin. "BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix." arXiv:2204.11425, 2022 (CVPR Workshops 2022) |
| Local copy | `FYP/data/raw/BCI_dataset/` (read only), with `FYP/data/raw/BCI_metadata_pairs_refined.csv` |
| Content | 4,873 registered H&E/IHC tile pairs, 1024×1024 RGB PNG: 3,896 official train, 977 official test |
| Licence | Non-commercial academic use. Cite the paper above (see the dataset README). |

## Which images FYP3 uses

FYP2's images are **HER2 IHC** patches, which show brown DAB membrane stain. Its augmentation keeps colour jitter mild precisely
because DAB intensity carries the score. So FYP3 uses `BCI_dataset/IHC/`. The H&E tiles are not used.

## Labels

- **Convention.** The BCI README names files `number_train/test_HER2level.png`. It warns: *"the HER2 level represents
  which category of WSI the image came from, not the image itself."* FYP3 parses `NNNNN_{train|test}_{0|1+|2+|3+}.png`
  into classes 0–3. Every file name matched, and the partition in each name matched its folder.
- **Class counts:**

  | Split | 0 | 1+ | 2+ | 3+ |
  |---|---|---|---|---|
  | Official train | 202 | 919 | 1,697 | 1,078 |
  | Official test | 38 | 235 | 446 | 258 |

  Class 0 is 5% of the data, whereas FYP2 was perfectly balanced at 600 per class.
- **Patch-level labels.** `BCI_metadata_pairs_refined.csv` gives a patch-level `her2_level_refined` for 3,513 tiles; its
  method is undocumented.
  - 4 values are non-standard (`1`, `1`, `2`, `0+`) and are treated as missing.
  - Of the 3,509 valid ones, 82.1% agree with the filename label. For filename-2+ tiles, 385 are refined as 1+ and 94 as 3+.
  - These labels are used **only for diagnosis**, never for training or model selection.

## Scan batches and stain measurement

- **Two scan types.** Background brightness (the mean of near-unsaturated pixels) is bimodal: *bright* scans
  sit around 240 and *grey* scans around 170. A tile counts as grey below 201 (Otsu threshold on training
  tiles). Scan type is a slide property: only 40 of 691 multi-tile fragments mix the two.
- **Scan type is associated with the label.** Grey scans make up 25% / 44% / 52% / 68% of tiles in classes
  0 / 1+ / 2+ / 3+. Background brightness alone predicts the class above chance (random-forest macro-F1 0.44,
  grouped CV), so models could use it as a shortcut. `analysis_labels.py` and `analysis_models.py` test this.
- **Raw stain optical density is not reliable on BCI.** Optical density measured against 255 (as
  `skimage.rgb2hed` does) counts a grey background cast as stain: within each class, raw DAB correlates
  ρ = −0.61 to −0.92 with background brightness. FYP3 therefore also measures stain relative to each
  tile's own background (`dabc_*` in `results/analysis/data_features.csv`). With that correction:
  - median DAB is 0.077 / 0.096 / 0.117 / 0.249 for classes 0 / 1+ / 2+ / 3+, so tile-level stain
    amount separates 3+ from the rest but barely separates 0, 1+ and 2+;
  - in FYP2's GAN images the same medians are 0.09 / 0.09 / 0.46 / 0.94.

## Split and leakage control

### Finding: the official split leaks

BCI has no slide or patient IDs. Tiles cut side by side from one slide share a continuous border, so `src/analysis_slides.py`
links tiles whose edges continue into each other. A link is accepted only if it is mutual, unambiguous, and as smooth as
adjacent pixel columns inside a tile.

| Check | Result |
|---|---|
| Accepted links | 2,763 |
| Linked pairs sharing a slide label | 99.96% (random pairs: 33%) |
| Fragments | 691, label-pure in 99.9% of cases |
| Tiles with no link | 1,645 |
| Official test tiles in a fragment that also has official training tiles | 619 of 977 (63%) |

The links are real, and the official split is therefore **not slide-independent**.

### FYP2's pHash grouping does not transfer

pHash ≤ 12 groups (FYP2's twin rule) mix slide labels in 39 of 53 cases. Even 6-bit pairs agree on label only 50% of the time,
and linked tiles are mostly background. In BCI they are look-alike backgrounds, not duplicates. There are no exact duplicates
(no shared MD5, no pHash distance below 6).

### Splits used

| Split | Construction | Files |
|---|---|---|
| **grouped (main)** | FYP2's recipe: StratifiedGroupKFold 70/15/15 over all tiles, groups = slide fragments. Train 3,388 / val 727 / test 758. No accepted edge link crosses a split. | `splits/split_manifest_grouped.csv`, `results/runs/` |
| official (reference) | Official BCI train/test, plus 15% of official train as validation (pHash groups). Leaks, as shown above. | `splits/split_manifest_official.csv`, `results/runs_official/` |

**Residual risk:** tiles of one slide that are not edge-connected can still fall in different splits of the grouped design.
