"""Shared configuration for FYP3: the FYP2 pipeline reproduced on the BCI HER2 IHC dataset.

Everything below the "Training schedule" line is copied unchanged from FYP2/src/config.py, so the
baseline runs are a like-for-like reproduction.  Only the data paths and the split differ.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FYP2_ROOT = ROOT.parent / "FYP2"                       # read-only reference project
BCI_RAW = ROOT.parent / "FYP" / "data" / "raw" / "BCI_dataset"   # read-only
BCI_IHC = BCI_RAW / "IHC"
BCI_REFINED_CSV = ROOT.parent / "FYP" / "data" / "raw" / "BCI_metadata_pairs_refined.csv"

# BCI tiles are 1024x1024. FYP2's GAN patches are 256x256, so the tiles are downsampled
# once (Lanczos, anti-aliased) to 256x256 and cached here. The pipeline then sees the same
# input geometry as FYP2: 256 px source -> 224 px network input.
SOURCE_SIZE = 256
DATA_DIR = ROOT / "data" / f"ihc_{SOURCE_SIZE}"
SPLIT_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"

# Two splits (select with the environment variable FYP3_SPLIT):
#   grouped  (main)      FYP2's recipe: StratifiedGroupKFold 70/15/15 over all tiles, groups = slide fragments
#                        reconstructed from tile-edge continuity (analysis_slides.py). Leakage-controlled.
#   official (reference) the official BCI train/test split + 15% validation from official train. It leaks:
#                        63% of official test tiles are edge-connected to official training tiles.
SPLIT = os.environ.get("FYP3_SPLIT", "grouped")
assert SPLIT in ("grouped", "official"), SPLIT
MANIFEST = SPLIT_DIR / f"split_manifest_{SPLIT}.csv"
SPLIT_REPORT = SPLIT_DIR / f"split_report_{SPLIT}.txt"
RUNS_DIR = RESULTS_DIR / ("runs" if SPLIT == "grouped" else f"runs_{SPLIT}")
SHARED_ANALYSIS = RESULTS_DIR / "analysis"                       # split-independent (dataset-level) analysis
ANALYSIS_DIR = SHARED_ANALYSIS if SPLIT == "grouped" else RESULTS_DIR / f"analysis_{SPLIT}"
SUFFIX = "" if SPLIT == "grouped" else f"_{SPLIT}"               # for metrics/selection/report file names
OFFICIAL_MANIFEST = SPLIT_DIR / "split_manifest_official.csv"

CLASS_NAMES = ["Score_0", "Score_1", "Score_2", "Score_3"]
BCI_LEVELS = ["0", "1+", "2+", "3+"]                  # filename convention: NNNNN_{train|test}_{level}.png
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224   # network input
SEED = 42

# official split: validation = 15% of the official train split. grouped split: 70/15/15 of all tiles.
# Test is never used for any decision in either split.
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------- Training schedule and model zoo: identical to FYP2 ----------------
MODELS = {
    "ResNet50":         dict(tier="heavy", app="ResNet50",          prep="caffe",   batch=16),
    "DenseNet121":      dict(tier="heavy", app="DenseNet121",       prep="torch",   batch=16),
    "ConvNeXtTiny":     dict(tier="heavy", app="ConvNeXtTiny",      prep="builtin", batch=16),
    "EfficientNetB0":   dict(tier="mid",   app="EfficientNetB0",    prep="builtin", batch=32),
    "EfficientNetV2B0": dict(tier="mid",   app="EfficientNetV2B0",  prep="builtin", batch=32),
    "NASNetMobile":     dict(tier="mid",   app="NASNetMobile",      prep="tf",      batch=32),
    "MobileNetV2":      dict(tier="lite",  app="MobileNetV2",       prep="tf",      batch=32),
    "MobileNetV3Large": dict(tier="lite",  app="MobileNetV3Large",  prep="builtin", batch=32),
    "MobileNetV3Small": dict(tier="lite",  app="MobileNetV3Small",  prep="builtin", batch=32),
}

LOSSES = ["ce", "corn"]

HEAD_EPOCHS = 5          # stage 1: frozen backbone
HEAD_LR = 1e-3
FT_EPOCHS = 40           # stage 2: full fine-tuning (upper bound, early stopping decides)
FT_LR = 1e-4
FT_WARMUP_EPOCHS = 2
PATIENCE = 10
DROPOUT = 0.3

# Improvement experiments (filled in after the diagnosis; baseline runs use none of them).
VARIANTS = {}
