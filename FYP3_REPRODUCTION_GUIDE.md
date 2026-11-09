# FYP3 reproduction guide (Windows laptop, WSL2 Ubuntu)

How to rebuild the complete FYP3 pipeline from scratch on another machine, using the project's real scripts.
FYP3 depends on FYP2 (its split, GAN images and trained networks), so FYP2 is reproduced first.

Markers used below:
- **[MANUAL]**: needs a person: downloads, copying, installers.
- **[NOT VERIFIED]**: written from the code and the documentation, never run on Windows/CUDA.
- **[NOT AVAILABLE]**: no project code exists for it.

The original runs used macOS on an Apple M1 with 8 GB, `tensorflow-macos 2.15.0` and `tensorflow-metal 1.1.0`. On CUDA the
numbers will match **within noise, not bit for bit**. Training is seeded (`config.SEED = 42`), but GPU kernels are not
deterministic; even on the Mac, a rerun differed in the 3rd decimal of the first epoch's loss.

---

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 22H2 / Windows 11 with WSL2 | Windows 11 |
| GPU | none (CPU works, but training is slow) | NVIDIA GPU, 8 GB+ VRAM (12 GB+ for 448 px heavy models) |
| RAM | 16 GB | 32 GB |
| Disk (inside WSL) | 40 GB free | 60 GB (the optional 1024 px cache adds ~7 GB) |
| Python | 3.11 (TensorFlow 2.15 supports 3.9–3.11, not 3.12) | 3.11 |

Disk use on the original machine:
- BCI extracted 2.8 GB;
- FYP3 caches: 256 px 0.55 GB, 512 px 1.8 GB, 448 px eval 0.43 GB;
- FYP3 results 0.95 GB;
- FYP2 results 2.8 GB.

## 2. Windows, WSL2 and VS Code [MANUAL]

1. Install the current NVIDIA Windows driver; it includes WSL support. **Do not install a Linux NVIDIA driver inside WSL.**
2. In PowerShell (as administrator): `wsl --install -d Ubuntu-22.04`, reboot, create the Linux user.
3. In Ubuntu: `nvidia-smi` must list the GPU.
4. Install VS Code on Windows, plus the extensions **WSL** (`ms-vscode-remote.remote-wsl`) and **Python**.
   Open the project from Ubuntu with `code ~/her2`.
5. Keep all project files **inside the WSL file system** (`~/her2`), not under `/mnt/c`. Reading thousands of
   small images through `/mnt/c` is very slow.

## 3. Python environment

```bash
sudo apt update && sudo apt install -y software-properties-common unzip
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv ~/her2/venv && source ~/her2/venv/bin/activate
pip install --upgrade pip
cat > ~/her2/requirements.txt <<'EOF'
tensorflow[and-cuda]==2.15.1
numpy==1.26.4
pandas==3.0.6
scikit-learn==1.9.1
scipy==1.17.1
pillow==12.3.0
matplotlib==3.11.2
opencv-python-headless==4.9.0.80
ImageHash==4.3.2
h5py==3.14.0
EOF
pip install -r ~/her2/requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Versions are those of the original venv (`pip freeze`). The Mac-only `tensorflow-macos` and `tensorflow-metal` are
replaced by `tensorflow[and-cuda]==2.15.1`, and `opencv-python` by the headless build (same API, no GUI libraries).

**[NOT VERIFIED]** that this exact set resolves on Linux. If pip refuses a version, keep `tensorflow==2.15.*` and
`numpy==1.26.*` fixed and relax the others to the nearest available version.

These are every third-party import in `FYP3/src` (plus TensorFlow). ImageNet weights are downloaded by Keras on first
use into `~/.keras/models/`, so internet access is needed once.

Activate the venv in every new shell: `source ~/her2/venv/bin/activate`.

## 4. Code and data placement

`config.py` finds everything relative to the parent folder: `FYP3/../FYP2` and `FYP3/../FYP/data/raw`.

```
~/her2/
├── FYP/data/raw/
│   ├── BCI_dataset/{HE,IHC}/{train,test}/NNNNN_{train|test}_{0|1+|2+|3+}.png    (unzipped, read-only)
│   ├── BCI_metadata_pairs_refined.csv                                           (read-only)
│   └── HER2-IHC-40x/wsi_test/class_{0,1+,2+,3+}/*.png                           (optional, read-only)
├── FYP2/
│   ├── HER2 SCORES DATA/Generated_HER2_images/Score_{0,1,2,3}/*.jpeg            (unzipped)
│   ├── src/  pi/  README.md  DATASET.md
│   ├── splits/        (rebuilt by step 5.1)
│   └── results/       (rebuilt)
└── FYP3/
    ├── src/  README.md  DATASET.md  NEXT_EXPERIMENTS.md  FYP3_REPRODUCTION_GUIDE.md
    ├── results/findings.md          (hand-written; report.py inserts it as section 0)
    ├── data/  splits/  results/     (rebuilt)
```

**Code [MANUAL]:** there is no git repository. On the Mac:

```bash
cd ~/Downloads && zip -r her2_code.zip FYP2/src FYP2/pi/README.md FYP2/pi/benchmark_pi.py FYP2/pi/requirements.txt \
  FYP2/README.md FYP2/DATASET.md FYP2/splits FYP3/src FYP3/*.md FYP3/results/findings.md FYP3/splits -x "*/__pycache__/*"
```

The original `splits/` are included only to check the rebuilt ones (section 8).

**Data [MANUAL]:**

| File | Source | Check |
|---|---|---|
| `BCI_dataset.zip` (2.9 GB) | BCI project page, https://bupt-ai-cz.github.io/BCI (also linked from github.com/bupt-ai-cz/BCI) | 4,873 IHC tiles: official train 3,896, test 977 |
| `BCI_metadata_pairs_refined.csv` | Zenodo, doi 10.5281/zenodo.17200604 | md5 `fc71ac9929a3e8a4f2c64ba85d76d177`, 3,513 rows |
| `HER2 SCORES DATA.zip` | Kaggle `jafarsleman/synthesized-her2-images-cgan` (see `FYP2/DATASET.md`) | 2,400 JPEGs, 600 per score |
| `her2-ihc-40x-wsi.zip` (optional) | Zenodo, doi 10.5281/zenodo.15179607 | test split 1,847 tiles |

```bash
mkdir -p ~/her2 && cd ~/her2 && unzip -q /mnt/c/Users/<you>/Downloads/her2_code.zip
mkdir -p FYP/data/raw && cd FYP/data/raw
unzip -q /mnt/c/Users/<you>/Downloads/BCI_dataset.zip                 # creates BCI_dataset/HE, BCI_dataset/IHC
cp /mnt/c/Users/<you>/Downloads/BCI_metadata_pairs_refined.csv .
md5sum BCI_metadata_pairs_refined.csv
chmod -R a-w BCI_dataset BCI_metadata_pairs_refined.csv
cd ~/her2/FYP2 && unzip -q "/mnt/c/Users/<you>/Downloads/HER2 SCORES DATA.zip"   # creates "HER2 SCORES DATA/"
find ~/her2/FYP/data/raw/BCI_dataset/IHC -name "*.png" | wc -l        # 4873
find "$HOME/her2/FYP2/HER2 SCORES DATA" -name "*.jpeg" | wc -l         # 2400
```

Note: the BCI files are named `.png` but are JPEG data. The code decodes them correctly, so leave them as they are.

**Machine-specific edits (required).**
- The runner scripts hard-code the Mac's Python path.
- `run_analysis_models.sh` uses the macOS-only `caffeinate`.

```bash
cd ~/her2/FYP3/src && sed -i 's#^PY=.*#PY="$(command -v python)"#' run_queue.sh run_variants.sh run_analysis_models.sh \
  && sed -i 's/caffeinate -ims //' run_analysis_models.sh && chmod +x *.sh
cd ~/her2/FYP2/src && sed -i 's#^PY=.*#PY="$(command -v python)"#' run_*.sh && chmod +x *.sh
```

## 5. FYP2 (prerequisite)

Run from `~/her2/FYP2/src` with the venv active (`PY=python`). From `FYP2/README.md`:

| Step | Command | Output |
|---|---|---|
| 5.1 Split | `python make_split.py` | `splits/split_manifest.csv` (420/90/90 per class) |
| 5.2 Train 18 runs (9 models × CE/CORN) | `./run_training.sh` | `results/runs/<model>__<loss>/` |
| 5.3 Metrics + overfitting | `python evaluate_all.py` | `results/metrics_all.csv`, `overfitting.csv` |
| 5.4 Quantise | `python quantize.py`, then `python retime_latency.py` on an idle machine | `results/tflite/`, `quant_results.csv` |
| 5.5 Select | `python select_models.py` | `results/selection.json` |
| 5.6 Pi bundle | `python package_pi.py` | `pi/her2_pi_bundle.tar.gz` |
| 5.7 Report | `python report.py` | `results/REPORT.md` |

FYP3 needs only 5.1–5.3. Steps 5.4–5.7 are the deployment path (section 7).

## 6. FYP3 pipeline, in execution order

Run from `~/her2/FYP3/src` with `PY=python`. **Run one TensorFlow job at a time.** Every runner skips finished work
(a run with `summary.json`, a metrics or analysis file that already exists).

| # | Command | Output | Check |
|---|---|---|---|
| 1 | `python prepare_data.py` | `data/ihc_256/`, `splits/split_manifest_official.csv`, `results/features/imagenet_effb0_bci.npz` | 4,873 tiles; official train 3,896 / test 977 |
| 2 | `python analysis_slides.py` | `results/analysis/slide_fragments.csv`, `edge_matches.csv` | 619 of 977 official test tiles share a fragment with train |
| 3 | `python make_grouped_split.py` | `splits/split_manifest_grouped.csv`, `split_report_grouped.txt` | train 3,388 / val 727 / test 758 |
| 4 | `FYP3_SPLIT=official ./run_queue.sh MobileNetV3Small:ce MobileNetV2:ce` | `results/runs_official/` (leaky reference) | 2 runs |
| 5 | `./run_queue.sh MobileNetV3Small:ce MobileNetV2:ce MobileNetV3Large:ce EfficientNetB0:ce EfficientNetV2B0:ce NASNetMobile:ce ResNet50:ce DenseNet121:ce ConvNeXtTiny:ce` | `results/runs/<model>__ce/` | 9 runs; test acc 0.91–0.95 |
| 6 | `python analysis_data.py` | `results/analysis/data_features.csv`, `data_separability.csv`, figs D1–D2 | |
| 7 | `python analysis_labels.py` | `label_noise.csv`, `scan_type_threshold.json`, `error_drivers.csv`, fig D3 | needed before step 9 |
| 8 | `python select_models.py`, then `./run_queue.sh DenseNet121:corn EfficientNetV2B0:corn MobileNetV3Large:corn` | `results/selection.json`; 3 CORN runs | CORN only for the tier winners in `selection.json` (originally these three) |
| 9 | `./run_analysis_models.sh`, then `python analysis_models.py` (redraws from cache) | `results/analysis/models/<arch>.json`, `model_summary.csv`, `cka.csv`, `stain_sensitivity.csv`, figs D4–D7, D10 | reload diff ≈ 0 for every arch |
| 10 | `for a in ResNet50 DenseNet121 ConvNeXtTiny EfficientNetB0 EfficientNetV2B0 NASNetMobile MobileNetV2 MobileNetV3Large MobileNetV3Small; do python analysis_haem_cue.py --arch $a; done; python analysis_haem_cue.py` | `haem_cue.csv`, `haem_cue_decision.json`, fig D11 | "NOT confirmed" originally |
| 11 | `./run_variants.sh MobileNetV3Small:ce:cw MobileNetV3Small:ce:ls MobileNetV3Small:ce:hires EfficientNetV2B0:ce:hires` | `results/runs/<model>__ce__<variant>/`; the first hires run builds `data/ihc_512*` | hires: batch 8, input 448 |
| 12 | `python evaluate_all.py` and `FYP3_SPLIT=official python evaluate_all.py` | `results/metrics/`, `metrics_all.csv`, `overfitting.csv` (+ `_official`) | 16 + 2 runs |
| 13 | `python analysis_labels.py`, `FYP3_SPLIT=official python analysis_labels.py`, `python analysis_leakage.py` | refreshed label, leakage and held-out tables, fig D9 | |
| 14 | `python select_models.py`, `python report.py`, `FYP3_SPLIT=official python report.py` | `results/REPORT.md`, `REPORT_official.md`, `results/figures/` | no "Pending" left |

Decisions and paired comparisons: `python paired.py RUN_A RUN_B --split val`, which prints Δ with a fragment-cluster 95% CI.

Notes on specific steps:
- **Step 8 and selection.** In the original, the CORN runs followed the 9 CE baselines' tier selection. The variant
  set in step 11 came from the diagnosis (cw, ls, hires) plus a pre-registered hires confirmation on EfficientNetV2B0.
  The `illum` and `hed` variants are not trained: their rules in `variants.py` were not met.
- **Smoke tests.** `python train.py --model X --loss ce --smoke` and `python train_variant.py ... --smoke` run 1+1 epochs into
  `results/smoke/`. Use them after any setup change.
- **macOS-only workarounds**, harmless elsewhere:
  - `analysis_models.py --archs ConvNeXtTiny --cpu`: Grad-CAM deadlocks on Metal.
  - `evaluate_all.py` forces the CPU for 448 px runs, but only on macOS.

  On CUDA, step 9's script runs ConvNeXtTiny on the GPU. **[NOT VERIFIED]**

## 7. Quantisation and deployment

**FYP3: [NOT AVAILABLE].** FYP3's scope excluded quantisation and deployment, so it has no quantisation code.
`FYP2/src/quantize.py` reads FYP2's own `config.py` (runs, data, calibration images). Quantising FYP3 networks needs a port
of `quantize.py` and `package_pi.py` to FYP3's config. That port has not been written.

**FYP2 → Raspberry Pi 4B** (from `FYP2/pi/README.md`; this was done on the Mac, while the Pi run itself is pending):
1. `python quantize.py && python retime_latency.py && python select_models.py && python package_pi.py` (in `FYP2/src`).
   This produces `pi/her2_pi_bundle.tar.gz`. Host latency figures are machine-specific.
2. Pi: Raspberry Pi OS **64-bit** (Bookworm), official 5V/3A supply and cooling. Then:
   `python3 -m venv ~/her2env && source ~/her2env/bin/activate && pip install -r requirements.txt` (numpy<2.0, psutil, ai-edge-litert;
   fall back to `tflite-runtime` if there is no wheel).
3. `scp her2_pi_bundle.tar.gz pi@<pi-ip>:~`, then on the Pi:
   - `tar xzf her2_pi_bundle.tar.gz && cd her2_pi_bundle`
   - `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
   - `python3 benchmark_pi.py --models models --data test_set.npz --threads 1 2 4 --out pi_results.csv`
4. Copy `pi_results.csv` to `FYP2/results/` and rerun `python report.py`.

**Kria KV260: [NOT AVAILABLE], no project code. [NOT VERIFIED]: the requirements below come from general Vitis AI
knowledge; check them against AMD's current Vitis AI and Kria documentation before planning work.**
- The DPU (DPUCZDX8G) runs models quantised and compiled with **Vitis AI**: the TF2 quantizer `vai_q_tensorflow2`, then
  `vai_c_tensorflow2` with the KV260 DPU `arch.json`. The toolchain ships as a Docker image for x86-64 Linux. Running it
  under WSL2 is not officially supported, so plan on a native Ubuntu host.
- Vitis AI's TF2 support is tied to a specific TensorFlow release, which may differ from 2.15. Re-export the FP32 Keras
  model in a format that release accepts. **The project's TFLite INT8 models cannot be deployed on the DPU.**
- The board runs Kria Ubuntu or PetaLinux with a DPU firmware overlay (`xmutil loadapp`), with VART/Vitis AI Runtime for inference.
- Expect layers the DPU lacks to fall back to the ARM CPU. Softmax and the CORN head are likely candidates; ConvNeXt's
  LayerNorm/GELU and parts of MobileNetV3 may be too. ResNet50, MobileNetV2 and EfficientNet-style CNNs are the safer candidates.
- None of the reported accuracy, latency or power numbers transfer. The DPU's INT8 quantisation is different from TFLite's.

## 8. Resuming and verifying

**Resuming:**
- `run_queue.sh` / `run_variants.sh` skip runs whose `summary.json` exists. An interrupted run has no `summary.json`
  and simply retrains from the start (there is no mid-run checkpoint).
- `evaluate_all.py` skips runs with `results/metrics/<run>.json` (`--force` redoes them).
- `analysis_models.py` skips architectures with `results/analysis/models/<arch>.json` (`--force`).
- `analysis_haem_cue.py --arch X` overwrites that architecture's file.
- **A hung job** (log silent for over 30 min, GPU idle): kill it and rerun the same command. Finished work is kept.

**Verifying the rebuild:**
1. Splits must be identical (they are seeded):

   ```bash
   cmp ~/her2/FYP3/splits/split_manifest_grouped.csv <original copy>
   ```

   Do the same for `split_manifest_official.csv` and `FYP2/splits/split_manifest.csv`.
2. Data: `split_manifest_official.csv` holds an md5 per raw tile. Compare that column with the original.
3. Models: compare `metrics_all.csv` with the original. Expect differences of about ±1–2 accuracy points per run
   (different hardware and GPU non-determinism; single seed). Compare *conclusions* with `paired.py` CIs, not single numbers.
4. Weight loading: step 9 prints `reload diff` for every architecture. It must be about 0 (≤ 1e-5).
5. Reproduced conclusions to check:
   - FYP2 ≈ 1.00 vs BCI 0.91–0.95 test accuracy;
   - frozen probe 0.98 vs 0.77;
   - the hires gain on EfficientNetV2B0 with a CI excluding 0;
   - cw, ls and CORN within noise.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `list_physical_devices('GPU')` is empty in WSL | Update the Windows NVIDIA driver; check `nvidia-smi`; reinstall `tensorflow[and-cuda]==2.15.1` in the venv |
| `ImportError: libGL.so.1` | You installed `opencv-python`; use `opencv-python-headless`, or `sudo apt install -y libgl1 libglib2.0-0` |
| Out of GPU memory | Close other TF jobs. Batch sizes are part of the method (`config.MODELS`), so changing them changes the experiment; record it if you must |
| Keras cannot load a ConvNeXt `model.keras` | Expected: `common.load_trained` falls back to rebuilding the network and loading `model.weights.h5` |
| `FileNotFoundError` under `FYP2/...` | FYP2 not reproduced or not placed next to FYP3 (section 4); the folder name `HER2 SCORES DATA` contains spaces, so quote it |
| `scan_type_threshold.json` missing in step 9 | Run step 7 first |
| Extremely slow data loading | The project is under `/mnt/c`; move it into `~/her2` |
| ImageNet weight download fails | Internet access is needed once; the weights are cached in `~/.keras/models/` |
| `run_*.sh: python: not found` | Activate the venv before running the scripts (they use `command -v python`) |
