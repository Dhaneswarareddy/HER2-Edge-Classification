# Cross-Dataset Generalisation, Stain Calibration, and Mobile Edge Deployment for HER2 Histopathology Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Platform: Android & Linux](https://img.shields.io/badge/Platform-Android%20%7C%20Linux%20%7C%20macOS-green.svg)](https://developer.android.com/)
[![Inference: TFLite XNNPACK](https://img.shields.io/badge/Inference-TFLite%20XNNPACK-blueviolet.svg)](https://www.tensorflow.org/lite)
[![Status: Complete & Reproducible](https://img.shields.io/badge/Status-Complete%20%26%20Reproducible-brightgreen.svg)]()

This repository contains the authoritative, reproducible implementation, experimental code, trained edge models, and Android deployment assets for our research on **Automated HER2 Breast Cancer Scoring from Clinical Immunohistochemistry (IHC) Biopsies**.

---

## 📌 Executive Research Overview

Automated scoring of Human Epidermal Growth Factor Receptor 2 (HER2) from immunohistochemistry (IHC) tissue sections is critical for breast cancer stratification and targeted anti-HER2 therapy (Trastuzumab). While baseline models trained on synthetic conditional GAN imagery achieve near-perfect performance, transferring these architectures to authentic clinical biopsy cohorts reveals severe vulnerabilities.

This project rigorously investigates the transition from synthetic benchmarks to real-world clinical tissue sections across **4,873 clinical tiles from the Breast Cancer Immunohistochemical (BCI) benchmark** and **9,940 clinical tiles from the external HER2-IHC-40x cohort**.

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              FYP3 SYSTEM ARCHITECTURE & WORKFLOW                                         |
|                                                                                                                         |
|   1. In-Domain Benchmark (BCI)      2. Unsupervised Stain Calibration     3. Physical Edge & Smartphone Deployment       |
|   +--------------------------+      +-------------------------------+     +------------------------------------------+  |
|   | 4,873 Clinical Biopsies  | ---> | HARMONISE Optical Density     | --->| Physical Samsung Galaxy M35 5G (46 FPS)  |  |
|   | 448px Spatial Resolution |      | Quantile Matching Transform   |     | Physical Realme 6 (26 FPS)               |  |
|   | 97.5% Acc | 0.959 QWK    |      | Recovers +20.2% Balanced Acc  |     | Production Native Android Application    |  |
|   +--------------------------+      +-------------------------------+     +------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------------------------+
```

### Key Scientific Findings:
1. **Spatial Resolution Outweighs Architectural Complexity**: Doubling spatial input resolution from $224\times224$ to $448\times448$ yields a statistically confirmed leap of **$\Delta\text{Val QWK} = +0.105$** ($95\%\text{ CI } [0.056, 0.168]$ on EfficientNetV2B0), establishing our primary in-domain model: `EfficientNetV2B0__ce__hires` (**97.5% Test Acc, 0.959 Test QWK**).
2. **Synthetic Pretraining Failure**: Transferring weights pretrained on synthetic cGAN images ($\Delta\text{QWK} = -0.015$) or augmenting real training with 25% synthetic tiles ($\Delta\text{QWK} = -0.023$) fails to improve real-tissue feature learning over ImageNet initialization.
3. **Cross-Dataset Stain Collapse & HARMONISE Calibration**: Evaluating raw CNNs zero-shot on an external diagnostic centre (HER2-IHC-40x) causes catastrophic collapse (accuracy plummets to 41.9%–50.8%). Unsupervised optical density calibration (**HARMONISE**) restores balanced accuracy on standard 224px models by **+20.2 percentage points** (0.301 $\to$ 0.503) without fine-tuning.
4. **The Clinical QWK Paradox**: High aggregate Quadratic Weighted Kappa ($>0.80$) frequently masks near-total algorithmic insensitivity to borderline equivocal classes (Score 1+ and 2+). Mandatory per-class recall auditing is essential.
5. **Point-of-Care Mobile Edge AI**: Native deployment on physical smartphones (**Samsung Galaxy M35 5G** and **Realme 6**) demonstrates **21.7 ms latency (46.1 FPS)** on INT8 with 4-thread Cortex-A78 core pinning and 67% RAM reduction (39.4 MB), running completely thermally stable (28.9°C) and outperforming the Raspberry Pi 4B by **4.5×–16.3×**.

---

## 📊 Summary of Authoritative Results

### 1. In-Domain Clinical Performance (BCI Benchmark)

| Model Architecture | Input Res | Loss | Val Acc | Val BalAcc | Val QWK | Test Acc | Test BalAcc | Test QWK |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`EfficientNetV2B0__ce__hires` (Primary)** | **$448\times 448$** | **CE** | **0.950** | **0.951** | **0.935** | **0.975** | **0.975** | **0.959** |
| `MobileNetV3Small__ce__hires` | $448\times 448$ | CE | 0.923 | 0.924 | 0.898 | 0.963 | 0.963 | 0.940 |
| `EfficientNetV2B0__ce` (Baseline) | $224\times 224$ | CE | 0.898 | 0.900 | 0.830 | 0.921 | 0.922 | 0.887 |
| `ResNet50__ce` | $224\times 224$ | CE | 0.908 | 0.910 | 0.869 | 0.923 | 0.925 | 0.890 |
| `DenseNet121__ce` | $224\times 224$ | CE | 0.904 | 0.906 | 0.858 | 0.908 | 0.910 | 0.869 |
| `MobileNetV3Small__ce` | $224\times 224$ | CE | 0.895 | 0.897 | 0.852 | 0.893 | 0.895 | 0.849 |

### 2. Tri-Hardware Edge Benchmark Comparison

All metrics measured on physical hardware over 50 timed iterations (10 warmup runs):

| Evaluation Dimension | Raspberry Pi 4 Model B (4GB) | Realme 6 (4GB) | Samsung Galaxy M35 5G (6GB) | Speedup / Efficiency Gain |
|---|:---:|:---:|:---:|:---:|
| **Processor SoC** | Broadcom BCM2711 (28nm Planar) | MediaTek Helio G90T (12nm FinFET) | Samsung Exynos 1380 (5nm FinFET EUV) | 5.6× finer lithography node |
| **CPU Cluster** | 4× Cortex-A72 @ 1.8 GHz (Uniform) | 2× A76 @ 2.05 GHz + 6× A55 @ 2.0 GHz | 4× A78 @ 2.4 GHz + 4× A55 @ 2.0 GHz | Modern big.LITTLE architecture |
| **Optimal Threads** | 2 Threads (Thermal Limiting) | 8 Threads (Burst) / 2T (Per-Watt) | 4 Threads (pinned to 4× Cortex-A78) | Matches performance core cluster |
| **Primary FP16 Latency** | 444.1 ms (2T) / 353.5 ms (4T) | **145.19 ms (8T)** | **78.91 ms (4T)** | **4.48× faster on M35 vs Pi 4B** |
| **Primary INT8 Latency** | 122.9 ms (4T, 8.1 FPS) | **38.63 ms (8T, 25.9 FPS)** | **21.71 ms (4T, 46.1 FPS)** | **16.28× faster on M35 vs Pi 4B** |
| **Peak Operating Temp** | **84.2°C (Thermal Soft Trip)** | **30.6°C (Thermally Flat)** | **28.9°C (Thermally Flat)** | **55.3°C cooler on smartphone** |
| **Thermal Throttling** | `throttled=0x80008` (Drops to 1.68 GHz)| `throttled=0x0` (Sustained 2.05 GHz) | `throttled=0x0` (Sustained 2.40 GHz) | Zero throttling on mobile phones |
| **Peak RAM Footprint** | 191.6 MB (Python + LiteRT) | **41.0 MB (INT8 Native)** | **39.4 MB (INT8 Native)** | **79.4% memory reduction** |

---

## 📱 Native Android Application: HER2 Mobile Edge

A production-ready Android Studio application (`HER2 Mobile Edge`, package `com.fyp3.her2classifier`) is provided in [`android_app/`](android_app/):
- **Instant APK Installation**: Pre-built APK available at [`deployment/apk/HER2_Mobile_Edge.apk`](deployment/apk/HER2_Mobile_Edge.apk).
- **Dynamic Hardware Detection**: Inspects processor architecture at runtime, displaying diagnostic badges for Exynos 1380, Helio G90T, Snapdragon, and Tensor SoCs.
- **On-Device Vectorized Inference**: Uses TensorFlow Lite with XNNPACK FP16 / INT8 delegates for real-time inference directly on the phone.
- **Clinical Decision Support**: Provides real-time multinomial probability bars, ASCO/CAP guideline statuses, and confirmatory reflex FISH testing alerts for borderline 2+ predictions.

![Figure 9: Physical Application Deployment on Samsung Galaxy M35 5G](final_report_figures/fig9_mobile_app_interface_and_inference.png)

![Figure 10: Physical Application Deployment on Realme 6](final_report_figures/fig10_realme6_app_interface_and_inference.png)

---

## 📂 Repository Directory Structure

```
HER2-Edge-Classification/
├── README.md                              # Main project documentation and quickstart
├── requirements.txt                       # Core Python package dependencies
├── DATASET.md                             # Dataset acquisition & directory layout guide
├── FYP3_REPRODUCTION_GUIDE.md             # Detailed step-by-step reproduction instructions
├── FYP3_FINAL_REPORT.md                   # Full 18-section BTech final technical report
├── FYP3_PRESENTATION_OUTLINE.md           # 15-slide defense presentation outline
├── FINAL_MODEL_SELECTION.md               # Model selection criteria and rationale
├── DEPLOYMENT_README.md                   # Edge & mobile deployment technical manual
│
├── src/                                   # Source code
│   ├── common.py                          # Metric computation, seeding, data loaders
│   ├── config.py                          # Hyperparameter configuration & path definitions
│   ├── prepare_data.py                    # Slide fragment grouping & tile preprocessing
│   ├── make_grouped_split.py              # Patient/slide grouped cross-validation splits
│   ├── train.py                           # Baseline deep learning training harness
│   ├── train_variant.py                   # Model variant training (resolution, batch, etc.)
│   ├── variants.py                        # Experimental ablation configurations
│   ├── evaluate_all.py                    # Comprehensive in-domain evaluation harness
│   ├── external_eval.py                   # Zero-shot cross-dataset & HARMONISE evaluation
│   ├── select_models.py                   # Automated primary model selection engine
│   ├── deploy.py                          # TensorFlow Lite conversion & INT8 quantization
│   ├── paired.py                          # Paired bootstrap effects & 95% confidence intervals
│   └── report.py                          # Summary tables & LaTeX metric report generator
│
├── splits/                                # Authoritative slide-grouped split manifests
│   ├── split_manifest_grouped.csv         # Grouped train/val/test split (4,873 tiles)
│   └── split_manifest_official.csv        # Official BCI benchmark split
│
├── deployment/                            # Mobile & edge deployment artifacts
│   ├── apk/                               # Pre-built Android debug APK
│   │   └── HER2_Mobile_Edge.apk           # Direct installation package (39 MB)
│   ├── models/                            # Exported TensorFlow Lite models
│   │   ├── primary_bci_hires_fp16.tflite  # Primary 448px FP16 model (11.8 MB)
│   │   ├── primary_bci_hires_int8.tflite  # Primary 448px INT8 quantized model (7.1 MB)
│   │   ├── primary_bci_hires_fp32.tflite  # Primary 448px FP32 baseline (23.5 MB)
│   │   └── experimental_stain224_fp16.tflite # Stain-calibrated 224px FP16 model (11.8 MB)
│   ├── sample_tiles/                      # Bundled reference clinical biopsy tiles
│   │   ├── sample_tile_class_0.png        # Negative biopsy (Score 0)
│   │   ├── sample_tile_class_1.png        # Negative biopsy (Score 1+)
│   │   ├── sample_tile_class_2.png        # Equivocal borderline biopsy (Score 2+)
│   │   └── sample_tile_class_3.png        # Strongly overexpressed biopsy (Score 3+)
│   ├── benchmark_results/                 # Authoritative physical hardware benchmark CSVs
│   │   ├── benchmark_samsung_m35.csv      # Samsung Galaxy M35 5G raw measurements
│   │   ├── benchmark_realme_6.csv         # Realme 6 raw measurements
│   │   ├── benchmark_primary_pi4.csv      # Raspberry Pi 4B Primary model measurements
│   │   └── benchmark_experimental_pi4.csv # Raspberry Pi 4B Experimental model measurements
│   ├── benchmark.py                       # Cross-platform edge benchmarking script
│   └── predict.py                         # Single tile command-line inference utility
│
├── android_app/                           # Complete Android Studio project (Kotlin)
│   ├── app/src/main/                      # App source code, UI layouts, assets
│   ├── build.gradle                       # Gradle build configuration
│   └── gradlew                            # Gradle wrapper script
│
├── final_report_figures/                  # Publication-quality 300 DPI figures
│   ├── fig1_bci_model_comparison.png      # In-domain BCI model comparison
│   ├── fig2_paired_effects_forest.png     # Paired effects forest plot with bootstrap CIs
│   ├── fig3_stain_optical_density_distributions.png # Optical density distributions
│   ├── fig4_cross_dataset_collapse_and_harmonise.png # Domain collapse & stain recovery
│   ├── fig5_classwise_performance_and_qwk_paradox.png # Classwise recall & QWK paradox
│   ├── fig6_harmonise2_bootstrap_and_flagging.png # Reliability & selective review analysis
│   ├── fig7_smartphone_edge_benchmark.png # Tri-hardware edge benchmark comparison
│   ├── fig8_quantization_and_agreement.png # Quantization efficiency & host-edge parity
│   ├── fig9_mobile_app_interface_and_inference.png # Samsung M35 live application composite
│   └── fig10_realme6_app_interface_and_inference.png # Realme 6 live application composite
│
├── generate_final_report_figures.py       # Programmatic figure generator for Figs 1-6 & 8
├── generate_smartphone_figures.py         # Programmatic figure generator for Fig 7
├── create_app_composite_figure.py         # Programmatic figure generator for Fig 9
├── create_realme_6_composite_figure.py    # Programmatic figure generator for Fig 10
│
├── final_reports/tables/                  # Authoritative CSV summary metric tables
└── results/                               # Authoritative evaluation logs, CSVs, and predictions
    ├── metrics_all.csv                    # In-domain evaluation metrics across all runs
    ├── selection.json                     # Primary model selection metadata
    └── external/                          # Zero-shot and HARMONISE cross-dataset outputs
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Dhaneswarareddy/HER2-Edge-Classification.git
cd HER2-Edge-Classification

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation

Follow [`DATASET.md`](DATASET.md) to download:
1. **BCI Benchmark Dataset** (Breast Cancer Immunohistochemical): Extract to `data/bci/`.
2. **HER2-IHC-40x Cohort**: Extract to `data/external_her2_40x/`.

Run data preprocessing:
```bash
python src/prepare_data.py
python src/make_grouped_split.py
```

### 3. Model Training & Evaluation

```bash
# Train the primary high-resolution EfficientNetV2B0 model (448px)
python src/train_variant.py --variant hires

# Evaluate all trained backbones on the in-domain test set
python src/evaluate_all.py

# Run cross-dataset zero-shot evaluation and HARMONISE stain calibration
python src/external_eval.py
```

### 4. Recreating Publication Figures

All 10 figures in [`final_report_figures/`](final_report_figures/) can be regenerated with zero synthetic extrapolation:
```bash
# Generate core research figures 1-6 and Figure 8:
python generate_final_report_figures.py

# Generate Tri-Hardware edge benchmark comparison (Figure 7):
python generate_smartphone_figures.py

# Generate Android App composite figures (Figures 9 and 10):
python create_app_composite_figure.py
python create_realme_6_composite_figure.py
```

### 5. Running the Android Application

- **Direct Installation**:
  ```bash
  adb install deployment/apk/HER2_Mobile_Edge.apk
  ```
- **Building from Source**: Open `android_app/` in Android Studio and click **Run** (or execute `./gradlew assembleDebug`).

---

## ⚖️ Scientific Disclosures & Ethical Considerations

1. **Investigational Prototype**: This software is an academic research prototype. It is not certified as a medical device and must never be used autonomously for clinical diagnosis, patient management, or therapeutic selection.
2. **Equivocal Vulnerability**: Deep learning models show severe vulnerability on borderline Score 1+ and Score 2+ cases (Score 2+ precision $\le 0.11$ on external tissue). Confirmatory reflex FISH/CISH gene amplification testing is strictly mandated per ASCO/CAP guidelines.
3. **Stain Calibration Nature**: HARMONISE provides empirical quantile-matching stain calibration between two specific clinical acquisition protocols; it does not represent an invariant physical stain normalization.

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{reddy2026her2edge,
  title={Cross-Dataset Generalisation, Stain Calibration, and Mobile Edge Deployment for HER2 Histopathology Classification},
  author={Karri Mohan Naga Durga Dhaneswara Reddy},
  year={2026},
  institution={Department of Computer Science and Engineering},
  url={https://github.com/Dhaneswarareddy/HER2-Edge-Classification}
}
```
