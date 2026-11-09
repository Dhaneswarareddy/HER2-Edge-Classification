#!/bin/bash
# analysis_models.py, one process per architecture: TF on Metal does not release memory between
# architectures, and a single process for all 9 grew to 18 GB and stalled at the third.
# Architectures with a cached results/analysis/models/<arch>.json are skipped by the script itself.
# Usage: ./run_analysis_models.sh [arch ...]   (default: all 9)
cd "$(dirname "$0")"
PY=/Users/dhaneswarareddy/Downloads/Final_Year_Project/Codes/Project/bin/python
LOG=../results/logs/analysis_models.log
ARCHS=${@:-ResNet50 DenseNet121 ConvNeXtTiny EfficientNetB0 EfficientNetV2B0 NASNetMobile MobileNetV2 MobileNetV3Large MobileNetV3Small}
for a in $ARCHS; do
  echo "=== $a $(date '+%H:%M') $(sysctl -n vm.swapusage)" >> $LOG
  caffeinate -ims $PY analysis_models.py --archs $a >> $LOG 2>&1
  echo "rc=$? $a $(date '+%H:%M')" >> $LOG
done
