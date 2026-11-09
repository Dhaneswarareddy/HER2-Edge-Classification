#!/bin/bash
# Wait for the last official-split reference run, preserve both official-split runs under
# results/runs_official/, then train the 9 CE baselines on the main (grouped) split.
cd "$(dirname "$0")"
REF_PID=$1
while kill -0 "$REF_PID" 2>/dev/null; do sleep 30; done
mkdir -p ../results/runs_official ../results/logs/official
for d in ../results/runs/*/; do [ -d "$d" ] && mv "$d" ../results/runs_official/; done
mv ../results/logs/train_*.log ../results/logs/queue_baseline.log ../results/logs/official/ 2>/dev/null
echo "$(date '+%H:%M:%S') official-split reference runs preserved:"; ls ../results/runs_official
FYP3_SPLIT=grouped ./run_queue.sh MobileNetV3Small:ce MobileNetV2:ce MobileNetV3Large:ce EfficientNetB0:ce \
  EfficientNetV2B0:ce NASNetMobile:ce ResNet50:ce DenseNet121:ce ConvNeXtTiny:ce
