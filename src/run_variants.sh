#!/bin/bash
# Train improvement variants sequentially (one process each; skips finished runs).
#   ./run_variants.sh MobileNetV3Small:ce:cw MobileNetV3Small:ce:ls ...
# Separate from run_queue.sh, which trains baselines with train.py.
cd "$(dirname "$0")"
PY=/Users/dhaneswarareddy/Downloads/Final_Year_Project/Codes/Project/bin/python
for spec in "$@"; do
  IFS=: read -r m l v <<< "$spec"
  name="${m}__${l}__${v}"
  if [ -f "../results/runs/$name/summary.json" ]; then echo "skip $name (done)"; continue; fi
  echo "$(date '+%H:%M:%S') start $name"
  "$PY" train_variant.py --model "$m" --loss "$l" --variant "$v" > "../results/logs/train_${name}.log" 2>&1
  rc=$?
  grep -E "^DONE|Error" "../results/logs/train_${name}.log" | tail -2
  echo "$(date '+%H:%M:%S') end $name rc=$rc"
done
echo "VARIANTS FINISHED $(date)"
