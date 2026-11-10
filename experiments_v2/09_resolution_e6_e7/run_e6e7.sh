#!/bin/bash
# Sequential E6 -> E7 queue (skips finished runs). Logs in logs/. Keeps the Mac awake (caffeinate).
cd "$(dirname "$0")"
PY=/Users/dhaneswarareddy/Downloads/Final_Year_Project/Codes/Project/bin/python
mkdir -p logs
for spec in DenseNet121:hires DenseNet121:lowbatch MobileNetV3Small:hires1024; do
  m=${spec%:*}; v=${spec#*:}; name="${m}__ce__${v}"
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; continue; fi
  echo "$(date '+%F %T') start $name"
  caffeinate -i "$PY" train_e6e7.py --model "$m" --variant "$v" > "logs/train_${name}.log" 2>&1
  echo "$(date '+%F %T') end $name rc=$? $(grep -E '^DONE' logs/train_${name}.log | tail -1)"
done
echo "E6E7 FINISHED $(date '+%F %T')"
