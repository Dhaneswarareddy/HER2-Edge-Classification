#!/system/bin/sh
# Whole-phone power during TFLite inference, from the phone's own battery fuel gauge (no root, no extra hardware).
# Runs on the phone, detached, so the USB cable can be unplugged (charging hides the battery drain).
#   sh phone_power.sh realme|samsung
# Protocol: wait for unplug -> 60 s settle -> idle 300 s -> [60 s rest -> FP16 448 px, T threads, 300 s] for T in 4 1
#           -> 60 s rest -> idle 300 s.  Screen stays on at a fixed manual brightness in every phase.
# Samples every ~1 s: time_ns, phase, current, voltage, charge counter, battery temperature, status/plug.
D=/data/local/tmp/power
MODE=$1
OUT=$D/samples.csv
PH=$D/phase
MODEL=/data/local/tmp/primary_448_fp16.tflite
rm -f $D/done $D/aborted $OUT $D/bench_*.txt $D/events.txt
echo waiting > $PH
echo "t_ns,phase,current_raw,voltage_mV,charge_counter_uAh,temp_dC,status" > $OUT
P=/sys/class/power_supply/battery
read_now() {
  if [ "$MODE" = realme ]; then
    echo "$(cat $P/current_now),$(cat $P/voltage_now),$(cat $P/charge_counter 2>/dev/null),$(cat $P/temp),$(cat $P/status)"
  else
    B=$(dumpsys battery)
    I=$(echo "$B" | grep "current now:" | head -1 | sed 's/.*: //')
    V=$(echo "$B" | grep "^  voltage:" | head -1 | sed 's/.*: //')
    Q=$(echo "$B" | grep "Charge counter:" | head -1 | sed 's/.*: //')
    T=$(echo "$B" | grep "^  temperature:" | head -1 | sed 's/.*: //')
    U=$(echo "$B" | grep "USB powered:" | head -1 | sed 's/.*: //')
    echo "$I,$V,$Q,$T,usb_$U"
  fi
}
plugged() {
  if [ "$MODE" = realme ]; then
    S=$(cat $P/status); [ "$S" = Charging ] || [ "$S" = Full ]
  else
    dumpsys battery | grep -q -E "(USB|AC) powered: true"
  fi
}
( while [ ! -f $D/done ] && [ ! -f $D/aborted ]; do
    echo "$(date +%s%N),$(cat $PH),$(read_now)" >> $OUT
    sleep 1
  done ) &
mark() { echo "$1" > $PH; echo "$(date +%s%N),$1" >> $D/events.txt; input keyevent 0; }
n=0
while plugged; do sleep 2; n=$((n+2)); if [ $n -gt 2400 ]; then mark aborted_never_unplugged; touch $D/aborted; exit 1; fi; done
mark settle; sleep 60
mark idle_pre; sleep 300
for T in 4 1; do
  mark rest; sleep 60
  mark fp16_${T}t
  /data/local/tmp/benchmark_model --graph=$MODEL --num_threads=$T --use_xnnpack=true --warmup_runs=5 \
     --num_runs=1000000 --min_secs=0 --max_secs=300 > $D/bench_${T}t.txt 2>&1
done
mark rest; sleep 60
mark idle_post; sleep 300
mark finished
plugged && echo "WARNING plugged in before end" >> $D/events.txt
touch $D/done
