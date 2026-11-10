# Raspberry Pi 4B precision benchmark (FP32 / FP16 / INT8): ready to run, not yet run

**Why.** The 31 Oct Pi benchmark measured FP16 only. The final report's Pi INT8 latency (122.9 ms) has no measurement
behind it (`../../REPORT_CORRECTIONS_v2.md`, item 6). This bundle measures all three precisions of the deployed 448 px
model with one protocol.

**Contents.**
- `models/primary_hires_{fp32,fp16,int8}.tflite`: byte-identical copies of
  `results/deploy/EfficientNetV2B0__ce__hires/model_*.tflite` (MD5 checked).
- `tiles120.npz`: the 120 preprocessed BCI-**validation** reference tiles (`deployment/reference/primary.npz`).
- `refs_primary_precisions.npz`: Mac TFLite outputs for each precision on the same tiles, written by
  `../mac_prepare.py`.
- `pi_precision_benchmark.py`.

**Measured.** For each precision and thread count (1, 2, 4), in a fresh process:
- load time;
- 10 warm-up and 100 timed invokes;
- median, p95 and p99 latency;
- peak RSS and process CPU %;
- `vcgencmd` temperature, ARM clock and throttle flags, before and after;
- prediction agreement with the Mac TFLite output at the same precision and with Keras;
- accuracy on the 120 tiles.

Raw output per configuration is kept in `raw/`. A row counts as a Pi 4 result only if `/proc/device-tree/model` says
so. **No power or energy is measured.**

## Run

```bash
# on the Mac (needs the Pi's address and login)
scp -r /Users/dhaneswarareddy/Downloads/FYP3/experiments_v2/08_edge/pi_bundle pi@<pi-ip>:~/pi_bundle_v2
# on the Pi: reuse the existing environment from the 31 Oct benchmark (ai_edge_litert 2.3.0)
source ~/her2env/bin/activate
cd ~/pi_bundle_v2
sudo cpufreq-set -g performance 2>/dev/null || echo performance | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
python3 pi_precision_benchmark.py --threads 1 2 4 --out results_pi.csv
# back on the Mac
scp -r pi@<pi-ip>:~/pi_bundle_v2/{results_pi.csv,raw} /Users/dhaneswarareddy/Downloads/FYP3/experiments_v2/08_edge/outputs/pi4/
```

It takes about 15 minutes: 9 configurations with a 60 s cool-down between them.

**Status (10 Nov 2026).** The Pi answers on the network, but SSH from this Mac was refused
(`Permission denied (publickey,password)`). The run needs the Pi login, or for someone at the Pi to run the two
commands above.
