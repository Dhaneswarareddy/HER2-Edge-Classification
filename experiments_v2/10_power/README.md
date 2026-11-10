# Step 30: smartphone power and energy per inference (next-semester work; measured 10 Nov 2026)

**Status: measured, analysis pending.** No power or energy number may be reported until `outputs/` holds the results.

## Method

- **Devices.** Samsung Galaxy M35 5G and Realme 6, USB unplugged (charging hides the battery drain). Wireless ADB was
  used only to start the run and to check every 60 s whether it had finished.
- **Script.** `phone_power.sh` runs on the phone, detached, and starts by itself when the cable is removed:
  60 s settle → idle 300 s → [60 s rest → deployed 448 px FP16 model, continuous `benchmark_model` invokes, 300 s] for
  4 threads and 1 thread → 60 s rest → idle 300 s.
- **Sampling.** About once a second: battery current, voltage, charge counter, battery temperature, plug status.
  - Realme: `/sys/class/power_supply/battery/current_now` (mA); the gauge updates about every second.
  - Samsung: `dumpsys battery` "current now" (µA); the gauge updates only about every 30 s, so Samsung values are
    coarser. The charge counter moves in 5 mAh steps.
- **Screen.** On throughout at a fixed manual brightness, so it is part of both idle and inference power.
- **Analysis** (`analyze_phone_power.py`): P = |I|·V; idle = mean of the two idle phases; energy per inference =
  P_run × mean invoke time (total) and (P_run − P_idle) × mean invoke time (incremental). The first 15 s of each phase
  are dropped.

## Limits

Whole-device power, not CPU power; fuel-gauge resolution, not a calibrated meter; one run per configuration. Pi 4B power
needs an external USB-C meter and was not measured.
