# Step 30: smartphone power and energy per inference (next-semester work; measured 10 Nov 2026)

**Status: complete** (one run per configuration). Results: `outputs/phone_power_summary.json`,
`outputs/phone_power_phases.csv`; raw data per phone in `outputs/<device>/` (samples, phase events, `benchmark_model` logs).

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

## Results (deployed 448 px FP16 model, continuous inference for 300 s, whole phone, screen on)

| | Samsung Galaxy M35 5G | Realme 6 |
|---|---|---|
| Idle, screen on (mean of before/after) | 737 mW (before 769, after 705) | 752 mW (before 674, after 830) |
| **4 threads:** power ± SE | **4.75 W ± 0.41** | **4.64 W ± 0.05** |
| 4 threads: added by inference | 4.01 W | 3.89 W |
| 4 threads: mean time per tile (sustained) | 98.1 ms (3,044 tiles) | 156.4 ms (1,914 tiles) |
| **4 threads: energy per tile, total / added** | **0.47 J / 0.39 J** | **0.73 J / 0.61 J** |
| 1 thread: power ± SE | 2.65 W ± 0.21 | 2.79 W ± 0.02 |
| 1 thread: added by inference | 1.91 W | 2.04 W |
| 1 thread: mean time per tile (sustained) | 258.9 ms (1,158 tiles) | 301.7 ms (993 tiles) |
| 1 thread: energy per tile, total / added | 0.69 J / 0.49 J | 0.84 J / 0.62 J |
| Battery temperature during the 4-thread phase | 30.9 → 31.9 °C | 30.0 → 36.4 °C |
| Distinct gauge readings per phase | 10–11 | 66–192 |

SE = standard deviation / √(number of distinct gauge readings). "Added" = phase power − idle power. Energy per tile =
power × mean time per tile.

**Findings**
1. **Inference adds about 3.9–4.0 W at 4 threads and 1.9–2.0 W at 1 thread** on top of a screen-on idle phone
   (0.74–0.75 W).
2. **Energy per tile at 4 threads: 0.39 J (Samsung) and 0.61 J (Realme) above idle; 0.47 J and 0.73 J in total.**
   From the added energy alone, one watt-hour of battery covers about 9,100 tiles (Samsung) and 5,900 (Realme).
3. **4 threads costs no more energy per tile than 1 thread** (Samsung 0.39 vs 0.49 J added; Realme 0.61 vs 0.62 J) and
   less in total, because the screen and idle cost are spread over more tiles. Faster is also cheaper here.
4. **Sustained inference on battery is slower than the short benchmark.** Median time per tile vs the 10 Nov short
   benchmark (100 runs, plugged in): Samsung 99.1 vs 81.4 ms at 4 threads (+22%) and 258.0 vs 214.9 ms at 1 thread
   (+20%); Realme 151.6 vs 142.6 ms (+6%) and 301.5 vs 301.7 ms (0%). The cause (temperature or a battery power
   policy) was not established: CPU temperatures and frequencies are not readable without root. For a phone used
   continuously, the sustained values are the realistic ones.

**Data quality**
- The **Realme** gauge updated about once a second, so its power values are precise (SE 1–2%). Its idle power rose by
  157 mW between the first and last idle phase as the phone warmed, so its "added" values carry roughly ±80 mW of
  baseline uncertainty (about ±0.01 J per tile at 4 threads).
- The **Samsung** gauge updated only about every 30 s (10–11 readings per phase), so its values are coarse (SE about
  9%).
- The **charge counter** (coulomb) estimate is in the CSV but **not used**. Over 5-minute windows it is too coarse:
  Samsung counts in 5 mAh steps (about ±35% at idle), and the Realme counter lags and even orders the phases
  inconsistently.

## Limits

Whole-device power, not CPU power; fuel-gauge resolution, not a calibrated meter; one run per configuration. Pi 4B power
needs an external USB-C meter and was not measured.
