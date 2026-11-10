"""Analyse the phone power runs (phone_power.sh) pulled into outputs/<device>/.

Whole-phone battery power P = |I| * V from the phone's own fuel gauge, screen on at fixed brightness.
  idle      = mean of idle_pre and idle_post (screen on, nothing running)
  fp16_4t / fp16_1t = continuous back-to-back TFLite invokes of the deployed 448 px FP16 model for 300 s
Energy per inference:
  total       = P_run * t_inf               (everything the phone draws while classifying, incl. screen)
  incremental = (P_run - P_idle) * t_inf    (what inference adds on top of an idle, screen-on phone)
t_inf is the mean invoke time reported by benchmark_model for the same 300 s window.
The first 15 s of every phase are dropped (transition). Two independent current estimates are reported per phase:
  * mean of the instantaneous fuel-gauge readings, and how many distinct readings occurred (update rate check);
  * coulomb counter: delta charge / delta time (only where the counter actually moved).
  python analyze_phone_power.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DEVICES = {"samsung_m35": "Samsung Galaxy M35 5G", "realme_6": "Realme 6"}
# current_raw units differ: Samsung dumpsys "current now" is in uA; Realme sysfs current_now is in mA (checked
# against the voltage and the charge-counter slope below; both estimates are written so the reader can verify).
CUR_TO_MA = {"samsung_m35": 1e-3, "realme_6": 1.0}
SKIP_S = 15
# 10 Nov 2026 short benchmark (10 warm-up + 100 runs, plugged in; experiments_v2/08_edge/outputs/edge_latency_table.csv)
SHORT_BENCH_MS = {"samsung_m35": {"4t": 81.4, "1t": 214.9}, "realme_6": {"4t": 142.6, "1t": 301.7}}
# The charge-counter (coulomb) estimate is written for completeness but NOT used: over 5-min windows it is too coarse
# (Samsung: 5 mAh steps, i.e. +/-35% at idle; Realme: the counter lags and even orders the phases inconsistently).


def bench(path):
    txt = path.read_text(errors="replace")
    avg = re.search(r"Inference \(avg\): ([\d.e+]+)", txt)
    cnt = re.findall(r"count=(\d+)", txt)
    med = re.findall(r"median=([\d.]+)", txt)
    return dict(t_inf_ms=float(avg.group(1)) / 1000 if avg else np.nan,          # mean: sets throughput / energy
                median_ms=float(med[-1]) / 1000 if med else np.nan, invokes=int(cnt[-1]) if cnt else 0)


def phase_stats(df, ph, dev):
    d = df[df.phase == ph].copy()
    if d.empty:
        return None
    d = d[d.t >= d.t.min() + SKIP_S]
    i_ma = d.current_raw.abs() * CUR_TO_MA[dev]
    p_mw = i_ma * d.voltage_mV / 1000.0
    q = d.charge_counter_uAh.astype(float)
    dt_h = (d.t.max() - d.t.min()) / 3600
    dq_ma = (q.iloc[0] - q.iloc[-1]) / 1000 / dt_h if dt_h > 0 else np.nan
    return dict(phase=ph, seconds=round(d.t.max() - d.t.min(), 1), samples=len(d),
                distinct_current_readings=int(d.current_raw.nunique()),
                mean_current_mA=round(i_ma.mean(), 1), sd_current_mA=round(i_ma.std(), 1),
                mean_voltage_mV=round(d.voltage_mV.mean(), 1), mean_power_mW=round(p_mw.mean(), 1),
                sd_power_mW=round(p_mw.std(), 1),
                # standard error over *distinct* gauge readings (repeated samples of one reading are not independent)
                se_power_mW=round(p_mw.std() / np.sqrt(max(d.current_raw.nunique(), 1)), 1),
                coulomb_current_mA=round(dq_ma, 1) if q.iloc[0] != q.iloc[-1] else None,
                coulomb_power_mW=round(dq_ma * d.voltage_mV.mean() / 1000, 1) if q.iloc[0] != q.iloc[-1] else None,
                temp_C_start=d.temp_dC.iloc[0] / 10, temp_C_end=d.temp_dC.iloc[-1] / 10)


def main():
    rows, summary = [], {}
    for dev, label in DEVICES.items():
        d = OUT / dev
        if not (d / "samples.csv").exists():
            print("missing", dev)
            continue
        df = pd.read_csv(d / "samples.csv")
        df["t"] = (df.t_ns - df.t_ns.min()) / 1e9
        for c in ["current_raw", "voltage_mV", "charge_counter_uAh", "temp_dC"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        stats = {ph: phase_stats(df, ph, dev) for ph in ["idle_pre", "fp16_4t", "fp16_1t", "idle_post"]}
        stats = {k: v for k, v in stats.items() if v}
        for v in stats.values():
            rows.append(dict(device=label, **v))
        idle = np.mean([stats[p]["mean_power_mW"] for p in ["idle_pre", "idle_post"] if p in stats])
        idle_c = [stats[p]["coulomb_power_mW"] for p in ["idle_pre", "idle_post"] if p in stats]
        res = dict(device=label, idle_power_mW=round(idle, 1),
                   idle_drift_mW=round(stats["idle_post"]["mean_power_mW"] - stats["idle_pre"]["mean_power_mW"], 1)
                   if "idle_post" in stats and "idle_pre" in stats else None,
                   unplugged_throughout=bool(not df.status.astype(str).str.contains("usb_true|Charging|Full").loc[
                       df.phase.isin(["idle_pre", "fp16_4t", "fp16_1t", "idle_post"])].any()),
                   runs={})
        for t in ["4t", "1t"]:
            ph = f"fp16_{t}"
            if ph not in stats or not (d / f"bench_{t}.txt").exists():
                continue
            b = bench(d / f"bench_{t}.txt")
            p_run = stats[ph]["mean_power_mW"]
            short = SHORT_BENCH_MS[dev][t]
            res["runs"][ph] = dict(**b, short_benchmark_median_ms=short,
                                   sustained_vs_short_median=round(b["median_ms"] / short, 2),
                                   run_power_mW=p_run, run_power_se_mW=stats[ph]["se_power_mW"],
                                   added_power_mW=round(p_run - idle, 1),
                                   energy_total_mJ_per_inference=round(p_run * b["t_inf_ms"] / 1000, 1),
                                   energy_incremental_mJ_per_inference=round((p_run - idle) * b["t_inf_ms"] / 1000, 1),
                                   inferences_per_Wh_incremental=round(3.6e6 / ((p_run - idle) * b["t_inf_ms"] / 1000))
                                   if p_run > idle else None)
        summary[dev] = res
    pd.DataFrame(rows).to_csv(OUT / "phone_power_phases.csv", index=False)
    json.dump(summary, open(OUT / "phone_power_summary.json", "w"), indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
