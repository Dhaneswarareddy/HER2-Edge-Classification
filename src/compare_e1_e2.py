"""Print the E1/E2 resolution vs batch comparison table once all four runs exist.

Usage (from FYP3/src/):
  python compare_e1_e2.py

Reads existing summary.json and predictions.npz. Runs paired.py comparisons.
Never uses test metrics for any decision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import paired as P

RUNS = {
    "EfficientNetV2B0": {
        "baseline":  "EfficientNetV2B0__ce",
        "lowbatch":  "EfficientNetV2B0__ce__lowbatch",
        "hires":     "EfficientNetV2B0__ce__hires",
    },
    "MobileNetV3Small": {
        "baseline":  "MobileNetV3Small__ce",
        "lowbatch":  "MobileNetV3Small__ce__lowbatch",
        "hires":     "MobileNetV3Small__ce__hires",
    },
}

METRICS_VAL  = ["qwk", "accuracy", "macro_f1"]
METRICS_RC   = ["recall_0", "recall_1", "recall_2", "recall_3"]


def load(run):
    p = C.RUNS_DIR / run / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v*100:.1f}" if pct else f"{v:.4f}"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_run_table(model_name, variants):
    section(f"{model_name} — absolute metrics")
    header = f"{'Variant':<12} {'Res':>4} {'Bat':>4} {'Params':>8}  " \
             f"{'Val QWK':>8} {'Val Acc':>8} {'Val F1':>8}  " \
             f"{'Tst QWK':>8} {'Tst Acc':>8} {'Tst F1':>8}  " \
             f"{'Train(min)':>10}"
    print(header)
    print("-" * len(header))
    for vname, run in variants.items():
        s = load(run)
        if s is None:
            print(f"  {vname:<12}  (not yet available)")
            continue
        v = s.get("val", {}); t = s.get("test", {})
        res  = s.get("image_size", 224)
        bat  = s.get("batch", 32 if "hires" not in run else 8)
        mins = s.get("train_time_s", 0) / 60
        print(f"  {vname:<12} {res:>4} {bat:>4} {s['params']:>8,}  "
              f"{fmt(v.get('qwk')):>8} {fmt(v.get('accuracy')):>8} {fmt(v.get('macro_f1')):>8}  "
              f"{fmt(t.get('qwk')):>8} {fmt(t.get('accuracy')):>8} {fmt(t.get('macro_f1')):>8}  "
              f"{mins:>10.1f}")


def print_recall_table(model_name, variants):
    section(f"{model_name} — per-class recall (val)")
    header = f"{'Variant':<12}  {'Score 0':>8} {'Score 1+':>9} {'Score 2+':>9} {'Score 3+':>9}"
    print(header)
    print("-" * len(header))
    for vname, run in variants.items():
        s = load(run)
        if s is None:
            print(f"  {vname:<12}  (not yet available)")
            continue
        v = s.get("val", {})
        r = [v.get(f"recall_{i}", None) for i in range(4)]
        print(f"  {vname:<12}  {fmt(r[0]):>8} {fmt(r[1]):>9} {fmt(r[2]):>9} {fmt(r[3]):>9}")


def print_paired(model_name, variants):
    bl = variants.get("baseline"); lb = variants.get("lowbatch"); hi = variants.get("hires")
    if not all([load(r) for r in [bl, lb, hi] if r]):
        print(f"\n  [{model_name}] Paired comparisons skipped — some runs not yet complete.")
        return
    section(f"{model_name} — paired val QWK differences (fragment-cluster bootstrap 95% CI)")
    print(f"  Effect            Comparison                              QWK diff [95% CI]")
    print(f"  {'─'*65}")
    for label, a, b in [
        ("Resolution",  hi, lb),   # hires − lowbatch  (same batch, different res)
        ("Batch size",  lb, bl),   # lowbatch − baseline (same res, different batch)
        ("Combined",    hi, bl),   # hires − baseline
    ]:
        try:
            r = P.compare(a, b, split="val")
            qwk = r["qwk"]
            sig = " ← CI excludes 0 ✓" if qwk["ci_lo"] > 0 or qwk["ci_hi"] < 0 else ""
            print(f"  {label:<18}  {a} − {b}")
            print(f"  {'':18}  Δ QWK = {qwk['diff']:+.4f}  [{qwk['ci_lo']:+.4f}, {qwk['ci_hi']:+.4f}]{sig}")
            for m in ["accuracy", "macro_f1"]:
                d = r[m]
                print(f"  {'':18}  Δ {m:<10} = {d['diff']:+.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
            print()
        except Exception as e:
            print(f"  {label}: ERROR — {e}")


def main():
    all_ready = True
    for model, variants in RUNS.items():
        for vname, run in variants.items():
            if load(run) is None:
                print(f"  MISSING: {run}")
                all_ready = False

    if not all_ready:
        print("\nNot all runs complete — printing available results only.\n")

    for model, variants in RUNS.items():
        print_run_table(model, variants)
        print_recall_table(model, variants)
        print_paired(model, variants)

    section("Resolution effect confirmed?")
    for model, variants in RUNS.items():
        hi = load(variants["hires"]); lb = load(variants.get("lowbatch"))
        bl = load(variants["baseline"])
        if hi and lb:
            res_effect = hi["val"]["qwk"] - lb["val"]["qwk"]
            bat_effect = lb["val"]["qwk"] - bl["val"]["qwk"] if bl else None
            print(f"  {model:<22}  hires−lowbatch Δ QWK = {res_effect:+.4f}  "
                  f"{'(batch effect: ' + f'{bat_effect:+.4f})' if bat_effect is not None else ''}")
        else:
            print(f"  {model:<22}  lowbatch not yet available")

    print()


if __name__ == "__main__":
    main()

