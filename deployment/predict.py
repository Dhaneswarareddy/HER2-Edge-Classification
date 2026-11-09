#!/usr/bin/env python3
"""Score IHC tiles. Research use only: not clinically validated.

  python3 predict.py --mode primary --input tile_or_folder --out primary.csv
  python3 predict.py --mode experimental --i-understand-experimental --input folder --out experimental.csv

Primary: EfficientNetV2B0 448 px FP16, for BCI-like tiles, no stain calibration.
Experimental: EfficientNetV2B0 224 px FP16 with the frozen stain_p99 gains, fitted for HER2-IHC-40x only.
Output is per tile. There is no slide-level score.
"""
import argparse, csv, sys
from pathlib import Path
from her2_infer import Her2Engine, load_config

ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["primary", "experimental"]); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--i-understand-experimental", action="store_true"); a = ap.parse_args()
if a.mode == "experimental" and not a.i_understand_experimental:
    sys.exit("Experimental mode is fitted to one external dataset and is weak on 1+ and 2+. Pass --i-understand-experimental to run it.")
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists. Use a new file: results from the two modes must not be mixed.")
src = Path(a.input); files = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
if not files:
    sys.exit("no images found")
for w in load_config(a.mode)["warnings"]:
    print("WARNING:", w, file=sys.stderr)
e = Her2Engine(a.mode, a.threads); rows = []
for f in files:
    rows.append({"file": str(f), **e.predict(f)})
with open(out, "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} tiles scored in {a.mode} mode -> {out}")
