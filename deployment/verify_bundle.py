#!/usr/bin/env python3
"""Run first on any new machine: checks hashes, mode separation, the runtime, and one prediction per mode."""
import json, platform, sys
from pathlib import Path
import numpy as np
from her2_infer import Her2Engine, MODES, load_config
from her2_infer.engine import BASE, interpreter_class, sha256
ok = True
print("python", platform.python_version(), "| machine", platform.machine(), "| runtime", interpreter_class()[1])
cfg = {m: load_config(m) for m in MODES}
if cfg["primary"]["model_sha256"] == cfg["experimental"]["model_sha256"]:
    ok = False; print("FAIL: both modes point at the same model")
for m in MODES:
    r = np.load(BASE / "reference" / f"{m}.npz"); e = Her2Engine(m, 4); p = e.invoke(e.from_resized(r["tiles_preprocessed"][0])); d = float(np.abs(p - r["tflite_mac"][0]).max())
    good = d < 1e-3; ok &= good; print(f"{m}: hash ok, input {e.inp['shape'].tolist()}, first-tile max |dp| vs Mac reference {d:.2e} {'OK' if good else 'FAIL'}")
print("BUNDLE OK" if ok else "BUNDLE FAILED"); sys.exit(0 if ok else 1)
