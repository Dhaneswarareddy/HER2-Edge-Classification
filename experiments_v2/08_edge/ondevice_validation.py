"""On-device accuracy of the Android app (v1.1), the first measurement of what the phone actually predicts.

The app's validation mode (MainActivity.runValidation) classifies every PNG pushed to its external files folder
through exactly the UI path: decode -> PilResample (LANCZOS 512 -> BILINEAR 448, bit-exact with PIL) -> TFLite FP16,
4 threads, XNNPACK. Tiles: the full BCI grouped TEST split (758 original 1024 px tiles, report-only) plus the 120
BCI-validation reference tiles. Nothing is tuned on these results.

Compared with: labels; the stored Keras predictions of EfficientNetV2B0__ce__hires (test) and the stored Keras
reference outputs (deployment/reference/primary.npz) for the 120 tiles.

  python ondevice_validation.py --serial <adb serial> --device-name samsung_m35 --tiles <staging dir> [--apk path]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
PKG = "com.fyp3.her2classifier"
DEV_DIR = f"/sdcard/Android/data/{PKG}/files"
LEVELS = ["0", "1+", "2+", "3+"]


def adb(serial, *a, check=True):
    return subprocess.run([ADB, "-s", serial, *a], capture_output=True, text=True, check=check).stdout


def metrics(y, p):
    from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
    return dict(n=len(y), accuracy=float((y == p).mean()),
                balanced_accuracy=float(recall_score(y, p, average="macro", labels=range(4), zero_division=0)),
                macro_f1=float(f1_score(y, p, average="macro", labels=range(4), zero_division=0)),
                qwk=float(cohen_kappa_score(y, p, weights="quadratic", labels=range(4))),
                **{f"recall_{c}": float(((p == k) & (y == k)).sum() / max((y == k).sum(), 1)) for k, c in enumerate(LEVELS)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--apk", default=str(ROOT / "android_app/app/build/outputs/apk/debug/app-debug.apk"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = HERE / "outputs" / f"ondevice_{a.device_name}"
    out.mkdir(parents=True, exist_ok=True)
    s = a.serial
    print(adb(s, "install", "-r", a.apk).strip()[-60:], flush=True)
    ver = adb(s, "shell", f"dumpsys package {PKG} | grep versionName").strip()
    adb(s, "shell", f"rm -rf {DEV_DIR}/validation {DEV_DIR}/validation_fp16_t{a.threads}.*", check=False)
    adb(s, "shell", f"mkdir -p {DEV_DIR}/validation")
    t0 = time.time()
    print(adb(s, "push", "--sync", str(Path(a.tiles)) + "/.", f"{DEV_DIR}/validation/").strip()[-120:], flush=True)
    n_dev = int(adb(s, "shell", f"ls {DEV_DIR}/validation | wc -l").strip())
    print("pushed", n_dev, "files in", round(time.time() - t0), "s", flush=True)
    adb(s, "shell", f"am force-stop {PKG}")
    adb(s, "shell", f"am start -n {PKG}/.MainActivity --ez validate true --es model fp16 --ei threads {a.threads}")
    tag = f"validation_fp16_t{a.threads}"
    t0 = time.time()
    while adb(s, "shell", f"test -f {DEV_DIR}/{tag}.done && echo yes", check=False).strip() != "yes":
        time.sleep(20)
        if time.time() - t0 > 4 * 3600:
            raise SystemExit("timeout")
    run_s = time.time() - t0
    adb(s, "pull", f"{DEV_DIR}/{tag}.csv", str(out / f"{tag}.csv"))
    d = pd.read_csv(out / f"{tag}.csv")
    idx = pd.read_csv(HERE / "ondevice_tiles_index.csv").set_index("file")
    d = d.join(idx, on="file")
    rows = {}
    # test split vs labels and stored Keras predictions
    z = np.load(ROOT / "results/runs/EfficientNetV2B0__ce__hires/predictions.npz")
    import sys
    sys.path.insert(0, str(HERE.parent))
    import v2lib as L
    man = L.manifest(); man = man[man.split == "test"].reset_index(drop=True)
    keras_test = pd.DataFrame(z["test"], index=man.path)
    te = d[d.set == "test"].set_index("path").loc[man.path]
    P = te[["p0", "p1", "p2", "p3"]].values
    y = man.label.values.astype(int)
    rows["test"] = dict(**metrics(y, P.argmax(1)), agreement_with_keras=float((P.argmax(1) == keras_test.values.argmax(1)).mean()),
                        max_abs_dp_vs_keras=float(np.abs(P - keras_test.values).max()))
    ref = np.load(ROOT / "deployment/reference/primary.npz", allow_pickle=True)
    vr = d[d.set == "valref"].set_index("path").loc[list(ref["paths"])]
    Q = vr[["p0", "p1", "p2", "p3"]].values
    rows["valref120"] = dict(**metrics(ref["labels"].astype(int), Q.argmax(1)),
                             agreement_with_keras=float((Q.argmax(1) == ref["keras"].argmax(1)).mean()),
                             max_abs_dp_vs_keras=float(np.abs(Q - ref["keras"]).max()))
    timing = {k: dict(median=float(d[k].median()), p95=float(d[k].quantile(0.95))) for k in ["preprocess_ms", "inference_ms", "total_ms"]}
    info = dict(device=a.device_name, serial=s, app_version=ver, threads=a.threads, model="FP16 448 px", n_files=int(len(d)),
                wall_clock_s=round(run_s), results=rows, single_tile_timing_ms=timing,
                note="timings are per tile inside the app (one pass each, not a controlled benchmark)")
    json.dump(info, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    adb(s, "shell", f"rm -rf {DEV_DIR}/validation", check=False)  # remove the pushed tiles from the phone


if __name__ == "__main__":
    main()
