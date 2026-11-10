"""Screenshots of the corrected Android app (v1.1) on a connected phone. Controls are located through the live UI
hierarchy (uiautomator dump, by resource-id), not hard-coded coordinates, so the script survives layout changes.

  python capture_screenshots_v11.py --serial <adb serial> --out <folder>

Sequence (same names as the original screenshots so the report figures can be rebuilt):
  01 overview (FP16, 4 threads); 02-05 reference tiles 0/1+/2+/3+ (FP16, 4 threads); 06 INT8 on tile 3+ (shows the
  rejection warning); 07 FP16, 1 thread on tile 3+.
The device's stay-awake setting is not changed; the screen is woken with a key event.
"""
import argparse
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
PKG = "com.fyp3.her2classifier"


class Phone:
    def __init__(self, serial):
        self.s = serial

    def sh(self, *a):
        return subprocess.run([ADB, "-s", self.s, *a], capture_output=True, text=True).stdout

    def size(self):
        w, h = map(int, re.search(r"(\d+)x(\d+)", self.sh("shell", "wm", "size")).groups())
        return w, h

    def dump(self):
        for _ in range(8):  # uiautomator fails transiently while the UI is animating
            self.sh("shell", "rm", "-f", "/sdcard/v11_ui.xml")
            self.sh("shell", "uiautomator", "dump", "/sdcard/v11_ui.xml")
            xml = self.sh("shell", "cat", "/sdcard/v11_ui.xml")
            if "<hierarchy" in xml:
                return ET.fromstring(xml[xml.index("<?xml"):] if "<?xml" in xml else xml)
            time.sleep(1.5)
        raise RuntimeError("uiautomator dump failed")

    def find(self, rid):
        for n in self.dump().iter("node"):
            if n.get("resource-id") == f"{PKG}:id/{rid}":
                x1, y1, x2, y2 = map(int, re.findall(r"\d+", n.get("bounds")))
                return (x1 + x2) // 2, (y1 + y2) // 2, y1, y2
        return None

    def dismiss_dialogs(self):
        """Close the Android 'app compatibility' dialog (debuggable app, 16 KB page alignment) if shown."""
        for _ in range(3):
            hit = None
            for n in self.dump().iter("node"):
                if n.get("text") in ("OK", "Ok") and n.get("clickable") == "true":
                    hit = n
            if hit is None:
                return
            x1, y1, x2, y2 = map(int, re.findall(r"\d+", hit.get("bounds")))
            self.sh("shell", "input", "tap", str((x1 + x2) // 2), str((y1 + y2) // 2))
            time.sleep(1.5)

    def swipe(self, dy):
        w, h = self.size()
        self.sh("shell", "input", "swipe", str(w // 2), str(h // 2 + dy // 2), str(w // 2), str(h // 2 - dy // 2), "400")
        time.sleep(1.0)

    def tap_id(self, rid, tries=6):
        w, h = self.size()
        for _ in range(tries):
            f = self.find(rid)
            if f and 0.12 * h < f[1] < 0.9 * h:
                self.sh("shell", "input", "tap", str(f[0]), str(f[1]))
                time.sleep(1.5)
                return True
            # scroll towards the element (down if below / not found, up if above)
            self.swipe(int(0.35 * h) if (f is None or f[1] >= 0.9 * h) else -int(0.35 * h))
        raise RuntimeError(f"control {rid} not found")

    def show(self, rid, frac=0.55):
        """Scroll so that element rid sits at about `frac` of the screen height."""
        w, h = self.size()
        for _ in range(6):
            f = self.find(rid)
            if f is None:
                self.swipe(int(0.3 * h)); continue
            dy = f[1] - int(frac * h)
            if abs(dy) < 0.08 * h:
                return
            self.swipe(max(min(dy, int(0.6 * h)), -int(0.6 * h)))

    def top(self):
        for _ in range(4):
            self.swipe(-int(0.6 * self.size()[1]))

    def shot(self, path):
        time.sleep(1.2)
        self.sh("shell", "screencap", "-p", "/sdcard/v11_shot.png")
        self.sh("pull", "/sdcard/v11_shot.png", str(path))
        print("saved", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = Phone(a.serial)
    p.sh("shell", "input", "keyevent", "224")          # wake
    p.sh("shell", "wm", "dismiss-keyguard")
    p.sh("shell", "am", "force-stop", PKG)
    p.sh("shell", "am", "start", "-n", f"{PKG}/.MainActivity")
    time.sleep(4)
    p.dismiss_dialogs()
    p.top()
    p.tap_id("rbFp16"); p.tap_id("rb4Threads")
    p.top(); p.shot(out / "01_app_overview_hardware_and_controls.png")
    names = ["02_app_inference_score_0_negative_fp16.png", "03_app_inference_score_1_negative_fp16.png",
             "04_app_inference_score_2_equivocal_fp16.png", "05_app_inference_score_3_positive_fp16.png"]
    for k, n in enumerate(names):
        p.tap_id(f"btnSample{k}")
        time.sleep(2.5)
        p.show("tvPredictionScore", 0.30)
        p.shot(out / n)
    p.top(); p.tap_id("rbInt8"); time.sleep(4)                       # INT8 re-runs tile 3+ and shows the warning
    p.show("tvPredictionScore", 0.30); p.shot(out / "06_app_inference_score_3_int8_quantized.png")
    p.top(); p.tap_id("rbFp16"); p.tap_id("rb1Thread"); time.sleep(3)
    p.show("tvPredictionScore", 0.30); p.shot(out / "07_app_inference_1thread_baseline.png")
    p.top(); p.tap_id("rb4Threads")
    p.sh("shell", "rm", "-f", "/sdcard/v11_ui.xml", "/sdcard/v11_shot.png")


if __name__ == "__main__":
    main()
