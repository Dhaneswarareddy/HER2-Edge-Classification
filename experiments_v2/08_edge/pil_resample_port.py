"""Pure-Python port of Pillow's separable resampler (libImaging/Resample.c, 8-bit fixed-point path), used as the
reference for the Android port in android_app/.../PilResample.kt. Verified here to be bit-identical to PIL on BCI
tiles for the deployed path 1024 -> LANCZOS 512 -> BILINEAR 448.

  python pil_resample_port.py      -> prints max |difference| vs PIL for 20 BCI validation tiles (must be 0)
"""
import math
import sys
from pathlib import Path

import numpy as np

PRECISION_BITS = 32 - 8 - 2


def _bilinear(x):
    x = -x if x < 0 else x
    return 1.0 - x if x < 1.0 else 0.0


def _sinc(x):
    if x == 0.0:
        return 1.0
    x *= math.pi
    return math.sin(x) / x


def _lanczos(x):
    return _sinc(x) * _sinc(x / 3.0) if -3.0 <= x < 3.0 else 0.0


FILTERS = {"bilinear": (_bilinear, 1.0), "lanczos": (_lanczos, 3.0)}


def coeffs(in_size, out_size, name):
    f, sup = FILTERS[name]
    scale = in_size / out_size
    fscale = max(scale, 1.0)
    support = sup * fscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds, kk = [], np.zeros((out_size, ksize), np.int64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / fscale
        xmin = int(center - support + 0.5)
        xmin = max(xmin, 0)
        xmax = int(center + support + 0.5)
        xmax = min(xmax, in_size) - xmin
        k = [f((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        ww = sum(k)
        if ww != 0.0:
            k = [v / ww for v in k]
        for x, v in enumerate(k):  # normalize_coeffs_8bpc: round half away from zero
            kk[xx, x] = int(-0.5 + v * (1 << PRECISION_BITS)) if v < 0 else int(0.5 + v * (1 << PRECISION_BITS))
        bounds.append((xmin, xmax))
    return bounds, kk


def _clip8(v):
    out = v >> PRECISION_BITS
    return np.clip(np.where(v <= 0, 0, out), 0, 255)


def resample_axis(img, out_size, name, axis):
    """img uint8 (H, W, C); resample along axis 1 (horizontal) or 0 (vertical)."""
    in_size = img.shape[axis]
    bounds, kk = coeffs(in_size, out_size, name)
    a = img.astype(np.int64)
    if axis == 0:
        a = a.transpose(1, 0, 2)
    out = np.empty((a.shape[0], out_size, a.shape[2]), np.int64)
    for xx, (xmin, xmax) in enumerate(bounds):
        ss = (1 << (PRECISION_BITS - 1)) + np.tensordot(a[:, xmin:xmin + xmax, :], kk[xx, :xmax], axes=([1], [0]))
        out[:, xx, :] = _clip8(ss)
    out = out.astype(np.uint8)
    return out.transpose(1, 0, 2) if axis == 0 else out


def resize(img, size, name):
    """Square resize as PIL Image.resize((size, size), filter): horizontal pass, then vertical."""
    if img.shape[1] != size:
        img = resample_axis(img, size, name, 1)
    if img.shape[0] != size:
        img = resample_axis(img, size, name, 0)
    return img


def main():
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import v2lib as L
    sys.path.insert(0, str(L.SRC))
    import config as C
    m = L.manifest(); m = m[m.split == "val"].reset_index(drop=True)
    worst = 0
    for p in m.path[:20]:
        im = Image.open(C.BCI_IHC / p).convert("RGB")
        ref = np.asarray(im.resize((512, 512), Image.LANCZOS).resize((448, 448), Image.BILINEAR))
        mine = resize(resize(np.asarray(im), 512, "lanczos"), 448, "bilinear")
        d = int(np.abs(ref.astype(int) - mine.astype(int)).max()); worst = max(worst, d)
        print(p, "max |diff| =", d, flush=True)
    print("WORST", worst)


if __name__ == "__main__":
    main()
