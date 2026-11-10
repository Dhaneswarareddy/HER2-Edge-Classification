"""Cache HER2-IHC-40x DEVELOPMENT tiles at the 224 px network input (1024 -> Lanczos 256 -> bilinear 224, the exact
path of harmonise_lib.prepare at scale 1), for experiments 5 (AdaBN) and 6 (few-shot).

Pools are rebuilt with next_experiment/harmonise2/h2_lib.h1_pools, which asserts that HARMONISE-1's sampling is
reproduced exactly:
  fit  (1,500)  HARMONISE-1 stain-gain fit pool (its labels were never read by HARMONISE-1)
  sel  (2,000)  HARMONISE-1 arm-selection pool
  R    (4,593)  the remaining tiles; HARMONISE-2 split them into R_cal (2,191) and R_eval (2,402) by index blocks
All three come from the 86 development slides; they are tile-disjoint but may share slides (no slide IDs exist).
The external TEST split is never read here (h2_lib replaces the test readers with a function that raises).
Output: experiments_v2/_cache/dev224_<pool>.npz with x (uint8), y, names.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import v2lib as L  # noqa: E402

sys.path.insert(0, str(L.NEXT / "harmonise2"))
import h2_lib as H2  # noqa: E402

CACHE = V2 / "_cache"
CACHE.mkdir(exist_ok=True)


def prep(pil):
    return np.asarray(pil.resize((256, 256), Image.LANCZOS).resize((224, 224), Image.BILINEAR), np.uint8)


def main():
    d, fit, sel, rest = H2.h1_pools()
    r = pd.read_csv(L.H2OUT / "R_index.csv")
    assert sorted(r.name) == sorted(rest), "R_index does not match the rebuilt remaining pool"
    for pool, names in [("fit", fit), ("sel", sel), ("R", list(r.name))]:
        out = CACHE / f"dev224_{pool}.npz"
        if out.exists():
            print("exists", out); continue
        x = np.stack([prep(d.read(n)) for n in names])
        y = np.array([H2.H.label_from_name(n) for n in names])
        np.savez(L.guard(out), x=x, y=y, names=np.array(names))
        print(pool, x.shape, np.bincount(y, minlength=4).tolist(), flush=True)


if __name__ == "__main__":
    main()
