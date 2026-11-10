"""E6 / E7 from NEXT_EXPERIMENTS.md, run in isolation under experiments_v2/09_resolution_e6_e7/.

  E6  DenseNet121, CE: `hires` (1024 -> 512 cache -> 448 px input, batch 4) and `lowbatch` (224 px, batch 4).
      Resolution effect = hires - lowbatch (paired fragment-cluster bootstrap on validation, pre-registered rule).
  E7  MobileNetV3Small, CE: `hires1024` (native 1024 px -> 896 px input, batch 2), compared with __ce__hires.

The training code is src/train_variant.py, imported unchanged. Only two things are redirected:
  * run folders  -> experiments_v2/09_resolution_e6_e7/runs/<model>__ce__<variant>/   (smoke: smoke/)
  * the new 1024 px cache (E7 only) -> experiments_v2/_cache/ihc_1024{,_eval896}/ (git-ignored).
The existing 512/448 cache under data/ is read only (it is complete; nothing is written there).
Model selection and early stopping use validation only; test metrics are written once, as for every FYP3 run.

  python train_e6e7.py --model DenseNet121 --variant hires [--smoke]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
import v2lib as L  # noqa: E402  (puts src/ on sys.path)

L.tf_setup()
import config as C  # noqa: E402
import train_variant as TV  # noqa: E402

C.RESULTS_DIR = L.guard(HERE)                       # train_variant writes smoke runs to RESULTS_DIR/"smoke"
C.RUNS_DIR = L.guard(HERE / "runs")

_orig = TV.ensure_cache


def ensure_cache(source_size, eval_size):
    if source_size == 512:
        d, e = C.ROOT / "data" / "ihc_512", C.ROOT / "data" / f"ihc_512_eval{eval_size}"
        assert d.exists() and e.exists(), "expected the existing 512/448 cache"
        return d, e
    root = L.guard(V2 / "_cache")
    import pandas as pd
    from PIL import Image
    man = pd.read_csv(C.MANIFEST)
    src_dir, ev_dir = root / f"ihc_{source_size}", root / f"ihc_{source_size}_eval{eval_size}"
    for p, s in zip(man.path, man.split):
        a, b = src_dir / p, ev_dir / p
        a.parent.mkdir(parents=True, exist_ok=True)
        if not a.exists():
            Image.open(C.BCI_IHC / p).convert("RGB").resize((source_size, source_size), Image.LANCZOS).save(a)
        if s != "train" and not b.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            Image.open(a).resize((eval_size, eval_size), Image.BILINEAR).save(b)
    return src_dir, ev_dir


TV.ensure_cache = ensure_cache
sys.argv = [sys.argv[0], "--loss", "ce"] + sys.argv[1:]
TV.main()
