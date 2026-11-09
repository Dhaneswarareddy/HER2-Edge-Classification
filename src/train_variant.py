"""Train one improvement variant (see variants.py).  Identical to train.py except for the variant.

  python train_variant.py --model EfficientNetB0 --loss ce --variant cw [--smoke]

Kept separate from train.py so the running baseline queue never picks up variant code.
Run folder: results/runs/<model>__<loss>__<variant>/ with the same files as a baseline run.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import common as U
import config as C
import train as T
import variants as V


def ensure_cache(source_size, eval_size):
    """Lanczos 1024 -> source_size cache for training, plus a PIL-bilinear eval cache at the network
    size (same two-step resize as the baseline: cache -> bilinear to input size)."""
    man = pd.read_csv(C.MANIFEST)
    src_dir = C.ROOT / "data" / f"ihc_{source_size}"
    ev_dir = C.ROOT / "data" / f"ihc_{source_size}_eval{eval_size}"
    for part in ["train", "test"]:
        (src_dir / part).mkdir(parents=True, exist_ok=True)
        (ev_dir / part).mkdir(parents=True, exist_ok=True)
    for p, s in zip(man.path, man.split):
        if not (src_dir / p).exists():
            Image.open(C.BCI_IHC / p).convert("RGB").resize((source_size, source_size), Image.LANCZOS).save(src_dir / p)
        if s != "train" and not (ev_dir / p).exists():
            Image.open(src_dir / p).resize((eval_size, eval_size), Image.BILINEAR).save(ev_dir / p)
    return src_dir, ev_dir


def fyp2_train_paths():
    """FYP2's GAN training images and labels (its train split only; its val/test are never used)."""
    m = pd.read_csv(C.FYP2_ROOT / "splits" / "split_manifest.csv")
    m = m[m.split == "train"]
    root = C.FYP2_ROOT / "HER2 SCORES DATA" / "Generated_HER2_images"
    return [str(root / p) for p in m.path], m.label.values.astype(np.int32)


def _read_any(path):
    """FYP2's GAN files are JPEG, the BCI cache is PNG: decode either."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.ensure_shape(img, [None, None, 3])


def init_from_fyp2(infer_model, model_name):
    """Copy the FYP2-trained CE network's weights (same code, same layer order; checked by the
    reload test in analysis_models.py) into the freshly built network."""
    d = C.FYP2_ROOT / "results" / "runs" / f"{model_name}__ce"
    if (d / "model.weights.h5").exists():
        infer_model.load_weights(d / "model.weights.h5")
    else:
        infer_model.set_weights(tf.keras.models.load_model(d / "model.keras", compile=False).get_weights())


def eval_ds_from_files(paths, batch=32):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    return ds.map(lambda p: tf.cast(U._read(p), tf.float32), num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(batch).prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--loss", required=True, choices=C.LOSSES)
    ap.add_argument("--variant", required=True, choices=list(V.VARIANTS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="Override C.FT_EPOCHS ceiling (used to budget-match mix_fyp2 vs baseline; "
                         "ignored when --smoke is set)")
    args = ap.parse_args()
    v = V.VARIANTS[args.variant]

    name = f"{args.model}__{args.loss}__{args.variant}"
    run_dir = (C.RESULTS_DIR / "smoke" if args.smoke else C.RUNS_DIR) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    U.set_seed()
    spec = C.MODELS[args.model]
    batch = spec["batch"]

    # resolution variant: point the shared pipeline at the larger cache / input size
    eval_dir = None
    if "image_size" in v:
        C.DATA_DIR, eval_dir = ensure_cache(v["source_size"], v["image_size"])
        C.IMAGE_SIZE = v["image_size"]
        batch = max(batch // 4, 4)            # 4x the pixels per image; keep memory constant
    if "batch_div" in v:                      # explicit batch size: lowbatch control, hires1024
        batch = max(spec["batch"] // v["batch_div"], 2)
    p_tr, y_tr = U.split_paths("train")
    if v.get("mix") == "fyp2":
        assert "image_size" not in v, "FYP2's GAN images are 256 px: mix only at the baseline resolution"
        p_f, y_f = fyp2_train_paths()
        p_tr, y_tr = p_tr + p_f, np.concatenate([y_tr, y_f])
        U._read = _read_any                   # make_train_ds looks _read up in common at trace time
    m = pd.read_csv(C.MANIFEST)
    if eval_dir is None:
        x_va, y_va = U.load_split("val", C.IMAGE_SIZE)
        x_te, y_te = U.load_split("test", C.IMAGE_SIZE)
        val_ds = U.make_eval_ds(x_va, y_va, 64).map(lambda a, b: a)
        test_ds = U.make_eval_ds(x_te, y_te, 64).map(lambda a, b: a)
    else:
        y_va = m[m.split == "val"].label.values.astype(np.int32)
        y_te = m[m.split == "test"].label.values.astype(np.int32)
        val_ds = eval_ds_from_files([str(eval_dir / p) for p in m[m.split == "val"].path])
        test_ds = eval_ds_from_files([str(eval_dir / p) for p in m[m.split == "test"].path])
    augment = V.AUGMENTS[v["augment"]](U._augment) if "augment" in v else U._augment
    train_ds = U.make_train_ds(p_tr, y_tr, batch, augment=augment)
    steps = len(p_tr) // batch

    cw = V.class_weights(y_tr) if v.get("class_weight") else None
    loss_fn = V.make_loss(args.loss, cw, v.get("label_smoothing", 0.0))
    train_model, infer_model, backbone = U.build_model(args.model, args.loss)
    if v.get("init_from") == "fyp2":
        assert args.loss == "ce", "only FYP2's CE networks are used as a starting point"
        init_from_fyp2(infer_model, args.model)
    eval_model = tf.keras.Model(train_model.input, [train_model.output, infer_model.output])
    state = {"history": [], "best": None, "weights": None}
    head_epochs = 1 if args.smoke else C.HEAD_EPOCHS
    ft_epochs = 1 if args.smoke else (args.ft_epochs if args.ft_epochs is not None else C.FT_EPOCHS)
    t_start = time.time()

    print(f"=== {name}  params={train_model.count_params():,}  batch={batch}  input={C.IMAGE_SIZE}  "
          f"class_weights={None if cw is None else np.round(cw, 3).tolist()} ===", flush=True)
    backbone.trainable = False
    train_model.compile(optimizer=T.make_optimizer(C.HEAD_LR), loss=loss_fn)
    train_model.fit(train_ds, epochs=head_epochs, verbose=0,
                    callbacks=[T.ValMonitor(eval_model, val_ds, y_va, loss_fn, 0, state, "head")])
    backbone.trainable = True
    sched = T.WarmupCosine(C.FT_LR, C.FT_WARMUP_EPOCHS * steps, ft_epochs * steps)
    train_model.compile(optimizer=T.make_optimizer(sched), loss=loss_fn)
    train_model.fit(train_ds, epochs=ft_epochs, verbose=0,
                    callbacks=[T.ValMonitor(eval_model, val_ds, y_va, loss_fn, C.PATIENCE, state, "finetune")])
    train_time = time.time() - t_start

    train_model.set_weights(state["weights"])
    _, p_va = eval_model.predict(val_ds, verbose=0)
    _, p_te = eval_model.predict(test_ds, verbose=0)
    m_va, m_te = U.compute_metrics(y_va, p_va), U.compute_metrics(y_te, p_te)
    infer_model.save(run_dir / "model.keras")
    infer_model.save_weights(run_dir / "model.weights.h5")
    np.savez_compressed(run_dir / "predictions.npz", val=p_va, test=p_te, y_val=y_va, y_test=y_te)
    pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
    summary = dict(model=args.model, loss=args.loss, variant=args.variant, variant_spec=v, tier=spec["tier"],
                   params=int(infer_model.count_params()), batch=batch, image_size=C.IMAGE_SIZE,
                   class_weights=None if cw is None else cw.tolist(),
                   best_epoch=state["best"][2], epochs_run=len(state["history"]), train_time_s=train_time,
                   val=m_va, test=m_te)
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)
    print(f"DONE {name}: best epoch {state['best'][2]}  val qwk {m_va['qwk']:.4f} acc {m_va['accuracy']:.4f} "
          f"f1 {m_va['macro_f1']:.4f} | test qwk {m_te['qwk']:.4f} acc {m_te['accuracy']:.4f}  "
          f"({train_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
