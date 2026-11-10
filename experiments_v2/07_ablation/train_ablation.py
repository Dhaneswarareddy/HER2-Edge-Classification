"""Experiment 8 (v2): backbone-freezing and head ablation for EfficientNetV2B0 (224 px, CE) on the BCI grouped split.

Existing (reused, not retrained): full fine-tuning baseline results/runs/EfficientNetV2B0__ce; resolution study
(448 vs 224 at batch 8 and 32: __ce__hires, __ce__lowbatch); frozen ImageNet EfficientNetB0 linear probe
(results/analysis/frozen_probe.json, logistic regression, no augmentation).

New variants -- identical to src/train.py (same data pipeline, augmentation, seed 42, batch 32, head stage 5 epochs at
1e-3, fine-tune stage <= 40 epochs, warm-up 2 epochs + cosine from 1e-4, early stopping on val QWK with patience 10,
best weights restored, BatchNorm in inference mode) except for what is listed:
  frozen_linear   backbone frozen for the whole run; the linear head trains for <= 45 epochs at 1e-3 with the same
                  early stopping (the baseline's 5 head epochs, continued to convergence)
  frozen_mlp      as frozen_linear, head = Dense(256, ReLU) -> Dropout(0.3) -> Dense(4)
  partial         as the baseline, but in stage 2 only 'block6*' and 'top_*' backbone layers are unfrozen
The training code is imported from src/ unchanged; outputs go to experiments_v2/07_ablation/runs/<variant>/.
Decisions use validation only; test is reported. One seed per arm (as every FYP3 run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["frozen_linear", "frozen_mlp", "partial"])
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
tf = L.tf_setup()
import common as U  # noqa: E402
import config as C  # noqa: E402
import train as T  # noqa: E402

RUNDIR = L.V2 / "07_ablation" / ("runs_smoke" if args.smoke else "runs") / args.variant


def build(variant):
    train_model, infer, bb = U.build_model("EfficientNetV2B0", "ce")
    if variant != "frozen_mlp":
        return train_model, infer, bb
    inp = train_model.input
    names = [l.name for l in train_model.layers]
    x = train_model.get_layer("preprocess")(inp) if "preprocess" in names else inp  # same input path as build_model
    x = bb(x, training=False)
    x = tf.keras.layers.Dense(256, activation="relu", name="mlp_hidden")(x)
    x = tf.keras.layers.Dropout(C.DROPOUT, name="dropout")(x)
    logits = tf.keras.layers.Dense(4, name="logits", dtype="float32")(x)
    probs = tf.keras.layers.Softmax(name="probs")(logits)
    return tf.keras.Model(inp, logits), tf.keras.Model(inp, probs), bb


def main():
    RUNDIR.mkdir(parents=True, exist_ok=True)
    L.guard(RUNDIR)
    U.set_seed()
    batch = C.MODELS["EfficientNetV2B0"]["batch"]
    p_tr, y_tr = U.split_paths("train")
    x_va, y_va = U.load_split("val", C.IMAGE_SIZE)
    x_te, y_te = U.load_split("test", C.IMAGE_SIZE)
    if args.smoke:
        p_tr, y_tr = p_tr[:256], y_tr[:256]
    train_ds = U.make_train_ds(p_tr, y_tr, batch)
    val_ds = U.make_eval_ds(x_va, y_va, 64).map(lambda a, b: a)
    test_ds = U.make_eval_ds(x_te, y_te, 64).map(lambda a, b: a)
    steps = len(p_tr) // batch
    train_model, infer, bb = build(args.variant)
    eval_model = tf.keras.Model(train_model.input, [train_model.output, infer.output])
    state = {"history": [], "best": None, "weights": None}
    t0 = time.time()
    frozen = args.variant.startswith("frozen")
    head_epochs = 1 if args.smoke else (C.HEAD_EPOCHS + C.FT_EPOCHS if frozen else C.HEAD_EPOCHS)
    print(f"=== EfficientNetV2B0__ce__{args.variant}  params={train_model.count_params():,} batch={batch} ===", flush=True)
    bb.trainable = False
    train_model.compile(optimizer=T.make_optimizer(C.HEAD_LR), loss=U.ce_loss)
    trainable_head = int(sum(np.prod(v.shape) for v in train_model.trainable_weights))
    train_model.fit(train_ds, epochs=head_epochs, verbose=0,
                    callbacks=[T.ValMonitor(eval_model, val_ds, y_va, U.ce_loss, C.PATIENCE if frozen else 0, state, "head")])
    trainable_ft = trainable_head
    if not frozen:
        bb.trainable = True
        for l in bb.layers:
            l.trainable = l.name.startswith(("block6", "top"))
        ft_epochs = 1 if args.smoke else C.FT_EPOCHS
        sched = T.WarmupCosine(C.FT_LR, C.FT_WARMUP_EPOCHS * steps, ft_epochs * steps)
        train_model.compile(optimizer=T.make_optimizer(sched), loss=U.ce_loss)
        trainable_ft = int(sum(np.prod(v.shape) for v in train_model.trainable_weights))
        train_model.fit(train_ds, epochs=ft_epochs, verbose=0,
                        callbacks=[T.ValMonitor(eval_model, val_ds, y_va, U.ce_loss, C.PATIENCE, state, "finetune")])
    train_time = time.time() - t0
    train_model.set_weights(state["weights"])
    _, p_va = eval_model.predict(val_ds, verbose=0)
    _, p_te = eval_model.predict(test_ds, verbose=0)
    m_va, m_te = U.compute_metrics(y_va, p_va), U.compute_metrics(y_te, p_te)
    infer.save(RUNDIR / "model.keras")
    infer.save_weights(RUNDIR / "model.weights.h5")
    np.savez_compressed(RUNDIR / "predictions.npz", val=p_va, test=p_te, y_val=y_va, y_test=y_te)
    pd.DataFrame(state["history"]).to_csv(RUNDIR / "history.csv", index=False)
    summary = dict(model="EfficientNetV2B0", loss="ce", variant=args.variant, params=int(infer.count_params()),
                   trainable_params_stage1=trainable_head, trainable_params_stage2=trainable_ft, batch=batch,
                   image_size=C.IMAGE_SIZE, seed=C.SEED, best_epoch=state["best"][2], epochs_run=len(state["history"]),
                   train_time_s=train_time, val=m_va, test=m_te)
    json.dump(summary, open(RUNDIR / "summary.json", "w"), indent=2)
    print(f"DONE {args.variant}: best epoch {state['best'][2]} / {len(state['history'])}  val qwk {m_va['qwk']:.4f} "
          f"acc {m_va['accuracy']:.4f} | test qwk {m_te['qwk']:.4f} acc {m_te['accuracy']:.4f} ({train_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
