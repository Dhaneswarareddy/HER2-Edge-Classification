"""Step 2: train one (model, loss) combination on the fixed split.

  python train.py --model ResNet50 --loss ce
  python train.py --model MobileNetV3Small --loss corn --smoke   # 1-epoch pipeline check

Stage 1: frozen backbone, train the head (HEAD_EPOCHS, lr HEAD_LR).
Stage 2: unfreeze everything (BatchNorm kept in inference mode), Adam with linear warm-up +
         cosine decay, early stopping on validation QWK (ties broken by val loss), best
         weights restored.  The test split is only touched once, after training.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import common as U
import config as C


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr, self.warmup_steps, self.total_steps = base_lr, warmup_steps, total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step + 1) / self.warmup_steps
        prog = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cos = 0.5 * self.base_lr * (1 + tf.cos(math.pi * tf.clip_by_value(prog, 0.0, 1.0)))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self):
        return dict(base_lr=self.base_lr, warmup_steps=self.warmup_steps, total_steps=self.total_steps)


class ValMonitor(tf.keras.callbacks.Callback):
    """Computes val loss/QWK/accuracy each epoch, keeps the best weights, early-stops."""

    def __init__(self, eval_model, val_ds, y_val, loss_fn, patience, state, stage):
        super().__init__()
        self.eval_model, self.val_ds, self.y_val = eval_model, val_ds, y_val
        self.loss_fn, self.patience, self.state, self.stage = loss_fn, patience, state, stage
        self.wait = 0
        self.t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logits, probs = self.eval_model.predict(self.val_ds, verbose=0)
        vloss = float(self.loss_fn(self.y_val, tf.constant(logits)))
        m = U.compute_metrics(self.y_val, probs)
        lr = self.model.optimizer.lr
        lr = float(lr(self.model.optimizer.iterations)) if callable(lr) else float(tf.keras.backend.get_value(lr))
        row = dict(stage=self.stage, epoch=len(self.state["history"]) + 1, train_loss=logs["loss"],
                   val_loss=vloss, val_acc=m["accuracy"], val_qwk=m["qwk"], val_f1=m["macro_f1"],
                   lr=lr, sec=time.time() - self.t0)
        self.state["history"].append(row)
        best = self.state["best"]
        improved = best is None or m["qwk"] > best[0] + 1e-4 or (abs(m["qwk"] - best[0]) <= 1e-4 and vloss < best[1])
        if improved:
            self.state["best"] = (m["qwk"], vloss, row["epoch"])
            self.state["weights"] = [w.copy() for w in self.model.get_weights()]
            self.wait = 0
        else:
            self.wait += 1
        print(f"  [{self.stage}] ep {row['epoch']:3d} loss {row['train_loss']:.4f} | val loss {vloss:.4f} "
              f"acc {m['accuracy']:.4f} qwk {m['qwk']:.4f} f1 {m['macro_f1']:.4f} | lr {lr:.2e} "
              f"{row['sec']:.0f}s{' *' if improved else ''}", flush=True)
        if self.patience and self.wait >= self.patience:
            print(f"  early stop (no val QWK improvement for {self.patience} epochs)")
            self.model.stop_training = True


def make_optimizer(lr):
    # The TF>=2.11 optimizers run slowly on Apple Silicon; TF recommends the legacy ones there.
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--loss", required=True, choices=C.LOSSES)
    ap.add_argument("--smoke", action="store_true", help="1+1 epochs, for pipeline testing")
    ap.add_argument("--variant", default=None, choices=list(C.VARIANTS),
                    help="improvement experiment (see config.VARIANTS); default = FYP2 baseline")
    args = ap.parse_args()
    variant = C.VARIANTS.get(args.variant, {})

    run_dir = C.RUNS_DIR / f"{args.model}__{args.loss}{'__' + args.variant if args.variant else ''}"
    if args.smoke:
        run_dir = C.RESULTS_DIR / "smoke" / f"{args.model}__{args.loss}"
    run_dir.mkdir(parents=True, exist_ok=True)
    U.set_seed()
    spec = C.MODELS[args.model]
    batch = spec["batch"]

    p_tr, y_tr = U.split_paths("train")
    x_va, y_va = U.load_split("val", C.IMAGE_SIZE)
    x_te, y_te = U.load_split("test", C.IMAGE_SIZE)
    train_ds = U.make_train_ds(p_tr, y_tr, batch)
    val_ds = U.make_eval_ds(x_va, y_va, 64).map(lambda a, b: a)
    test_ds = U.make_eval_ds(x_te, y_te, 64).map(lambda a, b: a)
    steps = len(p_tr) // batch

    train_model, infer_model, backbone = U.build_model(args.model, args.loss)
    eval_model = tf.keras.Model(train_model.input, [train_model.output, infer_model.output])
    loss_fn = U.LOSS_FNS[args.loss]
    state = {"history": [], "best": None, "weights": None}
    head_epochs = 1 if args.smoke else C.HEAD_EPOCHS
    ft_epochs = 1 if args.smoke else C.FT_EPOCHS
    t_start = time.time()

    print(f"=== {args.model} / {args.loss}  params={train_model.count_params():,}  batch={batch} ===", flush=True)
    backbone.trainable = False
    train_model.compile(optimizer=make_optimizer(C.HEAD_LR), loss=loss_fn)
    train_model.fit(train_ds, epochs=head_epochs, verbose=0,
                    callbacks=[ValMonitor(eval_model, val_ds, y_va, loss_fn, 0, state, "head")])

    backbone.trainable = True
    sched = WarmupCosine(C.FT_LR, C.FT_WARMUP_EPOCHS * steps, ft_epochs * steps)
    train_model.compile(optimizer=make_optimizer(sched), loss=loss_fn)
    train_model.fit(train_ds, epochs=ft_epochs, verbose=0,
                    callbacks=[ValMonitor(eval_model, val_ds, y_va, loss_fn, C.PATIENCE, state, "finetune")])
    train_time = time.time() - t_start

    train_model.set_weights(state["weights"])
    _, p_va = eval_model.predict(val_ds, verbose=0)
    _, p_te = eval_model.predict(test_ds, verbose=0)
    m_va, m_te = U.compute_metrics(y_va, p_va), U.compute_metrics(y_te, p_te)

    infer_model.save(run_dir / "model.keras")
    infer_model.save_weights(run_dir / "model.weights.h5")
    np.savez_compressed(run_dir / "predictions.npz", val=p_va, test=p_te, y_val=y_va, y_test=y_te)
    pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
    summary = dict(model=args.model, loss=args.loss, variant=args.variant, tier=spec["tier"],
                   params=int(infer_model.count_params()), best_epoch=state["best"][2],
                   epochs_run=len(state["history"]), train_time_s=train_time, val=m_va, test=m_te)
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)
    print(f"DONE {args.model}/{args.loss}: best epoch {state['best'][2]}  "
          f"val qwk {m_va['qwk']:.4f} acc {m_va['accuracy']:.4f} | test qwk {m_te['qwk']:.4f} "
          f"acc {m_te['accuracy']:.4f}  ({train_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
