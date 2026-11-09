"""Shared data pipeline, model builders, ordinal (CORN) loss and metrics."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, mean_absolute_error, roc_auc_score)

import config as C

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_split(split, size=None):
    """Return (uint8 images N x H x W x 3, int labels) for one split, in manifest order.

    size=None keeps the original 256x256 (used for training, where random crops are taken);
    size=IMAGE_SIZE gives the fixed evaluation input that is shared by Keras, TFLite and the Pi.
    """
    df = pd.read_csv(C.MANIFEST)
    df = df[df.split == split].reset_index(drop=True)
    imgs = []
    for p in df.path:
        im = Image.open(C.DATA_DIR / p).convert("RGB")
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.uint8))
    return np.stack(imgs), df.label.values.astype(np.int32)


def _augment(img):
    """Stain-safe augmentation for IHC patches.

    Orientation is meaningless in a tissue patch, so all 8 dihedral transforms are used.
    Colour jitter is deliberately mild: DAB (brown) membrane-stain intensity *is* the HER2
    label, so strong brightness/hue shifts could turn a 2+ into a 1+ or 3+.
    """
    img = tf.cast(img, tf.float32) / 255.0
    # random-resized crop: 70-100% of the side length, then back to network size
    side = tf.cast(tf.shape(img)[0], tf.float32)
    crop = tf.cast(side * tf.random.uniform([], 0.7, 1.0), tf.int32)
    img = tf.image.random_crop(img, [crop, crop, 3])
    img = tf.image.resize(img, [C.IMAGE_SIZE, C.IMAGE_SIZE], method="bilinear")
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def split_paths(split):
    """(file paths, labels) for one split, in manifest order."""
    df = pd.read_csv(C.MANIFEST)
    df = df[df.split == split].reset_index(drop=True)
    return [str(C.DATA_DIR / p) for p in df.path], df.label.values.astype(np.int32)


def _read(path):
    return tf.ensure_shape(tf.io.decode_png(tf.io.read_file(path), channels=3), [None, None, 3])


def make_train_ds(paths, y, batch, augment=None):
    """Same pipeline as FYP2, but images are streamed from the 256 px PNG cache instead of an
    in-memory array (BCI's training split is twice FYP2's; 8 GB unified memory)."""
    augment = augment or _augment
    ds = tf.data.Dataset.from_tensor_slices((paths, y)).shuffle(len(paths), seed=C.SEED,
                                                                reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (augment(_read(a)), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_eval_ds(x, y, batch):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds.map(lambda a, b: (tf.cast(a, tf.float32), b)).batch(batch).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable(package="her2")
class Preprocess(tf.keras.layers.Layer):
    """Backbone-specific input normalisation, embedded in the graph so every exported
    model (Keras, TFLite FP32/FP16/INT8) takes raw 0-255 RGB pixels."""

    def __init__(self, mode, **kw):
        super().__init__(**kw)
        self.mode = mode

    def call(self, x):
        if self.mode == "caffe":   # RGB->BGR, subtract ImageNet BGR mean
            x = tf.reverse(x, axis=[-1])
            return x - tf.constant([103.939, 116.779, 123.68])
        if self.mode == "torch":   # [0,1] then ImageNet mean/std
            x = x / 255.0
            return (x - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
        if self.mode == "tf":      # [-1, 1]
            return x / 127.5 - 1.0
        return x

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


@tf.keras.utils.register_keras_serializable(package="her2")
class CornToProbs(tf.keras.layers.Layer):
    """Convert K-1 CORN logits into K class probabilities.

    CORN models P(y > k | y > k-1) with K-1 conditional binary tasks; the chain rule gives
    the unconditional P(y > k) = prod_{j<=k} sigmoid(logit_j), which is rank-consistent by
    construction.  Class probabilities are the successive differences.  Written with plain
    element-wise ops so it converts to every TFLite quantisation mode.
    """

    def call(self, logits):
        s = tf.sigmoid(logits)
        cum = [s[:, 0:1]]
        for k in range(1, C.NUM_CLASSES - 1):
            cum.append(cum[-1] * s[:, k:k + 1])
        parts = [1.0 - cum[0]] + [cum[k] - cum[k + 1] for k in range(len(cum) - 1)] + [cum[-1]]
        return tf.concat(parts, axis=-1)


def grouped_to_depthwise(model):
    """Replace grouped Conv2D layers with groups == channels (ConvNeXt's 7x7 'depthwise' convs)
    by the mathematically identical DepthwiseConv2D, copying the weights.

    tensorflow-metal has no GPU kernel for grouped convolution, and TFLite/XNNPACK runs a
    DEPTHWISE_CONV_2D far faster than a grouped CONV_2D, so this helps training and the Pi.
    """
    L = tf.keras.layers

    def clone(layer):
        if type(layer) is L.Conv2D and layer.groups > 1 and layer.groups == layer.filters:
            return L.DepthwiseConv2D(layer.kernel_size, strides=layer.strides, padding=layer.padding,
                                     use_bias=layer.use_bias, name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    new = tf.keras.models.clone_model(model, clone_function=clone)
    for old, nl in zip(model.layers, new.layers):
        w = old.get_weights()
        if type(nl) is L.DepthwiseConv2D and type(old) is L.Conv2D:
            w[0] = w[0].transpose(0, 1, 3, 2)   # (kh, kw, 1, C) -> (kh, kw, C, 1)
        nl.set_weights(w)
    return new


def build_backbone(name):
    spec = C.MODELS[name]
    ctor = getattr(tf.keras.applications, spec["app"])
    kw = dict(include_top=False, weights="imagenet",
              input_shape=(C.IMAGE_SIZE, C.IMAGE_SIZE, 3), pooling="avg")
    if spec["prep"] == "builtin" and name.startswith(("ConvNeXt", "EfficientNetV2", "MobileNetV3")):
        kw["include_preprocessing"] = True
    model = ctor(**kw)
    if name.startswith("ConvNeXt"):
        model = grouped_to_depthwise(model)
    return model


def build_model(name, loss):
    """Returns (train_model, inference_model, backbone).

    train_model outputs logits (4 for CE, 3 for CORN); inference_model shares the weights
    and outputs 4 class probabilities for both heads, so evaluation/export is identical.
    The backbone is called with training=False so BatchNorm statistics stay frozen during
    fine-tuning (recommended for small datasets / small batches).
    """
    spec = C.MODELS[name]
    inp = tf.keras.Input((C.IMAGE_SIZE, C.IMAGE_SIZE, 3), name="image")
    x = Preprocess(spec["prep"], name="preprocess")(inp) if spec["prep"] != "builtin" else inp
    backbone = build_backbone(name)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(C.DROPOUT, name="dropout")(x)
    n_out = C.NUM_CLASSES if loss == "ce" else C.NUM_CLASSES - 1
    logits = tf.keras.layers.Dense(n_out, name="logits", dtype="float32")(x)
    train_model = tf.keras.Model(inp, logits, name=f"{name}_{loss}")
    if loss == "ce":
        probs = tf.keras.layers.Softmax(name="probs")(logits)
    else:
        probs = CornToProbs(name="probs")(logits)
    infer_model = tf.keras.Model(inp, probs, name=f"{name}_{loss}_infer")
    return train_model, infer_model, backbone


def load_trained(run_dir):
    """Load a run's inference model.  Falls back to rebuilding the architecture from code and
    loading the saved weights when Keras cannot deserialise the graph (Keras 2.15 fails on the
    depthwise-rewritten ConvNeXt)."""
    import json
    try:
        return tf.keras.models.load_model(run_dir / "model.keras", compile=False)
    except Exception:
        s = json.load(open(run_dir / "summary.json"))
        _, model, _ = build_model(s["model"], s["loss"])
        model.load_weights(run_dir / "model.weights.h5")
        return model


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------


def corn_loss(y_true, logits):
    """CORN loss (Shi, Cao & Raschka, 2023).  Task k is trained only on samples with
    y >= k (the conditional subset) with binary target y > k."""
    y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    total = 0.0
    for k in range(C.NUM_CLASSES - 1):
        mask = tf.cast(y >= k, tf.float32)
        target = tf.cast(y > k, tf.float32)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits[:, k])
        total += tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
    return total / (C.NUM_CLASSES - 1)


def ce_loss(y_true, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        tf.reshape(y_true, [-1]), logits, from_logits=True))


LOSS_FNS = {"ce": ce_loss, "corn": corn_loss}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

# Clinical grouping (ASCO/CAP 2018): 0/1+ negative, 2+ equivocal (reflex ISH), 3+ positive.
CLINICAL = np.array([0, 0, 1, 2])


def compute_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    pred = probs.argmax(1)
    m = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, pred),
        "off_by_2plus": float(np.mean(np.abs(y_true - pred) >= 2)),
        "clinical_accuracy": accuracy_score(CLINICAL[y_true], CLINICAL[pred]),
    }
    try:
        m["macro_auc"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        m["macro_auc"] = float("nan")
    cm = confusion_matrix(y_true, pred, labels=range(C.NUM_CLASSES))
    for k in range(C.NUM_CLASSES):
        m[f"recall_{k}"] = cm[k, k] / max(cm[k].sum(), 1)
    m["confusion_matrix"] = cm.tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def set_seed(seed=C.SEED):
    tf.keras.utils.set_random_seed(seed)
