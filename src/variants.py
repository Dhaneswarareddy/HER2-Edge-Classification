"""Improvement experiments for FYP3 (the baseline runs use none of this).

Each variant targets one diagnosed difference between BCI and FYP2:

  cw      class-weighted loss. BCI is imbalanced (score 0 = 5% of training tiles); FYP2 was perfectly
          balanced, so the FYP2 pipeline has no imbalance handling. Weights w_k = N / (K * n_k) from the
          training split (mean weight over training samples = 1).
  ls      label smoothing 0.1 (CE). BCI labels are slide-level; 18% of tiles disagree with the
          patch-level reference labels, so hard 0/1 targets over-trust noisy labels.
  hires   2x resolution: 1024 px tiles -> 512 px source -> 448 px input (baseline: 256 -> 224), to test
          whether 4x downsampling destroys membrane-level detail that the HER2 criteria depend on.
  hed     stain augmentation in haematoxylin-eosin-DAB space (Tellez et al. 2018), H and E channels only:
          DAB intensity is the label signal and is left untouched. Targets scanner/counterstain variation.
          Decision rule, fixed before the perturbation analysis was run: train it only if analysis_models.py
          shows BCI-trained networks shift their expected score by >= 0.10 when haematoxylin alone is scaled
          x0.8 or x1.25 (median over networks; the larger of the two directions). Both are well inside BCI's
          natural range: within a class, tile haematoxylin varies 4-20x between the 10th and 90th percentile.
  illum   illumination augmentation: the image is multiplied by a random gain in [0.85, 1.15], which
          rescales the illumination (background) while leaving stain optical density unchanged
          (Beer-Lambert). BCI's scan brightness differs between classes (grey scans are 25% of score-0
          tiles and 68% of 3+), so this takes away a possible shortcut. Decision rule, fixed before
          the perturbation analysis was run: train it only if analysis_models.py shows BCI-trained networks
          shift their expected score by >= 0.10 for a 15% illumination change (median over networks; x0.85
          or x1.15, the larger of the two directions).

Screening result (MobileNetV3Small, paired fragment-cluster bootstrap on validation, 29 Oct 2026):
cw +0.023 [-0.040, +0.088], ls +0.004 [-0.045, +0.054], hires +0.102 [+0.031, +0.173] val QWK. Only hires
meets the rule (CI excludes 0), although its test QWK does not move (-0.014 [-0.063, +0.038]).
Confirmation rule, fixed before the run: hires counts as a confirmed improvement only if EfficientNetV2B0
(mid-tier winner) at 448 px also gains val QWK over its 224 px baseline with a paired CI that excludes 0.
Caveat for both: hires trains with a quarter of the batch size (memory), so resolution and batch size change together.

Follow-up experiments (added 30 Oct 2026, see NEXT_EXPERIMENTS.md; decision rules fixed before any run):
  lowbatch   224 px at the batch size the hires run used (batch // 4): separates resolution from batch size.
             Resolution effect = hires - lowbatch; batch effect = lowbatch - baseline (paired, val).
  hires1024  native 1024 px source -> 896 px input, batch // 16 (min 2). Optional, lite model only.
  init_fyp2  start from the FYP2-trained (GAN) CE network instead of ImageNet, then the usual schedule.
  mix_fyp2   train on BCI train + FYP2's 1,680 GAN training images (FYP2 val/test never used); 224 px only.
  Each counts as an improvement only if the paired CI of delta val QWK against its reference run excludes 0.

Variants combine with '+' in run names only through explicit entries below.
"""
import numpy as np
import tensorflow as tf

import config as C

VARIANTS = {
    "cw": dict(class_weight=True),
    "ls": dict(label_smoothing=0.1),
    "hires": dict(source_size=512, image_size=448),
    "hed": dict(augment="hed"),
    "illum": dict(augment="illum"),
    "cw_hires": dict(class_weight=True, source_size=512, image_size=448),
    "cw_ls": dict(class_weight=True, label_smoothing=0.1),
    "lowbatch": dict(batch_div=4),
    "hires1024": dict(source_size=1024, image_size=896, batch_div=16),
    "init_fyp2": dict(init_from="fyp2"),
    "mix_fyp2": dict(mix="fyp2"),
}

# Ruifrok & Johnston stain matrix (same as analysis_data.py / skimage.rgb2hed)
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def class_weights(y):
    n = np.bincount(y, minlength=C.NUM_CLASSES).astype(np.float32)
    return len(y) / (C.NUM_CLASSES * n)


def make_loss(loss, class_weight=None, label_smoothing=0.0):
    """CE / CORN with optional per-class weights (and label smoothing for CE).
    With no options this is numerically the baseline loss of common.py."""
    w = tf.constant(class_weight if class_weight is not None else np.ones(C.NUM_CLASSES), tf.float32)

    def ce(y_true, logits):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        onehot = tf.one_hot(y, C.NUM_CLASSES)
        per = tf.keras.losses.categorical_crossentropy(onehot, logits, from_logits=True,
                                                       label_smoothing=label_smoothing)
        sw = tf.gather(w, y)
        return tf.reduce_sum(per * sw) / tf.reduce_sum(sw)

    def corn(y_true, logits):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        sw = tf.gather(w, y)
        total = 0.0
        for k in range(C.NUM_CLASSES - 1):
            mask = tf.cast(y >= k, tf.float32) * sw
            target = tf.cast(y > k, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits[:, k])
            total += tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1e-6)
        return total / (C.NUM_CLASSES - 1)

    if label_smoothing and loss != "ce":
        raise ValueError("label smoothing is defined for CE only")
    return ce if loss == "ce" else corn


def hed_augment(base_augment, sigma_scale=0.05, sigma_shift=0.05):
    """Wrap the FYP2 augmentation: perturb H and E optical densities, keep DAB unchanged."""
    m_in, m_out = tf.constant(_HED_FROM_RGB), tf.constant(_RGB_FROM_HED)

    def aug(img):
        img = base_augment(img)                                   # float 0..255
        od = -tf.math.log(tf.maximum(img / 255.0, 1e-6))
        hed = tf.reshape(od, [-1, 3]) @ m_in
        a = tf.concat([tf.random.uniform([2], 1 - sigma_scale, 1 + sigma_scale), [1.0]], 0)
        b = tf.concat([tf.random.uniform([2], -sigma_shift, sigma_shift), [0.0]], 0)
        rgb = tf.exp(-((hed * a + b) @ m_out))
        return tf.clip_by_value(tf.reshape(rgb, tf.shape(img)), 0.0, 1.0) * 255.0

    return aug


def illum_augment(base_augment, max_gain=0.15):
    """Wrap the FYP2 augmentation: a random illumination gain (scan brightness), stain OD unchanged."""
    def aug(img):
        img = base_augment(img)                                   # float 0..255
        return tf.clip_by_value(img * tf.random.uniform([], 1 - max_gain, 1 + max_gain), 0.0, 255.0)

    return aug


AUGMENTS = {"hed": hed_augment, "illum": illum_augment}
