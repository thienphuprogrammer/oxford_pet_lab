"""
Simplified callback factory using built‑in Keras utilities.

Highlights
----------
* EarlyStopping with restore_best_weights
* ModelCheckpoint (save_best_only)
* Cosine‑decay LR schedule with optional warm‑up implemented via LearningRateScheduler
* ReduceLROnPlateau as safety net
* TensorBoard & CSVLogger for monitoring
* No custom gradient clipping callback – set clipnorm/clipvalue directly in the optimizer
"""

from __future__ import annotations
import os
from typing import List

import tensorflow as tf


def cosine_decay_with_warmup(initial_lr: float,
                             total_epochs: int,
                             warmup_epochs: int = 0,
                             alpha: float = 0.0):
    """Return scalar LR for a given epoch."""
    def schedule(epoch: int) -> float:  # epoch is zero‑based
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(tf.math.pi) * progress))
        return initial_lr * ((1 - alpha) * cosine_decay + alpha)
    return schedule


def get_callbacks(task: str,
                  log_dir: str,
                  ckpt_dir: str,
                  monitor: str = "val_loss",
                  patience: int = 10,
                  total_epochs: int = 100,
                  initial_lr: float = 1e-3,
                  warmup_epochs: int = 0,
                  use_tensorboard: bool = True,
                  save_weights_only: bool = False) -> List[tf.keras.callbacks.Callback]:
    """Build a minimal yet powerful callback list for Keras training."""

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    mode = "min" if any(k in monitor.lower() for k in ["loss", "error"]) else "max"

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
        mode=mode,
    )

    ckpt_fmt = os.path.join(ckpt_dir, f"best_{task}_epoch_{{epoch:03d}}.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_fmt,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=save_weights_only,
        verbose=1,
    )

    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        cosine_decay_with_warmup(
            initial_lr=initial_lr,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
        ),
        verbose=0,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=max(1, patience // 3),
        min_lr=1e-8,
        mode=mode,
        verbose=1,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv"), append=True)

    callbacks: List[tf.keras.callbacks.Callback] = [early_stop, ckpt, lr_cb, reduce_lr, csv_logger]

    if use_tensorboard:
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tb)

    # Safety‑net callback to terminate on NaNs
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    return callbacks
