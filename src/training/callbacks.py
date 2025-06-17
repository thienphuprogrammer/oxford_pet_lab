from __future__ import annotations

from pathlib import Path
from typing import List, Union

import tensorflow as tf

__all__ = [
    "get_optimized_callbacks",
]

def get_callbacks(
    *,
    task: str,
    log_dir: Union[str, Path],
    ckpt_dir: Union[str, Path],
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 8,
    enable_reduce_lr: bool = True,
    optimizer_has_schedule: bool = False,
) -> List[tf.keras.callbacks.Callback]:

    log_dir = Path(log_dir)
    ckpt_dir = Path(ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cb: List[tf.keras.callbacks.Callback] = []

    # TensorBoard
    cb.append(tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1))

    # ModelCheckpoint
    ckpt_path = ckpt_dir / f"{task}_best.weights.h5"
    cb.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    # EarlyStopping
    cb.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # ReduceLROnPlateau (chỉ khi Optimizer chưa có schedule)
    if enable_reduce_lr and not optimizer_has_schedule:
        cb.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.3,
                patience=max(patience // 2, 2),
                verbose=1,
                mode=mode,
                min_lr=1e-7,
            )
        )

    return cb
