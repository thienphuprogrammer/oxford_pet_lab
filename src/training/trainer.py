from pathlib import Path
from typing import Optional

import tensorflow as tf

from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.training.losses import get_loss          # Giữ lại nếu cần custom
from src.training.metrics import get_metrics
from src.training.callbacks import get_callbacks


class Trainer:
    """Trainer tối giản – tận dụng hoàn toàn Keras Model.fit."""

    def __init__(
        self,
        model: tf.keras.Model,
        task_type: str,
        config: Optional[Config] = None,
        model_cfg: Optional[ModelConfigs] = None,
    ):
        self.model = model
        self.task_type = task_type
        self.cfg = config or Config()
        self.model_cfg = model_cfg or ModelConfigs()

        self._configure_precision_and_device()
        self.optimizer = self._build_optimizer()
        self.loss_fn = self._build_loss()
        self.metrics = get_metrics(task_type, self.cfg.NUM_CLASSES_DETECTION)
        self.callbacks = self._build_callbacks()

    # ---------- public API ----------
    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        epochs: int | None = None,
    ):
        train_ds = self._prepare_ds(train_ds, training=True)
        if val_ds is not None:
            val_ds = self._prepare_ds(val_ds, training=False)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics,
            run_eagerly=getattr(self.cfg, "DEBUG", False),
        )
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs or getattr(self.cfg, "EPOCHS", 100),
            callbacks=self.callbacks,
        )

    # ---------- helpers ----------
    def _configure_precision_and_device(self):
        if getattr(self.cfg, "USE_MIXED_PRECISION", False) and self.task_type != "detection":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            self._need_loss_scale = True
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            self._need_loss_scale = False

        # memory growth
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        # XLA
        tf.config.optimizer.set_jit(getattr(self.cfg, "USE_XLA", False))

    def _build_optimizer(self):
        opt_cfg = getattr(self.cfg, "OPTIMIZER", {"class_name": "SGD",
                                                  "config": {"learning_rate": 1e-2}})
        opt = tf.keras.optimizers.get(opt_cfg)
        if self._need_loss_scale:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        return opt

    def _build_loss(self):
        # Hỗ trợ dict/str; ưu tiên dùng get_loss nếu bạn đã có implement riêng
        cfg = self.model_cfg.LOSS_CONFIGS[self.task_type]
        if isinstance(cfg, str):
            return get_loss(cfg)
        loss_name = cfg.get("loss_type", next(iter(cfg)))
        kwargs = {k: v for k, v in cfg.items() if k != "loss_type"}
        return get_loss(loss_name, **kwargs)

    def _build_callbacks(self):
        log_dir = Path(self.cfg.LOGS_DIR) / self.task_type
        ckpt_dir = Path(self.cfg.RESULTS_DIR) / "checkpoints" / self.task_type
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return get_callbacks(task=self.task_type, log_dir=log_dir, ckpt_dir=ckpt_dir)

    def _prepare_ds(self, ds: tf.data.Dataset, training: bool):
        if training:
            ds = ds.shuffle(1000)
        return ds.prefetch(tf.data.AUTOTUNE)
