"""src/main.py
Entry-point script to train / evaluate Oxford-IIIT Pet models from the
command-line. The script wires together the high-level components already
implemented throughout *src/* so that the user can run e.g.

    $ python -m src.main --task detection --backbone resnet50 --epochs 30

Without touching the underlying library code.
"""
from __future__ import annotations

import argparse
import json
from mimetypes import suffix_map
from pathlib import Path
from typing import Any

import tensorflow as tf

# Internal imports ------------------------------------------------------------
from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.data import OxfordPetDatasetLoader, DataPreprocessor, DataAugmentor
from src.evaluation.evaluator import Evaluator
from src.models.base_model import ModelBuilder
from src.training.trainer import Trainer

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _build_model(task: str, cfg: Config) -> tf.keras.Model:
    """Factory wrapper around :class:`ModelBuilder`.

    The mapping below keeps CLI simple while still allowing the user to change
    backbones. Extend as needed when new architectures are added.
    """
    task = task.lower()

    if task == "detection":
        # For now we expose two flavours: simple (tiny custom CNN) or pretrained
        # backbone. We always use the *pretrained* builder here so that different
        # backbones can be selected via ``--backbone``.
        return ModelBuilder.build_detection_model(
            model_type="pretrained",
            num_classes=cfg.NUM_CLASSES_DETECTION,
            config=cfg,
        )

    if task == "segmentation":
        # Map backbone → segmentation model_type.
        seg_mapping = {
            "resnet50": "pretrained_unet",
            "mobilenetv2": "pretrained_unet",
        }
        model_type = seg_mapping.get(backbone, "simple_unet")
        return ModelBuilder.build_segmentation_model(
            model_type=model_type,
            num_classes=cfg.NUM_CLASSES_SEGMENTATION,
            backbone=backbone,
            config=cfg,
        )

    if task == "multitask":
        multi_mapping = {
            "resnet50": "resnet50_multitask",
            "efficientnetb0": "efficientnet_multitask",
        }
        model_type = multi_mapping.get(backbone, "resnet50_multitask")
        return ModelBuilder.build_multitask_model(
            model_type=model_type,
            num_detection_classes=cfg.NUM_CLASSES_DETECTION,
            num_segmentation_classes=cfg.NUM_CLASSES_SEGMENTATION,
            backbone_name=backbone,
            config=cfg,
        )

    raise ValueError(f"Unknown task type: {task}")


def _prepare_datasets(
    task: str,
    batch_size: int,
    cfg: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load TFDS splits and convert them into batched tensors."""
    loader = OxfordPetDatasetLoader(
        data_dir=cfg.DATA_DIR,
        download=cfg.DOWNLOAD,
        log_level=cfg.LOG_LEVEL,
    )

    train_ds, val_ds, test_ds = loader.create_train_val_test_splits(
        val_split=0.2,
        seed=42
    )

    # data augmentation
    augmentor = DataAugmentor(
        config=cfg,
        prob_geo=0.7,      # Tăng geometric augmentations
        prob_photo=0.8,    # Tăng photometric augmentations  
        prob_mixup=0.3,    # Thêm MixUp
        prob_cutout=0.2,   # Thêm Cutout
        prob_mosaic=0.3    # Thêm Mosaic
    )
    train_ds = augmentor.create_augmented_dataset(
        train_ds,
        augmentation_factor=3
    )

    prep = DataPreprocessor(config=cfg, shuffle_buffer=5000)
    train_ds = prep.create_training_dataset(
        train_ds,
        batch_size=batch_size,
        task=task,
        cache_filename="train_cache.tfrecord"
    )
    val_ds = prep.create_validation_dataset(
        val_ds,
        batch_size=batch_size,
        task=task,
    )
    test_ds = prep.create_validation_dataset(
        test_ds,
        batch_size=batch_size,
        task=task
    )

    # Monitor dataset quality
    # stats_train = prep.get_dataset_statistics(train_ds)
    # stats_val = prep.get_dataset_statistics(val_ds)
    # stats_test = prep.get_dataset_statistics(test_ds)

    # Save stats
    # with open(cfg.REPORT_DIR / "train_stats.json", "w") as f:
    #     json.dump(stats_train, f)
    # with open(cfg.REPORT_DIR / "val_stats.json", "w") as f:
    #     json.dump(stats_val, f)
    # with open(cfg.REPORT_DIR / "test_stats.json", "w") as f:
    #     json.dump(stats_test, f)

    return train_ds, val_ds, test_ds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Oxford-IIIT Pet Lab – train & evaluate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["detection", "segmentation", "multitask"], default="detection", help="Learning task to run")
    parser.add_argument("--backbone", default="resnet50", help="Backbone / architecture identifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--config_path", type=Path, default=None, help="Optional path to YAML config that overrides defaults")
    parser.add_argument("--output_dir", type=Path, default=Path("results/runs/latest"), help="Directory to save logs / checkpoints")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Setup & configuration
    # ------------------------------------------------------------------
    cfg = Config()
    cfg.LOGS_DIR = args.output_dir / "logs"  # type: ignore[attr-defined]
    cfg.CHECKPOINT_DIR = args.output_dir / "checkpoints"  # type: ignore[attr-defined]
    cfg.BACKBONE = args.backbone
    cfg.create_directories()  # ensure dirs exist

    Config.setup_gpu()

    # (Optional) user-provided YAML config for hyper-parameters / losses / etc.
    models_cfg: ModelConfigs | None = None
    if args.config_path and args.config_path.exists():
        try:
            models_cfg = ModelConfigs()
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Failed to load ModelConfigs: {exc}. Falling back to defaults.")
            models_cfg = None

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds = _prepare_datasets(args.task, args.batch_size, cfg)

    # ------------------------------------------------------------------
    # Model & trainer
    # ------------------------------------------------------------------
    model = _build_model(args.task, cfg)
    trainer = Trainer(
        model=model,
        task_type=args.task,
        backbone_name=cfg.BACKBONE,
        config=cfg,
        models_config=models_cfg,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting training – task={args.task} backbone={args.backbone} epochs={args.epochs}")
    trainer.fit(train_ds, val_dataset=val_ds, epochs=args.epochs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluator = Evaluator(
        model=trainer.model,
        task_type=args.task,
        num_classes=cfg.NUM_CLASSES_SEGMENTATION if args.task in {"segmentation", "multitask"} else None,
    )
    metrics: dict[str, Any] = evaluator.evaluate(test_ds)

    print("\n[RESULT] Test metrics:")
    for k, v in metrics.items():
        print(f" • {k}: {v:.4f}" if isinstance(v, (int, float)) else f" • {k}: {v}")

    # Persist metrics ---------------------------------------------------
    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump({k: float(v) if hasattr(v, "numpy") else v for k, v in metrics.items()}, fp, indent=2)

    print(f"\n[INFO] Finished – results saved to {args.output_dir}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
