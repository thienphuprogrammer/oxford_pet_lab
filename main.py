from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

# Internal imports
from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.data import OxfordPetDatasetLoader, DataPreprocessor, DataAugmentor
from src.evaluation.evaluator import Evaluator
from src.models.base_model import ModelBuilder
from src.training.trainer import *

def _build_model(task: str, cfg: Config, backbone: str) -> tf.keras.Model:
    task = task.lower()
    if task == "detection":
        return ModelBuilder.build_detection_model(
            model_type="pretrained",
            num_classes=cfg.NUM_CLASSES_DETECTION,
            config=cfg,
        )
    if task == "segmentation":
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
    loader = OxfordPetDatasetLoader(
        data_dir=cfg.DATA_DIR,
        download=cfg.DOWNLOAD,
        log_level=cfg.LOG_LEVEL,
    )

    # Initialize preprocessor to enforce fixed image sizes and normalization
    preprocessor = DataPreprocessor(config=cfg)

    train_ds, val_ds, test_ds = loader.create_train_val_test_splits(
        val_split=0.2,
        seed=42
    )
    # val_ds = val_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    # test_ds = test_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # augmentor = DataAugmentor(config=cfg, target_height=cfg.IMG_SIZE[0], target_width=cfg.IMG_SIZE[1])
    # train_ds = train_ds.map(augmentor, num_parallel_calls=tf.data.AUTOTUNE)

    if task == "detection":
        # Resize/normalize images and prepare detection targets
        train_ds = train_ds.map(preprocessor.for_detection, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocessor.for_detection, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocessor.for_detection, num_parallel_calls=tf.data.AUTOTUNE)
    elif task == "segmentation":
        # Resize/normalize images and prepare segmentation targets
        train_ds = train_ds.map(preprocessor.for_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocessor.for_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocessor.for_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    elif task == "multitask":
        # First standardize sample (resize/normalize), then format multitask targets
        def format_multitask(sample):
            image = sample['image']
            target = {
                'bbox': sample['head_bbox'],
                'mask': sample['segmentation_mask'],
                'label': sample['label'],
            }
            return image, target

        train_ds = train_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.map(format_multitask, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(format_multitask, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(format_multitask, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Batch + Prefetch
    train_ds = train_ds.shuffle(2048).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

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

    cfg = Config()
    cfg.LOGS_DIR = args.output_dir / "logs"  # type: ignore[attr-defined]
    cfg.CHECKPOINT_DIR = args.output_dir / "checkpoints"  # type: ignore[attr-defined]
    cfg.BACKBONE = args.backbone
    cfg.create_directories()
    Config.setup_gpu()

    # (Optional) user-provided YAML config
    models_cfg: ModelConfigs | None = None
    if args.config_path and args.config_path.exists():
        try:
            models_cfg = ModelConfigs()
        except Exception as exc:
            print(f"[WARNING] Failed to load ModelConfigs: {exc}. Falling back to defaults.")
            models_cfg = None

    train_ds, val_ds, test_ds = _prepare_datasets(args.task, args.batch_size, cfg)

    model = _build_model(args.task, cfg, args.backbone)
    
    model.summary()

    trainer = UniversalTrainer(
        model=model,
        task_type=args.task
    )

    print(f"\n[INFO] Starting training – task={args.task} backbone={args.backbone} epochs={args.epochs}")
    trainer.train(
        train_ds,
        val_ds,
        epochs=args.epochs,
    )

    evaluator = Evaluator(
        model=trainer.model,
        task_type=args.task,
        num_classes=cfg.NUM_CLASSES_SEGMENTATION if args.task in {"segmentation", "multitask"} else None,
    )
    metrics: dict[str, Any] = evaluator.evaluate(test_ds)

    print("\n[RESULT] Test metrics:")
    for k, v in metrics.items():
        print(f" • {k}: {v:.4f}" if isinstance(v, (int, float)) else f" • {k}: {v}")

    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump({k: float(v) if hasattr(v, "numpy") else v for k, v in metrics.items()}, fp, indent=2)

    print(f"\n[INFO] Finished – results saved to {args.output_dir}\n")

if __name__ == "__main__":
    main()
