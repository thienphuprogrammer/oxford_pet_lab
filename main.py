from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import tensorflow as tf

# Internal imports
from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.data import OxfordPetDatasetLoader, DataPreprocessor, DataAugmentor
from src.evaluation.evaluator import Evaluator
from src.models.base_model import ModelBuilder
from src.training.trainer import UniversalTrainer
from src.utils.result_manager import ResultManager

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Oxford-IIIT Pet Lab – train & evaluate models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["detection", "segmentation", "multitask"],
        default="detection",
        help="Learning task to run"
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        help="Backbone / architecture identifier"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Global batch size"
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional path to YAML config that overrides defaults"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/runs/latest"),
        help="Directory to save logs / checkpoints"
    )
    return parser.parse_args(argv)

def _build_model(task: str, cfg: Config, backbone: str) -> tf.keras.Model:
    """Build model based on task type and backbone."""
    task = task.lower()
    
    # Model type mappings
    model_mappings = {
        "segmentation": {
            "resnet50": "pretrained_unet",
            "mobilenetv2": "pretrained_unet",
        },
        "multitask": {
            "resnet50": "resnet50_multitask",
            "efficientnetb0": "efficientnet_multitask",
        }
    }
    
    # Build model based on task
    if task == "detection":
        return ModelBuilder.build_detection_model(
            model_type="pretrained",
            num_classes=cfg.NUM_CLASSES_DETECTION,
            config=cfg,
        )
    elif task == "segmentation":
        model_type = model_mappings["segmentation"].get(backbone, "simple_unet")
        return ModelBuilder.build_segmentation_model(
            model_type=model_type,
            num_classes=cfg.NUM_CLASSES_SEGMENTATION,
            backbone=backbone,
            config=cfg,
        )
    elif task == "multitask":
        model_type = model_mappings["multitask"].get(backbone, "resnet50_multitask")
        return ModelBuilder.build_multitask_model(
            model_type=model_type,
            num_detection_classes=cfg.NUM_CLASSES_DETECTION,
            num_segmentation_classes=cfg.NUM_CLASSES_SEGMENTATION,
            backbone_name=backbone,
            config=cfg,
        )
    else:
        raise ValueError(f"Unknown task type: {task}")

def _prepare_datasets(
    task: str,
    batch_size: int,
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Prepare datasets for training, validation, and testing."""
    # Initialize data loader and preprocessor
    loader = OxfordPetDatasetLoader(
        data_dir=cfg.DATA_DIR,
        download=cfg.DOWNLOAD,
        log_level=cfg.LOG_LEVEL,
    )
    preprocessor = DataPreprocessor(config=cfg)
    augmentor = DataAugmentor(
        config=cfg,
        target_height=cfg.IMG_SIZE[0],
        target_width=cfg.IMG_SIZE[1]
    )
    
    # Create dataset splits
    train_ds, val_ds, test_ds = loader.create_train_val_test_splits(
        val_split=0.2,
        seed=42
    )
    
    # Apply preprocessing to validation and test sets
    val_ds = val_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocessor.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation to training set if enabled
    if cfg.ENABLE_AUGMENTATION:
        train_ds = train_ds.map(augmentor, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply task-specific preprocessing
    task_preprocessors = {
        "detection": preprocessor.for_detection,
        "segmentation": preprocessor.for_segmentation,
        "multitask": lambda x: preprocessor.for_multitask(preprocessor.preprocess_sample(x))
    }
    
    if task not in task_preprocessors:
        raise ValueError(f"Unknown task: {task}")
    
    # Apply task-specific preprocessing to all datasets
    preprocess_fn = task_preprocessors[task]
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Configure dataset performance
    train_ds = (train_ds
                .shuffle(2048)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))
    
    val_ds = (val_ds
              .batch(batch_size, drop_remainder=False)
              .prefetch(tf.data.AUTOTUNE))
    
    test_ds = (test_ds
               .batch(batch_size, drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, val_ds, test_ds

def _save_results(
    result_manager: ResultManager,
    args: argparse.Namespace,
    model: tf.keras.Model,
    trainer: UniversalTrainer,
    metrics: dict[str, Any],
    models_cfg: ModelConfigs | None
) -> None:
    """Save all experiment results."""
    # Save experiment configuration
    config = {
        'task_type': args.task,
        'backbone': args.backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'config_path': args.config_path,
        'output_dir': args.output_dir,
        'model_configs': models_cfg.__dict__ if models_cfg else None
    }
    result_manager.save_experiment_config(config, args.task)
    
    # Save training history plots
    result_manager.save_plots(trainer.history.history, args.task)
    
    # Save model
    weights_path, arch_path = result_manager.save_model(model, args.task)
    print(f"Model saved to: {weights_path}")
    print(f"Model architecture saved to: {arch_path}")
    
    # Save evaluation metrics
    metrics_path = result_manager.save_metrics(metrics, args.task)
    print(f"Evaluation metrics saved to: {metrics_path}")

def main(argv: list[str] | None = None) -> None:
    """Main training function."""
    # Parse arguments and setup configuration
    args = parse_args(argv)
    cfg = Config()
    cfg.LOGS_DIR = args.output_dir / "logs"
    cfg.CHECKPOINT_DIR = args.output_dir / "checkpoints"
    cfg.BACKBONE = args.backbone
    cfg.create_directories()
    Config.setup_gpu()
    
    # Load model configurations
    models_cfg = None
    if args.config_path and args.config_path.exists():
        try:
            models_cfg = ModelConfigs(config_path=args.config_path)
        except Exception as exc:
            print(f"[WARNING] Failed to load ModelConfigs: {exc}. Falling back to defaults.")
    
    # Prepare datasets
    train_ds, val_ds, test_ds = _prepare_datasets(args.task, args.batch_size, cfg)
    
    # Build and train model
    model = _build_model(args.task, cfg, args.backbone)
    model.summary()
    
    trainer = UniversalTrainer(
        model=model,
        task_type=args.task,
        num_classes=cfg.NUM_CLASSES_DETECTION if args.task == "detection" else cfg.NUM_CLASSES_SEGMENTATION
    )
    
    print(f"\n[INFO] Starting training – task={args.task} backbone={args.backbone} epochs={args.epochs}")
    trainer.train(train_ds, val_ds, epochs=args.epochs)
    
    # Evaluate model
    evaluator = Evaluator(
        model=trainer.model,
        task_type=args.task,
        num_classes=cfg.NUM_CLASSES_SEGMENTATION if args.task in {"segmentation", "multitask"} else None,
    )
    metrics = evaluator.evaluate(test_ds)
    
    # Print results
    print("\n[RESULT] Test metrics:")
    for k, v in metrics.items():
        print(f" • {k}: {v:.4f}" if isinstance(v, (int, float)) else f" • {k}: {v}")
    
    # Save all results
    result_manager = ResultManager()
    _save_results(result_manager, args, model, trainer, metrics, models_cfg)
    
    print(f"\n[INFO] Finished – results saved to {args.output_dir}\n")

if __name__ == "__main__":
    main()
