# config/model_configs.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party
import yaml
from src.config.config import Config

@dataclass
class ModelConfigs:
    """Model-specific configurations."""
    
    # Detection model configurations
    DETECTION_MODELS: Dict[str, Dict[str, Any]] = None
    
    # Segmentation model configurations
    SEGMENTATION_MODELS: Dict[str, Dict[str, Any]] = None
    
    # Multitask model configurations
    MULTITASK_MODELS: Dict[str, Dict[str, Any]] = None

    # Loss function configurations
    LOSS_CONFIGS: Dict[str, Dict[str, Any]] = None

    # Optimizer configurations
    OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = None
    
    
    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _load_yaml_config() -> Optional[Dict[str, Any]]:
        yaml_path = Config.CONFIG_DIR / "model_config.yaml"
        if not yaml_path.exists():
            return None

        try:
            with yaml_path.open("r", encoding="utf-8") as fp:
                return yaml.safe_load(fp)
        except Exception as exc:  # pragma: no cover
            # Corrupted YAML etc. We fail silently and fall back to defaults.
            print(f"[ModelConfigs] Failed to read YAML config at '{yaml_path}': {exc}")
            return None

    def __post_init__(self):
        # Try to load YAML configuration first. If not available, we fall back
        # to the hard-coded defaults defined below.
        yaml_cfg = self._load_yaml_config()

        # Default configurations
        default_configs = {
            "detection_models": {},
            "segmentation_models": {},
            "multitask_models": {},
            "loss_configs": {},
            "optimizer_configs": {},
            "metrics_configs": {},
            "callbacks_configs": {},
            "tensorboard_configs": {}
        }

        # Update defaults with YAML config if available
        if yaml_cfg:
            for key in default_configs:
                if isinstance(yaml_cfg.get(key), dict):
                    default_configs[key].update(yaml_cfg[key])

        # Set configurations with defaults
        self.DETECTION_MODELS = default_configs["detection_models"]
        self.SEGMENTATION_MODELS = default_configs["segmentation_models"]
        self.MULTITASK_MODELS = default_configs["multitask_models"]
        self.LOSS_CONFIGS = default_configs["loss_configs"]
        self.OPTIMIZER_CONFIGS = default_configs["optimizer_configs"]
        self.METRICS_CONFIGS = default_configs["metrics_configs"]
        self.CALLBACKS_CONFIGS = default_configs["callbacks_configs"]
        self.TENSORBOARD_CONFIGS = default_configs["tensorboard_configs"]
        