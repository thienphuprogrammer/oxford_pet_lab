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

        # ------------------------------------------------------------------
        # Detection models configuration
        # ------------------------------------------------------------------

        if yaml_cfg and isinstance(yaml_cfg.get("detection_models"), dict):
            # Entire sub-dict is stored directly so it preserves any defaults
            # defined by the user.
            self.DETECTION_MODELS = yaml_cfg["detection_models"]
        else:
            raise ValueError("Detection models configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("segmentation_models"), dict):
            self.SEGMENTATION_MODELS = yaml_cfg["segmentation_models"]
        else:
            raise ValueError("Segmentation models configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("multitask_models"), dict):
            self.MULTITASK_MODELS = yaml_cfg["multitask_models"]
        else:
            raise ValueError("Multitask models configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("loss_configs"), dict):
            self.LOSS_CONFIGS = yaml_cfg["loss_configs"]
        else:
            raise ValueError("Loss functions configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("optimizer_configs"), dict):
            self.OPTIMIZER_CONFIGS = yaml_cfg["optimizer_configs"]
        else:
            raise ValueError("Optimizer configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("metrics_configs"), dict):
            self.METRICS_CONFIGS = yaml_cfg["metrics_configs"]
        else:
            raise ValueError("Metrics configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("callbacks_configs"), dict):
            self.CALLBACKS_CONFIGS = yaml_cfg["callbacks_configs"]
        else:
            raise ValueError("Callbacks configuration not found in YAML file")
        
        if yaml_cfg and isinstance(yaml_cfg.get("tensorboard_configs"), dict):
            self.TENSORBOARD_CONFIGS = yaml_cfg["tensorboard_configs"]
        else:
            raise ValueError("Tensorboard configuration not found in YAML file")
        