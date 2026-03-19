"""YAML config loader utility."""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_config(config_dir: str = "configs") -> Dict[str, Any]:
    """Load all config files from directory structure.

    Returns merged config dict with sections: data, model, train.
    """
    config_dir = Path(config_dir)
    config = {}

    sections = {
        "dataset": "data/dataset_v8.yaml",
        "phase_labeling": "data/phase_labeling.yaml",
        "augmentation": "data/augmentation.yaml",
        "model": "model/hybrid_diffusion.yaml",
        "encoders": "model/encoders.yaml",
        "inference": "model/inference.yaml",
        "train": "train/train_v8.yaml",
        "losses": "train/losses.yaml",
    }

    for key, rel_path in sections.items():
        full_path = config_dir / rel_path
        if full_path.exists():
            config[key] = load_yaml(str(full_path))
        else:
            logger.debug(f"Config not found: {full_path}")

    return config


def get_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract loss weights from config."""
    losses_cfg = config.get("losses", {})
    return {
        "lambda_arm": losses_cfg.get("lambda_arm", 1.0),
        "lambda_phase": losses_cfg.get("lambda_phase", 0.4),
        "lambda_grip": losses_cfg.get("lambda_grip", 0.8),
        "lambda_contact": losses_cfg.get("lambda_contact", 0.3),
        "lambda_smooth": losses_cfg.get("lambda_smooth", 0.05),
        "lambda_cons": losses_cfg.get("lambda_cons", 0.10),
        "phase_label_smoothing": losses_cfg.get("phase_label_smoothing", 0.05),
        "grip_focal_gamma": losses_cfg.get("grip_focal_gamma", 2.0),
        "phase_focal_gamma": losses_cfg.get("phase_focal_gamma", 1.5),
        "contact_pos_weight": losses_cfg.get("contact_pos_weight", 2.0),
    }
