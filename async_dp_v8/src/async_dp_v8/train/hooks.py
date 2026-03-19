"""Training hooks for logging and monitoring."""
import torch
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingHooks:
    def __init__(self, log_dir: str = "runs/v8", use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                logger.warning("tensorboard not available")

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        # Log to console
        parts = [f"epoch={epoch}"]
        for k, v in train_metrics.items():
            parts.append(f"train/{k}={v:.4f}")
        if val_metrics:
            for k, v in val_metrics.items():
                parts.append(f"val/{k}={v:.4f}")
        logger.info(" | ".join(parts))

        # Log to tensorboard
        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            if val_metrics:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

    def close(self):
        if self.writer:
            self.writer.close()
