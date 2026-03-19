"""Model checkpoint saving and loading."""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    ema_model: Optional[torch.nn.Module] = None,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if ema_model is not None:
        state["ema_model_state_dict"] = ema_model.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path} (epoch={epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if ema_model is not None and "ema_model_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model_state_dict"])

    logger.info(f"Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')})")
    return ckpt
