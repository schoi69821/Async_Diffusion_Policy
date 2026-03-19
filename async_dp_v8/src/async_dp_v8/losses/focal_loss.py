"""Focal loss for class-imbalanced classification (grip tokens, phases)."""
import torch
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss to handle class imbalance.

    logits: [B, C]
    targets: [B] (long)
    """
    ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    p = torch.softmax(logits, dim=-1)
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * ce).mean()
