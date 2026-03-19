"""Evaluation metrics for v8 policy."""
import torch
import numpy as np
from typing import Dict


def compute_phase_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean())


def compute_grip_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean())


def compute_contact_metrics(logit: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = logit.squeeze(-1).sigmoid()
    preds = (probs > threshold).long()
    targets_long = targets.long()

    tp = ((preds == 1) & (targets_long == 1)).sum().float()
    fp = ((preds == 1) & (targets_long == 0)).sum().float()
    fn = ((preds == 0) & (targets_long == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "contact_f1": float(f1),
    }


def compute_action_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
        return float(diff.sum() / mask.sum().clamp(min=1) / pred.shape[-1])
    return float(diff.mean())
