"""Combined multi-task loss for HybridPolicyV8."""
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .diffusion_loss import diffusion_mse_loss
from .focal_loss import focal_cross_entropy
from .smoothness_loss import smoothness_loss


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    arm_chunk_pred: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    lambda_arm: float = 1.0,
    lambda_phase: float = 0.4,
    lambda_grip: float = 0.8,
    lambda_contact: float = 0.3,
    lambda_smooth: float = 0.05,
    grip_focal_gamma: float = 2.0,
    phase_label_smoothing: float = 0.05,
) -> Dict[str, torch.Tensor]:
    losses = {}

    # Arm diffusion loss
    losses["arm_diff"] = diffusion_mse_loss(pred_noise, true_noise, mask)

    # Phase classification
    losses["phase_ce"] = focal_cross_entropy(
        out["phase_logits"],
        batch["target_phase_next"].long(),
        gamma=grip_focal_gamma,
        label_smoothing=phase_label_smoothing,
    )

    # Gripper token classification
    losses["grip_ce"] = focal_cross_entropy(
        out["grip_logits"],
        batch["target_grip_token"].long(),
        gamma=grip_focal_gamma,
    )

    # Contact prediction
    losses["contact_bce"] = F.binary_cross_entropy_with_logits(
        out["contact_logit"].squeeze(-1),
        batch["target_contact"].float(),
    )

    # Smoothness
    if arm_chunk_pred is not None:
        losses["smooth"] = smoothness_loss(arm_chunk_pred, mask)
    else:
        losses["smooth"] = torch.tensor(0.0, device=pred_noise.device)

    # Total
    total = (
        lambda_arm * losses["arm_diff"]
        + lambda_phase * losses["phase_ce"]
        + lambda_grip * losses["grip_ce"]
        + lambda_contact * losses["contact_bce"]
        + lambda_smooth * losses["smooth"]
    )
    losses["total"] = total
    return losses
