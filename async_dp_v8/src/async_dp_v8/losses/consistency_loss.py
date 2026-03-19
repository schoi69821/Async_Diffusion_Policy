"""Consistency loss between overlapping chunk predictions."""
import torch
import torch.nn.functional as F


def chunk_consistency_loss(
    prev_chunk: torch.Tensor,
    curr_chunk: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    """Encourage consistency in the overlapping region of consecutive chunks.

    prev_chunk: [B, H, A] previous prediction
    curr_chunk: [B, H, A] current prediction
    overlap: number of overlapping timesteps
    """
    if overlap <= 0:
        return torch.tensor(0.0, device=curr_chunk.device)

    prev_tail = prev_chunk[:, -overlap:]  # [B, overlap, A]
    curr_head = curr_chunk[:, :overlap]   # [B, overlap, A]
    return F.mse_loss(curr_head, prev_tail.detach())
