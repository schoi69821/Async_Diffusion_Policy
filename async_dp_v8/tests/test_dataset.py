"""Test dataset loader."""
import torch
from async_dp_v8.data.collate import v8_collate_fn


def test_collate_fn():
    batch = [
        {
            "obs_qpos": torch.randn(3, 6),
            "target_arm_chunk": torch.randn(12, 6),
            "phase_curr": torch.tensor(0),
        },
        {
            "obs_qpos": torch.randn(3, 6),
            "target_arm_chunk": torch.randn(12, 6),
            "phase_curr": torch.tensor(1),
        },
    ]
    collated = v8_collate_fn(batch)
    assert collated["obs_qpos"].shape == (2, 3, 6)
    assert collated["target_arm_chunk"].shape == (2, 12, 6)
    assert collated["phase_curr"].shape == (2,)
