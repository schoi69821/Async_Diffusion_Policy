"""Integration tests for training and inference."""
import torch
import numpy as np
import tempfile
from pathlib import Path

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.train.engine import TrainingEngine, NoiseScheduler
from async_dp_v8.losses.multitask_loss import compute_losses
from async_dp_v8.utils.checkpointing import save_checkpoint, load_checkpoint
from async_dp_v8.utils.normalization import Normalizer
from async_dp_v8.inference.state_machine import State, RuntimeContext
from async_dp_v8.control.chunk_blender import ChunkBlender
from async_dp_v8.inference.temporal_ensemble import TemporalEnsemble
from async_dp_v8.control.action_postprocess import clip_joint_step, ema_smooth


def _make_synthetic_batch(B=2, T=3, H=12, device="cpu"):
    return {
        "obs_image_wrist": torch.randn(B, T, 3, 224, 224, device=device),
        "obs_image_crop": torch.randn(B, T, 3, 96, 96, device=device),
        "obs_qpos": torch.randn(B, T, 6, device=device),
        "obs_qvel": torch.randn(B, T, 6, device=device),
        "obs_ee_pose": torch.randn(B, T, 7, device=device),
        "obs_gripper": torch.randn(B, T, 2, device=device),
        "obs_current": torch.randn(B, T, 7, device=device),
        "obs_pwm": torch.randn(B, T, 7, device=device),
        "target_phase_next": torch.randint(0, 6, (B,), device=device),
        "target_grip_token": torch.randint(0, 3, (B,), device=device),
        "target_contact": torch.rand(B, device=device),
        "target_arm_chunk": torch.randn(B, H, 6, device=device),
        "mask_valid": torch.ones(B, H, device=device),
        "phase_curr": torch.randint(0, 6, (B,), device=device),
        "contact_curr": torch.rand(B, device=device),
    }


def test_single_train_step():
    """Full forward + backward with synthetic batch."""
    model = HybridPolicyV8(pred_horizon=12, action_dim=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = NoiseScheduler(num_steps=10)

    batch = _make_synthetic_batch()
    model.train()
    target_actions = batch["target_arm_chunk"]
    timesteps = torch.randint(0, 10, (2,))
    noise = torch.randn_like(target_actions)
    noisy_actions = scheduler.add_noise(target_actions, noise, timesteps)
    out = model(batch, noisy_actions=noisy_actions, timestep=timesteps)

    losses = compute_losses(batch, out, out["pred_noise"], noise, arm_chunk_pred=target_actions)
    losses["total"].backward()
    optimizer.step()
    assert losses["total"].item() > 0


def test_checkpoint_roundtrip():
    """Save and load checkpoint, verify model state survives."""
    model = HybridPolicyV8(pred_horizon=12, action_dim=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Get initial weights
    w_before = model.phase_head.net[0].weight.data.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_ckpt.pt")
        save_checkpoint(model, optimizer, 5, {"loss": 0.5}, path)

        # Create fresh model and load
        model2 = HybridPolicyV8(pred_horizon=12, action_dim=6)
        load_checkpoint(path, model2, device="cpu")

        w_after = model2.phase_head.net[0].weight.data
        assert torch.equal(w_before, w_after)


def test_normalizer_roundtrip():
    """Normalize then denormalize should be identity."""
    data = {
        "qpos": np.random.randn(100, 6).astype(np.float32),
        "action_arm": np.random.randn(100, 6).astype(np.float32),
    }
    normalizer = Normalizer.fit(data)

    original = data["qpos"][0].copy()
    normed = normalizer.normalize("qpos", original)
    recovered = normalizer.denormalize("qpos", normed)
    np.testing.assert_allclose(original, recovered, atol=1e-5)


def test_normalizer_json_roundtrip():
    """Save and load normalizer stats from JSON."""
    data = {"qpos": np.random.randn(50, 6).astype(np.float32)}
    normalizer = Normalizer.fit(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "stats.json")
        normalizer.to_json(path)
        loaded = Normalizer.from_json(path)

        x = np.random.randn(6).astype(np.float32)
        np.testing.assert_allclose(
            normalizer.normalize("qpos", x),
            loaded.normalize("qpos", x),
            atol=1e-5,
        )


def test_temporal_ensemble_single():
    """Single prediction should pass through."""
    te = TemporalEnsemble(pred_horizon=12, execute_horizon=4)
    chunk = np.ones((12, 6))
    te.add_prediction(chunk)
    actions = te.get_actions(4)
    np.testing.assert_array_equal(actions, np.ones((4, 6)))


def test_temporal_ensemble_averaging():
    """Multiple overlapping predictions should be averaged."""
    te = TemporalEnsemble(pred_horizon=8, execute_horizon=4)
    te.add_prediction(np.ones((8, 6)))
    te.add_prediction(np.ones((8, 6)) * 3.0)
    actions = te.get_actions(4)
    # First 4 of the new chunk overlap with last 4 of previous
    # Should be weighted average, not exactly 1.0 or 3.0
    assert np.all(actions > 1.0)
    assert np.all(actions < 3.0)


def test_temporal_ensemble_empty_raises():
    te = TemporalEnsemble()
    try:
        te.get_actions()
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_clip_joint_step():
    """Verify joint step clipping works."""
    actions = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])  # big jump
    current = np.zeros(6)
    clipped = clip_joint_step(actions, current, max_step_rad=0.12)
    assert np.max(np.abs(clipped[0])) <= 0.12 + 1e-6


def test_ema_smooth():
    """EMA should reduce variance."""
    actions = np.random.randn(10, 6)
    smoothed = ema_smooth(actions, alpha=0.3)
    assert np.std(smoothed) < np.std(actions)
