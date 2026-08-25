#!/usr/bin/env python3
"""Offline evaluation of v8 policy on recorded episodes.

Measures:
1. Arm trajectory MSE (denoised vs ground truth)
2. Phase classification accuracy
3. Grip token classification accuracy
4. Contact prediction AUC / accuracy
5. Inference latency
"""
import argparse
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import logging

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.data.dataset_v8 import AsyncDPv8Dataset
from async_dp_v8.data.collate import v8_collate_fn
from async_dp_v8.train.engine import NoiseScheduler
from async_dp_v8.inference.ddim_sampler import DDIMSampler
from async_dp_v8.utils.checkpointing import load_checkpoint
from async_dp_v8.constants import NUM_PHASES, NUM_GRIP_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHASE_NAMES = {0: "REACH", 1: "ALIGN", 2: "CLOSE", 3: "LIFT", 4: "PLACE", 5: "RETURN"}


@torch.no_grad()
def denoise_actions(model, batch, scheduler, device, num_steps=100):
    """Run full DDPM denoising to get predicted clean actions."""
    B = batch["obs_qpos"].shape[0]
    H = 12  # pred_horizon
    A = 6   # action_dim

    x = torch.randn(B, H, A, device=device)

    for t in reversed(range(num_steps)):
        timestep = torch.full((B,), t, dtype=torch.long, device=device)
        out = model(batch, noisy_actions=x, timestep=timestep)
        pred_noise = out["pred_noise"]

        alpha_t = scheduler.alphas_cumprod[t]
        alpha_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(device)

        x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

        if t > 0:
            noise = torch.randn_like(x)
            x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise
        else:
            x = x0_pred

    return x


@torch.no_grad()
def single_step_predict(model, batch, scheduler, device):
    """Single-step noise prediction (faster, for loss comparison)."""
    target_actions = batch["target_arm_chunk"]
    B = target_actions.shape[0]

    timesteps = torch.randint(0, scheduler.num_steps, (B,), device=device)
    noise = torch.randn_like(target_actions)
    noisy_actions = scheduler.add_noise(target_actions, noise, timesteps)

    out = model(batch, noisy_actions=noisy_actions, timestep=timesteps)
    return out, noise


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation of v8 policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--index", default="data/interim/episodes_index_test.parquet")
    parser.add_argument("--stats", default="data/interim/stats.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--denoise-samples", type=int, default=5,
                        help="Number of batches to run full denoising (slow)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = HybridPolicyV8(pred_horizon=12, action_dim=6).to(device)
    ckpt_info = load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    logger.info(f"Checkpoint loaded (epoch={ckpt_info.get('epoch', '?')})")

    # Load data
    index_df = pd.read_parquet(args.index)
    ds = AsyncDPv8Dataset(
        data_dir=args.data_dir,
        index_df=index_df,
        stats_path=args.stats,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=v8_collate_fn,
    )
    logger.info(f"Test set: {len(ds)} samples, {len(loader)} batches")

    scheduler = NoiseScheduler(num_steps=100).to(device)

    # ========== Evaluation ==========
    all_phase_true, all_phase_pred = [], []
    all_grip_true, all_grip_pred = [], []
    all_contact_true, all_contact_pred = [], []
    noise_mse_list = []
    latencies = []
    denoise_mse_list = []
    ddim_mse_list = []

    print(f"\n{'='*60}")
    print(f"OFFLINE EVALUATION")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(ds)}")
    print(f"{'='*60}\n")

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # --- Classification + single-step noise prediction ---
        t0 = time.monotonic()
        out, true_noise = single_step_predict(model, batch, scheduler, device)
        dt = (time.monotonic() - t0) * 1000
        latencies.append(dt)

        # Noise MSE
        pred_noise = out["pred_noise"]
        mask = batch.get("mask_valid", None)
        if mask is not None:
            noise_mse = ((pred_noise - true_noise) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum() / pred_noise.shape[-1]
        else:
            noise_mse = ((pred_noise - true_noise) ** 2).mean()
        noise_mse_list.append(noise_mse.item())

        # Phase predictions
        phase_pred = out["phase_logits"].argmax(dim=-1).cpu().numpy()
        phase_true = batch["target_phase_next"].cpu().numpy()
        all_phase_pred.extend(phase_pred.tolist())
        all_phase_true.extend(phase_true.tolist())

        # Grip predictions
        grip_pred = out["grip_logits"].argmax(dim=-1).cpu().numpy()
        grip_true = batch["target_grip_token"].cpu().numpy()
        all_grip_pred.extend(grip_pred.tolist())
        all_grip_true.extend(grip_true.tolist())

        # Contact predictions
        contact_pred = out["contact_logit"].sigmoid().squeeze(-1).cpu().numpy()
        contact_true = batch["target_contact"].cpu().numpy()
        all_contact_pred.extend(contact_pred.tolist())
        all_contact_true.extend(contact_true.tolist())

        # --- Full denoising (slow, only a few batches) ---
        if batch_idx < args.denoise_samples:
            target = batch["target_arm_chunk"]

            # DDPM (100 steps)
            t0 = time.monotonic()
            denoised_ddpm = denoise_actions(model, batch, scheduler, device)
            ddpm_dt = (time.monotonic() - t0) * 1000

            if mask is not None:
                mse_ddpm = ((denoised_ddpm - target) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]
            else:
                mse_ddpm = ((denoised_ddpm - target) ** 2).mean()
            denoise_mse_list.append(mse_ddpm.item())

            # DDIM (10 steps)
            ddim = DDIMSampler(scheduler.alphas_cumprod, num_train_steps=100, num_inference_steps=10).to(device)
            B, H, A = target.shape
            t0 = time.monotonic()
            denoised_ddim = ddim.sample(model, batch, shape=(B, H, A), device=device)
            ddim_dt = (time.monotonic() - t0) * 1000

            if mask is not None:
                mse_ddim = ((denoised_ddim - target) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]
            else:
                mse_ddim = ((denoised_ddim - target) ** 2).mean()
            ddim_mse_list.append(mse_ddim.item())

            if batch_idx == 0:
                print(f"[DDPM 100-step] MSE={mse_ddpm.item():.4f}, latency={ddpm_dt:.0f}ms")
                print(f"[DDIM  10-step] MSE={mse_ddim.item():.4f}, latency={ddim_dt:.0f}ms  ({ddpm_dt/ddim_dt:.1f}x faster)")
                print(f"  DDIM predicted (first 4 steps, all joints):")
                print(f"    {denoised_ddim[0, :4].cpu().numpy().round(3)}")
                print(f"  Ground truth:")
                print(f"    {target[0, :4].cpu().numpy().round(3)}")

    # ========== Results ==========
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")

    # 1. Arm trajectory
    print("[1] ARM DIFFUSION")
    print(f"  Noise prediction MSE: {np.mean(noise_mse_list):.4f}")
    if ddim_mse_list:
        print(f"  DDIM 10-step trajectory MSE: {np.mean(ddim_mse_list):.4f}")
    if denoise_mse_list:
        print(f"  Denoised trajectory MSE: {np.mean(denoise_mse_list):.4f}")

    # 2. Phase classification
    print(f"\n[2] PHASE CLASSIFICATION")
    phase_true_arr = np.array(all_phase_true)
    phase_pred_arr = np.array(all_phase_pred)
    phase_acc = (phase_true_arr == phase_pred_arr).mean()
    print(f"  Overall accuracy: {phase_acc*100:.1f}%")

    present_phases = sorted(set(all_phase_true) | set(all_phase_pred))
    target_names = [PHASE_NAMES.get(p, f"Phase_{p}") for p in present_phases]
    print(classification_report(
        all_phase_true, all_phase_pred,
        labels=present_phases, target_names=target_names,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(all_phase_true, all_phase_pred, labels=present_phases)
    print("  Confusion Matrix:")
    header = "      " + "  ".join(f"{PHASE_NAMES.get(p, str(p)):>6}" for p in present_phases)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {PHASE_NAMES.get(present_phases[i], str(present_phases[i])):>4}: {row_str}")

    # 3. Grip classification
    print(f"\n[3] GRIP CLASSIFICATION")
    grip_true_arr = np.array(all_grip_true)
    grip_pred_arr = np.array(all_grip_pred)
    grip_acc = (grip_true_arr == grip_pred_arr).mean()
    print(f"  Overall accuracy: {grip_acc*100:.1f}%")
    grip_names = ["OPEN", "HOLD", "CLOSE"]
    present_grips = sorted(set(all_grip_true) | set(all_grip_pred))
    print(classification_report(
        all_grip_true, all_grip_pred,
        labels=present_grips,
        target_names=[grip_names[g] if g < 3 else f"Grip_{g}" for g in present_grips],
        zero_division=0,
    ))

    # 4. Contact prediction
    print(f"\n[4] CONTACT PREDICTION")
    contact_true_arr = np.array(all_contact_true)
    contact_pred_arr = np.array(all_contact_pred)
    contact_binary_true = (contact_true_arr > 0.5).astype(int)
    contact_binary_pred = (contact_pred_arr > 0.5).astype(int)
    contact_acc = (contact_binary_true == contact_binary_pred).mean()
    print(f"  Binary accuracy (threshold=0.5): {contact_acc*100:.1f}%")
    print(f"  True positives: {contact_true_arr.sum():.0f}/{len(contact_true_arr)} have contact>0.5")
    print(f"  Predicted positives: {(contact_pred_arr > 0.5).sum()}/{len(contact_pred_arr)}")
    print(f"  Mean predicted score: {contact_pred_arr.mean():.4f}")
    print(f"  Mean true score: {contact_true_arr.mean():.4f}")

    # 5. Latency
    print(f"\n[5] INFERENCE LATENCY")
    print(f"  Single-step (batch={args.batch_size}): "
          f"avg={np.mean(latencies):.1f}ms, p95={np.percentile(latencies, 95):.1f}ms")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Phase accuracy:   {phase_acc*100:.1f}%")
    print(f"  Grip accuracy:    {grip_acc*100:.1f}%")
    print(f"  Contact accuracy: {contact_acc*100:.1f}%")
    print(f"  Noise MSE:        {np.mean(noise_mse_list):.4f}")
    if denoise_mse_list:
        print(f"  DDPM 100-step MSE: {np.mean(denoise_mse_list):.4f}")
    if ddim_mse_list:
        print(f"  DDIM  10-step MSE: {np.mean(ddim_mse_list):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
