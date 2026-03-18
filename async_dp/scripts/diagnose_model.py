"""
Diagnose trained model: compare predictions vs actual actions from training data.
No hardware required — runs on saved episodes.

Usage:
    uv run python scripts/diagnose_model.py --checkpoint checkpoints/vision_policy/best.pth --data-dir episodes/20260310_141707
"""
import numpy as np
import torch
import json
import h5py
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vision_policy import VisionDiffusionPolicy
from src.models.scheduler import get_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-tests", type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VisionDiffusionPolicy(action_dim=7, qpos_dim=7).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded, val_loss={ckpt.get('val_loss', '?')}")

    # Load stats
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "dataset_stats.json")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    for k in stats:
        for sk in stats[k]:
            stats[k][sk] = np.array(stats[k][sk], dtype=np.float32)

    print(f"\nDataset stats:")
    print(f"  qpos  min: [{', '.join(f'{v:.3f}' for v in stats['qpos']['min'])}]")
    print(f"  qpos  max: [{', '.join(f'{v:.3f}' for v in stats['qpos']['max'])}]")
    print(f"  action min: [{', '.join(f'{v:.3f}' for v in stats['action']['min'])}]")
    print(f"  action max: [{', '.join(f'{v:.3f}' for v in stats['action']['max'])}]")

    # Normalization helpers
    def normalize_qpos(qpos):
        mn, mx = stats['qpos']['min'], stats['qpos']['max']
        rng = mx - mn
        rng[rng < 1e-6] = 1.0
        return (qpos - mn) / rng * 2 - 1

    def unnormalize_action(action):
        mn, mx = stats['action']['min'], stats['action']['max']
        return (action + 1) / 2 * (mx - mn) + mn

    scheduler = get_scheduler('ddim', num_train_timesteps=100)

    # Pick episodes to test
    episode_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.hdf5')])
    test_eps = episode_files[:args.num_tests]

    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS: Predict vs Actual (first step of 16-step trajectory)")
    print(f"{'='*70}")

    total_errors = []

    for ep_file in test_eps:
        ep_path = os.path.join(args.data_dir, ep_file)
        with h5py.File(ep_path, 'r') as f:
            qpos_all = f['observations/qpos'][:].astype(np.float32)
            action_all = f['action'][:].astype(np.float32)
            images = f['observations/images']

            T = len(qpos_all)
            # Test at 25%, 50%, 75% of episode
            test_frames = [T // 4, T // 2, 3 * T // 4]

            print(f"\n--- {ep_file} ({T} frames) ---")

            for t in test_frames:
                if t + 16 > T:
                    continue

                img = images[t]  # (224, 224, 3) uint8
                qpos = qpos_all[t]
                actual_action = action_all[t:t+16]  # (16, 7) ground truth

                qpos_norm = normalize_qpos(qpos)

                # Predict
                pred_norm = model.get_action(img, qpos_norm, scheduler,
                                              num_inference_steps=16, device=device)
                pred_action = unnormalize_action(pred_norm)

                # Compare first action step
                err_deg = np.abs(np.rad2deg(pred_action[0] - actual_action[0]))
                mean_err = np.mean(err_deg)
                max_err = np.max(err_deg)
                total_errors.append(mean_err)

                print(f"  Frame {t}:")
                print(f"    qpos   (deg): [{', '.join(f'{np.rad2deg(v):+6.1f}' for v in qpos)}]")
                print(f"    actual (deg): [{', '.join(f'{np.rad2deg(v):+6.1f}' for v in actual_action[0])}]")
                print(f"    pred   (deg): [{', '.join(f'{np.rad2deg(v):+6.1f}' for v in pred_action[0])}]")
                print(f"    error  (deg): [{', '.join(f'{v:6.1f}' for v in err_deg)}]  mean={mean_err:.1f}  max={max_err:.1f}")

                # Also check trajectory coherence (are predictions smooth?)
                pred_deltas = np.diff(pred_action, axis=0)
                max_jump = np.max(np.abs(np.rad2deg(pred_deltas)))
                print(f"    traj max jump: {max_jump:.1f} deg/step")

    print(f"\n{'='*70}")
    overall_mean = np.mean(total_errors)
    print(f"  Overall mean error: {overall_mean:.1f} deg")
    if overall_mean < 5:
        print(f"  → Good: model predictions are close to ground truth")
    elif overall_mean < 15:
        print(f"  → Moderate: model learned something but predictions are rough")
    elif overall_mean < 30:
        print(f"  → Poor: model barely learned the task, needs more data or training")
    else:
        print(f"  → Bad: model did not learn the task at all")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
