"""
Async DP Simulation - Full Pipeline Demo
Runs Diffusion Policy inference and control simulation without real hardware
"""
import sys
import os
import time
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import Config
from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
from src.utils.math_utils import linear_interpolate, apply_ema_filter


def run_simulation(duration_sec=5.0, visualize=True):
    """
    Run full Diffusion Policy simulation

    Args:
        duration_sec: Simulation duration in seconds
        visualize: Whether to plot results after simulation
    """
    print("=" * 60)
    print("  Async DP Simulation")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Sim] Device: {device}")

    # Load model (random weights for demo)
    print(f"[Sim] Loading DiffusionPolicy (action_dim={Config.ACTION_DIM}, obs_dim={Config.OBS_DIM})")
    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)
    model.eval()

    # Check for trained checkpoint
    if os.path.exists(Config.CKPT_PATH):
        print(f"[Sim] Loading checkpoint: {Config.CKPT_PATH}")
        model.load_state_dict(torch.load(Config.CKPT_PATH, map_location=device))
    else:
        print(f"[Sim] No checkpoint found. Using random weights (demo mode).")

    scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=16)

    # Simulation state
    current_obs = np.zeros(Config.OBS_DIM, dtype=np.float32)  # Simulated robot state
    current_action = np.zeros(Config.ACTION_DIM, dtype=np.float32)
    prev_action = None

    # Active trajectory from inference
    active_traj = np.zeros((Config.PRED_HORIZON, Config.ACTION_DIM), dtype=np.float32)
    traj_start_time = 0.0
    last_inference_time = -1.0

    # Timing
    dt_ctrl = 1.0 / Config.CONTROL_FREQ
    dt_inf = 1.0 / Config.INFERENCE_FREQ

    # History for visualization
    history = {
        'time': [],
        'obs': [],           # (T, OBS_DIM)
        'raw_action': [],    # (T, ACTION_DIM)
        'smooth_action': [], # (T, ACTION_DIM)
        'inference_times': [],
    }

    print(f"\n[Sim] Starting simulation ({duration_sec:.1f}s)")
    print(f"[Sim] Control: {Config.CONTROL_FREQ}Hz, Inference: {Config.INFERENCE_FREQ}Hz")
    print("-" * 60)

    sim_start = time.time()
    sim_time = 0.0
    step_count = 0
    inference_count = 0

    while sim_time < duration_sec:
        # === Inference (Brain - 15Hz) ===
        if sim_time - last_inference_time >= dt_inf:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(current_obs).float().to(device).unsqueeze(0)
                noisy_action = torch.randn(1, Config.PRED_HORIZON, Config.ACTION_DIM, device=device)

                scheduler.set_timesteps(16)
                for t in scheduler.timesteps:
                    noise_pred = model(noisy_action, t.unsqueeze(0).to(device), obs_tensor)
                    noise_pred = noise_pred.transpose(1, 2)
                    noisy_action_t = noisy_action.transpose(1, 2)
                    noisy_action_t = scheduler.step(noise_pred, t, noisy_action_t).prev_sample
                    noisy_action = noisy_action_t.transpose(1, 2)

                active_traj = noisy_action.squeeze(0).cpu().numpy()
                active_traj = np.clip(active_traj, -Config.EMERGENCY_STOP_LIMIT, Config.EMERGENCY_STOP_LIMIT)

            traj_start_time = sim_time
            last_inference_time = sim_time
            inference_count += 1
            history['inference_times'].append(sim_time)

            if inference_count % 10 == 0:
                print(f"[Sim] t={sim_time:.2f}s | Inference #{inference_count} | "
                      f"traj[0,0]={active_traj[0,0]:.3f}")

        # === Control (Muscle - 500Hz) ===
        elapsed = sim_time - traj_start_time
        traj_duration = Config.PRED_HORIZON / Config.INFERENCE_FREQ
        alpha = min(elapsed / traj_duration, 1.0)

        traj_progress = alpha * (Config.PRED_HORIZON - 1)
        idx_low = int(traj_progress)
        idx_high = min(idx_low + 1, Config.PRED_HORIZON - 1)
        idx_alpha = traj_progress - idx_low

        # Linear interpolation
        raw_action = linear_interpolate(active_traj[idx_low], active_traj[idx_high], idx_alpha)

        # EMA smoothing
        smooth_action = apply_ema_filter(raw_action, prev_action, alpha=0.3)
        prev_action = smooth_action.copy()

        # Velocity limiting
        if step_count > 0:
            delta = smooth_action - current_obs[:Config.ACTION_DIM]
            max_delta = Config.MAX_JOINT_VEL / Config.CONTROL_FREQ
            delta = np.clip(delta, -max_delta, max_delta)
            smooth_action = current_obs[:Config.ACTION_DIM] + delta

        # Update simulated robot state (simple dynamics)
        current_obs[:Config.ACTION_DIM] = smooth_action
        current_action = smooth_action

        # Log history
        history['time'].append(sim_time)
        history['obs'].append(current_obs.copy())
        history['raw_action'].append(raw_action.copy())
        history['smooth_action'].append(smooth_action.copy())

        # Advance simulation
        sim_time += dt_ctrl
        step_count += 1

    real_elapsed = time.time() - sim_start

    print("-" * 60)
    print(f"[Sim] Simulation complete!")
    print(f"[Sim] Simulated time: {sim_time:.2f}s")
    print(f"[Sim] Real time: {real_elapsed:.2f}s")
    print(f"[Sim] Speed ratio: {sim_time/real_elapsed:.2f}x")
    print(f"[Sim] Control steps: {step_count}")
    print(f"[Sim] Inference calls: {inference_count}")

    # Convert history to numpy
    history['time'] = np.array(history['time'])
    history['obs'] = np.array(history['obs'])
    history['raw_action'] = np.array(history['raw_action'])
    history['smooth_action'] = np.array(history['smooth_action'])
    history['inference_times'] = np.array(history['inference_times'])

    if visualize:
        plot_results(history)

    return history


def plot_results(history, show=False):
    """Plot simulation results"""
    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Sim] matplotlib not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot first 3 joints
    joint_names = ['Joint 0', 'Joint 1', 'Joint 2']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, (name, color) in enumerate(zip(joint_names, colors)):
        ax = axes[i]
        ax.plot(history['time'], history['raw_action'][:, i],
                '--', alpha=0.5, color=color, label='Raw')
        ax.plot(history['time'], history['smooth_action'][:, i],
                '-', color=color, label='Smoothed', linewidth=1.5)

        # Mark inference times
        for t_inf in history['inference_times']:
            ax.axvline(t_inf, color='gray', alpha=0.2, linewidth=0.5)

        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Async DP Simulation - Action Trajectories\n'
                      f'(Control: {Config.CONTROL_FREQ}Hz, Inference: {Config.INFERENCE_FREQ}Hz)')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'simulation_result.png')
    plt.savefig(output_path, dpi=150)
    print(f"[Sim] Plot saved to: {os.path.abspath(output_path)}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Async DP Simulation")
    parser.add_argument('--duration', type=float, default=3.0, help='Simulation duration (seconds)')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    run_simulation(duration_sec=args.duration, visualize=not args.no_viz)
