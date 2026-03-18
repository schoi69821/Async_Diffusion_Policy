"""
Async DP Core Engine - Dual Process Architecture
Brain (15Hz): Diffusion Policy Inference
Muscle (500Hz): Real-time Robot Control with Interpolation
"""
import multiprocessing
import time
import numpy as np
import torch
import os

from config.settings import Config
from src.core.shared_mem import SharedMemoryManager
from src.drivers.robot_driver import RobotDriver
from src.utils.math_utils import linear_interpolate, apply_ema_filter


def load_diffusion_model(device):
    """Load trained Diffusion Policy model"""
    from src.models.diffusion_net import DiffusionPolicy
    from src.models.scheduler import get_scheduler

    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)

    if os.path.exists(Config.CKPT_PATH):
        print(f"[Inference] Loading checkpoint: {Config.CKPT_PATH}")
        model.load_state_dict(torch.load(Config.CKPT_PATH, map_location=device))
    else:
        print(f"[Inference] No checkpoint found. Using random weights.")

    model.eval()
    scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=16)
    return model, scheduler


@torch.no_grad()
def diffusion_inference(model, scheduler, obs, device):
    """
    Run Diffusion Policy inference (DDIM denoising)

    Args:
        model: DiffusionPolicy network
        scheduler: DDIM scheduler
        obs: Current observation (OBS_DIM,)
        device: torch device

    Returns:
        action_traj: Predicted action trajectory (PRED_HORIZON, ACTION_DIM)
    """
    batch_size = 1

    # Prepare observation conditioning
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # (1, OBS_DIM)

    # Initialize with random noise
    noisy_action = torch.randn(
        (batch_size, Config.PRED_HORIZON, Config.ACTION_DIM),
        device=device
    )

    # DDIM Denoising Loop
    scheduler.set_timesteps(16)
    for t in scheduler.timesteps:
        # Predict noise
        noise_pred = model(noisy_action, t.unsqueeze(0).to(device), obs_tensor)
        noise_pred = noise_pred.transpose(1, 2)  # (B, Dim, Horizon) -> for scheduler
        noisy_action_t = noisy_action.transpose(1, 2)

        # DDIM step
        noisy_action_t = scheduler.step(noise_pred, t, noisy_action_t).prev_sample
        noisy_action = noisy_action_t.transpose(1, 2)  # Back to (B, Horizon, Dim)

    # Clamp to safety limits
    action_traj = noisy_action.squeeze(0).cpu().numpy()
    action_traj = np.clip(action_traj, -Config.EMERGENCY_STOP_LIMIT, Config.EMERGENCY_STOP_LIMIT)

    return action_traj


def run_inference_process(lock, stop_event):
    """
    Brain Process: Runs Diffusion Policy at 15Hz
    Reads observation from shared memory, writes action trajectory
    """
    print("[Inference] Process Started.")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Device: {device}")

    # Load model
    model, scheduler = load_diffusion_model(device)

    # Connect to shared memory
    sm = SharedMemoryManager(create=False)

    inference_count = 0
    try:
        while not stop_event.is_set():
            loop_start = time.time()

            # Read current observation
            with lock:
                obs = sm.obs_state.copy()

            # Run Diffusion inference
            action_traj = diffusion_inference(model, scheduler, obs, device)

            # Write action trajectory to shared memory
            with lock:
                np.copyto(sm.action_traj, action_traj)
                sm.update_counter[0] += 1

            inference_count += 1
            if inference_count % 100 == 0:
                print(f"[Inference] Count: {inference_count}, Obs[0]: {obs[0]:.3f}")

            # Maintain 15Hz
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / Config.INFERENCE_FREQ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[Inference] Error: {e}")
    finally:
        sm.close()
        print("[Inference] Process Stopped.")


def run_control_process(lock, stop_event):
    """
    Muscle Process: Runs at 500Hz
    Interpolates between trajectory points and applies EMA smoothing
    """
    print("[Control] Process Started.")

    # Connect to shared memory and robot
    sm = SharedMemoryManager(create=False)
    robot = RobotDriver(mode='dummy')  # Change to 'real' for hardware

    # Local state
    local_traj = np.zeros((Config.PRED_HORIZON, Config.ACTION_DIM))
    last_counter = -1
    traj_start_time = time.time()
    prev_action = None

    # EMA smoothing coefficient (higher = more responsive, lower = smoother)
    ema_alpha = 0.3

    control_count = 0
    try:
        while not stop_event.is_set():
            loop_start = time.time()

            # Read current robot state
            current_qpos = robot.get_qpos()

            # Update shared memory with current observation
            with lock:
                np.copyto(sm.obs_state, current_qpos)

                # Check for new trajectory from inference process
                if sm.update_counter[0] > last_counter:
                    np.copyto(local_traj, sm.action_traj)
                    last_counter = sm.update_counter[0]
                    traj_start_time = time.time()

            # Calculate interpolation alpha based on time elapsed
            elapsed = time.time() - traj_start_time
            traj_duration = Config.PRED_HORIZON / Config.INFERENCE_FREQ  # ~1.07s for 16 steps at 15Hz
            alpha = min(elapsed / traj_duration, 1.0)

            # Calculate trajectory index (which two points to interpolate between)
            traj_progress = alpha * (Config.PRED_HORIZON - 1)
            idx_low = int(traj_progress)
            idx_high = min(idx_low + 1, Config.PRED_HORIZON - 1)
            idx_alpha = traj_progress - idx_low

            # Linear interpolation between trajectory points
            target_action = linear_interpolate(
                local_traj[idx_low],
                local_traj[idx_high],
                idx_alpha
            )

            # Apply EMA smoothing for jerk reduction
            smoothed_action = apply_ema_filter(target_action, prev_action, ema_alpha)
            prev_action = smoothed_action

            # Velocity limiting for safety
            if prev_action is not None:
                delta = smoothed_action - current_qpos
                delta = np.clip(delta, -Config.MAX_JOINT_VEL / Config.CONTROL_FREQ,
                               Config.MAX_JOINT_VEL / Config.CONTROL_FREQ)
                smoothed_action = current_qpos + delta

            # Send command to robot
            robot.set_action(smoothed_action)

            control_count += 1
            if control_count % 5000 == 0:  # Every 10 seconds at 500Hz
                print(f"[Control] Count: {control_count}, Alpha: {alpha:.2f}, Idx: {idx_low}")

            # Maintain 500Hz
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / Config.CONTROL_FREQ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[Control] Error: {e}")
    finally:
        robot.close()
        sm.close()
        print("[Control] Process Stopped.")


class AsyncController:
    """
    Main controller that manages Brain and Muscle processes
    """
    def __init__(self):
        self.stop_event = None
        self.lock = None
        self.root_sm = None
        self.p_inf = None
        self.p_ctrl = None

    def start(self):
        """Start both inference and control processes"""
        print("=" * 50)
        print("  Async DP Controller Starting...")
        print("=" * 50)

        self.stop_event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()

        # Create shared memory (cleanup old if exists)
        try:
            old_sm = SharedMemoryManager(create=False)
            old_sm.shm.unlink()
            old_sm.close()
        except:
            pass

        self.root_sm = SharedMemoryManager(create=True)

        # Start processes
        self.p_inf = multiprocessing.Process(
            target=run_inference_process,
            args=(self.lock, self.stop_event),
            name="Brain-15Hz"
        )
        self.p_ctrl = multiprocessing.Process(
            target=run_control_process,
            args=(self.lock, self.stop_event),
            name="Muscle-500Hz"
        )

        self.p_inf.start()
        self.p_ctrl.start()

        print(f"[Main] Inference PID: {self.p_inf.pid}")
        print(f"[Main] Control PID: {self.p_ctrl.pid}")
        print("=" * 50)

    def stop(self):
        """Stop all processes gracefully"""
        print("\n[Main] Stopping controller...")
        self.stop_event.set()

        self.p_inf.join(timeout=2.0)
        self.p_ctrl.join(timeout=2.0)

        if self.p_inf.is_alive():
            self.p_inf.terminate()
        if self.p_ctrl.is_alive():
            self.p_ctrl.terminate()

        self.root_sm.close()
        print("[Main] Controller stopped.")

    def is_running(self):
        """Check if processes are still running"""
        return self.p_inf.is_alive() and self.p_ctrl.is_alive()
