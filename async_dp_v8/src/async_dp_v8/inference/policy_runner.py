"""Policy runner with state machine for inference."""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from .state_machine import State, RuntimeContext
from async_dp_v8.control.chunk_blender import ChunkBlender
from async_dp_v8.types import InferenceConfig

logger = logging.getLogger(__name__)

# Max steps before timeout per state
STATE_TIMEOUTS = {
    State.SAFE_HOME: 200,
    State.REACH: 500,
    State.ALIGN: 300,
    State.CLOSE_ARMED: 100,
    State.CLOSE_COMMIT: 30,
    State.LIFT: 200,
    State.PLACE: 300,
    State.RELEASE: 100,
    State.RETURN: 500,
    State.RECOVERY: 300,
}


class PolicyRunnerV8:
    def __init__(self, model, robot, cfg: InferenceConfig, safety_guard=None):
        self.model = model
        self.robot = robot
        self.cfg = cfg
        self.safety = safety_guard
        self.ctx = RuntimeContext()
        self.chunk_blender = ChunkBlender(
            execute_horizon=cfg.execute_horizon,
            alpha=0.6,
        )
        self._noise_scheduler = None
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_step: int = 0

    def set_noise_scheduler(self, scheduler):
        self._noise_scheduler = scheduler

    @torch.no_grad()
    def step(self, obs_batch: Dict[str, torch.Tensor]) -> Tuple[dict, dict]:
        """Execute one policy step. Returns (command, info)."""
        self.ctx.total_steps += 1

        # Check for E_STOP
        if self.ctx.state == State.E_STOP:
            return {"arm_actions": None, "gripper": "hold"}, {"state": "E_STOP"}

        # Check state timeout
        self._check_timeout()

        # Replan if needed (every execute_horizon steps or no chunk)
        need_replan = (
            self._current_chunk is None
            or self._chunk_step >= self.cfg.execute_horizon
        )

        if need_replan and self.ctx.state not in {
            State.BOOT, State.SAFE_HOME, State.CLOSE_COMMIT,
            State.DONE, State.E_STOP, State.RECOVERY,
        }:
            self._replan(obs_batch)

        # Get classification outputs for state machine
        try:
            out = self.model(obs_batch)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            self._trigger_estop(f"Model inference failed: {e}")
            return {"arm_actions": None, "gripper": "hold"}, {"state": "E_STOP"}

        phase_prob = out["phase_logits"].softmax(dim=-1)
        grip_prob = out["grip_logits"].softmax(dim=-1)
        contact_prob = out["contact_logit"].sigmoid()

        # State machine transition
        self._transition(
            phase_prob=phase_prob,
            grip_prob=grip_prob,
            contact_prob=contact_prob,
        )

        # Build and send command
        cmd = self._build_command(grip_prob, contact_prob)

        info = {
            "state": self.ctx.state.name,
            "phase_prob": phase_prob[0].cpu().numpy() if phase_prob.dim() > 1 else phase_prob.cpu().numpy(),
            "grip_prob": grip_prob[0].cpu().numpy() if grip_prob.dim() > 1 else grip_prob.cpu().numpy(),
            "contact_prob": float(contact_prob.flatten()[0].cpu()),
            "step": self.ctx.total_steps,
            "chunk_step": self._chunk_step,
        }
        return cmd, info

    def _replan(self, obs_batch):
        """Run diffusion denoising and update action chunk."""
        B = obs_batch["obs_qpos"].shape[0]
        device = obs_batch["obs_qpos"].device
        H = self.cfg.pred_horizon
        A = 6

        noisy_actions = torch.randn(B, H, A, device=device)

        if self._noise_scheduler is not None:
            arm_chunk = self._denoise(obs_batch, noisy_actions)
        else:
            timestep = torch.zeros(B, dtype=torch.long, device=device)
            out = self.model(obs_batch, noisy_actions=noisy_actions, timestep=timestep)
            arm_chunk = noisy_actions - out.get("pred_noise", noisy_actions)

        arm_np = arm_chunk[0].cpu().numpy()
        arm_np = self.chunk_blender.blend(arm_np)
        self._current_chunk = arm_np
        self._chunk_step = 0

    def _denoise(self, obs_batch, noisy_actions):
        """DDPM-style iterative denoising."""
        x = noisy_actions
        for t in reversed(range(self._noise_scheduler.num_steps)):
            timestep = torch.full(
                (x.shape[0],), t,
                dtype=torch.long, device=x.device,
            )
            out = self.model(obs_batch, noisy_actions=x, timestep=timestep)
            pred_noise = out["pred_noise"]

            alpha_t = self._noise_scheduler.alphas_cumprod[t]
            alpha_prev = self._noise_scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

            if t > 0:
                noise = torch.randn_like(x)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise
            else:
                x = x0_pred

        return x

    def _check_timeout(self):
        """Check if current state has timed out."""
        timeout = STATE_TIMEOUTS.get(self.ctx.state, None)
        if timeout is not None and self.ctx.steps_in_state() > timeout:
            logger.warning(
                f"Timeout in state {self.ctx.state.name} "
                f"after {self.ctx.steps_in_state()} steps"
            )
            if self.ctx.state == State.RECOVERY:
                self._trigger_estop(f"Recovery timeout after {self.ctx.steps_in_state()} steps")
            else:
                self.ctx.transition(State.RECOVERY)

    def _trigger_estop(self, reason: str):
        """Trigger emergency stop."""
        logger.critical(f"E_STOP: {reason}")
        self.ctx.transition(State.E_STOP)
        if self.safety:
            self.safety.trigger_estop(reason)

    def _transition(self, phase_prob, grip_prob, contact_prob):
        phase = int(phase_prob.argmax(dim=-1).flatten()[0].item())
        grip_close_p = float(grip_prob.flatten()[2].item()) if grip_prob.numel() > 2 else 0.0
        grip_open_p = float(grip_prob.flatten()[0].item()) if grip_prob.numel() > 0 else 0.0
        contact_p = float(contact_prob.flatten()[0].item())

        if self.ctx.state == State.BOOT:
            self.ctx.transition(State.SAFE_HOME)

        elif self.ctx.state == State.SAFE_HOME:
            if self.robot.is_at_home():
                self.ctx.transition(State.REACH)

        elif self.ctx.state == State.REACH:
            if phase == 1:  # ALIGN
                self.ctx.transition(State.ALIGN)

        elif self.ctx.state == State.ALIGN:
            if grip_close_p > self.cfg.close_prob_thresh:
                self.ctx.close_count += 1
            else:
                self.ctx.close_count = 0

            if self.ctx.close_count >= self.cfg.grip_commit_consecutive:
                self.ctx.transition(State.CLOSE_ARMED)
                self.ctx.close_count = 0

        elif self.ctx.state == State.CLOSE_ARMED:
            if contact_p > self.cfg.contact_prob_thresh or grip_close_p > self.cfg.close_prob_thresh:
                self.ctx.transition(State.CLOSE_COMMIT)
            elif grip_close_p < 0.3:
                self.ctx.close_count = 0
                self.ctx.transition(State.ALIGN)

        elif self.ctx.state == State.CLOSE_COMMIT:
            self.robot.close_gripper()
            if self.robot.gripper_contact_confirmed():
                self.ctx.has_object = True
                self.ctx.transition(State.LIFT)
            elif self.ctx.steps_in_state() > 10:
                self.ctx.transition(State.RECOVERY)

        elif self.ctx.state == State.LIFT:
            if self.robot.ee_z_above(self.cfg.lift_target_z_mm):
                self.ctx.transition(State.PLACE)

        elif self.ctx.state == State.PLACE:
            if phase == 4:  # PLACE phase
                self.ctx.transition(State.RELEASE)

        elif self.ctx.state == State.RELEASE:
            if grip_open_p > self.cfg.open_prob_thresh:
                self.ctx.open_count += 1
            else:
                self.ctx.open_count = 0

            if self.ctx.open_count >= self.cfg.grip_commit_consecutive:
                self.robot.open_gripper()
                self.ctx.has_object = False
                self.ctx.transition(State.RETURN)

        elif self.ctx.state == State.RETURN:
            if self.robot.is_at_home():
                self.ctx.transition(State.DONE)

        elif self.ctx.state == State.RECOVERY:
            # Try to go home safely
            try:
                if hasattr(self.robot, 'safe_go_home'):
                    self.robot.safe_go_home()
                self.robot.open_gripper()
                self.ctx.has_object = False
                self.ctx.transition(State.SAFE_HOME)
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                self._trigger_estop(f"Recovery failed: {e}")

    def _build_command(self, grip_prob, contact_prob) -> dict:
        if self.ctx.state in {State.REACH, State.ALIGN, State.PLACE, State.RETURN}:
            action = self._get_next_action()
            if action is not None:
                return {"arm_actions": action, "gripper": "hold"}
            return {"arm_actions": None, "gripper": "hold"}

        elif self.ctx.state == State.LIFT:
            return {
                "primitive": "lift_z",
                "delta_z_mm": self.cfg.lift_delta_z_mm,
                "gripper": "hold",
            }
        elif self.ctx.state == State.CLOSE_COMMIT:
            return {"arm_actions": None, "gripper": "close"}
        elif self.ctx.state == State.RELEASE:
            return {"arm_actions": None, "gripper": "open"}
        else:
            return {"arm_actions": None, "gripper": "hold"}

    def _get_next_action(self) -> Optional[np.ndarray]:
        """Get the next single action from current chunk."""
        if self._current_chunk is None:
            return None
        if self._chunk_step >= len(self._current_chunk):
            return None
        action = self._current_chunk[self._chunk_step]
        self._chunk_step += 1
        return action

    def reset(self):
        self.ctx = RuntimeContext()
        self.chunk_blender.reset()
        self._current_chunk = None
        self._chunk_step = 0
