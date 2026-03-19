"""Policy runner with state machine for inference."""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from .state_machine import State, RuntimeContext
from async_dp_v8.control.chunk_blender import ChunkBlender
from async_dp_v8.types import InferenceConfig

logger = logging.getLogger(__name__)


class PolicyRunnerV8:
    def __init__(self, model, robot, cfg: InferenceConfig):
        self.model = model
        self.robot = robot
        self.cfg = cfg
        self.ctx = RuntimeContext()
        self.chunk_blender = ChunkBlender(
            execute_horizon=cfg.execute_horizon,
            alpha=0.6,
        )
        self._noise_scheduler = None

    def set_noise_scheduler(self, scheduler):
        self._noise_scheduler = scheduler

    @torch.no_grad()
    def step(self, obs_batch: Dict[str, torch.Tensor]) -> Tuple[dict, dict]:
        self.ctx.total_steps += 1

        # Run model inference
        B = obs_batch["obs_qpos"].shape[0]
        device = obs_batch["obs_qpos"].device
        H = self.cfg.pred_horizon
        A = 6

        # Initialize with noise
        noisy_actions = torch.randn(B, H, A, device=device)

        # Denoise iteratively if scheduler is available
        if self._noise_scheduler is not None:
            arm_chunk = self._denoise(obs_batch, noisy_actions)
        else:
            timestep = torch.zeros(B, dtype=torch.long, device=device)
            out = self.model(obs_batch, noisy_actions=noisy_actions, timestep=timestep)
            arm_chunk = noisy_actions - out.get("pred_noise", noisy_actions)

        # Get classification outputs
        out = self.model(obs_batch)
        phase_prob = out["phase_logits"].softmax(dim=-1)
        grip_prob = out["grip_logits"].softmax(dim=-1)
        contact_prob = out["contact_logit"].sigmoid()

        # Blend chunks
        arm_np = arm_chunk[0].cpu().numpy()
        arm_np = self.chunk_blender.blend(arm_np)

        # State machine transition
        self._transition(
            phase_prob=phase_prob,
            grip_prob=grip_prob,
            contact_prob=contact_prob,
        )

        # Build command
        cmd = self._build_command(
            arm_chunk=arm_np,
            grip_prob=grip_prob,
            contact_prob=contact_prob,
        )
        self.robot.send_command(cmd)

        info = {
            "state": self.ctx.state.name,
            "phase_prob": phase_prob[0].cpu().numpy(),
            "grip_prob": grip_prob[0].cpu().numpy(),
            "contact_prob": float(contact_prob[0].cpu()),
            "step": self.ctx.total_steps,
        }
        return cmd, info

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

            # Predict x0
            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

            if t > 0:
                noise = torch.randn_like(x)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise
            else:
                x = x0_pred

        return x

    def _transition(self, phase_prob, grip_prob, contact_prob):
        phase = int(phase_prob.argmax(dim=-1).item())
        grip_close_p = float(grip_prob[..., 2].item())
        grip_open_p = float(grip_prob[..., 0].item())
        contact_p = float(contact_prob.item())

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
            if phase == 4:  # PLACE phase detected
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

    def _build_command(self, arm_chunk, grip_prob, contact_prob) -> dict:
        if self.ctx.state in {State.REACH, State.ALIGN, State.PLACE, State.RETURN}:
            return {
                "arm_actions": arm_chunk[:self.cfg.execute_horizon],
                "gripper": "hold",
            }
        elif self.ctx.state == State.LIFT:
            return {
                "primitive": "lift_z",
                "delta_z_mm": self.cfg.lift_delta_z_mm,
                "gripper": "hold",
            }
        elif self.ctx.state == State.CLOSE_COMMIT:
            return {
                "arm_actions": None,
                "gripper": "close",
            }
        elif self.ctx.state == State.RELEASE:
            return {
                "arm_actions": None,
                "gripper": "open",
            }
        else:
            return {
                "arm_actions": None,
                "gripper": "hold",
            }

    def reset(self):
        self.ctx = RuntimeContext()
        self.chunk_blender.reset()
