"""Type definitions for async_dp_v8."""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

import torch
import numpy as np

PhaseName = Literal["reach", "align", "close", "lift", "place", "return"]
GripToken = Literal["open", "hold", "close"]


@dataclass
class FrameRecord:
    """A single recorded frame from a demonstration episode."""

    episode_id: str
    frame_idx: int
    timestamp: float
    image_wrist: torch.Tensor
    image_wrist_crop: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    ee_pos: torch.Tensor
    ee_quat: torch.Tensor
    gripper_pos: torch.Tensor
    gripper_vel: torch.Tensor
    present_current: torch.Tensor
    present_pwm: torch.Tensor
    present_voltage: torch.Tensor
    phase: int
    grip_token: int
    contact_soft: float
    contact_hard: int
    success_liftable: int
    future_arm_delta_ee: torch.Tensor
    future_arm_delta_q: torch.Tensor


@dataclass
class PolicySample:
    """A training sample for the diffusion policy."""

    obs_image_wrist: torch.Tensor
    obs_image_crop: torch.Tensor
    obs_qpos: torch.Tensor
    obs_qvel: torch.Tensor
    obs_ee_pose: torch.Tensor
    obs_gripper: torch.Tensor
    obs_current: torch.Tensor
    obs_pwm: torch.Tensor
    phase_curr: torch.Tensor
    contact_curr: torch.Tensor
    target_phase_next: torch.Tensor
    target_grip_token: torch.Tensor
    target_contact: torch.Tensor
    target_arm_chunk: torch.Tensor
    target_keypose: Optional[torch.Tensor] = None
    mask_valid: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    meta_episode_id: str = ""
    meta_frame_idx: int = 0


@dataclass
class RobotCommand:
    """A command to send to the robot."""

    arm_actions: Optional[np.ndarray] = None
    gripper: str = "hold"  # "open", "hold", "close"
    primitive: Optional[str] = None
    delta_z_mm: float = 0.0


@dataclass
class InferenceConfig:
    """Configuration for inference-time behavior."""

    obs_horizon: int = 3
    pred_horizon: int = 12
    execute_horizon: int = 4
    control_hz: int = 15
    close_prob_thresh: float = 0.80
    open_prob_thresh: float = 0.80
    contact_prob_thresh: float = 0.55
    contact_current_delta_thresh: float = 35.0
    grip_commit_consecutive: int = 2
    lift_delta_z_mm: float = 20.0
    lift_target_z_mm: float = 35.0
    max_joint_step_rad: float = 0.12
