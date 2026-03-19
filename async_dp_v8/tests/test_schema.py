"""Test dataset schema definitions."""
import torch
from async_dp_v8.types import FrameRecord, PolicySample, InferenceConfig
from async_dp_v8.constants import NUM_PHASES, NUM_GRIP_TOKENS


def test_frame_record_creation():
    fr = FrameRecord(
        episode_id="ep_001",
        frame_idx=0,
        timestamp=0.0,
        image_wrist=torch.zeros(3, 224, 224),
        image_wrist_crop=torch.zeros(3, 96, 96),
        qpos=torch.zeros(6),
        qvel=torch.zeros(6),
        ee_pos=torch.zeros(3),
        ee_quat=torch.zeros(4),
        gripper_pos=torch.zeros(1),
        gripper_vel=torch.zeros(1),
        present_current=torch.zeros(7),
        present_pwm=torch.zeros(7),
        present_voltage=torch.zeros(1),
        phase=0,
        grip_token=0,
        contact_soft=0.0,
        contact_hard=0,
        success_liftable=0,
        future_arm_delta_ee=torch.zeros(12, 7),
        future_arm_delta_q=torch.zeros(12, 6),
    )
    assert fr.episode_id == "ep_001"
    assert fr.qpos.shape == (6,)


def test_policy_sample_creation():
    ps = PolicySample(
        obs_image_wrist=torch.zeros(3, 3, 224, 224),
        obs_image_crop=torch.zeros(3, 3, 96, 96),
        obs_qpos=torch.zeros(3, 6),
        obs_qvel=torch.zeros(3, 6),
        obs_ee_pose=torch.zeros(3, 7),
        obs_gripper=torch.zeros(3, 2),
        obs_current=torch.zeros(3, 7),
        obs_pwm=torch.zeros(3, 7),
        phase_curr=torch.tensor([0]),
        contact_curr=torch.tensor([0.0]),
        target_phase_next=torch.tensor([1]),
        target_grip_token=torch.tensor([0]),
        target_contact=torch.tensor([0.0]),
        target_arm_chunk=torch.zeros(12, 6),
    )
    assert ps.obs_qpos.shape == (3, 6)


def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.obs_horizon == 3
    assert cfg.pred_horizon == 12
    assert cfg.execute_horizon == 4
    assert cfg.close_prob_thresh == 0.80


def test_constants():
    assert NUM_PHASES == 6
    assert NUM_GRIP_TOKENS == 3
