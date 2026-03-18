import numpy as np
import os

class Config:
    PROJECT_NAME = "Async DP Wafer Inspector"
    SHM_NAME = "asyncdp_shm_v1"
    
    # Hardware Config
    ROBOT_MODEL = 'vx300s'
    CONTROL_FREQ = 500  # Hz (Muscle)
    GRIPPER_FORCE = 150 # mA

    # Serial ports — fixed by udev symlinks (serial number based, reboot-safe)
    # FT94EN6H = Follower (execution/puppet arm)
    # FT94ENC4 = Leader (teaching/master arm)
    PORT_FOLLOWER = "/dev/ttyDXL_puppet_right"
    PORT_LEADER   = "/dev/ttyDXL_master_right"
    
    # AI Model Config
    INFERENCE_FREQ = 15 # Hz (Brain)
    PRED_HORIZON = 16   # Steps
    OBS_HORIZON = 2     # Steps
    ACTION_DIM = 14     # 7 DOF x 2 Arms (or 6+1 gripper)
    OBS_DIM = 14
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "assets/data")
    CKPT_PATH = os.path.join(BASE_DIR, "assets/checkpoints/best_model.pth")
    
    # Home Position (7 joints: waist, shoulder, elbow, forearm, wrist_angle, wrist_rot, gripper)
    # Resting pose on table
    HOME_FOLLOWER = np.array([+0.0199, +1.6682, -1.6176, -0.0430, -0.1319, -0.0568, -0.4065], dtype=np.float32)
    HOME_LEADER   = np.array([-0.0476, +1.6974, -1.6230, -0.0522, +0.7869, +0.0460, -1.2563], dtype=np.float32)

    # Gripper Calibration (raw Dynamixel position units)
    # Measured by scripts/calibrate_gripper.py
    GRIPPER_LEADER_MIN   = 1545  # closed (raw)
    GRIPPER_LEADER_MAX   = 2187  # open (raw)
    GRIPPER_FOLLOWER_MIN = 1050  # closed (raw) — commands past physical limit for firm grip
    GRIPPER_FOLLOWER_MAX = 1965  # open (raw)

    # Mid Position (task-ready pose, arm raised)
    MID_FOLLOWER = np.array([-0.1227, +0.7148, -0.2094, +0.0828, +0.5891, -0.0261, -1.1766], dtype=np.float32)

    # Safety Limits
    MAX_JOINT_VEL = 2.0
    EMERGENCY_STOP_LIMIT = 3.1