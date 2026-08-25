"""Central constants for async_dp_v8."""

# Control frequencies
CONTROL_FREQ = 500
INFERENCE_FREQ = 15

# Horizons
OBS_HORIZON = 3
PRED_HORIZON = 12
EXECUTE_HORIZON = 4

# Dimensions
ARM_ACTION_DIM = 6
GRIPPER_DIM = 1
NUM_JOINTS = 6
NUM_MOTORS = 9
DOF_TOTAL = 7

# Phase IDs
PHASE_REACH = 0
PHASE_ALIGN = 1
PHASE_CLOSE = 2
PHASE_LIFT = 3
PHASE_PLACE = 4
PHASE_RETURN = 5
NUM_PHASES = 6

# Gripper tokens
GRIP_OPEN = 0
GRIP_HOLD = 1
GRIP_CLOSE = 2
NUM_GRIP_TOKENS = 3

# Image sizes
IMAGE_SIZE = (224, 224)
CROP_SIZE = (96, 96)

# Dynamixel
DXL_BAUDRATE = 1_000_000
FOLLOWER_PORT = "/dev/ttyDXL_puppet_right"
LEADER_PORT = "/dev/ttyDXL_master_right"

# Gripper calibration (raw Dynamixel units)
LEADER_GRIP_MIN = 1545
LEADER_GRIP_MAX = 2187
FOLLOWER_GRIP_MIN = 1050
FOLLOWER_GRIP_MAX = 1965

# Joint map: motor IDs -> logical joints
JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]

# Home position (median of episode start positions from data collection)
import numpy as np
HOME_QPOS = np.array([0.0, 1.71, -1.62, 0.0, 0.23, -0.04])
HOME_TOLERANCE_RAD = 0.35  # generous tolerance for wrist joints
