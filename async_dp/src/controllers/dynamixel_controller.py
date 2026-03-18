"""
Dynamixel Robot Controller
Ready-to-use controller for Dynamixel-based robots (VX300s, etc.)
"""
import numpy as np
import time
import logging
from typing import Optional, List
from dataclasses import dataclass, field

from src.controllers.base_controller import BaseRobotController, ControllerConfig
from src.interfaces import InterfaceConfig

logger = logging.getLogger(__name__)

# Try to import Dynamixel SDK
try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler,
        GroupSyncRead, GroupSyncWrite,
        COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE,
        DXL_LOWORD, DXL_HIWORD
    )
    DYNAMIXEL_AVAILABLE = True
except ImportError:
    DYNAMIXEL_AVAILABLE = False
    logger.warning("[Dynamixel] SDK not installed. Install with: pip install dynamixel-sdk")


# Dynamixel Protocol 2.0 Control Table Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108

# Data lengths
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4

# Protocol version
PROTOCOL_VERSION = 2.0


@dataclass
class DynamixelConfig(ControllerConfig):
    """Dynamixel-specific configuration"""
    # Port settings
    port: str = "/dev/ttyDXL_puppet_right"  # udev symlink (reboot-safe)
    baudrate: int = 1000000     # 1Mbps

    # Motor IDs (flat list of all physical motors, auto-derived from joint_map if set)
    motor_ids: List[int] = field(default_factory=lambda: list(range(1, 8)))  # [1,2,3,4,5,6,7]

    # Joint mapping: each entry is a list of motor IDs forming one logical joint.
    # Dual-motor joints (e.g. [2,3]) are synchronized: reads are averaged, writes are duplicated.
    # Example VX300s: [[1], [2,3], [4,5], [6], [7], [8], [9]]
    #   -> 7 logical joints (6 DOF + gripper) from 9 physical motors
    joint_map: Optional[List[List[int]]] = None

    # Gripper joint index (last joint by default, None to disable)
    gripper_joint_idx: Optional[int] = None

    # Position limits (in Dynamixel units, 0-4095)
    position_min: int = 0
    position_max: int = 4095

    # Conversion factors
    position_to_rad: float = 0.001534  # 4096 units = 2*pi rad
    velocity_to_rad_s: float = 0.229 * (2 * 3.14159 / 60)  # rpm to rad/s

    # Motion profile
    profile_velocity: int = 100      # 0-32767
    profile_acceleration: int = 50   # 0-32767

    def __post_init__(self):
        """Derive motor_ids from joint_map if provided."""
        if self.joint_map is not None:
            self.motor_ids = [mid for joint in self.joint_map for mid in joint]


class DynamixelController(BaseRobotController):
    """
    Controller for Dynamixel servo motors.

    Supports:
        - Dynamixel X-series (XM430, XM540, etc.)
        - Dynamixel Pro series
        - VX300s robot arm (Interbotix)

    Usage:
        config = DynamixelConfig(
            port="/dev/ttyDXL_puppet_right",
            motor_ids=[1, 2, 3, 4, 5, 6],
            interface_config=InterfaceConfig(
                action_dim=6,
                obs_dim=6,
                shm_name="vx300s_robot"
            )
        )

        controller = DynamixelController(config)
        controller.start()

        # Controller runs in background thread
        # Async DP connects to SharedMemory interface

        input("Press Enter to stop...")
        controller.stop()
    """

    def __init__(self, config: DynamixelConfig):
        # Determine number of logical joints
        if config.joint_map is not None:
            num_joints = len(config.joint_map)
        else:
            num_joints = len(config.motor_ids)

        # Update interface config dimensions based on logical joint count
        config.interface_config.action_dim = num_joints
        config.interface_config.obs_dim = num_joints

        super().__init__(config)
        self.dxl_config = config

        # Build joint map: if not provided, each motor is its own joint
        if config.joint_map is not None:
            self._joint_map = config.joint_map
        else:
            self._joint_map = [[mid] for mid in config.motor_ids]

        self._num_joints = len(self._joint_map)
        self._gripper_joint_idx = config.gripper_joint_idx

        # Dynamixel SDK objects
        self._port_handler = None
        self._packet_handler = None
        self._group_sync_read_pos = None
        self._group_sync_read_vel = None
        self._group_sync_write = None

        # Simulation mode flag
        self._simulation_mode = not DYNAMIXEL_AVAILABLE

        logger.info(f"[Dynamixel] Joint map: {self._joint_map} "
                     f"({self._num_joints} joints from {len(config.motor_ids)} motors)"
                     f"{f', gripper=joint[{self._gripper_joint_idx}]' if self._gripper_joint_idx is not None else ''}")

    def _connect_hardware(self) -> bool:
        """Connect to Dynamixel motors."""
        if self._simulation_mode:
            logger.info("[Dynamixel] Running in SIMULATION mode (SDK not available)")
            return True

        try:
            # Initialize port handler
            self._port_handler = PortHandler(self.dxl_config.port)

            if not self._port_handler.openPort():
                logger.error(f"[Dynamixel] Failed to open port: {self.dxl_config.port}")
                return False

            if not self._port_handler.setBaudRate(self.dxl_config.baudrate):
                logger.error(f"[Dynamixel] Failed to set baudrate: {self.dxl_config.baudrate}")
                return False

            logger.info(f"[Dynamixel] Port opened: {self.dxl_config.port} @ {self.dxl_config.baudrate}")

            # Initialize packet handler
            self._packet_handler = PacketHandler(PROTOCOL_VERSION)

            # Initialize sync read/write
            self._init_sync_handlers()

            # Enable torque and set motion profile
            self._setup_motors()

            logger.info(f"[Dynamixel] Connected to {len(self.dxl_config.motor_ids)} motors")
            return True

        except Exception as e:
            logger.error(f"[Dynamixel] Connection error: {e}")
            return False

    def _init_sync_handlers(self):
        """Initialize sync read/write handlers."""
        # Sync read for position
        self._group_sync_read_pos = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION
        )

        # Sync read for velocity
        self._group_sync_read_vel = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY
        )

        # Sync write for goal position
        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )

        # Add motors to sync read
        for motor_id in self.dxl_config.motor_ids:
            self._group_sync_read_pos.addParam(motor_id)
            self._group_sync_read_vel.addParam(motor_id)

    def _setup_motors(self):
        """Setup motor parameters."""
        for motor_id in self.dxl_config.motor_ids:
            # Set operating mode to position control (3)
            self._write_register(motor_id, ADDR_OPERATING_MODE, 3, 1)

            # Set motion profile
            self._write_register(motor_id, ADDR_PROFILE_VELOCITY,
                               self.dxl_config.profile_velocity, 4)
            self._write_register(motor_id, ADDR_PROFILE_ACCELERATION,
                               self.dxl_config.profile_acceleration, 4)

            # Enable torque
            self._write_register(motor_id, ADDR_TORQUE_ENABLE, 1, 1)

    def _write_register(self, motor_id: int, address: int, value: int, length: int):
        """Write to motor register."""
        if length == 1:
            self._packet_handler.write1ByteTxRx(
                self._port_handler, motor_id, address, value
            )
        elif length == 2:
            self._packet_handler.write2ByteTxRx(
                self._port_handler, motor_id, address, value
            )
        elif length == 4:
            self._packet_handler.write4ByteTxRx(
                self._port_handler, motor_id, address, value
            )

    def _disconnect_hardware(self) -> None:
        """Disconnect from Dynamixel motors."""
        if self._simulation_mode:
            return

        try:
            # Disable torque
            if self._packet_handler and self._port_handler:
                for motor_id in self.dxl_config.motor_ids:
                    self._write_register(motor_id, ADDR_TORQUE_ENABLE, 0, 1)

            # Close port
            if self._port_handler:
                self._port_handler.closePort()

            logger.info("[Dynamixel] Disconnected")

        except Exception as e:
            logger.error(f"[Dynamixel] Disconnect error: {e}")

    def _read_motor_position(self, motor_id: int) -> float:
        """Read a single motor's position in radians."""
        if self._group_sync_read_pos.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            raw_pos = self._group_sync_read_pos.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            return (raw_pos - 2048) * self.dxl_config.position_to_rad
        return 0.0

    def _read_motor_velocity(self, motor_id: int) -> float:
        """Read a single motor's velocity in rad/s."""
        if self._group_sync_read_vel.isAvailable(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
            raw_vel = self._group_sync_read_vel.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if raw_vel > 0x7FFFFFFF:
                raw_vel -= 0x100000000
            return raw_vel * self.dxl_config.velocity_to_rad_s
        return 0.0

    def _read_state(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Read current joint positions and velocities.

        For dual-motor joints, motor positions/velocities are averaged
        to produce a single logical joint value.
        """
        if self._simulation_mode:
            return (
                self._current_qpos.copy(),
                np.zeros(self._num_joints, dtype=np.float32)
            )

        positions = np.zeros(self._num_joints, dtype=np.float32)
        velocities = np.zeros(self._num_joints, dtype=np.float32)

        try:
            # Read all motor positions
            result_pos = self._group_sync_read_pos.txRxPacket()
            # Read all motor velocities
            result_vel = self._group_sync_read_vel.txRxPacket()

            for joint_idx, motor_ids in enumerate(self._joint_map):
                # Average positions from all motors in this joint
                if result_pos == COMM_SUCCESS:
                    pos_values = [self._read_motor_position(mid) for mid in motor_ids]
                    positions[joint_idx] = np.mean(pos_values)

                # Average velocities from all motors in this joint
                if result_vel == COMM_SUCCESS:
                    vel_values = [self._read_motor_velocity(mid) for mid in motor_ids]
                    velocities[joint_idx] = np.mean(vel_values)

        except Exception as e:
            logger.error(f"[Dynamixel] Read error: {e}")

        return positions, velocities

    def _rad_to_raw(self, rad_value: float) -> int:
        """Convert radians to clamped Dynamixel raw position."""
        raw_pos = int(rad_value / self.dxl_config.position_to_rad + 2048)
        return int(np.clip(raw_pos, self.dxl_config.position_min, self.dxl_config.position_max))

    def _pack_position(self, raw_pos: int) -> list:
        """Pack raw position into 4-byte Dynamixel parameter."""
        return [
            DXL_LOBYTE(DXL_LOWORD(raw_pos)),
            DXL_HIBYTE(DXL_LOWORD(raw_pos)),
            DXL_LOBYTE(DXL_HIWORD(raw_pos)),
            DXL_HIBYTE(DXL_HIWORD(raw_pos))
        ]

    def _write_command(self, action: np.ndarray) -> bool:
        """Write position command to motors.

        For dual-motor joints, the same command is sent to all motors
        in the joint group to keep them synchronized.

        Args:
            action: Array of joint positions in radians (length = num_joints)
        """
        if self._simulation_mode:
            self._current_qpos = action.copy()
            return True

        try:
            self._group_sync_write.clearParam()

            for joint_idx, motor_ids in enumerate(self._joint_map):
                raw_pos = self._rad_to_raw(action[joint_idx])
                param = self._pack_position(raw_pos)

                # Send same command to all motors in this joint
                for motor_id in motor_ids:
                    self._group_sync_write.addParam(motor_id, param)

            result = self._group_sync_write.txPacket()
            return result == COMM_SUCCESS

        except Exception as e:
            logger.error(f"[Dynamixel] Write error: {e}")
            return False

    def enable_torque(self, enable: bool = True) -> None:
        """Enable or disable motor torque."""
        if self._simulation_mode:
            return

        value = 1 if enable else 0
        for motor_id in self.dxl_config.motor_ids:
            self._write_register(motor_id, ADDR_TORQUE_ENABLE, value, 1)

        logger.info(f"[Dynamixel] Torque {'enabled' if enable else 'disabled'}")

    def set_profile(self, velocity: int, acceleration: int) -> None:
        """Set motion profile for all motors."""
        if self._simulation_mode:
            return

        for motor_id in self.dxl_config.motor_ids:
            self._write_register(motor_id, ADDR_PROFILE_VELOCITY, velocity, 4)
            self._write_register(motor_id, ADDR_PROFILE_ACCELERATION, acceleration, 4)

        logger.info(f"[Dynamixel] Profile set: vel={velocity}, acc={acceleration}")

    def go_home(self, home_position: Optional[np.ndarray] = None) -> None:
        """Move robot to home position."""
        if home_position is None:
            home_position = np.zeros(self._num_joints, dtype=np.float32)

        logger.info("[Dynamixel] Moving to home position...")
        self._write_command(home_position)
        time.sleep(2.0)  # Wait for motion to complete


# =============================================================================
# Convenience functions
# =============================================================================

# VX300s joint map: 9 motors -> 7 joints (6 DOF + gripper)
# Joint 0: Waist       [ID 1]
# Joint 1: Shoulder    [ID 2, 3] (dual motor)
# Joint 2: Elbow       [ID 4, 5] (dual motor)
# Joint 3: Forearm     [ID 6]
# Joint 4: Wrist Angle [ID 7]
# Joint 5: Wrist Rot   [ID 8]
# Joint 6: Gripper     [ID 9]
VX300S_JOINT_MAP = [[1], [2, 3], [4, 5], [6], [7], [8], [9]]
VX300S_GRIPPER_IDX = 6


def create_vx300s_controller(
    port: str = "/dev/ttyDXL_puppet_right",
    shm_name: str = "vx300s_robot"
) -> DynamixelController:
    """
    Create controller for Interbotix VX300s robot arm.

    Joint mapping: 9 physical motors -> 7 logical joints (6 DOF + gripper).
    Dual-motor joints (shoulder ID 2/3, elbow ID 4/5) are synchronized.

    Args:
        port: Serial port (udev symlink or /dev/ttyUSBx)
        shm_name: Shared memory name for Async DP interface

    Returns:
        Configured DynamixelController
    """
    config = DynamixelConfig(
        port=port,
        joint_map=VX300S_JOINT_MAP,
        gripper_joint_idx=VX300S_GRIPPER_IDX,
        baudrate=1000000,
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=7,  # 6 DOF + gripper
            obs_dim=7,
            shm_name=shm_name
        )
    )
    return DynamixelController(config)


def create_dual_arm_controller(
    port_follower: str = "/dev/ttyDXL_puppet_right",
    port_leader: str = "/dev/ttyDXL_master_right",
    shm_name: str = "dual_arm_robot"
) -> tuple['DynamixelController', 'DynamixelController']:
    """
    Create controllers for leader-follower (teaching-execution) arm pair.

    - Leader (teaching arm, ttyDXL_master_right): Human moves this arm to demonstrate motions
    - Follower (execution arm, ttyDXL_puppet_right): Replays learned motions autonomously

    Each arm has 9 physical motors mapped to 7 logical joints (6 DOF + gripper).

    Args:
        port_follower: Serial port for follower/execution arm (default: /dev/ttyDXL_puppet_right)
        port_leader: Serial port for leader/teaching arm (default: /dev/ttyDXL_master_right)
        shm_name: Shared memory name prefix

    Returns:
        Tuple of (leader_controller, follower_controller)
    """
    leader_config = DynamixelConfig(
        port=port_leader,
        joint_map=VX300S_JOINT_MAP,
        gripper_joint_idx=VX300S_GRIPPER_IDX,
        baudrate=1000000,
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=7,
            obs_dim=7,
            shm_name=f"{shm_name}_leader"
        )
    )
    follower_config = DynamixelConfig(
        port=port_follower,
        joint_map=VX300S_JOINT_MAP,
        gripper_joint_idx=VX300S_GRIPPER_IDX,
        baudrate=1000000,
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=7,
            obs_dim=7,
            shm_name=f"{shm_name}_follower"
        )
    )
    return DynamixelController(leader_config), DynamixelController(follower_config)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 60)
    print("  Dynamixel Controller Example")
    print("=" * 60)

    # Check if running on Windows
    if sys.platform == 'win32':
        port = "COM3"
    else:
        port = "/dev/ttyDXL_puppet_right"

    # Create controller (9 motors -> 7 joints: 6 DOF + gripper)
    controller = create_vx300s_controller(port=port, shm_name="example_robot")

    # Set callbacks
    def on_state_change(old, new):
        print(f"State: {old.value} -> {new.value}")

    def on_error(e):
        print(f"Error: {e}")

    controller.set_callbacks(on_state_change=on_state_change, on_error=on_error)

    # Start controller
    if not controller.start():
        print("Failed to start controller!")
        sys.exit(1)

    print("\nController running. Press Ctrl+C to stop...")
    print("Connect Async DP to shared memory 'example_robot'\n")

    try:
        while True:
            # Print statistics every 5 seconds
            stats = controller.get_statistics()
            print(f"Loops: {stats['loop_count']}, "
                  f"Avg: {stats['avg_loop_time_ms']:.2f}ms, "
                  f"qpos[0]: {stats['current_qpos'][0]:.3f}")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\nStopping...")

    controller.stop()
    print("Done.")
