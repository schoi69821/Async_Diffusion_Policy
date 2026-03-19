"""Low-level Dynamixel SDK client with SyncRead and error handling."""
import numpy as np
from typing import Dict, List, Optional
import logging

from async_dp_v8.constants import DXL_BAUDRATE, JOINT_MAP
from .dxl_modes import (
    ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE,
    ADDR_GOAL_POSITION, ADDR_GOAL_CURRENT,
    ADDR_PRESENT_POSITION, ADDR_PRESENT_VELOCITY,
    ADDR_PRESENT_CURRENT, ADDR_PRESENT_PWM,
    ADDR_PRESENT_INPUT_VOLTAGE, ADDR_BUS_WATCHDOG,
    ADDR_PROFILE_VELOCITY, ADDR_PROFILE_ACCELERATION,
    MODE_POSITION, MODE_CURRENT_BASED_POSITION,
    PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)

# SyncRead block: PWM(124,2) + Current(126,2) + Velocity(128,4) + Position(132,4) = 12 bytes
SYNC_READ_ADDR = 124
SYNC_READ_LEN = 12


class DxlReadError(Exception):
    """Raised when Dynamixel communication fails."""
    pass


class DxlClient:
    def __init__(
        self,
        port: str,
        baudrate: int = DXL_BAUDRATE,
        ids: Optional[List[int]] = None,
        bus_watchdog_20ms: int = 50,
    ):
        self.port = port
        self.baudrate = baudrate
        self.ids = ids or list(range(1, 10))
        self.bus_watchdog_20ms = bus_watchdog_20ms
        self._port_handler = None
        self._packet_handler = None
        self._sync_reader = None
        self._connected = False
        self._read_errors = 0
        self._max_consecutive_errors = 5

    def connect(self) -> bool:
        try:
            from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead
            self._port_handler = PortHandler(self.port)
            self._packet_handler = PacketHandler(PROTOCOL_VERSION)

            if not self._port_handler.openPort():
                logger.error(f"Failed to open port {self.port}")
                return False
            if not self._port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate {self.baudrate}")
                return False

            # Setup SyncRead for state block
            self._sync_reader = GroupSyncRead(
                self._port_handler, self._packet_handler,
                SYNC_READ_ADDR, SYNC_READ_LEN,
            )
            for motor_id in self.ids:
                self._sync_reader.addParam(motor_id)

            # Enable bus watchdog on all motors
            if self.bus_watchdog_20ms > 0:
                for motor_id in self.ids:
                    self._write1(motor_id, ADDR_BUS_WATCHDOG, self.bus_watchdog_20ms)

            self._connected = True
            self._read_errors = 0
            logger.info(f"Connected to {self.port} @ {self.baudrate}, {len(self.ids)} motors")
            return True
        except ImportError:
            logger.warning("dynamixel_sdk not available, running in dummy mode")
            self._connected = False
            return False

    def disconnect(self):
        if self._connected:
            # Clear bus watchdog before disconnect
            for motor_id in self.ids:
                try:
                    self._write1(motor_id, ADDR_BUS_WATCHDOG, 0)
                except Exception:
                    pass
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._connected = False

    def set_torque(self, motor_id: int, enable: bool):
        if not self._connected:
            return
        self._write1(motor_id, ADDR_TORQUE_ENABLE, int(enable))

    def torque_off_all(self):
        """Emergency: disable torque on all motors."""
        for motor_id in self.ids:
            try:
                if self._packet_handler and self._port_handler:
                    self._packet_handler.write1ByteTxRx(
                        self._port_handler, motor_id,
                        ADDR_TORQUE_ENABLE, 0,
                    )
            except Exception:
                pass  # Best-effort during emergency

    def set_operating_mode(self, motor_id: int, mode: int):
        if not self._connected:
            return
        self.set_torque(motor_id, False)
        self._write1(motor_id, ADDR_OPERATING_MODE, mode)

    def set_joint_mode(self, motor_id: int, velocity: int = 100, acceleration: int = 50):
        self.set_operating_mode(motor_id, MODE_POSITION)
        if self._connected:
            self._write4(motor_id, ADDR_PROFILE_VELOCITY, velocity)
            self._write4(motor_id, ADDR_PROFILE_ACCELERATION, acceleration)

    def set_current_based_position_mode(self, motor_id: int, current_limit: int = 150):
        self.set_operating_mode(motor_id, MODE_CURRENT_BASED_POSITION)
        if self._connected:
            self._write2(motor_id, ADDR_GOAL_CURRENT, current_limit)

    def read_state(self) -> Dict[str, np.ndarray]:
        """Read all motor states via SyncRead (single bus transaction)."""
        n = len(self.ids)
        state = {
            "present_position": np.zeros(n),
            "present_velocity": np.zeros(n),
            "present_current": np.zeros(n),
            "present_pwm": np.zeros(n),
        }

        if not self._connected or self._sync_reader is None:
            return state

        result = self._sync_reader.txRxPacket()
        if result != 0:
            self._read_errors += 1
            error_msg = self._packet_handler.getTxRxResult(result)
            logger.warning(f"SyncRead failed: {error_msg} (errors={self._read_errors})")
            if self._read_errors >= self._max_consecutive_errors:
                raise DxlReadError(
                    f"SyncRead failed {self._read_errors} consecutive times: {error_msg}"
                )
            return state

        self._read_errors = 0

        for i, motor_id in enumerate(self.ids):
            if not self._sync_reader.isAvailable(motor_id, SYNC_READ_ADDR, SYNC_READ_LEN):
                logger.warning(f"Motor {motor_id} data not available")
                continue

            pwm = self._sync_reader.getData(motor_id, 124, 2)
            cur = self._sync_reader.getData(motor_id, 126, 2)
            vel = self._sync_reader.getData(motor_id, 128, 4)
            pos = self._sync_reader.getData(motor_id, 132, 4)

            state["present_pwm"][i] = self._to_signed16(pwm)
            state["present_current"][i] = self._to_signed16(cur)
            state["present_velocity"][i] = self._to_signed32(vel)
            state["present_position"][i] = self._to_signed32(pos)

        return state

    def read_voltage(self) -> float:
        if not self._connected:
            return 0.0
        val, result, error = self._packet_handler.read2ByteTxRx(
            self._port_handler, self.ids[0], ADDR_PRESENT_INPUT_VOLTAGE)
        if result != 0:
            logger.warning(f"Voltage read failed: {self._packet_handler.getTxRxResult(result)}")
            return 0.0
        return val * 0.1

    def goal_position(self, motor_id: int, position: int):
        if not self._connected:
            return
        self._write4(motor_id, ADDR_GOAL_POSITION, position)

    def goal_position_rad(self, motor_id: int, rad: float):
        """Convert radians to ticks. Supports full rotation range."""
        ticks = int(rad / (2 * np.pi) * 4096 + 2048)
        self.goal_position(motor_id, ticks)

    # --- Private helpers with error checking ---

    def _write1(self, motor_id: int, addr: int, value: int):
        result, error = self._packet_handler.write1ByteTxRx(
            self._port_handler, motor_id, addr, value)
        if result != 0:
            logger.warning(
                f"Write1 failed motor={motor_id} addr={addr}: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            logger.warning(
                f"DXL error motor={motor_id}: "
                f"{self._packet_handler.getRxPacketError(error)}"
            )

    def _write2(self, motor_id: int, addr: int, value: int):
        result, error = self._packet_handler.write2ByteTxRx(
            self._port_handler, motor_id, addr, value)
        if result != 0:
            logger.warning(
                f"Write2 failed motor={motor_id} addr={addr}: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            logger.warning(
                f"DXL error motor={motor_id}: "
                f"{self._packet_handler.getRxPacketError(error)}"
            )

    def _write4(self, motor_id: int, addr: int, value: int):
        result, error = self._packet_handler.write4ByteTxRx(
            self._port_handler, motor_id, addr, value)
        if result != 0:
            logger.warning(
                f"Write4 failed motor={motor_id} addr={addr}: "
                f"{self._packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            logger.warning(
                f"DXL error motor={motor_id}: "
                f"{self._packet_handler.getRxPacketError(error)}"
            )

    @staticmethod
    def _to_signed32(val: int) -> int:
        if val > 0x7FFFFFFF:
            val -= 0x100000000
        return val

    @staticmethod
    def _to_signed16(val: int) -> int:
        if val > 0x7FFF:
            val -= 0x10000
        return val

    @property
    def connected(self) -> bool:
        return self._connected
