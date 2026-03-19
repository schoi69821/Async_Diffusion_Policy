"""Low-level Dynamixel SDK client for reading/writing motor registers."""
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


class DxlClient:
    def __init__(
        self,
        port: str,
        baudrate: int = DXL_BAUDRATE,
        ids: Optional[List[int]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.ids = ids or list(range(1, 10))
        self._port_handler = None
        self._packet_handler = None
        self._connected = False

    def connect(self) -> bool:
        try:
            from dynamixel_sdk import PortHandler, PacketHandler
            self._port_handler = PortHandler(self.port)
            self._packet_handler = PacketHandler(PROTOCOL_VERSION)

            if not self._port_handler.openPort():
                logger.error(f"Failed to open port {self.port}")
                return False
            if not self._port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate {self.baudrate}")
                return False

            self._connected = True
            logger.info(f"Connected to {self.port} @ {self.baudrate}")
            return True
        except ImportError:
            logger.warning("dynamixel_sdk not available, running in dummy mode")
            self._connected = False
            return False

    def disconnect(self):
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._connected = False

    def set_torque(self, motor_id: int, enable: bool):
        if not self._connected:
            return
        self._packet_handler.write1ByteTxRx(
            self._port_handler, motor_id,
            ADDR_TORQUE_ENABLE, int(enable),
        )

    def set_operating_mode(self, motor_id: int, mode: int):
        if not self._connected:
            return
        self.set_torque(motor_id, False)
        self._packet_handler.write1ByteTxRx(
            self._port_handler, motor_id,
            ADDR_OPERATING_MODE, mode,
        )

    def set_joint_mode(self, motor_id: int, velocity: int = 100, acceleration: int = 50):
        self.set_operating_mode(motor_id, MODE_POSITION)
        if self._connected:
            self._packet_handler.write4ByteTxRx(
                self._port_handler, motor_id,
                ADDR_PROFILE_VELOCITY, velocity,
            )
            self._packet_handler.write4ByteTxRx(
                self._port_handler, motor_id,
                ADDR_PROFILE_ACCELERATION, acceleration,
            )

    def set_current_based_position_mode(self, motor_id: int, current_limit: int = 150):
        self.set_operating_mode(motor_id, MODE_CURRENT_BASED_POSITION)
        if self._connected:
            self._packet_handler.write2ByteTxRx(
                self._port_handler, motor_id,
                ADDR_GOAL_CURRENT, current_limit,
            )

    def set_bus_watchdog(self, motor_id: int, watchdog_20ms_units: int = 50):
        if not self._connected:
            return
        self._packet_handler.write1ByteTxRx(
            self._port_handler, motor_id,
            ADDR_BUS_WATCHDOG, watchdog_20ms_units,
        )

    def read_state(self) -> Dict[str, np.ndarray]:
        """Read all motor states (position, velocity, current, pwm)."""
        n = len(self.ids)
        state = {
            "present_position": np.zeros(n),
            "present_velocity": np.zeros(n),
            "present_current": np.zeros(n),
            "present_pwm": np.zeros(n),
        }

        if not self._connected:
            return state

        for i, motor_id in enumerate(self.ids):
            pos, _, _ = self._packet_handler.read4ByteTxRx(
                self._port_handler, motor_id, ADDR_PRESENT_POSITION)
            vel, _, _ = self._packet_handler.read4ByteTxRx(
                self._port_handler, motor_id, ADDR_PRESENT_VELOCITY)
            cur, _, _ = self._packet_handler.read2ByteTxRx(
                self._port_handler, motor_id, ADDR_PRESENT_CURRENT)
            pwm, _, _ = self._packet_handler.read2ByteTxRx(
                self._port_handler, motor_id, ADDR_PRESENT_PWM)

            state["present_position"][i] = self._to_signed32(pos)
            state["present_velocity"][i] = self._to_signed32(vel)
            state["present_current"][i] = self._to_signed16(cur)
            state["present_pwm"][i] = self._to_signed16(pwm)

        return state

    def read_voltage(self) -> float:
        if not self._connected:
            return 0.0
        val, _, _ = self._packet_handler.read2ByteTxRx(
            self._port_handler, self.ids[0], ADDR_PRESENT_INPUT_VOLTAGE)
        return val * 0.1

    def goal_position(self, motor_id: int, position: int):
        if not self._connected:
            return
        self._packet_handler.write4ByteTxRx(
            self._port_handler, motor_id,
            ADDR_GOAL_POSITION, position,
        )

    def goal_position_rad(self, motor_id: int, rad: float):
        ticks = int(rad / (2 * np.pi) * 4096 + 2048)
        ticks = max(0, min(4095, ticks))
        self.goal_position(motor_id, ticks)

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
