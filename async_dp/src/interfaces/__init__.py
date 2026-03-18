"""
Robot Communication Interfaces
Supports: SharedMemory (local), gRPC (remote)
"""
from src.interfaces.base import RobotInterface, InterfaceConfig
from src.interfaces.factory import create_interface, InterfaceType

__all__ = [
    'RobotInterface',
    'InterfaceConfig',
    'create_interface',
    'InterfaceType',
]
