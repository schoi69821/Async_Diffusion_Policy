#!/usr/bin/env python3
"""Deploy v8 policy on real robot."""
import argparse
import torch
import logging

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.inference.policy_runner import PolicyRunnerV8
from async_dp_v8.inference.async_executor import AsyncExecutor
from async_dp_v8.types import InferenceConfig
from async_dp_v8.utils.checkpointing import load_checkpoint
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridPolicyV8().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    dxl.connect()
    robot = RobotInterface(dxl)

    cfg = InferenceConfig()
    runner = PolicyRunnerV8(model, robot, cfg)

    executor = AsyncExecutor(runner)
    executor.start()

    try:
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        executor.stop()
        dxl.disconnect()


if __name__ == "__main__":
    main()
