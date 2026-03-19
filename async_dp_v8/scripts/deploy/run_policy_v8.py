#!/usr/bin/env python3
"""Deploy v8 policy on real robot."""
import argparse
import torch
import logging
from pathlib import Path

from async_dp_v8.models.hybrid_policy_v8 import HybridPolicyV8
from async_dp_v8.robot.dxl_client import DxlClient
from async_dp_v8.robot.robot_interface import RobotInterface
from async_dp_v8.robot.safety import SafetyGuard
from async_dp_v8.inference.policy_runner import PolicyRunnerV8
from async_dp_v8.inference.async_executor import AsyncExecutor
from async_dp_v8.types import InferenceConfig
from async_dp_v8.utils.checkpointing import load_checkpoint
from async_dp_v8.utils.normalization import Normalizer
from async_dp_v8.constants import FOLLOWER_PORT, DXL_BAUDRATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--stats", default="data/interim/stats.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridPolicyV8().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    dxl = DxlClient(args.port, DXL_BAUDRATE)
    dxl.connect()

    # Safety
    safety = SafetyGuard(dxl_client=dxl)

    # Robot with safety
    robot = RobotInterface(dxl, safety_guard=safety)

    cfg = InferenceConfig()
    runner = PolicyRunnerV8(model, robot, cfg, safety_guard=safety)

    # Normalization
    if Path(args.stats).exists():
        normalizer = Normalizer.from_json(args.stats)
        runner.set_normalizer(normalizer)
        logger.info(f"Loaded normalization stats from {args.stats}")
    else:
        logger.warning(f"No stats file found at {args.stats}, running without denormalization")

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
