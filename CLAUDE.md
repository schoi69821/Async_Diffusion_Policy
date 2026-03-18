# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Async DP (Asynchronous Diffusion Policy) is a robotics control system for wafer inspection using the Interbotix VX300s robot arm. It implements a dual-process architecture separating AI inference (15 Hz "brain") from real-time control (500 Hz "muscle") via pluggable IPC backends (shared memory or gRPC).

## Commands

```bash
# Install dependencies (requires uv package manager)
cd async_dp
python install_deps.py

# Run tests
cd async_dp
uv run pytest

# Run a single test
uv run pytest tests/test_model.py -v

# Run application modes
uv run python main.py --mode train     # Train diffusion model
uv run python main.py --mode run       # Production control loop
uv run python main.py --mode optimize  # Gripper optimization

# Visualize control logic
uv run python scripts/visualize_traj.py

# Compile proto definitions (after modifying .proto files)
cd async_dp/src/interfaces/proto
python compile_proto.py
```

## Architecture

### Dual-Process Control System
- **Inference Process** (`src/core/async_engine.py:run_inference_process`): Runs diffusion model at 15 Hz, writes predicted action trajectories via the interface layer
- **Control Process** (`src/core/async_engine.py:run_control_process`): Runs at 500 Hz, reads trajectories, applies interpolation and EMA smoothing, sends commands to robot
- **Shared Memory (legacy)** (`src/core/shared_mem.py`): Lock-protected numpy arrays for action trajectories (16x14), observation state (14), and update counter

### IPC Abstraction Layer (interfaces/)

All IPC goes through `RobotInterface` (ABC in `src/interfaces/base.py`), with backends swappable via `create_interface()` factory (`src/interfaces/factory.py`):

| Backend | Class | Latency | Use Case |
|---------|-------|---------|----------|
| **SHM** | `SharedMemoryInterface` | ~1μs | Both processes on same machine |
| **gRPC** | `GrpcInterface` (client) + `GrpcRobotServer` (server) | ~1-10ms | Remote robot controller, cross-language |
| **Dummy** | `DummyInterface` | N/A | Testing without hardware |

**Server vs client semantics:** `is_server=True` is the robot controller side (updates state, reads actions); `is_server=False` is the Async DP side (reads state, writes actions).

SHM memory layout: `[action_cmd (14)] | [action_traj (16x14)] | [robot_state (14)] | [robot_vel (14)] | [metadata (4x float64)]`

### Controller Hierarchy (controllers/)

- **`BaseRobotController`** (`src/controllers/base_controller.py`): Abstract base with thread-safe state (`RLock`), automatic reconnection with exponential backoff, watchdog timer for safety, velocity limiting, and state machine (IDLE → CONNECTING → RUNNING → ERROR → STOPPING). Subclasses implement `_read_state()`, `_write_command()`, `_connect_hardware()`, `_disconnect_hardware()`.
- **`DynamixelController`** (`src/controllers/dynamixel_controller.py`): Concrete implementation for Dynamixel servos (XM430/XM540). Supports VX300s (6 DOF) and dual-arm (14 DOF) via `create_vx300s_controller()` / `create_dual_arm_controller()` helpers. Falls back to simulation if SDK unavailable.

Configuration hierarchy: `InterfaceConfig` → `ControllerConfig` (adds safety params) → `DynamixelConfig` (adds hardware specifics).

### Diffusion Policy Model
- `src/models/diffusion_net.py`: Uses UNet1D from diffusers with FiLM conditioning and pad-crop architecture — input padded from 16→32 before UNet, cropped back to 16
- `src/models/train_engine.py`: Training loop with early stopping
- `src/models/scheduler.py`: Noise scheduler for diffusion process
- Configuration: 14 DOF (7 per arm), 16-step prediction horizon, 2-step observation horizon

### Robot Driver (legacy)
- `src/drivers/robot_driver.py`: Simple driver with lazy Interbotix SDK import, falls back to dummy mode. Superseded by the controller hierarchy for production use.

### Key Configuration (`config/settings.py`)
- `CONTROL_FREQ = 500`, `INFERENCE_FREQ = 15`, `PRED_HORIZON = 16`, `ACTION_DIM = 14`

## Directory Structure

```
async_dp/
├── config/settings.py        # Central configuration constants
├── src/
│   ├── core/                  # Async engine, shared memory
│   ├── controllers/           # Robot controller hierarchy (base + Dynamixel)
│   ├── interfaces/            # Pluggable IPC: base, factory, SHM, gRPC
│   │   └── proto/             # Protobuf definitions + compiled stubs
│   ├── drivers/               # Legacy robot hardware interface
│   ├── models/                # Diffusion policy, scheduler, training
│   └── utils/                 # Math utilities (interpolation, EMA), dataset loader (HDF5/Aloha)
├── scripts/                   # Visualization, simulation, gripper optimization
├── tests/                     # Pytest tests
└── main.py                    # Entry point with train/run/optimize modes
```
