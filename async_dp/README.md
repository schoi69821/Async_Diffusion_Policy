# Async DP (Asynchronous Diffusion Policy)

Dual-process robot control system using Diffusion Policy for wafer inspection with Interbotix VX300s.

## Overview

Async DP addresses the limitations of Action Chunking Transformer (ACT) by implementing an asynchronous architecture that separates AI inference from real-time control:

| Process | Frequency | Role |
|---------|-----------|------|
| **Brain** | 15 Hz | Diffusion Policy inference |
| **Muscle** | 500 Hz | Real-time control with interpolation |

This architecture eliminates motor vibration and instability issues caused by synchronous action chunking.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Async DP Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Brain (15Hz)   │         │  Muscle (500Hz)  │         │
│  │                  │         │                  │         │
│  │  DiffusionPolicy │ ──SHM──▶│  Interpolation   │         │
│  │  DDIM Denoising  │         │  EMA Smoothing   │         │
│  │  FiLM Condition  │◀──SHM── │  Velocity Limit  │         │
│  └──────────────────┘         └────────┬─────────┘         │
│                                        │                    │
│                                        ▼                    │
│                               ┌──────────────────┐         │
│                               │   Robot Driver   │         │
│                               │   VX300s / Dummy │         │
│                               └──────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **DiffusionPolicy**: UNet1D with FiLM conditioning, pad-crop for horizon 16
- **SharedMemoryManager**: Lock-protected IPC with retry logic
- **Control Loop**: Linear interpolation + EMA smoothing + velocity limiting

## Installation

### Requirements

- **Python 3.12+** (required)
- **[uv](https://github.com/astral-sh/uv)** package manager (required)

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via pip
pip install uv
```

### Setup

```bash
cd async_dp
python install_deps.py
```

The installer will:
1. Check Python version (3.12+)
2. Check uv is installed
3. Install all dependencies via `uv sync`

## Usage

### Training

```bash
uv run python main.py --mode train
```

Trains the Diffusion Policy model on HDF5 demonstration data.

### Production Run

```bash
uv run python main.py --mode run
```

Starts the dual-process controller (requires robot hardware or runs in dummy mode).

### Simulation

```bash
# Run 3-second simulation with visualization
uv run python scripts/run_simulation.py --duration 3

# Run without plot
uv run python scripts/run_simulation.py --duration 5 --no-viz
```

### Trajectory Visualization

```bash
uv run python scripts/visualize_traj.py
```

Real-time animation of control logic (interpolation + smoothing).

## Configuration

All settings are centralized in `config/settings.py`:

```python
# Frequencies
CONTROL_FREQ = 500      # Muscle process (Hz)
INFERENCE_FREQ = 15     # Brain process (Hz)

# Dimensions
ACTION_DIM = 14         # 7 DOF x 2 arms
OBS_DIM = 14            # Joint positions
PRED_HORIZON = 16       # Prediction steps

# Safety
EMERGENCY_STOP_LIMIT = 1.0  # Radians
MAX_JOINT_VEL = 2.0         # Rad/s
```

## Project Structure

```
async_dp/
├── config/
│   └── settings.py          # Central configuration
├── src/
│   ├── core/
│   │   ├── async_engine.py  # Dual-process controller
│   │   └── shared_mem.py    # IPC shared memory
│   ├── drivers/
│   │   └── robot_driver.py  # Hardware interface
│   ├── models/
│   │   ├── diffusion_net.py # DiffusionPolicy + FiLM
│   │   ├── scheduler.py     # DDPM/DDIM schedulers
│   │   └── train_engine.py  # Training loop
│   └── utils/
│       ├── dataset.py       # HDF5 data loader
│       └── math_utils.py    # Interpolation, EMA
├── scripts/
│   ├── run_simulation.py    # Full pipeline simulation
│   └── visualize_traj.py    # Control logic visualization
├── tests/
│   ├── test_model.py        # UNet padding test
│   ├── test_inference.py    # Diffusion inference tests
│   ├── test_shm.py          # Shared memory tests
│   ├── test_math_utils.py   # Math utilities tests
│   └── test_driver.py       # Robot driver tests
├── main.py                  # Entry point
└── README.md
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_inference.py -v
```

### Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_model.py | 1 | UNet pad-crop architecture |
| test_inference.py | 3 | Diffusion inference, scheduler, clipping |
| test_shm.py | 5 | Shared memory operations |
| test_math_utils.py | 2 | Interpolation, EMA filter |
| test_driver.py | 1 | Dummy robot driver |

**Total: 12 tests**

## Technical Details

### Diffusion Policy

- **Architecture**: UNet1D (128, 256 channels) with 2 down/up blocks
- **Conditioning**: FiLM layers modulate features with obs + timestep embeddings
- **Padding**: Input padded 16→32 for UNet, cropped back to 16
- **Scheduler**: DDIM with 16 inference steps (100 training steps)

### Control Pipeline

1. **Trajectory Interpolation**: Linear interpolation between prediction points
2. **EMA Smoothing**: `y = α * x + (1-α) * y_prev` with α=0.3
3. **Velocity Limiting**: Clamp joint velocity to MAX_JOINT_VEL

### Shared Memory Layout

```
[0:action_size]                    → action_traj (16, 14) float32
[action_size:action_size+obs_size] → obs_state (14,) float32
[action_size+obs_size:]            → update_counter (1,) int64
```

## Data Format

### HDF5 Structure

```
episode.hdf5
├── observations/
│   ├── qpos    (T, 14)  # Joint positions
│   └── qvel    (T, 14)  # Joint velocities (optional)
└── action      (T, 14)  # Actions
```

### Creating Sample Data

```python
from src.utils.dataset import create_sample_hdf5
create_sample_hdf5("data/sample.hdf5", num_timesteps=500)
```

## Performance

| Metric | Value |
|--------|-------|
| Inference latency (CPU) | ~300ms |
| Inference latency (GPU) | ~20ms |
| Control loop jitter | <1ms |
| Memory usage (SHM) | ~1KB |

## License

MIT License

## References

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Chi et al.
- [FiLM Conditioning](https://arxiv.org/abs/1709.07871) - Perez et al.
- [Interbotix VX300s](https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx)
