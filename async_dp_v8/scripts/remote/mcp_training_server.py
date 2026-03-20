#!/usr/bin/env python3
"""MCP server for remote GPU training management.

Run on GPU server:
    pip install mcp aiofiles
    python mcp_training_server.py --port 8719

Exposes tools for:
  - GPU status monitoring
  - Training job management (start/stop/status)
  - File sync (upload/download)
  - Log tailing
  - Checkpoint listing
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get(
    "MCP_PROJECT_ROOT",
    str(Path.home() / "devAI" / "taskbotics" / "Async_Diffusion_Policy" / "async_dp_v8"),
))
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

mcp = FastMCP("gpu-training-server")

# Track running training process
_training_proc: Optional[asyncio.subprocess.Process] = None
_training_log: Optional[Path] = None


# ---------------------------------------------------------------------------
# GPU Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def gpu_status() -> str:
    """Show GPU utilization, memory, temperature, and running processes."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=10
        )
        return out
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def gpu_memory() -> str:
    """Show GPU memory usage in compact format."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        lines = []
        for row in out.strip().split("\n"):
            idx, name, used, total, util, temp = [x.strip() for x in row.split(",")]
            lines.append(f"GPU {idx}: {name} | {used}/{total} MiB ({util}% util) | {temp}C")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Training Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def train_start(
    epochs: int = 100,
    batch_size: int = 12,
    lr: float = 1e-4,
    gpu: int = 0,
    extra_args: str = "",
) -> str:
    """Start a training job in the background.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        gpu: GPU device index (0 or 1)
        extra_args: Additional CLI arguments
    """
    global _training_proc, _training_log

    if _training_proc and _training_proc.returncode is None:
        return "Training already running! Use train_status() or train_stop() first."

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _training_log = LOG_DIR / f"train_{timestamp}.log"

    cmd = (
        f"cd {PROJECT_ROOT} && "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"uv run python scripts/train/train_v8.py "
        f"--data-dir data/processed/train "
        f"--index data/interim/episodes_index_train.parquet "
        f"--val-index data/interim/episodes_index_val.parquet "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--lr {lr} "
        f"--checkpoint-dir checkpoints/hybrid_policy_v8 "
        f"{extra_args}"
    )

    log_file = open(_training_log, "w")
    _training_proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )

    return (
        f"Training started (PID: {_training_proc.pid})\n"
        f"GPU: {gpu}, epochs: {epochs}, batch_size: {batch_size}, lr: {lr}\n"
        f"Log: {_training_log}\n"
        f"Command: {cmd}"
    )


@mcp.tool()
async def train_status() -> str:
    """Check status of current training job and show recent log lines."""
    global _training_proc, _training_log

    if not _training_proc:
        return "No training job has been started in this session."

    status = "RUNNING" if _training_proc.returncode is None else f"FINISHED (exit code: {_training_proc.returncode})"

    log_tail = ""
    if _training_log and _training_log.exists():
        lines = _training_log.read_text().strip().split("\n")
        log_tail = "\n".join(lines[-30:])

    return f"Status: {status}\nPID: {_training_proc.pid}\nLog: {_training_log}\n\n--- Last 30 lines ---\n{log_tail}"


@mcp.tool()
async def train_stop() -> str:
    """Stop the running training job gracefully."""
    global _training_proc

    if not _training_proc or _training_proc.returncode is not None:
        return "No training job is currently running."

    _training_proc.send_signal(signal.SIGINT)
    try:
        await asyncio.wait_for(_training_proc.wait(), timeout=15)
        return f"Training stopped (PID: {_training_proc.pid})"
    except asyncio.TimeoutError:
        _training_proc.kill()
        return f"Training force-killed (PID: {_training_proc.pid})"


@mcp.tool()
def train_log(lines: int = 50) -> str:
    """Read the last N lines of the training log.

    Args:
        lines: Number of lines to show (default 50)
    """
    global _training_log

    if not _training_log or not _training_log.exists():
        # Try to find the most recent log
        if LOG_DIR.exists():
            logs = sorted(LOG_DIR.glob("train_*.log"))
            if logs:
                _training_log = logs[-1]
            else:
                return "No training logs found."
        else:
            return "No log directory found."

    all_lines = _training_log.read_text().strip().split("\n")
    return f"Log: {_training_log}\n\n" + "\n".join(all_lines[-lines:])


# ---------------------------------------------------------------------------
# Checkpoint Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def list_checkpoints() -> str:
    """List all saved model checkpoints with file sizes."""
    ckpt_dir = CHECKPOINT_DIR / "hybrid_policy_v8"
    if not ckpt_dir.exists():
        return f"No checkpoint directory: {ckpt_dir}"

    files = sorted(ckpt_dir.glob("*.pt"))
    if not files:
        return "No checkpoints found."

    lines = []
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        mtime = f.stat().st_mtime
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"  {f.name:30s}  {size_mb:8.1f} MB  {ts}")

    return f"Checkpoints in {ckpt_dir}:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# File Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def read_remote_file(path: str, tail: int = 0) -> str:
    """Read a file on the server. Relative paths resolve from project root.

    Args:
        path: File path (relative to project root or absolute)
        tail: If > 0, only return last N lines
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return f"File not found: {p}"
    if p.stat().st_size > 5 * 1024 * 1024:
        return f"File too large: {p.stat().st_size / 1024 / 1024:.1f} MB"

    content = p.read_text(errors="replace")
    if tail > 0:
        lines = content.strip().split("\n")
        content = "\n".join(lines[-tail:])
    return content


@mcp.tool()
def list_remote_dir(path: str = ".") -> str:
    """List directory contents on the server.

    Args:
        path: Directory path (relative to project root or absolute)
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return f"Directory not found: {p}"

    try:
        out = subprocess.check_output(
            ["ls", "-lah", str(p)], text=True, timeout=5
        )
        return f"Contents of {p}:\n{out}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Shell Tool
# ---------------------------------------------------------------------------
@mcp.tool()
def run_command(command: str, timeout_sec: int = 120) -> str:
    """Run a shell command on the server and return output.

    Args:
        command: Shell command to execute
        timeout_sec: Timeout in seconds (default 120)
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout_sec, cwd=str(PROJECT_ROOT),
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[EXIT CODE: {result.returncode}]"
        return output[:50000]  # cap output size
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_sec}s"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Sync Tool
# ---------------------------------------------------------------------------
@mcp.tool()
def sync_status() -> str:
    """Show git status of the project on the server."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--short", "-b"],
            text=True, cwd=str(PROJECT_ROOT), timeout=10,
        )
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            text=True, cwd=str(PROJECT_ROOT), timeout=10,
        )
        return f"Git status:\n{out}\nRecent commits:\n{log}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def git_pull() -> str:
    """Pull latest code from origin on the server."""
    try:
        out = subprocess.check_output(
            ["git", "pull", "--ff-only"],
            text=True, cwd=str(PROJECT_ROOT), timeout=30,
        )
        return f"Git pull result:\n{out}"
    except subprocess.CalledProcessError as e:
        return f"Git pull failed:\n{e.stderr or e.stdout}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Data Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def data_status() -> str:
    """Show status of data directories (counts, sizes)."""
    sections = []

    raw_dir = PROJECT_ROOT / "data" / "raw" / "pen_fixed_hdf5"
    if raw_dir.exists():
        files = list(raw_dir.glob("*.hdf5"))
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        sections.append(f"Raw HDF5: {len(files)} episodes, {total_mb:.0f} MB")

    proc_dir = PROJECT_ROOT / "data" / "processed" / "train"
    if proc_dir.exists():
        files = list(proc_dir.glob("*.parquet"))
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        sections.append(f"Processed Parquet: {len(files)} episodes, {total_mb:.1f} MB")

    interim_dir = PROJECT_ROOT / "data" / "interim"
    if interim_dir.exists():
        files = list(interim_dir.iterdir())
        sections.append(f"Interim files: {', '.join(f.name for f in files)}")

    return "\n".join(sections) if sections else "No data found."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="MCP transport: stdio (for SSH) or sse (HTTP)")
    parser.add_argument("--port", type=int, default=8719)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"MCP Training Server | project: {PROJECT_ROOT}", file=sys.stderr)

    if args.transport == "sse":
        print(f"Listening on {args.host}:{args.port}", file=sys.stderr)
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        print("Running in stdio mode (for SSH)", file=sys.stderr)
        mcp.run(transport="stdio")
