#!/bin/bash
# Setup MCP training server on remote GPU machine.
# Run this on the server: bash setup_server.sh

set -e

echo "=== Setting up MCP Training Server ==="

# Install dependencies
pip3 install --user "mcp[cli]>=1.0" aiofiles

# Ensure project structure
cd ~/Async_Diffusion_Policy/async_dp_v8
mkdir -p logs checkpoints/hybrid_policy_v8 data/interim data/processed/train data/raw

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the MCP server:"
echo "  cd ~/Async_Diffusion_Policy/async_dp_v8"
echo "  python3 scripts/remote/mcp_training_server.py --port 8719"
echo ""
echo "To start in background with nohup:"
echo "  nohup python3 scripts/remote/mcp_training_server.py --port 8719 > mcp_server.log 2>&1 &"
echo ""
echo "Then configure Claude Code on local machine (see README)."
