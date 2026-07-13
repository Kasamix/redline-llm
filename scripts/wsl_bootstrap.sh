#!/usr/bin/env bash
# One-time WSL Ubuntu toolchain setup for building Redline locally.
# Installs the CUDA *toolkit* only - never the driver (WSL uses the Windows driver).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
    python3-venv python3-pip wget ca-certificates

wget -qO /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

if ! grep -q 'cuda-12.6/bin' ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi

# Python env for the e2e tests and bench harness (torch wheel matches the
# CUDA 12.6 toolkit above). Idempotent: venv is created once, pip re-runs are
# no-ops when the packages are already present.
VENV="$HOME/venvs/redline"
if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu126
"$VENV/bin/pip" install transformers tokenizers safetensors numpy pytest

/usr/local/cuda-12.6/bin/nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader
echo BOOTSTRAP_OK
