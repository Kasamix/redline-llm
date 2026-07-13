#!/usr/bin/env bash
# Provision a RunPod GPU pod (RTX 4090, CUDA 12.9 PyTorch base image) to build
# redline and run benchmarks - unattended.
#
# Run ON the pod as root AFTER the repository has been synced (create_pod.py
# --setup performs the sync and then invokes this script). Idempotent: safe to
# re-run - completed stages resolve quickly (apt/pip no-op, the model download
# skips files already present, CMake/ninja rebuild incrementally).
#
# Layout (override via environment):
#   REDLINE_REPO    repo checkout           default /workspace/redline-llm
#   REDLINE_BUILD   out-of-tree build dir   default /workspace/build
#   REDLINE_MODELS  model parent dir        default /workspace/models
#   REDLINE_LOGS    log dir                 default /workspace/logs
#
# Stages: apt deps -> pip deps -> model fetch (background, overlaps build) ->
# pod-release configure + build -> ctest -> model wait -> environment capture.
# The final line is "READY" - printed only when every stage succeeded.
set -euo pipefail

REPO="${REDLINE_REPO:-/workspace/redline-llm}"
BUILD="${REDLINE_BUILD:-/workspace/build}"
MODELS="${REDLINE_MODELS:-/workspace/models}"
LOGS="${REDLINE_LOGS:-/workspace/logs}"
MODEL_DIR="$MODELS/Qwen2.5-1.5B-Instruct"

export PATH="/usr/local/cuda/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive

mkdir -p "$LOGS" "$MODELS"
exec > >(tee -a "$LOGS/pod_setup.log") 2>&1

T0=$(date +%s)
stage() { echo "[pod_setup +$(($(date +%s) - T0))s] $*"; }

stage "start $(date -u +%FT%TZ) repo=$REPO build=$BUILD"

if [ ! -f "$REPO/CMakePresets.json" ]; then
    echo "ERROR: repo not found at $REPO - sync it first (create_pod.py --setup does this)." >&2
    exit 1
fi

# --- 1. apt deps (skipped when the toolchain is already present) -------------
if ! command -v ninja >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 \
        || ! command -v g++ >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
    stage "apt: installing toolchain"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake ninja-build git rsync python3-pip python3-venv >/dev/null
else
    stage "apt: toolchain already present, skipping"
fi

# --- 2. python deps (PEP 668 images need --break-system-packages) ------------
stage "pip: python deps"
PIP_BREAK=""
if python3 -m pip install --help 2>/dev/null | grep -q break-system-packages; then
    PIP_BREAK="--break-system-packages"
fi
python3 -m pip install --no-cache-dir -q $PIP_BREAK \
    numpy transformers tokenizers safetensors pytest "huggingface_hub>=0.23"

# --- 3. model fetch, backgrounded so it overlaps the build -------------------
stage "model: fetch (pinned revision) -> $MODEL_DIR (background)"
python3 "$REPO/scripts/fetch_model.py" --dest "$MODEL_DIR" \
    > "$LOGS/fetch_model.log" 2>&1 &
MODEL_PID=$!

# --- 4. configure: pod-release preset into the out-of-tree build dir ---------
stage "cmake: configure pod-release -> $BUILD"
if ! (cd "$REPO" && cmake --preset pod-release -B "$BUILD" \
        > "$LOGS/cmake_configure.log" 2>&1) || [ ! -f "$BUILD/CMakeCache.txt" ]; then
    # Older CMake cannot override a preset's binaryDir with -B; expand the
    # pod-release preset (base: Ninja, tests+python ON; Release, sm_89).
    stage "cmake: preset -B override unavailable, using explicit pod-release cache vars"
    cmake -S "$REPO" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DREDLINE_BUILD_TESTS=ON -DREDLINE_BUILD_PYTHON=ON \
        >> "$LOGS/cmake_configure.log" 2>&1
fi
[ -f "$BUILD/CMakeCache.txt" ] || { echo "ERROR: configure failed"; tail -40 "$LOGS/cmake_configure.log"; exit 1; }

# --- 5. build -----------------------------------------------------------------
stage "build: cmake --build -j$(nproc)"
if ! cmake --build "$BUILD" -j"$(nproc)" > "$LOGS/build.log" 2>&1; then
    echo "ERROR: build failed - tail of $LOGS/build.log:" >&2
    tail -40 "$LOGS/build.log" >&2
    exit 1
fi
tail -n 1 "$LOGS/build.log"

# --- 6. ctest smoke -----------------------------------------------------------
stage "ctest: full suite"
if ! ctest --test-dir "$BUILD" --output-on-failure > "$LOGS/ctest.log" 2>&1; then
    echo "ERROR: ctest failed - tail of $LOGS/ctest.log:" >&2
    tail -60 "$LOGS/ctest.log" >&2
    exit 1
fi
grep -E "tests passed" "$LOGS/ctest.log" | tail -n 1

# --- 7. wait for the model download -------------------------------------------
stage "model: waiting for download"
if ! wait "$MODEL_PID"; then
    echo "ERROR: model fetch failed - tail of $LOGS/fetch_model.log:" >&2
    tail -20 "$LOGS/fetch_model.log" >&2
    exit 1
fi
for f in model.safetensors tokenizer.json config.json generation_config.json; do
    [ -f "$MODEL_DIR/$f" ] || { echo "ERROR: model incomplete - missing $f"; exit 1; }
done
echo "model bytes: $(du -sb "$MODEL_DIR" | cut -f1)"

# --- 8. environment capture ----------------------------------------------------
stage "env: capturing toolchain + GPU state -> $LOGS/env_capture.log"
{
    echo "=== captured $(date -u +%FT%TZ)"
    uname -a
    echo "--- nvidia-smi"; nvidia-smi
    echo "--- versions"
    nvcc --version | tail -n 2
    cmake --version | head -n 1
    echo "ninja $(ninja --version)"
    gcc --version | head -n 1
    python3 -V
    echo "--- lscpu"; lscpu | head -n 25
    echo "--- memory"; free -g | head -n 2
    echo "--- disk"; df -h "$BUILD" | tail -n 1
    echo "--- pip freeze"; python3 -m pip freeze
    echo "--- nvidia-smi -q"; nvidia-smi -q
} > "$LOGS/env_capture.log" 2>&1

stage "done in $(($(date +%s) - T0))s - remember to record this session's pod-hours and cost in your ops spend log"
echo "READY"
