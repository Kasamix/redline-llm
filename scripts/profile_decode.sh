#!/usr/bin/env bash
# Capture an Nsight Systems trace (CUDA + NVTX) of a decode-heavy
# redline_bench run. Feeds the kernel-time breakdown and timeline gap
# analysis in docs/PROFILING.md (docs/DESIGN.md section 14).
#
# Usage: scripts/profile_decode.sh <model_dir> [output_basename]
#
# The workload is one closed wave of REQUESTS == MAX_BATCH identical
# requests with short prompts and long outputs, so after the prefill steps
# the engine executes GEN_LEN-1 consecutive decode steps at the full batch
# (511 by default -- the section 14 "500-step decode run"). The trace is
# collected via --capture-range=cudaProfilerApi: redline_bench calls
# cudaProfilerStart() only after engine init and its warmup request
# (--profile-after-init), so weight upload, allocation, Lt probing, and
# graph capture are all outside the capture window and every remaining
# CUDA API call in the trace belongs to steady-state stepping.
#
# NVTX ranges emitted by the engine (docs/DESIGN.md section 14): step /
# prefill[len,chunk] / decode[b] / graph_replay[b] / sample / h2d_inputs /
# d2h_tokens. With GRAPHS=1 every full-bucket decode step replays, so the
# decode window shows graph_replay[b]; set GRAPHS=0 to trace the eager
# path (decode[b] + sample) instead.
#
# Environment overrides:
#   BENCH_BIN   redline_bench binary (default: first existing of
#               build/pod-release, build/wsl-release relative to the repo
#               root, then $HOME/build/redline/redline_bench)
#   KV_GB       --kv-gb        (default 1.0; bench preset uses 8)
#   MAX_BATCH   --max-batch    (default 8;   bench preset uses 64)
#   REQUESTS    request count  (default MAX_BATCH -- one closed wave)
#   PROMPT_LEN  --prompt-len   (default 128 -- short inputs)
#   GEN_LEN     --gen-len      (default 512 -- 511 full-batch decode steps)
#   GRAPHS      --graphs       (default 1)
#   SEED        --seed         (default 0)
#   NSYS_EXTRA  extra nsys profile arguments (e.g. "--stats=true")
set -euo pipefail

MODEL_DIR="${1:?usage: scripts/profile_decode.sh <model_dir> [output_basename]}"
OUT="${2:-decode_profile}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${BENCH_BIN:-}" ]]; then
  for candidate in \
      "${REPO_ROOT}/build/pod-release/redline_bench" \
      "${REPO_ROOT}/build/wsl-release/redline_bench" \
      "${HOME}/build/redline/redline_bench"; do
    if [[ -x "${candidate}" ]]; then
      BENCH_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BENCH_BIN:-}" || ! -x "${BENCH_BIN}" ]]; then
  echo "profile_decode.sh: redline_bench not found; set BENCH_BIN" >&2
  exit 1
fi

KV_GB="${KV_GB:-1.0}"
MAX_BATCH="${MAX_BATCH:-8}"
REQUESTS="${REQUESTS:-${MAX_BATCH}}"
PROMPT_LEN="${PROMPT_LEN:-128}"
GEN_LEN="${GEN_LEN:-512}"
GRAPHS="${GRAPHS:-1}"
SEED="${SEED:-0}"

echo "[profile_decode] bench: ${BENCH_BIN}"
echo "[profile_decode] trace: ${OUT}.nsys-rep  bench json: ${OUT}.bench.json"

# shellcheck disable=SC2086  # NSYS_EXTRA is intentionally word-split.
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="${OUT}" \
  --force-overwrite=true \
  ${NSYS_EXTRA:-} \
  "${BENCH_BIN}" \
  --model "${MODEL_DIR}" \
  --kv-gb "${KV_GB}" \
  --max-batch "${MAX_BATCH}" \
  --requests "${REQUESTS}" \
  --prompt-len "${PROMPT_LEN}" \
  --gen-len "${GEN_LEN}" \
  --graphs "${GRAPHS}" \
  --ignore-eos 1 \
  --seed "${SEED}" \
  --profile-after-init 1 \
  --json "${OUT}.bench.json"

echo "[profile_decode] done. Suggested reports:"
echo "  nsys stats -r cuda_api_sum   ${OUT}.nsys-rep   # API mix (no-alloc audit)"
echo "  nsys stats -r cuda_gpu_trace ${OUT}.nsys-rep   # per-item GPU timeline (gap analysis)"
echo "  nsys stats -r cuda_gpu_kern_sum ${OUT}.nsys-rep # kernel-time breakdown"
echo "  nsys stats -r nvtx_sum       ${OUT}.nsys-rep   # NVTX range summary"
