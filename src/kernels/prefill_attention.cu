#include "kernels/kernels.cuh"

#include <cassert>
#include <cfloat>
#include <cstdint>

// Prefill causal softmax (docs/DESIGN.md section 6.2, kernel #7) - the middle
// stage of the chunked-prefill attention composite of docs/DESIGN.md
// section 6.3:
//
//   kv_gather -> scores GEMMs -> [this kernel] -> PV GEMMs
//
// The scores GEMMs (two strided-batched cuBLASLt calls, one per KV head,
// owned by the executor's prefill composite) write
// S = (Q K^T) * 1/sqrt(head_dim) as FP32 into a dense
// [num_q_heads, chunk_len, kv_len] buffer: head h's plane starts at element
// h * chunk_len * kv_len and query row i sits at row offset i * kv_len (the
// column-major dual with ldd = kv_len and batch stride chunk_len * kv_len).
// This kernel softmaxes every (head, query-row) over the key dimension under
// the causal mask and writes FP16 `probs` in the identical layout, which the
// PV GEMMs then read (ldb = kv_len, batch stride chunk_len * kv_len). Both
// tensors are densely packed by construction of those GEMM descriptors, so
// unlike the qkv_out-strided kernels there are no stride parameters here.
//
// Causal mask across chunks: the gathered khat/vhat buffers hold key
// positions 0 .. kv_len-1 of the one sequence being prefilled, and query row
// i of this chunk has global position q = chunk_start + i, so key j is live
// iff j <= q (a token attends to itself and everything before it). kv_len >
// chunk_len is the normal case for every chunk after the first: chunk n
// attends to all previously scattered positions plus itself, and the final
// masked-out upper triangle only ever spans this chunk's own future rows.
//
// Masking is select-style, mirroring the decode kernel's policy (docs/
// DESIGN.md section 6.2 #6): masked entries are excluded by never being read
// at all during the max/sum passes - not by adding -inf to whatever the
// buffer holds - so the masked region's contents can never contaminate a
// result. Their probs are written as exact zeros because the PV GEMMs
// contract over the full kv_len extent; a masked key participates in that
// GEMM and must contribute exactly nothing.
//
// Numerics (matches HF's FP32-softmax-then-downcast attention path): FP32
// running max, FP32 sum of exponentials, per-element FP32 normalize, single
// round to FP16 at the end. Probabilities lie in [0, 1] - safe in FP16.
//
// Mapping: one 256-thread block per (query row, head) - grid
// (chunk_len, num_q_heads) - with the block's threads striding over the key
// dimension. Three passes over the row (max, sum, write); correctness-first
// v1, tuning is a later measured concern. There are no padded rows in
// prefill (the composite processes exactly chunk_len rows of one sequence),
// so unlike the seq_len-driven kernels there is no padded-row exit here.

namespace redline::kernels {

namespace {

constexpr std::int32_t kBlockThreads = 256;

// Deterministic block-wide FP32 reductions, the same fixed halving tree as
// src/kernels/rmsnorm.cu: every thread deposits its partial in a fixed shared
// slot and the slots collapse pairwise in the same order every launch - no
// atomics, no warp-arrival ordering anywhere - so identical inputs at
// identical shapes reduce bitwise-identically, a premise of the
// docs/DESIGN.md section 12 same-shape equivalence suites. Both functions
// assume blockDim.x == kBlockThreads (a power of two), which the launcher
// guarantees, and return the block-wide result to every thread (the final
// __syncthreads() of the loop makes partials[0] visible block-wide). Threads
// whose key columns are exhausted contribute the identity of their reduction
// (-FLT_MAX for max, +0.0f for sum).

__device__ float BlockReduceMax(float v) {
  __shared__ float partials[kBlockThreads];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  partials[tid] = v;
  __syncthreads();
#pragma unroll
  for (std::int32_t offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partials[tid] = fmaxf(partials[tid], partials[tid + offset]);
    }
    __syncthreads();
  }
  return partials[0];
}

__device__ float BlockReduceSum(float v) {
  __shared__ float partials[kBlockThreads];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  partials[tid] = v;
  __syncthreads();
#pragma unroll
  for (std::int32_t offset = kBlockThreads / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partials[tid] += partials[tid + offset];
    }
    __syncthreads();
  }
  return partials[0];
}

// One block softmaxes one (query row, head) score row of kv_len FP32 entries
// into the matching FP16 probs row. blockIdx.x = query row within the chunk,
// blockIdx.y = Q head.
__global__ __launch_bounds__(kBlockThreads) void PrefillCausalSoftmaxKernel(
    half* __restrict__ probs, const float* __restrict__ scores, std::int32_t chunk_len,
    std::int32_t kv_len, std::int32_t chunk_start) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t head = static_cast<std::int32_t>(blockIdx.y);
  if (row >= chunk_len) {
    return; // grid is sized exactly to chunk_len; guard kept uniform across the block
  }

  // Dense [num_q_heads, chunk_len, kv_len] packing (see file header): the
  // score row and its probs row share one element offset. int64 keeps the
  // arithmetic overflow-free with headroom (12 heads x 2048 x 4096 ~= 1e8
  // elements at the bench preset).
  const std::int64_t row_offset = (static_cast<std::int64_t>(head) * chunk_len + row) * kv_len;
  const float* __restrict__ s_row = scores + row_offset;
  half* __restrict__ p_row = probs + row_offset;

  // MASK BOUNDARY - the single causal switch point of this kernel. Query row
  // `row` has global position q = chunk_start + row; key j is live iff
  // j <= q, so key j == q (the query's own position) is the last live key
  // and key j == q + 1 is the first masked one. `live` is the count of live
  // keys. The clamp to kv_len never engages under the launcher's
  // chunk_start + chunk_len <= kv_len precondition; it only keeps every read
  // in bounds if a caller violates that precondition in a build with
  // assertions stripped.
  const std::int32_t q_pos = chunk_start + row;
  const std::int32_t live = (q_pos + 1 < kv_len) ? (q_pos + 1) : kv_len;

  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);

  // Pass 1: FP32 max over the live prefix [0, live). Select-style masking:
  // the loop bound *is* the mask - masked entries are never loaded.
  float local_max = -FLT_MAX;
  for (std::int32_t j = tid; j < live; j += kBlockThreads) {
    local_max = fmaxf(local_max, s_row[j]);
  }
  const float row_max = BlockReduceMax(local_max);

  // Pass 2: FP32 sum of exp(s - max) over the live prefix. expf is the
  // precise-path exponential (this build does not enable fast-math). Every
  // argument is <= 0, so each term lies in (0, 1] and the max element
  // contributes exactly 1: with at least one live key - guaranteed, see the
  // launcher's fully-masked-row assert - the denominator is >= 1 and the
  // division below can never see 0/0.
  float local_sum = 0.0f;
  for (std::int32_t j = tid; j < live; j += kBlockThreads) {
    local_sum += expf(s_row[j] - row_max);
  }
  const float denom = BlockReduceSum(local_sum);

  // Pass 3: normalize and write the full row width. Live keys get
  // fp16(exp(s - max) / denom) - the exponential is recomputed and
  // normalized in FP32 so the value is rounded to FP16 exactly once. Masked
  // keys get exact +0.0f so the PV GEMMs' full-kv_len contraction drops
  // them.
  for (std::int32_t j = tid; j < kv_len; j += kBlockThreads) {
    const float p = (j < live) ? expf(s_row[j] - row_max) / denom : 0.0f;
    p_row[j] = __float2half_rn(p);
  }
}

} // namespace

void LaunchPrefillSoftmax(half* probs, const float* scores, std::int32_t num_q_heads,
                          std::int32_t chunk_len, std::int32_t kv_len, std::int32_t chunk_start,
                          cudaStream_t stream) {
  if (num_q_heads <= 0 || chunk_len <= 0 || kv_len <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to normalize
  }

  // Fully masked rows are impossible by construction - debug-asserted here
  // rather than handled: the chunk being softmaxed is the newest content of
  // its sequence, so the gathered key range covers every query's own
  // position and each query attends at least to itself
  // (docs/DESIGN.md section 6.2 #7). The composite always calls with
  // chunk_start + chunk_len == kv_len (kv_gather copies [0, ctx_end) with
  // ctx_end = chunk_start + chunk_len); the kernel itself only needs <=,
  // which is exactly the no-fully-masked-row guarantee. positions are
  // 0-based, so chunk_start >= 0.
  assert(chunk_start >= 0);
  assert(chunk_start + chunk_len <= kv_len);

  // One block per (query row, head); threads stride the key dimension.
  // grid.y is bounded by 65535, far above any head count; grid.x carries
  // chunk_len (dev preset 1024, bench preset 2048).
  const dim3 grid(static_cast<unsigned int>(chunk_len), static_cast<unsigned int>(num_q_heads));
  const dim3 block(kBlockThreads);
  PrefillCausalSoftmaxKernel<<<grid, block, 0, stream>>>(probs, scores, chunk_len, kv_len,
                                                         chunk_start);
}

} // namespace redline::kernels
