#include "kernels/kernels.cuh"

#include <cstdint>

#include <math_constants.h>

// Greedy argmax (docs/DESIGN.md section 6.2, kernel #9) - the sampling step
// at the end of every decode step and of the final prefill chunk: for each
// sequence, the int32 index of the largest logit in its vocab row.
//
// Mapping: grid = one block per sequence row; the block's threads stride
// across the vocab dimension, each thread carrying a running (value, index)
// champion in FP32, then a deterministic reduction collapses the champions
// to the row winner. Single kernel, single pass over the row, no atomics.
// The primary path (measured tuning pass, docs/PROFILING.md section 2.1)
// reads the row as 16-byte words - eight
// logits per load, compared in ascending column order - with a per-arch
// block size (1024 threads on sm_80+, where the row scan is DRAM-bound at
// decode grids and wants maximum loads in flight per SM; 256 elsewhere,
// matching the kernel-suite-validated shape). A scalar kernel remains as the
// totality fallback for vocab sizes off the 8-column grain (never this
// model's 151,936).
//
// The contract (src/kernels/kernels.cuh) carries no row stride: `logits` is
// the dense [num_seqs, vocab_size] output of the lm_head GEMM (the
// column-major TN dual of docs/DESIGN.md section 6.1 lands D exactly as
// row-major [rows, 151936]), so rows sit vocab_size elements apart - and a
// vocab_size divisible by 8 keeps every row base on the allocation's 16-byte
// alignment.
//
// Ordering - must match torch.argmax for HF greedy parity (docs/DESIGN.md
// sections 6.2, 12a):
//   - compares are FP32; __half2float is exact and order-preserving, so the
//     FP32 comparison induces exactly the FP16 ordering of the stored row;
//   - a strictly greater value wins; equal values resolve to the LOWEST
//     index (torch.argmax's first-occurrence rule). Ties are broken on the
//     column index itself - never on thread id, warp id, or arrival order -
//     so ties straddling thread, warp, vector-word, or reduction-tree
//     boundaries still resolve to the lowest column. +0.0 and -0.0 compare
//     equal and therefore also resolve by index, matching torch.
//
// NaN policy (documented behavior, tested by the docs/DESIGN.md section 12a
// exactness cases): a NaN logit never wins while the row holds any non-NaN
// logit. Both arms of the ordering predicate are IEEE comparisons, which are
// false against NaN, so a NaN candidate can never displace an incumbent -
// and the incumbent is never NaN by construction (it starts at -inf and only
// ever adopts candidates that passed the predicate). If the ENTIRE row is
// NaN, no candidate is ever adopted and the kernel deterministically writes
// token 0 - in-range for downstream token append, and an all-NaN logit row
// is hard upstream corruption that the parity suites flag regardless. This
// deviates from torch.argmax on purpose: torch propagates NaN as the
// maximum, which would let a single corrupted lane silently steer token
// selection.
//
// Determinism: the ordering predicate is a total order over every value the
// reduction can see (adopted candidates are non-NaN; the init sentinel sits
// last in the order), so ANY reduction shape yields the identical
// (max value, min index) winner; the shapes used - per-thread scan in
// ascending column order, lane-xor shuffle tree inside each warp, ordered
// scan of the warp champions - are additionally fixed per launch
// configuration, so identical logits produce bitwise-identical tokens, a
// premise of the docs/DESIGN.md section 12 same-shape equivalence suites.
// The winner is exactly the pre-tuning kernel's: total order makes the
// reduction-tree change observationally invisible.

namespace redline::kernels {

namespace {

// Per-arch tuning constant (docs/PROFILING.md section 2.1): threads per row block on sm_80+
// (measured on sm_89 at the decode shape: 1024 -> 22.0 us, 512 -> 22.2,
// 256 -> 23.2; DRAM-bound at ~85-89% of measured peak, so deeper splits
// plateau). The guarded define lets the tuning harness sweep candidates
// without touching this file.
#ifndef REDLINE_ARGMAX_BLOCK_SM80
#define REDLINE_ARGMAX_BLOCK_SM80 1024
#endif
constexpr std::int32_t kBlockThreadsSm80 = REDLINE_ARGMAX_BLOCK_SM80;
static_assert(kBlockThreadsSm80 >= 64 && kBlockThreadsSm80 <= 1024 && (kBlockThreadsSm80 & 31) == 0,
              "argmax block must be a multiple of the warp size");
constexpr std::int32_t kBlockThreads = 256; // pre-sm_80 + scalar-fallback block

// Ordering predicate: does candidate (v, i) beat incumbent (best_v, best_i)?
// Strictly greater value wins; equal values resolve to the lower index
// (torch.argmax first-occurrence rule). Both arms are IEEE compares, so a
// NaN candidate yields false on both and is never adopted. The incumbent is
// never NaN (it starts at -inf and only adopts candidates that passed this
// predicate), so the predicate is a total order over everything it is ever
// asked to compare.
__device__ bool Better(float v, std::int32_t i, float best_v, std::int32_t best_i) {
  return (v > best_v) || (v == best_v && i < best_i);
}

// Deterministic block reduction of per-thread (value, index) champions:
// lane-xor shuffle tree inside each warp, then thread 0 scans the warp
// champions in ascending warp order. Returns the winner to thread 0 only.
// Assumes blockDim.x == kBlock; init sentinels (-inf, vocab_size) flow
// through harmlessly (they lose to every adopted candidate).
template <std::int32_t kBlock>
__device__ void BlockReduceArgmax(float& best_v, std::int32_t& best_i) {
  constexpr std::int32_t kWarps = kBlock / 32;
  __shared__ float warp_v[kWarps];
  __shared__ std::int32_t warp_i[kWarps];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
#pragma unroll
  for (std::int32_t offset = 16; offset > 0; offset >>= 1) {
    const float other_v = __shfl_xor_sync(0xffffffffu, best_v, offset);
    const std::int32_t other_i = __shfl_xor_sync(0xffffffffu, best_i, offset);
    if (Better(other_v, other_i, best_v, best_i)) {
      best_v = other_v;
      best_i = other_i;
    }
  }
  if ((tid & 31) == 0) {
    warp_v[tid >> 5] = best_v;
    warp_i[tid >> 5] = best_i;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (std::int32_t w = 1; w < kWarps; ++w) {
      if (Better(warp_v[w], warp_i[w], best_v, best_i)) {
        best_v = warp_v[w];
        best_i = warp_i[w];
      }
    }
  }
}

// Primary kernel: 16-byte row scan. Requires vocab_size % 8 == 0 and a
// 16-byte-aligned logits base (then every row base is 16-byte aligned too);
// the launcher checks both. Within each uint4 word the eight columns are
// compared in ascending order, so the champion update order is a fixed
// function of the launch configuration.
template <std::int32_t kBlock>
__global__ __launch_bounds__(kBlock) void GreedyArgmaxVecKernel(
    std::int32_t* __restrict__ out_tokens, const half* __restrict__ logits, std::int32_t num_seqs,
    std::int32_t vocab_size) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_seqs) {
    return; // bound guard, uniform across the block (grid is sized to num_seqs)
  }

  const uint4* __restrict__ row8 =
      reinterpret_cast<const uint4*>(logits + static_cast<std::int64_t>(row) * vocab_size);
  const std::int32_t words = vocab_size >> 3; // uint4 words per row
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);

  // Thread-local champion over this thread's words (tid, tid + kBlock, ...).
  // Init: value -inf, index vocab_size - an index no real column carries,
  // ordered after every real candidate, so any non-NaN logit (including a
  // real -inf logit, via the index tiebreak) is adopted over it. A sentinel
  // index surviving the reduction therefore identifies an all-NaN row.
  float best_v = -CUDART_INF_F;
  std::int32_t best_i = vocab_size;
  for (std::int32_t w = tid; w < words; w += kBlock) {
    const uint4 raw = row8[w];
    const half* h = reinterpret_cast<const half*>(&raw);
    const std::int32_t base = w << 3;
#pragma unroll
    for (std::int32_t l = 0; l < 8; ++l) {
      const float v = __half2float(h[l]);
      if (Better(v, base + l, best_v, best_i)) {
        best_v = v;
        best_i = base + l;
      }
    }
  }

  BlockReduceArgmax<kBlock>(best_v, best_i);
  if (tid == 0) {
    // Sentinel index == vocab_size means no candidate was ever adopted: the
    // whole row was NaN. Emit token 0 deterministically (documented above).
    out_tokens[row] = (best_i >= vocab_size) ? 0 : best_i;
  }
}

// Scalar fallback: one half per load, thread-strided columns (tid, tid+256,
// ...) - total over every vocab_size/alignment the contract admits. Never
// taken for this model's 151,936 vocab.
__global__ __launch_bounds__(kBlockThreads) void GreedyArgmaxKernel(
    std::int32_t* __restrict__ out_tokens, const half* __restrict__ logits, std::int32_t num_seqs,
    std::int32_t vocab_size) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_seqs) {
    return; // bound guard, uniform across the block (grid is sized to num_seqs)
  }

  const half* __restrict__ logits_row = logits + static_cast<std::int64_t>(row) * vocab_size;
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);

  float best_v = -CUDART_INF_F;
  std::int32_t best_i = vocab_size;
  for (std::int32_t c = tid; c < vocab_size; c += kBlockThreads) {
    const float v = __half2float(logits_row[c]);
    if (Better(v, c, best_v, best_i)) {
      best_v = v;
      best_i = c;
    }
  }

  BlockReduceArgmax<kBlockThreads>(best_v, best_i);
  if (tid == 0) {
    out_tokens[row] = (best_i >= vocab_size) ? 0 : best_i;
  }
}

// Cached once at first use (engine warmup, before any CUDA graph capture);
// afterwards a plain load - same pattern as paged_attention.cu's per-arch
// dispatch.
int DeviceComputeMajor() {
  static const int major = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return 0;
    }
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      return 0;
    }
    return value;
  }();
  return major;
}

} // namespace

void LaunchGreedyArgmax(std::int32_t* out_tokens, const half* logits, std::int32_t num_seqs,
                        std::int32_t vocab_size, cudaStream_t stream) {
  if (num_seqs <= 0 || vocab_size <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to select
  }
  const dim3 grid(static_cast<unsigned int>(num_seqs)); // one block per sequence row

  // 16-byte path: whole uint4 words per row plus a 16-byte-aligned base
  // keeps every row base aligned (row offset = row * vocab_size * 2 bytes, a
  // multiple of 16 when vocab_size % 8 == 0).
  const bool vectorizable =
      vocab_size % 8 == 0 && reinterpret_cast<std::uintptr_t>(logits) % sizeof(uint4) == 0;
  if (vectorizable) {
    if (DeviceComputeMajor() >= 8) {
      GreedyArgmaxVecKernel<kBlockThreadsSm80>
          <<<grid, dim3(kBlockThreadsSm80), 0, stream>>>(out_tokens, logits, num_seqs, vocab_size);
    } else {
      GreedyArgmaxVecKernel<kBlockThreads>
          <<<grid, dim3(kBlockThreads), 0, stream>>>(out_tokens, logits, num_seqs, vocab_size);
    }
    return;
  }
  GreedyArgmaxKernel<<<grid, dim3(kBlockThreads), 0, stream>>>(out_tokens, logits, num_seqs,
                                                               vocab_size);
}

} // namespace redline::kernels
