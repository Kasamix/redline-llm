#include "kernels/kernels.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <cuda_pipeline_primitives.h>

// Paged grouped-query decode attention: the primary KV-head-centric kernel
// plus the naive per-Q-head test oracle.
//
// Block-table indirection and the paged-cache concept come from the
// PagedAttention paper (Kwon et al., 2023, arXiv:2309.06180). The kernels
// below are original implementations.
//
// Primary kernel (docs/DESIGN.md section 6.2 #6): one CTA per
// (sequence, KV head). The CTA serves the num_q_heads / num_kv_heads = 6
// query heads of its GQA group (q_head -> kv_head = q_head / 6), so every
// 16-token K/V tile is staged from DRAM once and shared across the group
// instead of being re-streamed per query head. Per tile: lane-strided dot
// products + warp shuffle reductions produce the 6x16 scores with
// SELECT-style masking (pos < seq_len ? dot : -inf); a per-head online
// softmax (FP32 running max m[h], denominator l[h], rescale
// alpha = exp(m_old - m_new)) folds the tile into a 6x128 FP32 shared-memory
// accumulator; the epilogue normalizes by l[h] and writes FP16 through the
// caller's row stride.
//
// The kernel is DRAM-traffic-bound by design (it must stream the sequence's
// whole cached K/V once per step), so the launch configuration and staging
// path are tuned per arch - see the tuning constants below. The math,
// masking, and output contract are identical across arches.
//
// Oracle kernel (docs/DESIGN.md section 12a): one warp per
// (sequence, Q head), textbook two-pass softmax over the same block-table
// walk, register accumulator, no shared memory and no tile staging. It is
// deliberately structured unlike the primary kernel so the unit-test
// cross-check exercises independent code paths; it is compiled for tests
// only and never called on the hot path.

namespace redline::kernels {
namespace {

// Frozen contract constants. kKvBlockSize matches the in-kernel slot math of
// the pool layout (p >> 4, p & 15 - docs/DESIGN.md section 5); the head
// geometry is the Qwen2.5-1.5B shape the v1 kernels are specialized to
// (docs/MODEL_SPEC.md section 1: 12 Q heads, 2 KV heads, head_dim 128).
constexpr int kKvBlockSize = 16; // tokens per KV pool block
constexpr int kHeadDim = 128;    // elements per head
constexpr int kGroupSize = 6;    // query heads per KV head (12 / 2)
constexpr int kWarpSize = 32;
constexpr int kDimsPerLane = kHeadDim / kWarpSize; // oracle: dims owned per lane

// Per-arch launch configuration (measured tuning pass, docs/PROFILING.md
// section 4; the kernels.cuh contract and
// the kernel's math are unchanged). One CTA serves one (sequence, KV head)
// pair in both configurations:
//   pre-sm_80 (dev floor, sm_75): 128 threads / 4 warps with single-buffered
//     synchronous staging. Decode grids (num_seqs x 2 KV heads) oversubscribe
//     that part's SM count several times over, so DRAM latency is hidden
//     across resident CTAs.
//   sm_80+ (bench arch, sm_89): 256 threads / 8 warps and a two-stage
//     cp.async pipeline that prefetches tile t+1's K/V while tile t is
//     computed. At the bench decode shape the grid is ~1 CTA per SM
//     (64 seqs x 2 KV heads = 128 CTAs on 128 SMs), so warps-per-SM and
//     prefetch depth - not CTA count - set how much DRAM latency each SM
//     can hide.
constexpr int kThreadsPreAmpere = 128;
constexpr int kThreadsAmpere = 256;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define REDLINE_PA_CPASYNC 1
#else
#define REDLINE_PA_CPASYNC 0
#endif
constexpr int kKvStages = REDLINE_PA_CPASYNC ? 2 : 1; // staged K/V smem buffers

constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

static_assert(kDimsPerLane * kWarpSize == kHeadDim,
              "the oracle's register accumulator tiles head_dim exactly across one warp");

// K/V tiles are staged as 16-byte vectors: one (block, kv, kv_head) tile is
// 16x128 contiguous halves at a 4 KiB-aligned pool offset (PoolRowOffset is
// a multiple of 2048 elements), so a tile moves as 256 uint4 copies instead
// of 2048 scalar half loads (512 B per warp per load instruction).
constexpr int kVecHalves = 8; // halves per uint4
constexpr int kTileVecs = kKvBlockSize * kHeadDim / kVecHalves;
static_assert(kKvBlockSize * kHeadDim % kVecHalves == 0);

// Element offset of (block, kv, kv_head, slot, dim 0) inside ONE layer's pool
// slice [num_blocks, 2 (0=K,1=V), num_kv_heads, kKvBlockSize, kHeadDim]
// (docs/DESIGN.md section 5). The innermost [kKvBlockSize, kHeadDim] tile of a
// (block, kv, kv_head) triple is contiguous, giving coalesced 256 B rows.
__device__ inline std::int64_t PoolRowOffset(std::int64_t block, int kv, int kv_head,
                                             int num_kv_heads, int slot) {
  return (((block * 2 + kv) * num_kv_heads + kv_head) * kKvBlockSize + slot) *
         static_cast<std::int64_t>(kHeadDim);
}

// Shared memory of one primary-kernel CTA. Budget: FP32 Q tile 3 KiB + K/V
// tiles 8 KiB per stage + softmax state - 11.6 KiB single-stage (pre-sm_80)
// / 19.6 KiB double-buffered (sm_80+), both well inside the 48 KiB/block
// static shared-memory limit of every target arch (Turing is the floor,
// docs/DESIGN.md section 6.2). The online-softmax output accumulator lives
// in registers (each thread owns fixed (head, dim) output elements across
// all tiles), not in shared memory.
struct alignas(16) SharedStorage {
  float q_tile[kGroupSize][kHeadDim];                         // group's queries, pre-scaled FP32
  alignas(16) half k_tile[kKvStages][kKvBlockSize][kHeadDim]; // staged K tile(s)
  alignas(16) half v_tile[kKvStages][kKvBlockSize][kHeadDim]; // staged V tile(s)
  float scores[kGroupSize][kKvBlockSize];                     // raw scores, then probabilities
  float m[kGroupSize];                                        // running max per head
  float l[kGroupSize];                                        // running denominator per head
  float alpha[kGroupSize]; // exp(m_old - m_new) for the current tile
};
static_assert(sizeof(SharedStorage) <= 48 * 1024, "decode attention smem budget exceeded");

// Primary kernel. Grid (num_seqs, num_kv_heads); block kThreads (per-arch
// tuning constant, selected by the launcher at run time).
template <int kThreads>
__global__ __launch_bounds__(kThreads) void PagedAttentionDecodeKernel(
    half* __restrict__ out, std::int64_t out_row_stride, const half* __restrict__ q,
    std::int64_t q_row_stride, const half* __restrict__ kv_pool_layer,
    const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ seq_lens,
    std::int32_t max_blocks_per_seq, std::int32_t num_kv_heads, float scale) {
  constexpr int kWarps = kThreads / kWarpSize;
  static_assert(kThreads % kWarpSize == 0);
  static_assert(kGroupSize * kKvBlockSize <= kThreads,
                "the probability phase assigns one thread per (head, slot) score");
  static_assert(kGroupSize * kHeadDim % kThreads == 0,
                "the accumulate/epilogue mapping assumes kThreads tiles the group's output");
  static_assert(kThreads % kHeadDim == 0,
                "the accumulate/epilogue mapping fixes one output column per thread");
  // Register accumulator mapping: thread `tid` owns output column
  // d = tid % 128 of heads h0 + j*kHeadStride - all tiles, then the epilogue.
  constexpr int kHeadsPerThread = kGroupSize * kHeadDim / kThreads; // 6 (128t) / 3 (256t)
  constexpr int kHeadStride = kThreads / kHeadDim;                  // 1 (128t) / 2 (256t)

  const int seq = static_cast<int>(blockIdx.x);
  const int kv_head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  // seq_lens is uniform across the CTA, so this early exit retires the whole
  // block together (no divergent __syncthreads later). The scheduler
  // guarantees seq_len <= max_blocks_per_seq * kKvBlockSize; the clamp keeps
  // every block-table read in bounds even if that contract is ever violated.
  const int seq_len = min(seq_lens[seq], max_blocks_per_seq * kKvBlockSize);
  if (seq_len <= 0) {
    return; // padded batch slot: CTA exits immediately
  }

  __shared__ SharedStorage s;

  // Stage the group's 6 query heads as FP32 pre-scaled by 1/sqrt(head_dim)
  // so the per-tile dot products need no conversion or extra multiply; init
  // the softmax state. This CTA's heads are q_head = kv_head*kGroupSize + h.
  const half* q_row = q + static_cast<std::int64_t>(seq) * q_row_stride +
                      static_cast<std::int64_t>(kv_head) * kGroupSize * kHeadDim;
  for (int i = tid; i < kGroupSize * kHeadDim; i += kThreads) {
    s.q_tile[i / kHeadDim][i % kHeadDim] = __half2float(q_row[i]) * scale;
  }
  if (tid < kGroupSize) {
    s.m[tid] = kNegInf;
    s.l[tid] = 0.0f;
  }
  const int d_own = tid % kHeadDim; // this thread's output column
  const int h_own = tid / kHeadDim; // first owned head
  float acc[kHeadsPerThread] = {};  // FP32 output accumulator, in registers

  const std::int32_t* table_row =
      block_tables + static_cast<std::int64_t>(seq) * max_blocks_per_seq;
  const int num_tiles = (seq_len + kKvBlockSize - 1) / kKvBlockSize;

  // Stage one tile's K and V into smem buffer `buf` as 16-byte vectors
  // (kTileVecs note above). Physical blocks are arbitrary: every tile
  // re-reads its own table entry (no contiguity assumption). Entries past
  // num_tiles hold the dummy block (0) and are never dereferenced; even a
  // speculative touch would stay in bounds because block 0 is real,
  // reserved pool memory. Slots >= tile_valid may hold garbage or NaN from
  // never-written pool memory; the select mask and the bounded V loop below
  // keep those values out of every result. On sm_80+ the copies are issued
  // as one cp.async commit group and complete asynchronously (two-stage
  // pipeline overlapping the previous tile's compute); pre-sm_80 they are
  // plain vector loads into the single buffer.
  const auto stage_tile = [&](int tile, int buf) {
    const std::int64_t block = table_row[tile];
    const uint4* k4 = reinterpret_cast<const uint4*>(
        kv_pool_layer + PoolRowOffset(block, /*kv=*/0, kv_head, num_kv_heads, 0));
    const uint4* v4 = reinterpret_cast<const uint4*>(
        kv_pool_layer + PoolRowOffset(block, /*kv=*/1, kv_head, num_kv_heads, 0));
    uint4* kt = reinterpret_cast<uint4*>(&s.k_tile[buf][0][0]);
    uint4* vt = reinterpret_cast<uint4*>(&s.v_tile[buf][0][0]);
#pragma unroll
    for (int i = tid; i < kTileVecs; i += kThreads) {
#if REDLINE_PA_CPASYNC
      __pipeline_memcpy_async(&kt[i], &k4[i], sizeof(uint4));
      __pipeline_memcpy_async(&vt[i], &v4[i], sizeof(uint4));
#else
      kt[i] = k4[i];
      vt[i] = v4[i];
#endif
    }
  };

#if REDLINE_PA_CPASYNC
  stage_tile(0, 0);
  __pipeline_commit();
#endif

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_start = tile * kKvBlockSize;
    const int tile_valid = min(kKvBlockSize, seq_len - tile_start);

#if REDLINE_PA_CPASYNC
    // (a) Prefetch tile+1 into the other buffer, then wait for tile's own
    // group. Overwriting buffer (tile+1)&1 is safe: its last consumers ran
    // in iteration tile-1, which ended with __syncthreads().
    const int buf = tile & 1;
    if (tile + 1 < num_tiles) {
      stage_tile(tile + 1, (tile + 1) & 1);
      __pipeline_commit();
      __pipeline_wait_prior(1); // tile's group done; tile+1's stays in flight
    } else {
      __pipeline_wait_prior(0);
    }
#else
    // (a) Single-buffer synchronous staging; the trailing __syncthreads() of
    // the previous iteration already retired all reads of the buffer.
    const int buf = 0;
    stage_tile(tile, buf);
#endif
    __syncthreads(); // staged K/V visible CTA-wide

    const half(*k_tile)[kHeadDim] = s.k_tile[buf];
    const half(*v_tile)[kHeadDim] = s.v_tile[buf];

    // (b) 6x16 scores: each warp takes (head, slot) pairs in a strided loop;
    // lanes stride the 128 dims and a shuffle tree reduces the partials
    // (fixed tree, no atomics - deterministic). SELECT-style masking
    // (docs/DESIGN.md section 6.2): a dot computed over poisoned smem values
    // may be garbage or NaN, and the select DISCARDS it entirely. An additive
    // `dot + (-inf)` mask would propagate NaN and is forbidden - the
    // NaN-poisoned-pool unit test (section 12a) enforces this.
    // Eight lanes cooperate on each (head, slot) score: lane sub-index
    // `sub` owns dims [64c + 8*sub, +8) for c in {0,1} - one 16-byte FP32 Q
    // pair and one 16-byte K read per chunk (16-byte alignment: Q rows are
    // 512 B, K rows 256 B) - then a 3-step shuffle tree reduces the eight
    // partials (fixed tree, no atomics - deterministic).
    constexpr int kScoreLanes = 8; // lanes per (head, slot) dot product
    const int sub = lane % kScoreLanes;
    const int wpair = lane / kScoreLanes;
    for (int pair = warp * (kWarpSize / kScoreLanes) + wpair; pair < kGroupSize * kKvBlockSize;
         pair += kWarps * (kWarpSize / kScoreLanes)) {
      const int h = pair / kKvBlockSize;
      const int t = pair % kKvBlockSize;
      float dot = 0.0f;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int d = 64 * c + 8 * sub;
        const float4* qf4 = reinterpret_cast<const float4*>(&s.q_tile[h][d]);
        const float4 qa = qf4[0];
        const float4 qb = qf4[1];
        const uint4 kbits = *reinterpret_cast<const uint4*>(&k_tile[t][d]);
        const half2* kh = reinterpret_cast<const half2*>(&kbits);
        const float2 k0 = __half22float2(kh[0]);
        const float2 k1 = __half22float2(kh[1]);
        const float2 k2 = __half22float2(kh[2]);
        const float2 k3 = __half22float2(kh[3]);
        dot = fmaf(qa.x, k0.x, dot);
        dot = fmaf(qa.y, k0.y, dot);
        dot = fmaf(qa.z, k1.x, dot);
        dot = fmaf(qa.w, k1.y, dot);
        dot = fmaf(qb.x, k2.x, dot);
        dot = fmaf(qb.y, k2.y, dot);
        dot = fmaf(qb.z, k3.x, dot);
        dot = fmaf(qb.w, k3.y, dot);
      }
#pragma unroll
      for (int off = kScoreLanes / 2; off > 0; off >>= 1) {
        dot += __shfl_down_sync(kFullWarpMask, dot, off);
      }
      if (sub == 0) {
        // SELECT-style masking (docs/DESIGN.md section 6.2): a dot computed
        // over poisoned smem values may be garbage or NaN, and the select
        // DISCARDS it entirely. An additive `dot + (-inf)` mask would
        // propagate NaN and is forbidden - the NaN-poisoned-pool unit test
        // (section 12a) enforces this.
        s.scores[h][t] = (tile_start + t < seq_len) ? dot : kNegInf;
      }
    }
    __syncthreads();

    // (c) Per-head online-softmax bookkeeping fused with the
    // scores -> probabilities pass: 16 lanes per head reduce the tile max
    // and the probability sum with xor-shuffle trees (offsets < 16 keep the
    // two heads of a warp separate; fixed trees, deterministic); every lane
    // recomputes m_new/alpha redundantly and lane t == 0 commits the head
    // state. Every processed tile contains at least one in-range slot
    // (tile_start < seq_len), so m becomes finite on the first tile;
    // exp(-inf - finite) == 0 covers the initial state. Masked slots get an
    // explicit probability 0 (select again, not exp(-inf - m), so no
    // arithmetic on the discarded value can ever reach the accumulator).
    if (tid < kGroupSize * kKvBlockSize) {
      const int h = tid / kKvBlockSize;
      const int t = tid % kKvBlockSize;
      const float sc = s.scores[h][t];
      float tile_max = sc;
#pragma unroll
      for (int off = kKvBlockSize / 2; off > 0; off >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullWarpMask, tile_max, off));
      }
      const float m_new = fmaxf(s.m[h], tile_max);
      const float prob = (t < tile_valid) ? expf(sc - m_new) : 0.0f;
      s.scores[h][t] = prob;
      float tile_sum = prob;
#pragma unroll
      for (int off = kKvBlockSize / 2; off > 0; off >>= 1) {
        tile_sum += __shfl_xor_sync(kFullWarpMask, tile_sum, off);
      }
      if (t == 0) {
        const float alpha = expf(s.m[h] - m_new);
        s.alpha[h] = alpha;
        s.m[h] = m_new;
        s.l[h] = s.l[h] * alpha + tile_sum;
      }
    }
    __syncthreads();

    // (e) Fold the tile into the register accumulator. Thread `tid` owns
    // column d_own of its kHeadsPerThread heads, so the v_tile reads stay
    // coalesced across the CTA, and one V element read per token serves
    // every owned head. The V loop is bounded by tile_valid: probabilities
    // beyond it are 0, but the staged V values there can be NaN and 0 * NaN
    // would poison the accumulator. Full tiles (every tile but possibly the
    // last) take the compile-time-unrolled path.
#pragma unroll
    for (int j = 0; j < kHeadsPerThread; ++j) {
      acc[j] *= s.alpha[h_own + j * kHeadStride];
    }
    const auto fold_token = [&](int t) {
      const float vf = __half2float(v_tile[t][d_own]);
#pragma unroll
      for (int j = 0; j < kHeadsPerThread; ++j) {
        acc[j] = fmaf(s.scores[h_own + j * kHeadStride][t], vf, acc[j]);
      }
    };
    if (tile_valid == kKvBlockSize) {
#pragma unroll
      for (int t = 0; t < kKvBlockSize; ++t) {
        fold_token(t);
      }
    } else {
      for (int t = 0; t < tile_valid; ++t) {
        fold_token(t);
      }
    }
    __syncthreads(); // next tile rewrites scores and reuses a K/V buffer
  }

  // Epilogue: out[seq, (kv_head*6 + h)*128 + d] = fp16(acc[h][d] / l[h]),
  // straight from each thread's register accumulator. l[h] >=
  // exp(m_final - m_final) == 1 (the running-max position always
  // contributes 1), so the division is well-defined.
  half* out_row = out + static_cast<std::int64_t>(seq) * out_row_stride +
                  static_cast<std::int64_t>(kv_head) * kGroupSize * kHeadDim;
#pragma unroll
  for (int j = 0; j < kHeadsPerThread; ++j) {
    const int h = h_own + j * kHeadStride;
    out_row[h * kHeadDim + d_own] = __float2half(acc[j] / s.l[h]);
  }
}

// Test oracle. Grid (num_seqs, num_q_heads); block = one warp. Textbook
// two-pass softmax: pass 1 finds the exact global max, pass 2 accumulates the
// denominator and the weighted V sum. K is deliberately read twice and every
// (seq, q_head) re-streams its group's K/V - obviously correct beats fast
// here. Only positions < seq_len are ever addressed, so poisoned pool slots
// cannot enter any intermediate value.
__global__ __launch_bounds__(kWarpSize) void PagedAttentionDecodeOracleKernel(
    half* __restrict__ out, std::int64_t out_row_stride, const half* __restrict__ q,
    std::int64_t q_row_stride, const half* __restrict__ kv_pool_layer,
    const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ seq_lens,
    std::int32_t max_blocks_per_seq, std::int32_t num_kv_heads, std::int32_t q_heads_per_kv,
    float scale) {
  const int seq = static_cast<int>(blockIdx.x);
  const int q_head = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);

  const int seq_len = min(seq_lens[seq], max_blocks_per_seq * kKvBlockSize);
  if (seq_len <= 0) {
    return; // padded batch slot
  }
  const int kv_head = q_head / q_heads_per_kv; // GQA head mapping
  const std::int32_t* table_row =
      block_tables + static_cast<std::int64_t>(seq) * max_blocks_per_seq;

  // This head's query, scaled once in FP32; lane owns dims lane + 32*j.
  const half* q_row = q + static_cast<std::int64_t>(seq) * q_row_stride +
                      static_cast<std::int64_t>(q_head) * kHeadDim;
  float qv[kDimsPerLane];
#pragma unroll
  for (int j = 0; j < kDimsPerLane; ++j) {
    qv[j] = __half2float(q_row[lane + j * kWarpSize]) * scale;
  }

  // score(pos) = <scaled q, K[pos]> through the block table: lane-strided
  // partial sums + butterfly shuffle, every lane ends with the full dot.
  const auto score_at = [&](int pos) -> float {
    const std::int64_t k_base = PoolRowOffset(table_row[pos / kKvBlockSize], /*kv=*/0, kv_head,
                                              num_kv_heads, pos % kKvBlockSize);
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kDimsPerLane; ++j) {
      dot = fmaf(qv[j], __half2float(kv_pool_layer[k_base + lane + j * kWarpSize]), dot);
    }
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      dot += __shfl_xor_sync(kFullWarpMask, dot, off);
    }
    return dot;
  };

  // Pass 1: global max.
  float m = kNegInf;
  for (int pos = 0; pos < seq_len; ++pos) {
    m = fmaxf(m, score_at(pos));
  }

  // Pass 2: denominator + weighted V accumulation (FP32 registers).
  float l = 0.0f;
  float acc[kDimsPerLane] = {};
  for (int pos = 0; pos < seq_len; ++pos) {
    const float p = expf(score_at(pos) - m);
    l += p;
    const std::int64_t v_base = PoolRowOffset(table_row[pos / kKvBlockSize], /*kv=*/1, kv_head,
                                              num_kv_heads, pos % kKvBlockSize);
#pragma unroll
    for (int j = 0; j < kDimsPerLane; ++j) {
      acc[j] = fmaf(p, __half2float(kv_pool_layer[v_base + lane + j * kWarpSize]), acc[j]);
    }
  }

  // l >= exp(m - m) == 1, so the division is well-defined.
  half* out_row = out + static_cast<std::int64_t>(seq) * out_row_stride +
                  static_cast<std::int64_t>(q_head) * kHeadDim;
#pragma unroll
  for (int j = 0; j < kDimsPerLane; ++j) {
    out_row[lane + j * kWarpSize] = __float2half(acc[j] / l);
  }
}

// Host-side guard for wiring errors against the frozen kernels.cuh contract.
// These are programming errors (not runtime conditions callers could handle),
// so they fail loudly instead of silently mis-launching.
void RequireContract(bool ok, const char* function, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "redline: %s: launch contract violated: %s\n", function, what);
    std::abort();
  }
}

void CheckCommonContract(const char* function, const half* out, std::int64_t out_row_stride,
                         const half* q, std::int64_t q_row_stride, const half* kv_pool_layer,
                         const std::int32_t* block_tables, const std::int32_t* seq_lens,
                         std::int32_t max_blocks_per_seq, std::int32_t num_q_heads,
                         std::int32_t num_kv_heads, std::int32_t head_dim) {
  RequireContract(out != nullptr && q != nullptr && kv_pool_layer != nullptr &&
                      block_tables != nullptr && seq_lens != nullptr,
                  function, "null device pointer");
  RequireContract(head_dim == kHeadDim, function,
                  "head_dim must be 128 (v1 kernels are specialized to the model shape, "
                  "docs/MODEL_SPEC.md section 1)");
  RequireContract(num_kv_heads >= 1 && num_q_heads >= num_kv_heads &&
                      num_q_heads % num_kv_heads == 0,
                  function, "num_q_heads must be a positive multiple of num_kv_heads");
  RequireContract(max_blocks_per_seq >= 1, function, "max_blocks_per_seq must be >= 1");
  const std::int64_t row_width = static_cast<std::int64_t>(num_q_heads) * head_dim;
  RequireContract(q_row_stride >= row_width && out_row_stride >= row_width, function,
                  "row stride smaller than the row itself");
}

// The primary kernel's block size is a per-arch tuning constant, so the
// launcher picks the instantiation from the device's compute capability -
// queried once and cached (first use is engine warmup, before any CUDA
// graph capture; afterwards this is a plain load).
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

void LaunchPagedAttentionDecode(half* out, std::int64_t out_row_stride, const half* q,
                                std::int64_t q_row_stride, const half* kv_pool_layer,
                                const std::int32_t* block_tables, const std::int32_t* seq_lens,
                                std::int32_t num_seqs, std::int32_t max_blocks_per_seq,
                                std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                std::int32_t head_dim, float scale, cudaStream_t stream) {
  if (num_seqs <= 0) {
    return;
  }
  CheckCommonContract("LaunchPagedAttentionDecode", out, out_row_stride, q, q_row_stride,
                      kv_pool_layer, block_tables, seq_lens, max_blocks_per_seq, num_q_heads,
                      num_kv_heads, head_dim);
  RequireContract(num_q_heads == num_kv_heads * kGroupSize, "LaunchPagedAttentionDecode",
                  "primary kernel requires the 6-queries-per-KV-head grouping (12 Q / 2 KV)");

  const dim3 grid(static_cast<unsigned>(num_seqs), static_cast<unsigned>(num_kv_heads));
  if (DeviceComputeMajor() >= 8) {
    PagedAttentionDecodeKernel<kThreadsAmpere><<<grid, kThreadsAmpere, 0, stream>>>(
        out, out_row_stride, q, q_row_stride, kv_pool_layer, block_tables, seq_lens,
        max_blocks_per_seq, num_kv_heads, scale);
  } else {
    PagedAttentionDecodeKernel<kThreadsPreAmpere><<<grid, kThreadsPreAmpere, 0, stream>>>(
        out, out_row_stride, q, q_row_stride, kv_pool_layer, block_tables, seq_lens,
        max_blocks_per_seq, num_kv_heads, scale);
  }
}

void LaunchPagedAttentionDecodeOracle(half* out, std::int64_t out_row_stride, const half* q,
                                      std::int64_t q_row_stride, const half* kv_pool_layer,
                                      const std::int32_t* block_tables,
                                      const std::int32_t* seq_lens, std::int32_t num_seqs,
                                      std::int32_t max_blocks_per_seq, std::int32_t num_q_heads,
                                      std::int32_t num_kv_heads, std::int32_t head_dim, float scale,
                                      cudaStream_t stream) {
  if (num_seqs <= 0) {
    return;
  }
  CheckCommonContract("LaunchPagedAttentionDecodeOracle", out, out_row_stride, q, q_row_stride,
                      kv_pool_layer, block_tables, seq_lens, max_blocks_per_seq, num_q_heads,
                      num_kv_heads, head_dim);

  const dim3 grid(static_cast<unsigned>(num_seqs), static_cast<unsigned>(num_q_heads));
  PagedAttentionDecodeOracleKernel<<<grid, kWarpSize, 0, stream>>>(
      out, out_row_stride, q, q_row_stride, kv_pool_layer, block_tables, seq_lens,
      max_blocks_per_seq, num_kv_heads, num_q_heads / num_kv_heads, scale);
}

} // namespace redline::kernels
