#include "kernels/kernels.cuh"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

// rope (docs/DESIGN.md section 6.2, kernel #3) - rotary position embeddings
// applied in place to the Q and K head slices of the fused QKV GEMM output.
// One launch rotates every head of the step: for this model 12 Q + 2 K heads,
// i.e. 14 heads x 64 rotation pairs per token row at head_dim 128.
//
// Convention (docs/MODEL_SPEC.md section 3): GPT-NeoX half-rotation, exactly
// HF Qwen2's rotate_half - element d of each head pairs with element
// d + head_dim/2 (first half with second half), NOT the GPT-J interleaved
// (2d, 2d+1) pairing. The pair addresses are formed at a single commented
// site inside the kernel body (the convention switch point); nothing else in
// the engine encodes the pairing.
//
// Mapping (per-arch tuning constant, docs/PROFILING.md section 2.1): the
// angle of rotation pair d -
// positions[r] * inv_freq[d] - depends on the token row and d only, never on
// the head, so every head of a row shares the same head_dim/2 cos/sin values.
//   sm_80+ (the bench 4090): grid (num_tokens, ceil(total_heads / HPB)) with
//     HPB heads served per block (blockDim = HPB * head_dim/2, one thread per
//     rotation pair). Threads d < head_dim/2 evaluate sincosf once into
//     shared memory; all the block's pairs then read it - cutting the
//     sincosf count per row from total_heads * head_dim/2 to
//     ceil(total_heads / HPB) * head_dim/2 and giving each block enough
//     resident warps to hide its own load latency (the pre-tuning 64-thread
//     blocks left every latency exposed at decode grids).
//   pre-sm_80 (the sm_75 dev card): the original mapping - grid
//     (num_tokens, total_heads), 64-thread blocks, each thread computing its
//     own sincosf - kept verbatim: the dev preset's small batches with
//     34 SMs favor many small blocks, and the kernel suite validated this shape.
// Q heads occupy head indices [0, num_q_heads); K heads follow. Row
// addressing goes through the explicit q/k row strides of the kernels.cuh
// strided-view rule: the canonical decode call rotates q = qkv_out (row
// stride 2048) and k = qkv_out + 1536 (row stride 2048) in place;
// dense-packed tensors simply pass stride == row width. V is never touched -
// the grid does not cover it.
//
// Frequencies: inv_freq[d] = rope_theta^(-2d/head_dim), d < head_dim/2,
// mirroring HF's float32 `1.0 / (base ** (arange(0, dim, 2) / dim))`
// (docs/MODEL_SPEC.md section 3; theta = 1e6 for this model). The table is
// computed on the host in FP32 and uploaded ONCE into __constant__ memory,
// guarded by a (theta, head_dim) cache: repeat launches - every launch after
// the first - take a compare-and-return fast path that issues no CUDA API
// call besides the kernel launch itself, which keeps this launcher legal
// inside CUDA graph capture (docs/DESIGN.md section 9: the pre-capture bucket
// warmup performs the first launch, so the upload never runs in a capture
// region; that same warmup populates the cached compute-capability query
// below). A (theta, head_dim) change re-uploads after a device synchronize;
// that path serves unit tests sweeping rope_theta, never the engine hot path.
//
// Numerics (docs/MODEL_SPEC.md section 3): the angle is one FP32 multiply,
// positions[r] * inv_freq[d] - bit-identical to HF's float32
// position @ inv_freq outer product (single multiply per element, correctly
// rounded). sin/cos come from sincosf on the precise path: the build sets no
// --use_fast_math (docs/DESIGN.md section 3), and fast-math must not be
// re-enabled for this file without re-passing the tolerance suites. HF then
// casts the FP32 cos/sin cache to the activation dtype at use and applies
// `q*cos + rotate_half(q)*sin` as FP16 elementwise torch ops, so the exact HF
// rounding chain has three FP16 rounding points per output element - cos/sin
// to FP16, each product to FP16, the sum to FP16 - all reproduced below and
// marked LOAD-BEARING at their sites. (torch evaluates half elementwise ops
// at FP32 opmath and rounds once per op; fp16(fp32(a) * fp32(b)) is the
// correctly rounded half product, so each __float2half_rn here is
// bit-identical to the native half op torch runs.) Residual FP32-level
// differences against torch - sincosf vs torch's sin/cos kernels, host pow in
// the inv_freq table - are last-ulp FP32 effects far below the FP16 quantum
// and inside the unit tolerances of docs/DESIGN.md section 12(a). Sharing the
// FP16-rounded cos/sin through shared memory moves NO rounding site: the
// stored values are exactly what every thread of the pre-tuning kernel
// computed privately, so both mappings produce bitwise-identical outputs.
//
// In-place contract: each thread loads both elements of its pair into
// registers before storing either, and no two threads share an element, so
// the in-place update is race-free with no synchronization. q and k may be
// views into one buffer (the canonical qkv_out call): the Q-head and K-head
// element sets are disjoint there, so the __restrict__ qualifiers hold.
//
// Padded rows: positions is read once per row; padded batch slots carry
// position 0 (docs/DESIGN.md section 9 padding policy), giving angle 0,
// cos 1, sin 0 - an identity rotation. Padded rows pass through unchanged and
// nothing downstream reads them, so like embed_gather there is no early-exit
// path.
//
// Determinism: pure elementwise math - no reductions, no atomics - so
// identical inputs at identical shapes produce bitwise-identical outputs, a
// premise of the docs/DESIGN.md section 12 same-shape equivalence suites.

#if defined(__USE_FAST_MATH__)
#error "rope numerics require the precise-path math functions (sincosf)"
#endif

namespace redline::kernels {

namespace {

// head_dim/2 rotation pairs; sized for the model's head_dim 128 (the largest
// this engine admits - the launcher rejects anything wider).
constexpr std::int32_t kMaxRotaryPairs = 64;

// Per-arch tuning constant (docs/PROFILING.md section 2.1): heads served
// per block on sm_80+. At the
// model shape (14 heads, 64 pairs) HPB 7 gives grid (num_tokens, 2) of
// 448-thread blocks - 128 blocks at the bench decode batch 64, one per 4090
// SM - measured fastest on sm_89 across HPB {4, 7, 14} (1.527 / 1.516 /
// 1.595 us at the batch-64 decode shape vs 1.748 for the per-head mapping).
// The guarded define lets the tuning harness sweep candidates without
// touching this file.
#ifndef REDLINE_ROPE_HPB_SM80
#define REDLINE_ROPE_HPB_SM80 7
#endif
constexpr std::int32_t kHeadsPerBlockSm80 = REDLINE_ROPE_HPB_SM80;
static_assert(kHeadsPerBlockSm80 >= 1 && kHeadsPerBlockSm80 * kMaxRotaryPairs <= 1024,
              "rope heads-per-block must keep the block within 1024 threads");

// FP32 inv_freq table, one entry per rotation pair. Written only by
// EnsureInvFreqTable below; read by every rope kernel launch.
__constant__ float g_inv_freq[kMaxRotaryPairs];

// Host-side record of what g_inv_freq currently holds. theta == 0 marks
// "never uploaded" (valid thetas are positive). The mutex keeps concurrent
// launcher calls safe (the engine itself is single-threaded by contract, but
// unit tests need not be); std::mutex is constexpr-constructible, so this
// global has no initialization-order hazard.
struct InvFreqCacheState {
  std::mutex mutex;
  float theta = 0.0f;
  std::int32_t num_pairs = 0;
};
InvFreqCacheState g_inv_freq_state;

// Ensures g_inv_freq holds the table for (rope_theta, head_dim), uploading it
// at most once per distinct configuration:
//   inv_freq[d] = rope_theta^(-2d/head_dim), computed on the host in FP32 in
//   the same shape as HF's `1.0 / (base ** (arange(0, dim, 2).float() / dim))`
//   (docs/MODEL_SPEC.md section 3).
// The match path (every call after the first for the engine's fixed theta and
// head_dim) performs no CUDA API call - required during graph capture. The
// upload path synchronizes the device first so an in-flight kernel can never
// observe a half-rewritten table (a theta change while kernels using the old
// table are still running is otherwise possible in test sweeps), then copies
// synchronously: once cudaMemcpyToSymbol returns, the table is visible to
// every subsequently launched kernel on any stream. On failure the cache
// state is left unchanged (a later call retries) and the error is returned.
cudaError_t EnsureInvFreqTable(float rope_theta, std::int32_t head_dim) {
  const std::int32_t num_pairs = head_dim / 2;
  std::lock_guard<std::mutex> lock(g_inv_freq_state.mutex);
  if (g_inv_freq_state.theta == rope_theta && g_inv_freq_state.num_pairs == num_pairs) {
    return cudaSuccess;
  }

  float table[kMaxRotaryPairs];
  for (std::int32_t d = 0; d < num_pairs; ++d) {
    // Exponent 2d/head_dim is exact in FP32 for power-of-two head_dim (the
    // model's 128). d == 0 gives exactly 1.0f.
    const float exponent = static_cast<float>(2 * d) / static_cast<float>(head_dim);
    table[d] = 1.0f / std::pow(rope_theta, exponent);
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMemcpyToSymbol(g_inv_freq, table, static_cast<std::size_t>(num_pairs) * sizeof(float));
  if (err != cudaSuccess) {
    return err;
  }
  g_inv_freq_state.theta = rope_theta;
  g_inv_freq_state.num_pairs = num_pairs;
  return cudaSuccess;
}

// Rotates one pair (i_lo, i_hi) = (d, d + head_dim/2) of one head row, given
// the FP16-rounded-then-upcast cos/sin for pair index d. Shared verbatim by
// both mappings - every rounding site lives here.
__device__ void RotatePair(half* __restrict__ head_base, std::int32_t d, std::int32_t half_dim,
                           float cos_h, float sin_h) {
  // THE CONVENTION SWITCH POINT (docs/MODEL_SPEC.md section 3). NeoX-style
  // half-rotation: element d pairs with element d + head_dim/2 - first half
  // with second half, HF Qwen2's rotate_half(x) = cat(-x2, x1) - NOT the
  // GPT-J interleaved (2d, 2d+1) pairing. This is the only place the pair
  // addresses are formed; a convention change edits these two indices only.
  const std::int32_t i_lo = d;            // x1 element (first half)
  const std::int32_t i_hi = d + half_dim; // x2 element (second half)

  // Both pair elements load before either store: the in-place update of a
  // pair is confined to this thread's registers.
  const float x1 = __half2float(head_base[i_lo]);
  const float x2 = __half2float(head_base[i_hi]);

  // ROUNDING POINT 2 - LOAD-BEARING. torch evaluates q*cos and
  // rotate_half(q)*sin as FP16 elementwise ops (FP32 opmath, one correctly
  // rounded FP16 result each), so each product rounds to FP16 before the add.
  // rotate_half feeds -x2 to the first-half output and +x1 to the second-half
  // output; the FP16 negation is exact.
  const half lo_cos_term = __float2half_rn(x1 * cos_h);
  const half lo_sin_term = __float2half_rn((-x2) * sin_h);
  const half hi_cos_term = __float2half_rn(x2 * cos_h);
  const half hi_sin_term = __float2half_rn(x1 * sin_h);

  // ROUNDING POINT 3 - LOAD-BEARING. The final add is itself an FP16 torch op:
  // upcast both FP16 terms, add in FP32, round once.
  head_base[i_lo] = __float2half_rn(__half2float(lo_cos_term) + __half2float(lo_sin_term));
  head_base[i_hi] = __float2half_rn(__half2float(hi_cos_term) + __half2float(hi_sin_term));
}

// Head base pointer for (row, head): Q heads fill [0, num_q_heads); K heads
// follow with their own base pointer and row stride. Both are commonly
// strided views into qkv_out.
__device__ half* HeadBase(half* __restrict__ q, half* __restrict__ k, std::int64_t q_row_stride,
                          std::int64_t k_row_stride, std::int32_t row, std::int32_t head,
                          std::int32_t num_q_heads, std::int32_t head_dim) {
  return (head < num_q_heads)
             ? q + static_cast<std::int64_t>(row) * q_row_stride + head * head_dim
             : k + static_cast<std::int64_t>(row) * k_row_stride + (head - num_q_heads) * head_dim;
}

// sm_80+ mapping: one block serves kHeadsPerBlockSm80 heads of one token row
// (grid (num_tokens, ceil(total_heads / HPB)); blockDim = HPB * head_dim/2).
// Threads d < head_dim/2 evaluate the row's angle table once into shared
// memory - including ROUNDING POINT 1, so consumers read exactly the values
// the private-sincosf mapping computes - then thread t serves head
// blockIdx.y * HPB + t / half_dim, pair t % half_dim. A trailing grid-y
// block with fewer live heads simply retires its out-of-range threads after
// the barrier.
__global__ __launch_bounds__(kHeadsPerBlockSm80* kMaxRotaryPairs) void RopeInplaceSharedKernel(
    half* __restrict__ q, half* __restrict__ k, std::int64_t q_row_stride,
    std::int64_t k_row_stride, const std::int32_t* __restrict__ positions, std::int32_t num_q_heads,
    std::int32_t total_heads, std::int32_t head_dim) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  const std::int32_t half_dim = head_dim >> 1;

  // One positions load per block, the same address across the block (a
  // broadcast read). Padded rows carry position 0 -> identity rotation.
  const float pos = static_cast<float>(positions[row]);

  // {cos_h, sin_h} per rotation pair, already through ROUNDING POINT 1 -
  // LOAD-BEARING (docs/MODEL_SPEC.md section 3): HF computes the cos/sin
  // cache in FP32 and casts it to the activation dtype (FP16) at use; the
  // FP16-rounded values are what multiply q and k. The angle is one FP32
  // multiply, matching HF's float32 position x inv_freq product; HF builds
  // its cache as cat(freqs, freqs), so both elements of the pair share this
  // angle (cos[d] == cos[d + half_dim]).
  __shared__ float2 rot[kMaxRotaryPairs];
  if (tid < half_dim) {
    const float angle = pos * g_inv_freq[tid];
    float sin32;
    float cos32;
    sincosf(angle, &sin32, &cos32); // precise path - this build has no fast-math
    rot[tid] =
        make_float2(__half2float(__float2half_rn(cos32)), __half2float(__float2half_rn(sin32)));
  }
  __syncthreads();

  const std::int32_t head =
      static_cast<std::int32_t>(blockIdx.y) * kHeadsPerBlockSm80 + tid / half_dim;
  if (head >= total_heads) {
    return; // trailing grid-y block covering fewer than HPB heads
  }
  const std::int32_t d = tid % half_dim; // rotation pair index
  half* head_base = HeadBase(q, k, q_row_stride, k_row_stride, row, head, num_q_heads, head_dim);
  const float2 cs = rot[d];
  RotatePair(head_base, d, half_dim, cs.x, cs.y);
}

// Pre-sm_80 mapping (the original): one block per (token row, head), one
// thread per rotation pair (blockDim = head_dim/2), each thread evaluating
// its own sincosf. gridDim.x == num_tokens exactly (set by the launcher), so
// no row bound check is needed.
__global__ __launch_bounds__(kMaxRotaryPairs) void RopeInplaceKernel(
    half* __restrict__ q, half* __restrict__ k, std::int64_t q_row_stride,
    std::int64_t k_row_stride, const std::int32_t* __restrict__ positions, std::int32_t num_q_heads,
    std::int32_t head_dim) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t head = static_cast<std::int32_t>(blockIdx.y);
  const std::int32_t d = static_cast<std::int32_t>(threadIdx.x); // rotation pair index
  const std::int32_t half_dim = head_dim >> 1;                   // == blockDim.x

  const float pos = static_cast<float>(positions[row]);
  half* head_base = HeadBase(q, k, q_row_stride, k_row_stride, row, head, num_q_heads, head_dim);

  const float angle = pos * g_inv_freq[d];
  float sin32;
  float cos32;
  sincosf(angle, &sin32, &cos32); // precise path - this build has no fast-math

  // ROUNDING POINT 1 - LOAD-BEARING; see the shared-memory kernel's site.
  const float cos_h = __half2float(__float2half_rn(cos32));
  const float sin_h = __half2float(__float2half_rn(sin32));

  RotatePair(head_base, d, half_dim, cos_h, sin_h);
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

void LaunchRopeInplace(half* q, half* k, std::int64_t q_row_stride, std::int64_t k_row_stride,
                       const std::int32_t* positions, std::int32_t num_tokens,
                       std::int32_t num_q_heads, std::int32_t num_kv_heads, std::int32_t head_dim,
                       float rope_theta, cudaStream_t stream) {
  const std::int32_t total_heads = num_q_heads + num_kv_heads;
  if (num_tokens <= 0 || total_heads <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to rotate
  }

  // Contract violations: debug-asserted, and hard-skipped in release so a bad
  // head_dim can never index past the constant table or launch an odd block.
  assert(num_q_heads >= 0 && num_kv_heads >= 0);
  assert(head_dim > 0 && head_dim % 2 == 0 && head_dim / 2 <= kMaxRotaryPairs);
  assert(rope_theta > 0.0f);
  if (num_q_heads < 0 || num_kv_heads < 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      head_dim / 2 > kMaxRotaryPairs || rope_theta <= 0.0f) {
    return;
  }

  if (EnsureInvFreqTable(rope_theta, head_dim) != cudaSuccess) {
    // Launching would read a stale or uninitialized table; skip. The upload
    // error stays observable through cudaGetLastError, and only a broken
    // context reaches here (first launch runs at init, long before capture).
    return;
  }

  const std::int32_t half_dim = head_dim / 2;
  if (DeviceComputeMajor() >= 8 && kHeadsPerBlockSm80 > 1) {
    // sm_80+: HPB heads per block, shared cos/sin (see the mapping note in
    // the file header).
    const std::int32_t head_groups = (total_heads + kHeadsPerBlockSm80 - 1) / kHeadsPerBlockSm80;
    const dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(head_groups));
    const dim3 block(static_cast<unsigned int>(kHeadsPerBlockSm80 * half_dim));
    RopeInplaceSharedKernel<<<grid, block, 0, stream>>>(q, k, q_row_stride, k_row_stride, positions,
                                                        num_q_heads, total_heads, head_dim);
    return;
  }

  const dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(total_heads));
  const dim3 block(static_cast<unsigned int>(half_dim)); // one thread per rotation pair
  RopeInplaceKernel<<<grid, block, 0, stream>>>(q, k, q_row_stride, k_row_stride, positions,
                                                num_q_heads, head_dim);
}

} // namespace redline::kernels
