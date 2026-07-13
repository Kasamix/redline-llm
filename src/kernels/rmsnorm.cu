#include "kernels/kernels.cuh"

#include <cassert>
#include <cstdint>

// rmsnorm / fused_add_rmsnorm (docs/DESIGN.md section 6.2, kernel #2) - the
// pre-norm normalization of the model. Plain RMSNorm feeds only layer 0's
// input_layernorm; the fused residual-add variant serves the other 56 norm
// sites of the 28-layer stack plus the final model.norm (docs/DESIGN.md
// section 6.3 step composition; 57 norm vectors total).
//
// Mapping: one block per token row, thread-strided over the hidden dimension.
// Two logical passes over the row:
//   pass 1: accumulate the FP32 sum of squares (the fused variant also
//           materializes the new residual during this pass);
//   reduce: deterministic block reduction + rsqrt;
//   pass 2: normalize, round, apply the weight.
// The primary (vectorized) kernels move the row as half2 words and keep the
// pass-1 values in registers, so pass 2 issues no second global read of the
// row; a scalar two-pass kernel remains as the totality fallback for shapes
// the vector premise does not cover (odd hidden_size, misaligned bases, or
// rows wider than the register cache - never the engine's buffers).
//
// Block size is a per-arch tuning constant (docs/PROFILING.md section 2.1):
// at decode batch sizes
// the grid is one block per row (64 blocks at the bench decode shape - half
// of a 4090's 128 SMs), so each row's latency must be hidden by its own
// block's warps; 512 threads measured fastest on sm_89 (tuning harness:
// 256 -> 512 took the fused kernel 1.77 -> 1.64 us at batch 64). Pre-sm_80
// keeps the kernel-suite-validated 256-thread shape.
//
// Numerics contract (docs/MODEL_SPEC.md section 4; kernels.cuh): FP16
// storage, FP32 accumulation, and the HF Qwen2RMSNorm operation order
// reproduced exactly. Two FP16 roundings are load-bearing for token parity
// and are marked LOAD-BEARING at their exact sites below:
//   ROUNDING POINT 1 (fused variant only) - the residual sum
//     h = fp16(fp32(residual) + fp32(input)) is rounded to FP16 *before* the
//     mean-square accumulates over fp32(h). HF materializes the FP16 tensor
//     `hidden_states = residual + x` and Qwen2RMSNorm upcasts that *rounded*
//     tensor; normalizing the unrounded FP32 sum is a systematic drift across
//     all 57 norm sites that unit tolerances never catch.
//   ROUNDING POINT 2 (both kernels) - the normalized value is rounded to
//     FP16 *before* the FP16 weight multiply, matching Qwen2RMSNorm's
//     `weight * hidden_states.to(input_dtype)`.
// eps sits inside the rsqrt argument - rsqrt(mean_sq + eps) - exactly as
// HF's torch.rsqrt(variance + self.variance_epsilon).
//
// Determinism: per-thread partials accumulate in a fixed column order and
// collapse through a fixed reduction tree (lane-xor shuffle inside each warp,
// then an ordered scan of the warp partials - no atomics, no warp-arrival
// ordering anywhere), so identical inputs at identical shapes give
// bitwise-identical outputs, a premise of the docs/DESIGN.md section 12
// same-shape equivalence suites. The tree SHAPE is fixed per launch
// configuration; it pairs additions differently than the pre-tuning shared
// -memory halving tree (and the vector path partitions columns per thread
// differently than the scalar path), which moves the FP32 sum by at most a
// couple of ulp - far below the FP16 quantum and inside the section 12(a)
// tolerances, re-verified by suites (a)/(e) (plus (c) as prudence) on the
// tuned build.

// The numerics above assume the precise-path device math functions (rsqrtf
// below; the rope kernel's sincosf rests on the same premise). nvcc defines
// __USE_FAST_MATH__ under --use_fast_math, which would silently swap in the
// approximate intrinsics - make that a compile error instead.
#if defined(__USE_FAST_MATH__)
#error "rmsnorm numerics require the precise-path math functions"
#endif

namespace redline::kernels {

namespace {

// Per-arch block sizes (see the mapping note above). The guarded define lets
// the tuning harness sweep the sm_80+ candidate without touching this
// file.
#ifndef REDLINE_RMSNORM_BLOCK_SM80
#define REDLINE_RMSNORM_BLOCK_SM80 512
#endif
constexpr std::int32_t kBlockSm80 = REDLINE_RMSNORM_BLOCK_SM80;
constexpr std::int32_t kBlockPre80 = 256;
static_assert(kBlockSm80 >= 64 && kBlockSm80 <= 1024 && (kBlockSm80 & 31) == 0,
              "rmsnorm block must be a multiple of the warp size");

// Vector path: widest per-thread row slice, in half2 words. hidden_size 1536
// is exactly 2 words per thread at 512 threads (3 at 256); 4 admits every
// even hidden_size through 8 * block halves before falling back to the
// scalar kernel.
constexpr std::int32_t kMaxWordsPerThread = 4;

// Deterministic block-wide FP32 sum. Fixed tree in two stages: a lane-xor
// shuffle tree inside each warp (fixed pairing by lane id), then every
// thread sums the warp partials from shared memory in ascending warp order -
// so all threads return the identical FP32 total and the addition order is
// a fixed function of the launch configuration alone. No atomics. One
// __syncthreads() (the pre-tuning halving tree took eight). Assumes
// blockDim.x == kBlock; threads whose columns are exhausted contribute
// +0.0f.
template <std::int32_t kBlock> __device__ float BlockReduceSum(float v) {
  constexpr std::int32_t kWarps = kBlock / 32;
  __shared__ float warp_sums[kWarps];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
#pragma unroll
  for (std::int32_t offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  if ((tid & 31) == 0) {
    warp_sums[tid >> 5] = v;
  }
  __syncthreads();
  float total = warp_sums[0];
#pragma unroll
  for (std::int32_t w = 1; w < kWarps; ++w) {
    total += warp_sums[w];
  }
  return total;
}

// 1 / rms of the row: mean of the squares, then eps *inside* the rsqrt
// argument - rsqrtf(sum_sq / hidden_size + eps) - exactly HF's
// torch.rsqrt(variance + self.variance_epsilon). rsqrtf is the precise-path
// FP32 reciprocal square root (this build does not enable fast-math); its
// couple-of-ulp FP32 error sits far below the FP16 rounding that follows.
// Every thread computes this from the identical reduced total, so every
// thread holds the identical FP32 result.
__device__ float InvRms(float sum_sq, std::int32_t hidden_size, float eps) {
  return rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
}

// ROUNDING POINT 2 - LOAD-BEARING (docs/DESIGN.md section 6.2). The
// normalized value h32 * inv_rms is rounded to FP16 *before* the weight
// multiply: Qwen2RMSNorm returns `weight * hidden_states.to(input_dtype)`,
// i.e. the norm result re-enters FP16 first and the weight multiply is an
// FP16 x FP16 elementwise op. Rounding only after multiplying by the weight
// (one round instead of two) drifts from HF near rounding boundaries. The
// product itself is computed in FP32 and rounded once - an FP16 x FP16
// product is exact in FP32, so this is bit-identical both to torch's
// float-opmath half multiply and to a native __hmul.
__device__ half NormalizeAndWeight(float h32, float inv_rms, half w) {
  const half normed = __float2half_rn(h32 * inv_rms);
  return __float2half_rn(__half2float(normed) * __half2float(w));
}

// Both half2 lanes of NormalizeAndWeight; lanes are the independent logical
// columns (2w, 2w+1), each rounded exactly as the scalar site.
__device__ half2 NormalizeAndWeightTwo(float2 h32, float inv_rms, half2 w) {
  return make_half2(NormalizeAndWeight(h32.x, inv_rms, __low2half(w)),
                    NormalizeAndWeight(h32.y, inv_rms, __high2half(w)));
}

// ===========================================================================
// Vectorized primary kernels (measured tuning pass, docs/PROFILING.md
// section 2.1): half2 loads/stores, pass-1 row
// values held in registers so pass 2 re-reads nothing from global memory.
// WPT = half2 words owned per thread; the launcher instantiates the smallest
// WPT covering ceil((hidden_size/2) / kBlock).
// ===========================================================================

// out[t, :] = rmsnorm(input[t, :]) * weight - vector path.
template <std::int32_t kBlock, std::int32_t WPT>
__global__ __launch_bounds__(kBlock) void RmsNormVecKernel(
    half2* __restrict__ out, const half2* __restrict__ input, const half2* __restrict__ weight,
    std::int32_t num_tokens, std::int32_t words, std::int32_t hidden_size, float eps) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_tokens) {
    return; // padded-row exit on the num_tokens bound (uniform across the block)
  }

  const half2* __restrict__ in_row = input + static_cast<std::int64_t>(row) * words;
  half2* __restrict__ out_row = out + static_cast<std::int64_t>(row) * words;
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);

  // Pass 1: FP32 sum of squares over the FP32-upcast row, fixed per-thread
  // word order (w, w + kBlock, ...), lane .x before lane .y. The upcast
  // values stay in registers for pass 2.
  float2 cache[WPT];
  float sum_sq = 0.0f;
#pragma unroll
  for (std::int32_t i = 0; i < WPT; ++i) {
    const std::int32_t w = tid + i * kBlock;
    if (w < words) {
      const float2 v = __half22float2(in_row[w]);
      cache[i] = v;
      sum_sq += v.x * v.x;
      sum_sq += v.y * v.y;
    }
  }

  const float inv_rms = InvRms(BlockReduceSum<kBlock>(sum_sq), hidden_size, eps);

  // Pass 2: normalize (FP16 round), then weight - see NormalizeAndWeight.
#pragma unroll
  for (std::int32_t i = 0; i < WPT; ++i) {
    const std::int32_t w = tid + i * kBlock;
    if (w < words) {
      out_row[w] = NormalizeAndWeightTwo(cache[i], inv_rms, weight[w]);
    }
  }
}

// Fused residual-add + RMSNorm - vector path (contract as the scalar kernel
// below; kernels.cuh). The rounded residual h is written back during pass 1
// and its FP32 upcast is kept in registers, so pass 2 performs the
// same-thread reuse through registers instead of a global read-after-write.
template <std::int32_t kBlock, std::int32_t WPT>
__global__ __launch_bounds__(kBlock) void RmsNormResidualVecKernel(
    half2* __restrict__ out, half2* __restrict__ residual, const half2* __restrict__ input,
    const half2* __restrict__ weight, std::int32_t num_tokens, std::int32_t words,
    std::int32_t hidden_size, float eps) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_tokens) {
    return; // padded-row exit on the num_tokens bound (uniform across the block)
  }

  half2* __restrict__ res_row = residual + static_cast<std::int64_t>(row) * words;
  const half2* __restrict__ in_row = input + static_cast<std::int64_t>(row) * words;
  half2* __restrict__ out_row = out + static_cast<std::int64_t>(row) * words;
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);

  // Pass 1: materialize the new residual and accumulate its mean-square.
  float2 cache[WPT];
  float sum_sq = 0.0f;
#pragma unroll
  for (std::int32_t i = 0; i < WPT; ++i) {
    const std::int32_t w = tid + i * kBlock;
    if (w < words) {
      const float2 r = __half22float2(res_row[w]);
      const float2 x = __half22float2(in_row[w]);
      // ROUNDING POINT 1 - LOAD-BEARING (docs/DESIGN.md section 6.2). h is
      // the FP16-rounded residual sum - fp16(fp32(r) + fp32(x)) per lane,
      // identical to a native half add since both are correctly rounded -
      // written back as the new residual. The squares accumulate over
      // fp32(h), the *rounded* value, never the raw FP32 sum: HF
      // materializes `residual + x` as an FP16 tensor and Qwen2RMSNorm
      // upcasts that tensor.
      const half2 h = make_half2(__float2half_rn(r.x + x.x), __float2half_rn(r.y + x.y));
      res_row[w] = h;
      const float2 h32 = __half22float2(h);
      cache[i] = h32;
      sum_sq += h32.x * h32.x;
      sum_sq += h32.y * h32.y;
    }
  }

  const float inv_rms = InvRms(BlockReduceSum<kBlock>(sum_sq), hidden_size, eps);

  // Pass 2: normalize the register-held rounded residual, then weight.
#pragma unroll
  for (std::int32_t i = 0; i < WPT; ++i) {
    const std::int32_t w = tid + i * kBlock;
    if (w < words) {
      out_row[w] = NormalizeAndWeightTwo(cache[i], inv_rms, weight[w]);
    }
  }
}

// ===========================================================================
// Scalar fallback kernels - the pre-tuning two-pass bodies, kept so the
// launchers stay total over every shape the contract admits (odd
// hidden_size, 2-byte-aligned bases, oversized rows). Never taken for the
// engine's buffers.
// ===========================================================================

// out[t, :] = rmsnorm(input[t, :]) * weight. `input` is read twice (sum of
// squares, then normalize); `out` must not alias it.
template <std::int32_t kBlock>
__global__ __launch_bounds__(kBlock) void RmsNormKernel(half* __restrict__ out,
                                                        const half* __restrict__ input,
                                                        const half* __restrict__ weight,
                                                        std::int32_t num_tokens,
                                                        std::int32_t hidden_size, float eps) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_tokens) {
    return; // padded-row exit on the num_tokens bound (uniform across the block)
  }

  const half* __restrict__ in_row = input + static_cast<std::int64_t>(row) * hidden_size;
  half* __restrict__ out_row = out + static_cast<std::int64_t>(row) * hidden_size;

  // Pass 1: FP32 sum of squares over the FP32-upcast row, fixed per-thread
  // column order (c, c + kBlock, ...).
  float sum_sq = 0.0f;
  for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < hidden_size; c += kBlock) {
    const float v = __half2float(in_row[c]);
    sum_sq += v * v;
  }

  const float inv_rms = InvRms(BlockReduceSum<kBlock>(sum_sq), hidden_size, eps);

  // Pass 2: normalize (FP16 round), then weight - see NormalizeAndWeight.
  for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < hidden_size; c += kBlock) {
    out_row[c] = NormalizeAndWeight(__half2float(in_row[c]), inv_rms, weight[c]);
  }
}

// Fused residual-add + RMSNorm per the kernels.cuh contract:
//   h = fp16(fp32(residual) + fp32(input));  residual = h;
//   mean_sq over fp32(h);  out = fp16(fp32(h) * rsqrt(mean_sq + eps)) * w.
// `out`, `residual`, and `input` must be pairwise non-aliasing (the decode
// sequence always passes three distinct buffers: x / attn_out or mlp_out /
// normed). `residual` is updated in place by design.
template <std::int32_t kBlock>
__global__ __launch_bounds__(kBlock) void RmsNormResidualKernel(
    half* __restrict__ out, half* __restrict__ residual, const half* __restrict__ input,
    const half* __restrict__ weight, std::int32_t num_tokens, std::int32_t hidden_size, float eps) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= num_tokens) {
    return; // padded-row exit on the num_tokens bound (uniform across the block)
  }

  half* __restrict__ res_row = residual + static_cast<std::int64_t>(row) * hidden_size;
  const half* __restrict__ in_row = input + static_cast<std::int64_t>(row) * hidden_size;
  half* __restrict__ out_row = out + static_cast<std::int64_t>(row) * hidden_size;

  // Pass 1: materialize the new residual and accumulate its mean-square.
  float sum_sq = 0.0f;
  for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < hidden_size; c += kBlock) {
    // ROUNDING POINT 1 - LOAD-BEARING (docs/DESIGN.md section 6.2); see the
    // vector kernel's site for the full note.
    const half h = __float2half_rn(__half2float(res_row[c]) + __half2float(in_row[c]));
    res_row[c] = h;
    const float h32 = __half2float(h);
    sum_sq += h32 * h32;
  }

  const float inv_rms = InvRms(BlockReduceSum<kBlock>(sum_sq), hidden_size, eps);

  // Pass 2: each thread re-reads exactly the rounded residual elements it
  // wrote in pass 1 (same-thread global read-after-write; no cross-thread
  // traffic, so no visibility concern), then normalizes and weights.
  for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < hidden_size; c += kBlock) {
    out_row[c] = NormalizeAndWeight(__half2float(res_row[c]), inv_rms, weight[c]);
  }
}

// True when the pointer may be dereferenced as half2 (4-byte words).
bool IsHalf2Aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % alignof(half2) == 0;
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

// Block-size-resolved launch bodies. The vector path applies when
// hidden_size is even, every base is 4-byte aligned, and the row fits the
// register cache (words <= kMaxWordsPerThread * kBlock); the scalar kernel
// covers everything else the contract admits.
template <std::int32_t kBlock>
void LaunchRmsNormImpl(half* out, const half* input, const half* weight, std::int32_t num_tokens,
                       std::int32_t hidden_size, float eps, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_tokens)); // one block per token row
  const dim3 block(kBlock);
  const std::int32_t words = hidden_size / 2;
  const std::int32_t wpt = (words + kBlock - 1) / kBlock;
  if (hidden_size % 2 == 0 && wpt <= kMaxWordsPerThread && IsHalf2Aligned(out) &&
      IsHalf2Aligned(input) && IsHalf2Aligned(weight)) {
    half2* out2 = reinterpret_cast<half2*>(out);
    const half2* in2 = reinterpret_cast<const half2*>(input);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    switch (wpt) {
    case 1:
      RmsNormVecKernel<kBlock, 1>
          <<<grid, block, 0, stream>>>(out2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    case 2:
      RmsNormVecKernel<kBlock, 2>
          <<<grid, block, 0, stream>>>(out2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    case 3:
      RmsNormVecKernel<kBlock, 3>
          <<<grid, block, 0, stream>>>(out2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    default:
      RmsNormVecKernel<kBlock, kMaxWordsPerThread>
          <<<grid, block, 0, stream>>>(out2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    }
  }
  RmsNormKernel<kBlock>
      <<<grid, block, 0, stream>>>(out, input, weight, num_tokens, hidden_size, eps);
}

template <std::int32_t kBlock>
void LaunchRmsNormResidualImpl(half* out, half* residual, const half* input, const half* weight,
                               std::int32_t num_tokens, std::int32_t hidden_size, float eps,
                               cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_tokens)); // one block per token row
  const dim3 block(kBlock);
  const std::int32_t words = hidden_size / 2;
  const std::int32_t wpt = (words + kBlock - 1) / kBlock;
  if (hidden_size % 2 == 0 && wpt <= kMaxWordsPerThread && IsHalf2Aligned(out) &&
      IsHalf2Aligned(residual) && IsHalf2Aligned(input) && IsHalf2Aligned(weight)) {
    half2* out2 = reinterpret_cast<half2*>(out);
    half2* res2 = reinterpret_cast<half2*>(residual);
    const half2* in2 = reinterpret_cast<const half2*>(input);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    switch (wpt) {
    case 1:
      RmsNormResidualVecKernel<kBlock, 1>
          <<<grid, block, 0, stream>>>(out2, res2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    case 2:
      RmsNormResidualVecKernel<kBlock, 2>
          <<<grid, block, 0, stream>>>(out2, res2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    case 3:
      RmsNormResidualVecKernel<kBlock, 3>
          <<<grid, block, 0, stream>>>(out2, res2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    default:
      RmsNormResidualVecKernel<kBlock, kMaxWordsPerThread>
          <<<grid, block, 0, stream>>>(out2, res2, in2, w2, num_tokens, words, hidden_size, eps);
      return;
    }
  }
  RmsNormResidualKernel<kBlock>
      <<<grid, block, 0, stream>>>(out, residual, input, weight, num_tokens, hidden_size, eps);
}

} // namespace

void LaunchRmsNorm(half* out, const half* input, const half* weight, std::int32_t num_tokens,
                   std::int32_t hidden_size, float eps, cudaStream_t stream) {
  if (num_tokens <= 0 || hidden_size <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to normalize
  }
  // Debug-build guard for the aliasing contract (exact equality only - the
  // realistic misuse is passing one buffer twice for an in-place norm, which
  // the kernel's two-pass reads plus __restrict__ make undefined).
  assert(out != input && "rmsnorm: out must not alias input");
  if (DeviceComputeMajor() >= 8) {
    LaunchRmsNormImpl<kBlockSm80>(out, input, weight, num_tokens, hidden_size, eps, stream);
  } else {
    LaunchRmsNormImpl<kBlockPre80>(out, input, weight, num_tokens, hidden_size, eps, stream);
  }
}

void LaunchRmsNormResidual(half* out, half* residual, const half* input, const half* weight,
                           std::int32_t num_tokens, std::int32_t hidden_size, float eps,
                           cudaStream_t stream) {
  if (num_tokens <= 0 || hidden_size <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to normalize
  }
  // Debug-build guard for the aliasing contract (exact equality only - the
  // realistic misuse is reusing one buffer for two roles; `residual` alone is
  // updated in place by design).
  assert(out != residual && out != input && residual != input &&
         "fused rmsnorm: out/residual/input must be pairwise distinct");
  if (DeviceComputeMajor() >= 8) {
    LaunchRmsNormResidualImpl<kBlockSm80>(out, residual, input, weight, num_tokens, hidden_size,
                                          eps, stream);
  } else {
    LaunchRmsNormResidualImpl<kBlockPre80>(out, residual, input, weight, num_tokens, hidden_size,
                                           eps, stream);
  }
}

} // namespace redline::kernels
