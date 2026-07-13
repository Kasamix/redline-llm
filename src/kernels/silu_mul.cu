#include "kernels/kernels.cuh"

#include <cstdint>

// silu_mul (docs/DESIGN.md section 6.2, kernel #8) - SwiGLU activation over
// the fused gate+up GEMM output. The gateup GEMM (docs/DESIGN.md section 4
// fusion table) writes one [num_tokens, 2 * intermediate_size] buffer whose
// row layout is gate(0:I) | up(I:2I), I = intermediate_size (8960 for
// Qwen2.5-1.5B). This kernel computes, elementwise over [num_tokens, I]:
//
//   out[t, c]   = fp16( silu_f32(gate_up[t, c]) * fp32(gate_up[t, I + c]) )
//   silu_f32(x) = x / (1 + expf(-x))
//
// Output placement - the contract signature carries no output stride, so the
// launcher derives it from pointer identity:
//   - aliased (out == gate_up), the decode/prefill hot path: the result is
//     written in place over the gate half of each row, out row stride 2*I.
//     The downstream down GEMM then reads the [num_tokens, I] slice with
//     lda = 17920 (docs/DESIGN.md section 6.3
//     allocates no separate activation buffer for this stage).
//   - separate (out != gate_up): a dense [num_tokens, I] buffer, row stride
//     I, exactly as the kernels.cuh contract comment documents. Partial
//     overlap between out and gate_up is unsupported; callers pass either
//     the identical pointer or a disjoint buffer.
//
// In-place safety (aliased mode): the thread computing (t, c) reads
// gate_up[t, c] and gate_up[t, I + c] and writes only gate_up[t, c]. The up
// half is never written, and no other thread reads or writes row t's gate
// element c, so the single write cannot race any other thread's accesses.
// Within the thread the stored value depends on both loads, so the store
// cannot be reordered above them. No synchronization is required. `out` and
// `gate_up` are deliberately NOT declared __restrict__: in aliased mode they
// are the same allocation by design, and this elementwise kernel has no
// reuse for the compiler to exploit anyway.
//
// Numerics (docs/MODEL_SPEC.md section 4; kernels.cuh conventions): FP32
// silu using the precise-path expf (this build does not enable fast-math),
// FP32 product, one FP16 round-to-nearest-even of the final value. Purely
// elementwise - no reductions - so outputs are bitwise deterministic for
// identical inputs, a premise of the docs/DESIGN.md section 12 same-shape
// equivalence suites.
//
// Mapping: 2-D grid-stride - blockIdx.y walks token rows, blockIdx.x x 256
// threads walk the row's I output columns with adjacent threads on adjacent
// columns, so warp loads/stores coalesce. When every access stays 4-byte
// aligned (I even plus aligned base pointers - always true for the
// cudaMalloc'd step buffers at I = 8960), a half2 variant moves two columns
// per thread; a scalar variant covers the general case. Wider vectorization
// was measured on sm_89 (16-byte uint4 tier, batch-64 decode shape;
// docs/PROFILING.md section 2.1)
// and REGRESSED -16% - the 3.6x-fewer resident threads hide the L2 latency
// of the disjoint gate/up streams worse than the half2 grid, which already
// runs at ~1.2 TB/s effective - so half2 stays the widest tier by evidence,
// not by omission.

namespace redline::kernels {

namespace {

constexpr std::int32_t kBlockThreads = 256;
constexpr unsigned int kMaxGridY = 65535; // CUDA grid.y limit; rows grid-stride past it

// FP32 SiLU: x / (1 + e^{-x}), with expf the precise-path FP32 exponential
// required by docs/DESIGN.md section 6.2 kernel #8 (never __expf). Extremes
// behave like the same formula in HF's float opmath: large +x returns x,
// large -x underflows cleanly to a signed zero.
__device__ float SiluF32(float x) {
  return x / (1.0f + expf(-x));
}

// One output element: fp16( silu_f32(gate) * fp32(up) ). The product is
// computed in FP32 and rounded exactly once.
__device__ half SiluMulOne(half gate, half up) {
  return __float2half_rn(SiluF32(__half2float(gate)) * __half2float(up));
}

// Two output elements; the half2 lanes are the independent logical columns
// (c, c+1), so the gate/up pairing per lane is preserved exactly.
__device__ half2 SiluMulTwo(half2 gate, half2 up) {
  const float2 g = __half22float2(gate);
  const float2 u = __half22float2(up);
  return __floats2half2_rn(SiluF32(g.x) * u.x, SiluF32(g.y) * u.y);
}

// General-case scalar kernel. Pointers and strides are in half elements;
// gate_up rows are 2 * intermediate_size wide, out rows out_row_stride wide
// (2 * intermediate_size when aliased over the gate half, intermediate_size
// when dense - computed by the launcher).
__global__ __launch_bounds__(kBlockThreads) void SiluMulScalarKernel(
    half* out, const half* gate_up, std::int64_t out_row_stride, std::int32_t num_tokens,
    std::int32_t intermediate_size) {
  const std::int64_t in_row_stride = 2 * static_cast<std::int64_t>(intermediate_size);
  for (std::int32_t row = static_cast<std::int32_t>(blockIdx.y); row < num_tokens;
       row += static_cast<std::int32_t>(gridDim.y)) {
    const half* gate_row = gate_up + row * in_row_stride;
    const half* up_row = gate_row + intermediate_size;
    half* out_row = out + row * out_row_stride;
    for (std::int32_t c = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         c < intermediate_size; c += static_cast<std::int32_t>(gridDim.x * blockDim.x)) {
      out_row[c] = SiluMulOne(gate_row[c], up_row[c]);
    }
  }
}

// half2-vectorized kernel: `pairs` = intermediate_size / 2 half2 columns per
// row; all pointers and strides are in half2 units (gate_up rows are
// 2 * pairs wide, the up half starts `pairs` in). Launched only when the
// launcher has proven 4-byte alignment of every access.
__global__ __launch_bounds__(kBlockThreads) void SiluMulHalf2Kernel(half2* out,
                                                                    const half2* gate_up,
                                                                    std::int64_t out_row_stride,
                                                                    std::int32_t num_tokens,
                                                                    std::int32_t pairs) {
  const std::int64_t in_row_stride = 2 * static_cast<std::int64_t>(pairs);
  for (std::int32_t row = static_cast<std::int32_t>(blockIdx.y); row < num_tokens;
       row += static_cast<std::int32_t>(gridDim.y)) {
    const half2* gate_row = gate_up + row * in_row_stride;
    const half2* up_row = gate_row + pairs;
    half2* out_row = out + row * out_row_stride;
    for (std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         p < pairs; p += static_cast<std::int32_t>(gridDim.x * blockDim.x)) {
      out_row[p] = SiluMulTwo(gate_row[p], up_row[p]);
    }
  }
}

} // namespace

void LaunchSiluMul(half* out, const half* gate_up, std::int32_t num_tokens,
                   std::int32_t intermediate_size, cudaStream_t stream) {
  if (num_tokens <= 0 || intermediate_size <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to activate
  }

  // Output row stride by pointer identity (see file header): in place over
  // the gate half (stride 2*I, down GEMM lda = 17920) when aliased, dense
  // [num_tokens, I] otherwise.
  const bool aliased = out == gate_up;

  const unsigned int grid_y = static_cast<unsigned int>(num_tokens) > kMaxGridY
                                  ? kMaxGridY
                                  : static_cast<unsigned int>(num_tokens);

  // half2 path whenever every access stays 4-byte aligned: an even I keeps
  // the up-half offset (I halves) and both possible out row strides (I or
  // 2*I halves) at whole half2 counts, leaving base-pointer alignment as the
  // only remaining condition. The canonical step buffers (cudaMalloc'd,
  // I = 8960) always qualify.
  const bool vectorize = intermediate_size % 2 == 0 &&
                         reinterpret_cast<std::uintptr_t>(out) % alignof(half2) == 0 &&
                         reinterpret_cast<std::uintptr_t>(gate_up) % alignof(half2) == 0;

  if (vectorize) {
    const std::int32_t pairs = intermediate_size / 2;
    const std::int64_t out_row_stride = aliased ? 2 * static_cast<std::int64_t>(pairs) : pairs;
    const unsigned int grid_x =
        static_cast<unsigned int>((pairs + kBlockThreads - 1) / kBlockThreads);
    SiluMulHalf2Kernel<<<dim3(grid_x, grid_y), dim3(kBlockThreads), 0, stream>>>(
        reinterpret_cast<half2*>(out), reinterpret_cast<const half2*>(gate_up), out_row_stride,
        num_tokens, pairs);
  } else {
    const std::int64_t out_row_stride =
        aliased ? 2 * static_cast<std::int64_t>(intermediate_size) : intermediate_size;
    const unsigned int grid_x =
        static_cast<unsigned int>((intermediate_size + kBlockThreads - 1) / kBlockThreads);
    SiluMulScalarKernel<<<dim3(grid_x, grid_y), dim3(kBlockThreads), 0, stream>>>(
        out, gate_up, out_row_stride, num_tokens, intermediate_size);
  }
}

} // namespace redline::kernels
