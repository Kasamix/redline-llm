#include "kernels/kernels.cuh"

#include <cstdint>

// embed_gather (docs/DESIGN.md section 6.2, kernel #1) - the first launch of
// every decode step and prefill chunk:
//
//   out[t, :] = embed[input_ids[t], :]
//
// A pure row copy out of the embedding table: no reduction, no shared memory.
// One block serves one token row. Three width tiers, widest applicable
// chosen at launch (measured tuning pass, docs/PROFILING.md section 2.1):
// 16-byte uint4 words (for this model a row
// is hidden_size = 1536 halves = 3 KiB = 192 words, one per thread of a
// 192-thread block), half2 words for 4-byte-but-not-16-byte-aligned shapes,
// and a scalar fallback for everything else the contract admits. On every
// tier consecutive threads touch consecutive words, so both the table read
// and the activation write coalesce into full cache lines; the copy is a
// pure bit move on all tiers, so tier choice is observationally invisible.
//
// input_ids[t] is a single int32 load, uniform across the block. Ids must be
// in [0, vocab_size); range validation belongs to request admission, not to
// this kernel. Padded batch slots carry token id 0 (docs/DESIGN.md section 9
// padding policy), which gathers embedding row 0 into a row nothing reads -
// harmless by construction, so unlike the seq_len-driven kernels there is no
// early-exit path here.
//
// The out row stride is explicit per the strided-view rule (kernels.cuh
// header comment); the canonical caller passes the dense residual buffer `x`
// with stride == hidden_size. The embedding table is a dense device weight
// ([vocab_size, hidden_size], docs/MODEL_SPEC.md section 8) and carries no
// stride parameter.

namespace redline::kernels {

namespace {

constexpr std::int32_t kBlockThreads = 256;

// Per-arch/width tuning constant (docs/PROFILING.md section 2.1): block size for the 16-byte tier.
// 192 threads move a 1536-half row as exactly one uint4 word per thread
// (measured fastest on sm_89 across {128, 192, 256}); rows on other shapes
// grid-stride. The guarded define lets the tuning harness sweep candidates
// without touching this file.
#ifndef REDLINE_EMBED_VEC8_BLOCK
#define REDLINE_EMBED_VEC8_BLOCK 192
#endif
constexpr std::int32_t kVec8BlockThreads = REDLINE_EMBED_VEC8_BLOCK;
static_assert(kVec8BlockThreads >= 32 && kVec8BlockThreads <= 1024 && (kVec8BlockThreads & 31) == 0,
              "embed_gather vec8 block must be a multiple of the warp size");

// 16-byte path: copies the row as uint4 words (eight halves each). The
// launcher dispatches here only when every row base is 16-byte aligned
// (hidden_size and out_row_stride multiples of 8, 16-byte base pointers).
__global__ __launch_bounds__(kVec8BlockThreads) void EmbedGatherVec8Kernel(
    half* __restrict__ out, std::int64_t out_row_stride, const half* __restrict__ embed,
    const std::int32_t* __restrict__ input_ids, std::int32_t hidden_size) {
  const std::int64_t row = blockIdx.x;
  const std::int64_t token = input_ids[row]; // uniform across the block

  const uint4* __restrict__ src = reinterpret_cast<const uint4*>(embed + token * hidden_size);
  uint4* __restrict__ dst = reinterpret_cast<uint4*>(out + row * out_row_stride);

  const std::int32_t width8 = hidden_size >> 3; // 192 uint4 words for hidden_size 1536
  for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < width8;
       i += kVec8BlockThreads) {
    dst[i] = src[i];
  }
}

// Vectorized path: copies the row as half2 words. The launcher dispatches
// here only when both row bases are 4-byte aligned for every row (even
// hidden_size, even out_row_stride, aligned base pointers).
__global__ __launch_bounds__(kBlockThreads) void EmbedGatherHalf2Kernel(
    half* __restrict__ out, std::int64_t out_row_stride, const half* __restrict__ embed,
    const std::int32_t* __restrict__ input_ids, std::int32_t hidden_size) {
  const std::int64_t row = blockIdx.x;
  const std::int64_t token = input_ids[row]; // uniform across the block

  const half2* __restrict__ src = reinterpret_cast<const half2*>(embed + token * hidden_size);
  half2* __restrict__ dst = reinterpret_cast<half2*>(out + row * out_row_stride);

  const std::int32_t width2 = hidden_size >> 1; // 768 half2 words for hidden_size 1536
  for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < width2; i += kBlockThreads) {
    dst[i] = src[i];
  }
}

// Scalar fallback for inputs where the half2 alignment premise does not hold
// (odd hidden_size, odd out_row_stride, or misaligned base pointers). Never
// taken for this model - hidden_size 1536 and the canonical strides (1536
// dense, 2048 qkv_out-style) are all even - but it keeps the launcher total
// over everything the contract admits.
__global__ __launch_bounds__(kBlockThreads) void EmbedGatherScalarKernel(
    half* __restrict__ out, std::int64_t out_row_stride, const half* __restrict__ embed,
    const std::int32_t* __restrict__ input_ids, std::int32_t hidden_size) {
  const std::int64_t row = blockIdx.x;
  const std::int64_t token = input_ids[row];

  const half* __restrict__ src = embed + token * hidden_size;
  half* __restrict__ dst = out + row * out_row_stride;

  for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < hidden_size;
       i += kBlockThreads) {
    dst[i] = src[i];
  }
}

// True when the pointer may be dereferenced as half2 (4-byte words).
bool IsHalf2Aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % alignof(half2) == 0;
}

// True when the pointer may be dereferenced as uint4 (16-byte words).
bool IsVec8Aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % sizeof(uint4) == 0;
}

} // namespace

void LaunchEmbedGather(half* out, std::int64_t out_row_stride, const half* embed,
                       const std::int32_t* input_ids, std::int32_t num_tokens,
                       std::int32_t hidden_size, cudaStream_t stream) {
  if (num_tokens <= 0 || hidden_size <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to gather
  }

  const dim3 grid(static_cast<unsigned int>(num_tokens)); // one block per token row

  // A uint4 word spans eight elements: element count and row stride must be
  // multiples of 8 (so every embed row base and out row base keeps the base
  // pointers' 16-byte alignment). The canonical shapes - hidden 1536, out
  // strides 1536 and 2048, cudaMalloc'd bases - always qualify.
  const bool vectorizable8 = (hidden_size % 8 == 0) && (out_row_stride % 8 == 0) &&
                             IsVec8Aligned(out) && IsVec8Aligned(embed);
  if (vectorizable8) {
    EmbedGatherVec8Kernel<<<grid, dim3(kVec8BlockThreads), 0, stream>>>(out, out_row_stride, embed,
                                                                        input_ids, hidden_size);
    return;
  }

  const dim3 block(kBlockThreads);

  // A half2 word spans two elements, so beyond base-pointer alignment the
  // vector path needs an even element count and an even row stride to keep
  // every row base 4-byte aligned.
  const bool vectorizable = (hidden_size % 2 == 0) && (out_row_stride % 2 == 0) &&
                            IsHalf2Aligned(out) && IsHalf2Aligned(embed);
  if (vectorizable) {
    EmbedGatherHalf2Kernel<<<grid, block, 0, stream>>>(out, out_row_stride, embed, input_ids,
                                                       hidden_size);
  } else {
    EmbedGatherScalarKernel<<<grid, block, 0, stream>>>(out, out_row_stride, embed, input_ids,
                                                        hidden_size);
  }
}

} // namespace redline::kernels
