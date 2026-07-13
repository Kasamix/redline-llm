#include "kernels/kernels.cuh"

#include <bit>
#include <cstdint>

#include "core/types.hpp"

// kv_scatter (docs/DESIGN.md section 6.2, kernel #4) - after RoPE, every
// layer publishes the current step's K and V vectors into the paged KV pool,
// the single persistent home of cached context. Block-table indirection
// follows the PagedAttention design (Kwon et al., 2023, arXiv:2309.06180);
// this implementation is original.
//
// Sources are the strided qkv_out views of the strided-view rule
// (docs/DESIGN.md section 5): on the canonical path K sits at qkv_out + 1536
// and V at qkv_out + 1792, both with row stride 2048, and K already carries
// RoPE. Both tensors are read through explicit k_row_stride / v_row_stride,
// so dense-packed test tensors (stride == num_kv_heads * head_dim) go
// through the identical code path.
//
// The destination slot derives ENTIRELY in-kernel from positions[t] and the
// block table - there is no host-computed slot-index array anywhere, which
// keeps the per-step graph-replay upload set at the four arrays of
// docs/DESIGN.md section 5 at the cost of one extra int32 load per token
// (uniform across the block, so it broadcasts from one cache line):
//     p     = positions[t]
//     block = block_tables[t * block_table_row_stride + (p >> 4)]
//     slot  = p & 15
//     elem  = (((block * 2 + kv) * num_kv_heads + h) * 16 + slot) * head_dim + d
// where kv = 0 selects the K plane and kv = 1 the V plane of one layer's
// pool slice [num_blocks, 2, num_kv_heads, kKvBlockSize, head_dim].
//
// block_table_row_stride selects the row mapping (kernels.cuh contract):
//   decode  - stride = max_blocks_per_seq: token row t is batch slot t;
//   prefill - stride = 0: every chunk token shares the one sequence's row.
//
// Padded decode rows need no early exit and the kernel takes no seq_lens:
// padding carries position 0 and a dummy-block table row (docs/DESIGN.md
// section 9), so a padded row's K/V lands in reserved physical block 0,
// which nothing ever reads - harmless by construction.
//
// Memory behavior: the pool's innermost [kKvBlockSize, head_dim] tile makes
// each (head, token-slot) row a contiguous 256 B run (128 halves). One
// 128-thread block serves one token; consecutive threads move consecutive
// words of each such row, so the strided reads and the pool writes both
// coalesce. Copies are pure bit moves (no arithmetic, no float
// interpretation - the vector path copies uint4 words), so a kv_gather of
// the same positions returns bit-identical values; the unit suite asserts
// that round trip. A scalar fallback keeps the launcher total over
// arbitrary strides/alignments the contract admits; the canonical decode
// and prefill calls always satisfy the 128-bit alignment premise.

namespace redline::kernels {

namespace {

// Per-block work is one token's K+V payload - canonically 2 planes x
// 2 heads x 128 halves = 512 halves, i.e. 64 uint4 words or 512 scalar
// copies - so anything beyond launch cost is immaterial here; 128 threads
// matches the one-block-per-row sibling kernels, and the strided loops stay
// total over any num_kv_heads / head_dim. Block-size tuning is a later,
// measured concern.
constexpr std::int32_t kBlockThreads = 128;

// 128-bit copy unit of the vector path: 8 halves per uint4 word.
constexpr std::int32_t kVecHalves = 8;

// The frozen launch contract documents the slot derivation as p >> 4 and
// p & 15. Derive shift and mask from the shared core constant instead of
// restating 16, so this file and core/types.hpp cannot drift apart.
static_assert(kKvBlockSize > 0 && (kKvBlockSize & (kKvBlockSize - 1)) == 0,
              "in-kernel slot derivation uses shift/mask arithmetic");
constexpr std::int32_t kKvBlockShift = std::countr_zero(static_cast<std::uint32_t>(kKvBlockSize));

// Element offset of the contiguous [head_dim]-half row for (block, kv, head,
// slot) inside ONE layer's pool slice
//   [num_blocks, 2 (0 = K, 1 = V), num_kv_heads, kKvBlockSize, head_dim]
// - the docs/DESIGN.md section 5 index formula with the layer term dropped
// (kernels receive the layer slice base, so `l` contributes nothing here).
// int64 math: a bench-sized pool slice already holds ~1.5e8 elements, and
// nothing caps future pool sizes at int32 range.
__device__ __forceinline__ std::int64_t PoolRowBase(std::int32_t block, std::int32_t kv,
                                                    std::int32_t head, std::int32_t slot,
                                                    std::int32_t num_kv_heads,
                                                    std::int32_t head_dim) {
  const std::int64_t plane = static_cast<std::int64_t>(block) * 2 + kv;
  return ((plane * num_kv_heads + head) * kKvBlockSize + slot) * head_dim;
}

// One block per token row; threads stride over the token's K and V words.
// Word = uint4 is the 128-bit fast path (launcher checks the alignment
// premise); Word = half is the scalar fallback - same body, kHalvesPerWord
// degenerates to 1, keeping the addressing logic in exactly one place.
// Aliasing contract: the pool never overlaps k or v (it is a dedicated
// init-time allocation; k and v are views into activation scratch). k and v
// may share an underlying buffer (both are qkv_out views) - both are
// read-only here, so that is well-formed.
template <typename Word>
__global__ __launch_bounds__(kBlockThreads) void KvScatterKernel(
    half* __restrict__ pool, const half* __restrict__ k, const half* __restrict__ v,
    std::int64_t k_row_stride, std::int64_t v_row_stride,
    const std::int32_t* __restrict__ block_tables, std::int64_t block_table_row_stride,
    const std::int32_t* __restrict__ positions, std::int32_t num_kv_heads, std::int32_t head_dim) {
  constexpr std::int32_t kHalvesPerWord = static_cast<std::int32_t>(sizeof(Word) / sizeof(half));

  const std::int64_t t = blockIdx.x; // token row; grid.x == num_tokens exactly
  // In-kernel slot derivation (see file comment). All three loads are
  // uniform across the block and broadcast from cache.
  const std::int32_t p = positions[t];
  const std::int32_t block = block_tables[t * block_table_row_stride + (p >> kKvBlockShift)];
  const std::int32_t slot = p & (kKvBlockSize - 1);

  const std::int32_t words_per_head = head_dim / kHalvesPerWord;
  const std::int32_t words_per_side = num_kv_heads * words_per_head; // K plane or V plane
  const std::int32_t words_per_token = 2 * words_per_side;

  const Word* __restrict__ k_words = reinterpret_cast<const Word*>(k + t * k_row_stride);
  const Word* __restrict__ v_words = reinterpret_cast<const Word*>(v + t * v_row_stride);
  Word* __restrict__ pool_words = reinterpret_cast<Word*>(pool);

  for (std::int32_t w = static_cast<std::int32_t>(threadIdx.x); w < words_per_token;
       w += kBlockThreads) {
    const std::int32_t kv = w < words_per_side ? 0 : 1; // 0 = K, 1 = V
    const std::int32_t r = w - kv * words_per_side;     // word within that plane's token row
    const std::int32_t h = r / words_per_head;
    const std::int32_t dw = r - h * words_per_head; // word within the (head, slot) row

    // Source word r of the K/V view row t (its element h*head_dim + dw*
    // kHalvesPerWord); destination word dw of the pool's (block, kv, h,
    // slot) row. PoolRowBase is a multiple of head_dim, so the word
    // division below is exact whenever the vector path is taken
    // (head_dim % kVecHalves == 0 is a dispatch precondition).
    const std::int64_t dst =
        PoolRowBase(block, kv, h, slot, num_kv_heads, head_dim) / kHalvesPerWord + dw;
    pool_words[dst] = (kv == 0 ? k_words : v_words)[r];
  }
}

// True when the pointer may be dereferenced as uint4 (16-byte words).
bool IsUint4Aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % alignof(uint4) == 0;
}

} // namespace

void LaunchKvScatter(half* kv_pool_layer, const half* k, const half* v, std::int64_t k_row_stride,
                     std::int64_t v_row_stride, const std::int32_t* block_tables,
                     std::int64_t block_table_row_stride, const std::int32_t* positions,
                     std::int32_t num_tokens, std::int32_t num_kv_heads, std::int32_t head_dim,
                     cudaStream_t stream) {
  if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to scatter
  }

  const dim3 grid(static_cast<unsigned int>(num_tokens)); // one block per token row
  const dim3 block(kBlockThreads);

  // Vector path premise: every 128-bit word the kernel touches must be
  // 16-byte aligned - base pointers aligned, row strides in whole words,
  // and head rows (multiples of head_dim relative to the bases) in whole
  // words. Canonical calls satisfy this: qkv_out views at +1536/+1792 with
  // stride 2048 land on 16 B boundaries of a 256 B-aligned cudaMalloc
  // buffer, and head_dim 128 keeps every pool row 256 B-aligned.
  const bool vectorizable = head_dim % kVecHalves == 0 && k_row_stride % kVecHalves == 0 &&
                            v_row_stride % kVecHalves == 0 && IsUint4Aligned(kv_pool_layer) &&
                            IsUint4Aligned(k) && IsUint4Aligned(v);
  if (vectorizable) {
    KvScatterKernel<uint4>
        <<<grid, block, 0, stream>>>(kv_pool_layer, k, v, k_row_stride, v_row_stride, block_tables,
                                     block_table_row_stride, positions, num_kv_heads, head_dim);
  } else {
    KvScatterKernel<half>
        <<<grid, block, 0, stream>>>(kv_pool_layer, k, v, k_row_stride, v_row_stride, block_tables,
                                     block_table_row_stride, positions, num_kv_heads, head_dim);
  }
}

} // namespace redline::kernels
