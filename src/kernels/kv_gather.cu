#include "kernels/kernels.cuh"

#include <bit>
#include <cstdint>

#include "core/types.hpp"

// kv_gather (docs/DESIGN.md section 6.2, kernel #5) - prefill-only inverse
// of kv_scatter: copies one sequence's cached K and V for positions
// [0, ctx_len) out of the paged pool into the dense khat/vhat staging
// buffers, so the prefill attention composite can run plain strided-batched
// GEMMs over contiguous [ctx_len, head_dim] head tiles (docs/DESIGN.md
// section 6.3 descriptor recipes, lda = head_dim). The paged pool stays the
// single source of truth - no shadow linear cache exists to keep coherent -
// and the gathered bytes (~1 MiB per layer per chunk at 4K context) are
// noise next to the GEMMs they feed. Block-table indirection follows the
// PagedAttention design (Kwon et al., 2023, arXiv:2309.06180); this
// implementation is original.
//
// Addressing derives entirely in-kernel from the destination row index -
// gathered position p IS destination row t, so no positions array and no
// host-computed slot array exist:
//     block = block_table_row[t >> 4]
//     slot  = t & 15
//     elem  = (((block * 2 + kv) * num_kv_heads + h) * 16 + slot) * head_dim + d
// on the layer's pool slice, with kv = 0 for K (into khat) and kv = 1 for V
// (into vhat). ctx_len is NOT rounded to a block boundary: the final block
// may be partial, and only its slots holding positions < ctx_len are read.
//
// Destination layout is [num_kv_heads, ctx_len, head_dim] per buffer,
// addressed through explicit strides - head plane h starts at
// h * *_head_stride and token rows are spaced *_row_stride elements - so
// both the section 5 allocation shape (head stride = max_seq_len * head_dim,
// dense rows) and fully packed test tensors (head stride = ctx_len *
// head_dim) go through the identical code path.
//
// Memory behavior mirrors kv_scatter: each pool (head, slot) row is a
// contiguous 256 B run; one 128-thread block serves one position, with
// consecutive threads moving consecutive words, so pool reads and staging
// writes both coalesce. Copies are pure bit moves (uint4 words on the
// vector path, half on the scalar fallback), so scatter-then-gather of the
// same positions is bit-exact - the unit suite asserts that round trip.
// This kernel runs only inside eager prefill chunks and is never part of a
// CUDA graph capture (docs/DESIGN.md section 9 captures decode only).

namespace redline::kernels {

namespace {

// Per-block work is one position's K+V payload - canonically 512 halves,
// i.e. 64 uint4 words - matching kv_scatter's mapping; block-size tuning is
// a later, measured concern.
constexpr std::int32_t kBlockThreads = 128;

// 128-bit copy unit of the vector path: 8 halves per uint4 word.
constexpr std::int32_t kVecHalves = 8;

// The launch contract documents the slot derivation as p >> 4 and p & 15.
// Derive shift and mask from the shared core constant instead of restating
// 16, so this file and core/types.hpp cannot drift apart.
static_assert(kKvBlockSize > 0 && (kKvBlockSize & (kKvBlockSize - 1)) == 0,
              "in-kernel slot derivation uses shift/mask arithmetic");
constexpr std::int32_t kKvBlockShift = std::countr_zero(static_cast<std::uint32_t>(kKvBlockSize));

// Element offset of the contiguous [head_dim]-half row for (block, kv, head,
// slot) inside ONE layer's pool slice
//   [num_blocks, 2 (0 = K, 1 = V), num_kv_heads, kKvBlockSize, head_dim]
// - the docs/DESIGN.md section 5 index formula with the layer term dropped
// (kernels receive the layer slice base). int64 math: a bench-sized pool
// slice already holds ~1.5e8 elements.
__device__ __forceinline__ std::int64_t PoolRowBase(std::int32_t block, std::int32_t kv,
                                                    std::int32_t head, std::int32_t slot,
                                                    std::int32_t num_kv_heads,
                                                    std::int32_t head_dim) {
  const std::int64_t plane = static_cast<std::int64_t>(block) * 2 + kv;
  return ((plane * num_kv_heads + head) * kKvBlockSize + slot) * head_dim;
}

// One block per gathered position; threads stride over that position's K and
// V words. Word = uint4 is the 128-bit fast path (launcher checks the
// alignment premise); Word = half is the scalar fallback - same body,
// kHalvesPerWord degenerates to 1, keeping the addressing logic in exactly
// one place.
// Aliasing contract: khat and vhat are written and must not overlap each
// other or the pool (canonically they are dedicated staging regions of the
// activation scratch; the pool is a separate init-time allocation).
template <typename Word>
__global__ __launch_bounds__(kBlockThreads) void KvGatherKernel(
    half* __restrict__ khat, half* __restrict__ vhat, std::int64_t khat_head_stride,
    std::int64_t khat_row_stride, std::int64_t vhat_head_stride, std::int64_t vhat_row_stride,
    const half* __restrict__ pool, const std::int32_t* __restrict__ block_table_row,
    std::int32_t num_kv_heads, std::int32_t head_dim) {
  constexpr std::int32_t kHalvesPerWord = static_cast<std::int32_t>(sizeof(Word) / sizeof(half));

  const std::int64_t t = blockIdx.x; // gathered position == destination row; grid.x == ctx_len
  // The block-table load is uniform across the block and broadcasts from
  // one cache line. Entry t >> 4 exists whenever the caller's row covers
  // ceil(ctx_len / 16) blocks - every position in [0, ctx_len) was
  // previously scattered, so its block is allocated and mapped.
  const std::int32_t block = block_table_row[t >> kKvBlockShift];
  const std::int32_t slot = static_cast<std::int32_t>(t) & (kKvBlockSize - 1);

  const std::int32_t words_per_head = head_dim / kHalvesPerWord;
  const std::int32_t words_per_side = num_kv_heads * words_per_head; // K plane or V plane
  const std::int32_t words_per_token = 2 * words_per_side;

  const Word* __restrict__ pool_words = reinterpret_cast<const Word*>(pool);

  for (std::int32_t w = static_cast<std::int32_t>(threadIdx.x); w < words_per_token;
       w += kBlockThreads) {
    const std::int32_t kv = w < words_per_side ? 0 : 1; // 0 = K -> khat, 1 = V -> vhat
    const std::int32_t r = w - kv * words_per_side;     // word within that plane's row
    const std::int32_t h = r / words_per_head;
    const std::int32_t dw = r - h * words_per_head; // word within the (head, slot) row

    // Source: word dw of the pool's (block, kv, h, slot) row - PoolRowBase
    // is a multiple of head_dim, so the word division is exact whenever the
    // vector path is taken. Destination: word dw of row t in head plane h
    // of khat or vhat (dispatch guarantees the strides are whole words on
    // the vector path).
    const std::int64_t src =
        PoolRowBase(block, kv, h, slot, num_kv_heads, head_dim) / kHalvesPerWord + dw;
    half* const dst_row = kv == 0 ? khat + h * khat_head_stride + t * khat_row_stride
                                  : vhat + h * vhat_head_stride + t * vhat_row_stride;
    reinterpret_cast<Word*>(dst_row)[dw] = pool_words[src];
  }
}

// True when the pointer may be dereferenced as uint4 (16-byte words).
bool IsUint4Aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % alignof(uint4) == 0;
}

} // namespace

void LaunchKvGather(half* khat, half* vhat, std::int64_t khat_head_stride,
                    std::int64_t khat_row_stride, std::int64_t vhat_head_stride,
                    std::int64_t vhat_row_stride, const half* kv_pool_layer,
                    const std::int32_t* block_table_row, std::int32_t ctx_len,
                    std::int32_t num_kv_heads, std::int32_t head_dim, cudaStream_t stream) {
  if (ctx_len <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return; // a zero-extent grid is a CUDA launch error; there is nothing to gather
  }

  const dim3 grid(static_cast<unsigned int>(ctx_len)); // one block per gathered position
  const dim3 block(kBlockThreads);

  // Vector path premise: every 128-bit word the kernel touches must be
  // 16-byte aligned - base pointers aligned and all strides in whole words
  // (head rows sit at multiples of head_dim relative to the pool base, so
  // head_dim must be whole words too). The canonical prefill call - dense
  // rows (row stride == head_dim == 128), section 5 allocation head stride
  // max_seq_len * head_dim, cudaMalloc'd buffers - always qualifies.
  const bool vectorizable = head_dim % kVecHalves == 0 && khat_head_stride % kVecHalves == 0 &&
                            khat_row_stride % kVecHalves == 0 &&
                            vhat_head_stride % kVecHalves == 0 &&
                            vhat_row_stride % kVecHalves == 0 && IsUint4Aligned(khat) &&
                            IsUint4Aligned(vhat) && IsUint4Aligned(kv_pool_layer);
  if (vectorizable) {
    KvGatherKernel<uint4><<<grid, block, 0, stream>>>(
        khat, vhat, khat_head_stride, khat_row_stride, vhat_head_stride, vhat_row_stride,
        kv_pool_layer, block_table_row, num_kv_heads, head_dim);
  } else {
    KvGatherKernel<half><<<grid, block, 0, stream>>>(
        khat, vhat, khat_head_stride, khat_row_stride, vhat_head_stride, vhat_row_stride,
        kv_pool_layer, block_table_row, num_kv_heads, head_dim);
  }
}

} // namespace redline::kernels
