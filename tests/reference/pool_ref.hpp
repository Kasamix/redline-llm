#pragma once

// CPU mirror of the paged KV pool layout (docs/DESIGN.md section 5) and of
// the scatter/gather slot derivation (section 6.2 #4/#5). Header-only,
// independent of the kernels under test: the index math is restated here
// from the design document, not shared with the .cu implementations, so a
// transcription error on either side shows up as a test failure instead of
// cancelling out.

#include <cstdint>
#include <vector>

namespace redline::reference {

// Frozen by the kernels.cuh contract ("kKvBlockSize = 16").
inline constexpr std::int32_t kKvBlockSize = 16;

// Element index into ONE layer's pool slice
//   [num_blocks, 2 (0 = K, 1 = V), num_kv_heads, kKvBlockSize, head_dim]
// per the section 5 formula (layer factored out by the caller's slice).
inline std::int64_t PoolElemIndex(std::int32_t block, std::int32_t kv, std::int32_t head,
                                  std::int32_t slot, std::int32_t d, std::int32_t num_kv_heads,
                                  std::int32_t head_dim) {
  return (
      (((static_cast<std::int64_t>(block) * 2 + kv) * num_kv_heads + head) * kKvBlockSize + slot) *
          head_dim +
      d);
}

inline std::int64_t PoolSliceElems(std::int32_t num_blocks, std::int32_t num_kv_heads,
                                   std::int32_t head_dim) {
  return static_cast<std::int64_t>(num_blocks) * 2 * num_kv_heads * kKvBlockSize * head_dim;
}

// CPU scatter mirroring the LaunchKvScatter contract: for token t,
// p = positions[t], block = block_table_row[p >> 4], slot = p & 15;
// block_table_row_stride 0 = prefill (all tokens share one row).
// k/v are strided CPU mirrors ([num_tokens, num_kv_heads * head_dim] rows
// spaced k/v_row_stride elements). Pure bit moves.
inline void ScatterReference(std::vector<std::uint16_t>& pool, const std::uint16_t* k,
                             const std::uint16_t* v, std::int64_t k_row_stride,
                             std::int64_t v_row_stride, const std::int32_t* block_tables,
                             std::int64_t block_table_row_stride, const std::int32_t* positions,
                             std::int32_t num_tokens, std::int32_t num_kv_heads,
                             std::int32_t head_dim) {
  for (std::int32_t t = 0; t < num_tokens; ++t) {
    const std::int32_t p = positions[t];
    const std::int32_t block =
        block_tables[static_cast<std::int64_t>(t) * block_table_row_stride + p / kKvBlockSize];
    const std::int32_t slot = p % kKvBlockSize;
    for (std::int32_t h = 0; h < num_kv_heads; ++h) {
      for (std::int32_t d = 0; d < head_dim; ++d) {
        const std::int64_t src = static_cast<std::int64_t>(h) * head_dim + d;
        pool[static_cast<std::size_t>(
            PoolElemIndex(block, 0, h, slot, d, num_kv_heads, head_dim))] =
            k[static_cast<std::int64_t>(t) * k_row_stride + src];
        pool[static_cast<std::size_t>(
            PoolElemIndex(block, 1, h, slot, d, num_kv_heads, head_dim))] =
            v[static_cast<std::int64_t>(t) * v_row_stride + src];
      }
    }
  }
}

// CPU gather mirroring LaunchKvGather: destination row t IS the position
// (block = block_table_row[t >> 4], slot = t & 15); khat/vhat head plane h
// starts at h * *_head_stride, token rows spaced *_row_stride.
inline void GatherReference(std::uint16_t* khat, std::uint16_t* vhat, std::int64_t khat_head_stride,
                            std::int64_t khat_row_stride, std::int64_t vhat_head_stride,
                            std::int64_t vhat_row_stride, const std::vector<std::uint16_t>& pool,
                            const std::int32_t* block_table_row, std::int32_t ctx_len,
                            std::int32_t num_kv_heads, std::int32_t head_dim) {
  for (std::int32_t t = 0; t < ctx_len; ++t) {
    const std::int32_t block = block_table_row[t / kKvBlockSize];
    const std::int32_t slot = t % kKvBlockSize;
    for (std::int32_t h = 0; h < num_kv_heads; ++h) {
      for (std::int32_t d = 0; d < head_dim; ++d) {
        khat[h * khat_head_stride + static_cast<std::int64_t>(t) * khat_row_stride + d] =
            pool[static_cast<std::size_t>(
                PoolElemIndex(block, 0, h, slot, d, num_kv_heads, head_dim))];
        vhat[h * vhat_head_stride + static_cast<std::int64_t>(t) * vhat_row_stride + d] =
            pool[static_cast<std::size_t>(
                PoolElemIndex(block, 1, h, slot, d, num_kv_heads, head_dim))];
      }
    }
  }
}

} // namespace redline::reference
