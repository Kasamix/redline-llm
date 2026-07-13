#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace redline {

// Per-request mapping from logical block index (token_position / kKvBlockSize)
// to physical BlockId in the paged KV pool.
//
// Before each step the engine writes the tables of every scheduled sequence
// into the pinned host mirror of the device int32 tensor
// [max_seqs, max_blocks_per_seq] and uploads it as one contiguous copy.
// Unused entries and padded rows are filled with kDummyBlockId (block 0,
// reserved at init) so any address a kernel computes from padding stays in
// bounds; kInvalidBlockId is host-side-only and is debug-asserted never to
// reach the mirror (docs/DESIGN.md section 9 sentinel policy). Block-table
// indirection follows the PagedAttention design (Kwon et al., 2023);
// implementation is original.
class BlockTable {
 public:
  void Append(BlockId block) { blocks_.push_back(block); }

  std::int32_t num_blocks() const { return static_cast<std::int32_t>(blocks_.size()); }
  bool empty() const { return blocks_.empty(); }
  const std::vector<BlockId>& blocks() const { return blocks_; }

  // Capacity in tokens of the blocks currently owned.
  std::int32_t token_capacity() const { return num_blocks() * kKvBlockSize; }

  // Hand every owned id back to the caller (destined for BlockAllocator::Free)
  // and leave this table empty.
  std::vector<BlockId> Release() { return std::exchange(blocks_, {}); }

 private:
  std::vector<BlockId> blocks_;
};

} // namespace redline
