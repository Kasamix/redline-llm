#pragma once

#include <cstdint>
#include <vector>

#include "core/types.hpp"

namespace redline {

// Host-side reservation allocator for physical KV-cache blocks - the exact
// contract of docs/DESIGN.md section 7 (this header and that section are
// kept in lockstep on purpose).
//
// The device pool itself (one cudaMalloc at init, layer-outer layout,
// docs/DESIGN.md section 5) is owned by the engine; this class only tracks
// which block ids are free and how many are promised. Block id 0 is the
// reserved dummy block (kDummyBlockId): it never enters the free list, so
// usable capacity is total_blocks - 1. A BlockId is valid in every layer -
// all layers share one index space, so a sequence owns the same ids across
// layers. Paged-cache concept from Kwon et al., 2023 (PagedAttention);
// implementation is original.
//
// Reservation model (what makes "no preemption in v1" sound): admission
// under the default reserve_full policy calls
// Reserve(BlocksForTokens(prompt_len + max_new)) up front, and blocks are
// then materialized lazily via AllocReserved() - prompt blocks during
// prefill scatter, decode blocks at 16-token boundaries. The invariant
// free_.size() >= reserved_ guarantees a lazily requested block always
// exists, so decode can never run out of blocks mid-flight. On finish/abort,
// owned blocks return via FreeBlock and the unused remainder of the
// reservation is released via Unreserve.
//
// All operations are O(1); the constructor is O(total_blocks).
// Not thread-safe; the engine serializes scheduler/allocator access.
class BlockAllocator {
 public:
  explicit BlockAllocator(std::int32_t total_blocks);

  // True when `count` more blocks could be promised without breaking the
  // free >= reserved invariant.
  bool CanReserve(std::int64_t count) const;

  // Promise `count` blocks to an admitted sequence (admission time). Must be
  // preceded by a CanReserve check (debug-asserted).
  void Reserve(std::int64_t count);

  // Release promised-but-never-allocated blocks (finish/abort time).
  void Unreserve(std::int64_t count);

  // Materialize one previously reserved block (lazy, at first use). The
  // invariant guarantees success; an empty free list here is a programming
  // error (debug-asserted).
  BlockId AllocReserved();

  // Return one block to the free list. Freeing an id twice, or freeing
  // kDummyBlockId / kInvalidBlockId, is a programming error (debug-asserted).
  void FreeBlock(BlockId block);

  std::int32_t total_blocks() const { return total_blocks_; }
  // total minus the reserved dummy block.
  std::int32_t usable_blocks() const { return total_blocks_ > 0 ? total_blocks_ - 1 : 0; }
  std::int32_t free_blocks() const { return static_cast<std::int32_t>(free_list_.size()); }
  std::int64_t reserved_blocks() const { return reserved_; }

 private:
  // Debug-asserts the class invariant (free_list_.size() >= reserved_ plus
  // counter sanity) after every mutation. Compiles to nothing under NDEBUG.
  void DebugCheckInvariant() const;

  std::int32_t total_blocks_ = 0;
  std::int64_t reserved_ = 0;      // promised to admitted sequences, not yet allocated
  std::vector<BlockId> free_list_; // stack; init [total_blocks-1 .. 1] (0 = dummy)
  // Shadow occupancy map: outstanding_[id] == 1 iff `id` is currently
  // allocated (popped via AllocReserved and not yet freed). Maintained in
  // every build type (one byte store per alloc/free) so the layout does not
  // depend on NDEBUG, but consulted only by the debug asserts that catch
  // double frees and frees of foreign ids. Entry 0 (kDummyBlockId) stays 0.
  std::vector<std::uint8_t> outstanding_;
};

} // namespace redline
