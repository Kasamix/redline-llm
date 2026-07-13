#include "core/block_allocator.hpp"

#include <cassert>
#include <cstddef>

namespace redline {

BlockAllocator::BlockAllocator(std::int32_t total_blocks) : total_blocks_(total_blocks) {
  assert(total_blocks_ >= 1 && "pool must contain at least the reserved dummy block 0");
  // Free stack holds [total_blocks-1 .. 1], so pop_back() hands out ascending
  // ids starting at 1. Block 0 (kDummyBlockId) never enters the list - it is
  // the padding target for device block tables (docs/DESIGN.md section 9).
  free_list_.reserve(static_cast<std::size_t>(usable_blocks()));
  for (BlockId id = total_blocks_ - 1; id >= 1; --id) {
    free_list_.push_back(id);
  }
  outstanding_.assign(total_blocks_ > 0 ? static_cast<std::size_t>(total_blocks_) : 0u, 0u);
  DebugCheckInvariant();
}

bool BlockAllocator::CanReserve(std::int64_t count) const {
  assert(count >= 0 && "reservation size must be non-negative");
  return static_cast<std::int64_t>(free_list_.size()) - reserved_ >= count;
}

void BlockAllocator::Reserve(std::int64_t count) {
  assert(count >= 0 && "reservation size must be non-negative");
  assert(CanReserve(count) && "Reserve() without a passing CanReserve() check");
  reserved_ += count;
  DebugCheckInvariant();
}

void BlockAllocator::Unreserve(std::int64_t count) {
  assert(count >= 0 && "unreserve size must be non-negative");
  assert(count <= reserved_ && "Unreserve() exceeds the outstanding reservation total");
  reserved_ -= count;
  DebugCheckInvariant();
}

BlockId BlockAllocator::AllocReserved() {
  assert(reserved_ > 0 && "AllocReserved() without a live reservation");
  // free_.size() >= reserved_ > 0, so the pop below cannot underflow when the
  // reservation contract above is honored.
  assert(!free_list_.empty() && "free list empty despite the free >= reserved invariant");
  const BlockId block = free_list_.back();
  free_list_.pop_back();
  --reserved_;
  assert(block != kDummyBlockId && block != kInvalidBlockId && "corrupt id on the free list");
  assert(block >= 1 && block < total_blocks_ && "free-list id outside the pool");
  assert(outstanding_[static_cast<std::size_t>(block)] == 0 && "id was already outstanding");
  outstanding_[static_cast<std::size_t>(block)] = 1;
  DebugCheckInvariant();
  return block;
}

void BlockAllocator::FreeBlock(BlockId block) {
  assert(block != kInvalidBlockId && "FreeBlock(kInvalidBlockId): host-side sentinel, never owned");
  assert(block != kDummyBlockId && "FreeBlock(kDummyBlockId): dummy block never enters the pool");
  assert(block >= 1 && block < total_blocks_ && "FreeBlock() id outside the pool");
  assert(outstanding_[static_cast<std::size_t>(block)] == 1 &&
         "double free (or freeing an id that was never allocated)");
  outstanding_[static_cast<std::size_t>(block)] = 0;
  free_list_.push_back(block);
  DebugCheckInvariant();
}

void BlockAllocator::DebugCheckInvariant() const {
#ifndef NDEBUG
  assert(reserved_ >= 0 && "reserved count went negative");
  assert(static_cast<std::int64_t>(free_list_.size()) >= reserved_ &&
         "invariant violated: fewer free blocks than promised (free >= reserved)");
  assert(static_cast<std::int64_t>(free_list_.size()) <= usable_blocks() &&
         "free list exceeds usable capacity (double free?)");
#endif
}

} // namespace redline
