// BlockAllocator unit tests - written against the reservation API of
// docs/DESIGN.md section 7 (CanReserve / Reserve / Unreserve / AllocReserved
// / FreeBlock). Block 0 is the reserved dummy block and is never allocatable.
//
// Coverage map (ticket acceptance):
//   - reservation arithmetic for both admission patterns
//     (ReserveFullAdmissionPattern, WatermarkAdmissionPattern)
//   - alloc until exhaustion honoring reservations
//     (AllocUntilExhaustionHonorsReservations, AllocReservedNeverFailsWithinReservation)
//   - Unreserve of the unused remainder (UnreserveReleasesUnusedRemainder)
//   - free-list conservation after full churn (asserted at the end of every
//     churny test plus RandomizedChurnConservesBlocks)
//   - block 0 never returned (exhaustion + churn tests check every id)
//   - double-free asserts in debug (death tests, skipped under NDEBUG)
//   - randomized reserve/alloc/free property test with
//     free + outstanding == usable checked after every operation.

#include "core/block_allocator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "core/types.hpp"

namespace redline {
namespace {

// Materializes `count` reserved blocks, as prefill scatter / decode boundary
// crossings would, returning the ids in allocation order.
std::vector<BlockId> AllocN(BlockAllocator& allocator, std::int64_t count) {
  std::vector<BlockId> ids;
  ids.reserve(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    ids.push_back(allocator.AllocReserved());
  }
  return ids;
}

void FreeAll(BlockAllocator& allocator, const std::vector<BlockId>& ids) {
  for (const BlockId id : ids) {
    allocator.FreeBlock(id);
  }
}

TEST(BlockAllocatorTest, ConstructionReportsTotals) {
  const BlockAllocator allocator(64);
  EXPECT_EQ(allocator.total_blocks(), 64);
  EXPECT_EQ(allocator.usable_blocks(), 63); // block 0 = dummy
  EXPECT_EQ(allocator.free_blocks(), 63);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(0));
  EXPECT_TRUE(allocator.CanReserve(63));
  EXPECT_FALSE(allocator.CanReserve(64)); // the dummy block is not reservable capacity
}

TEST(BlockAllocatorTest, DummyOnlyPoolHasNoUsableCapacity) {
  const BlockAllocator allocator(1); // holds only kDummyBlockId
  EXPECT_EQ(allocator.total_blocks(), 1);
  EXPECT_EQ(allocator.usable_blocks(), 0);
  EXPECT_EQ(allocator.free_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(0));
  EXPECT_FALSE(allocator.CanReserve(1));
}

// Pins the specified free-stack initialization [total_blocks-1 .. 1]: a
// fresh pool hands out ascending ids starting at 1 via pop_back().
TEST(BlockAllocatorTest, FreshPoolHandsOutAscendingIdsFromOne) {
  BlockAllocator allocator(8);
  allocator.Reserve(7);
  for (BlockId expected = 1; expected <= 7; ++expected) {
    EXPECT_EQ(allocator.AllocReserved(), expected);
  }
  EXPECT_EQ(allocator.free_blocks(), 0);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(BlockAllocatorTest, ReserveTracksAgainstFreeBlocks) {
  BlockAllocator allocator(64); // usable 63
  allocator.Reserve(30);
  EXPECT_EQ(allocator.reserved_blocks(), 30);
  // A reservation is a promise, not a materialization: nothing left the pool.
  EXPECT_EQ(allocator.free_blocks(), 63);

  // Exact boundary: n == free - reserved admits, n + 1 does not.
  EXPECT_TRUE(allocator.CanReserve(33));
  EXPECT_FALSE(allocator.CanReserve(34));

  allocator.Reserve(33);
  EXPECT_EQ(allocator.reserved_blocks(), 63);
  EXPECT_TRUE(allocator.CanReserve(0));
  EXPECT_FALSE(allocator.CanReserve(1));

  allocator.Unreserve(33); // headroom returns without any block traffic
  EXPECT_TRUE(allocator.CanReserve(33));
  allocator.Unreserve(30);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(63));
}

// reserve_full admission (docs/DESIGN.md section 8): admit only when the
// worst case ceil((prompt_len + max_new) / 16) blocks fit; materialize
// lazily; on finish, free the owned blocks and Unreserve the untouched
// remainder of the promise.
TEST(BlockAllocatorTest, ReserveFullAdmissionPattern) {
  BlockAllocator allocator(64); // usable 63

  // A: prompt 100 + max_new 50 -> 150 tokens -> 10 blocks.
  const std::int64_t need_a = BlocksForTokens(100 + 50);
  ASSERT_EQ(need_a, 10);
  ASSERT_TRUE(allocator.CanReserve(need_a));
  allocator.Reserve(need_a);

  // B: prompt 833 + max_new 15 -> 848 tokens -> 53 blocks - exactly the
  // remaining headroom, so this is the boundary admit.
  const std::int64_t need_b = BlocksForTokens(833 + 15);
  ASSERT_EQ(need_b, 53);
  ASSERT_TRUE(allocator.CanReserve(need_b));
  allocator.Reserve(need_b);
  EXPECT_EQ(allocator.reserved_blocks(), 63);

  // C: even a one-block request is refused while the pool is fully promised.
  EXPECT_FALSE(allocator.CanReserve(BlocksForTokens(16)));

  // A prefills its 7 prompt blocks, crosses one decode boundary
  // (100 + 20 = 120 tokens -> 8 blocks), then finishes early on EOS.
  std::vector<BlockId> a_blocks = AllocN(allocator, BlocksForTokens(100));
  a_blocks.push_back(allocator.AllocReserved()); // boundary crossing at position 112
  ASSERT_EQ(static_cast<std::int64_t>(a_blocks.size()), BlocksForTokens(120));
  EXPECT_EQ(allocator.free_blocks(), 63 - 8);
  EXPECT_EQ(allocator.reserved_blocks(), 63 - 8); // invariant is tight here

  FreeAll(allocator, a_blocks);
  allocator.Unreserve(need_a - static_cast<std::int64_t>(a_blocks.size())); // 2 unused promises
  EXPECT_EQ(allocator.free_blocks(), 63);
  EXPECT_EQ(allocator.reserved_blocks(), need_b);

  // The headroom C needed is back.
  EXPECT_TRUE(allocator.CanReserve(BlocksForTokens(16)));

  // B runs to its full reservation (nothing left to unreserve) and finishes.
  const std::vector<BlockId> b_blocks = AllocN(allocator, need_b);
  EXPECT_EQ(allocator.free_blocks(), 10);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  FreeAll(allocator, b_blocks);

  // Full-churn conservation: free == usable == num_blocks - 1.
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
  EXPECT_EQ(allocator.free_blocks(), 63);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

// watermark admission (docs/DESIGN.md section 8): admit while
// free - ceil(prompt_len / 16) >= 0.10 * num_blocks. Only prompt blocks are
// promised, and they materialize immediately during prefill; decode growth
// is a Reserve(1) + AllocReserved() pair per boundary, and CanReserve(1)
// returning false at that moment is the abort-youngest trigger.
TEST(BlockAllocatorTest, WatermarkAdmissionPattern) {
  BlockAllocator allocator(100);           // usable 99
  const std::int64_t watermark_floor = 10; // 0.10 * num_blocks

  const auto admit = [&](std::int32_t prompt_tokens) {
    const std::int64_t prompt_blocks = BlocksForTokens(prompt_tokens);
    EXPECT_GE(allocator.free_blocks() - prompt_blocks, watermark_floor);
    allocator.Reserve(prompt_blocks);
    return AllocN(allocator, prompt_blocks);
  };

  const std::vector<BlockId> seq1 = admit(640); // 40 blocks -> free 59
  const std::vector<BlockId> seq2 = admit(640); // 40 blocks -> free 19
  EXPECT_EQ(allocator.free_blocks(), 19);

  // A 20-block prompt fails the watermark check (19 - 20 < 10)...
  EXPECT_LT(allocator.free_blocks() - BlocksForTokens(320), watermark_floor);
  // ...while an 8-block prompt still clears it (19 - 8 >= 10).
  const std::vector<BlockId> seq3 = admit(128); // free 11
  EXPECT_EQ(allocator.free_blocks(), 11);
  // Watermark reservations are instantaneous: nothing stays promised.
  EXPECT_EQ(allocator.reserved_blocks(), 0);

  // Decode growth with no up-front reservation can drain the pool dry...
  std::vector<BlockId> seq1_decode;
  while (allocator.CanReserve(1)) {
    allocator.Reserve(1);
    seq1_decode.push_back(allocator.AllocReserved());
  }
  EXPECT_EQ(allocator.free_blocks(), 0);
  EXPECT_EQ(seq1_decode.size(), 11u);

  // ...and the next boundary cannot allocate: the scheduler aborts the
  // youngest sequence (seq3), whose freed blocks restore decode headroom.
  EXPECT_FALSE(allocator.CanReserve(1));
  FreeAll(allocator, seq3);
  EXPECT_EQ(allocator.free_blocks(), 8);
  EXPECT_TRUE(allocator.CanReserve(1));

  FreeAll(allocator, seq1);
  FreeAll(allocator, seq1_decode);
  FreeAll(allocator, seq2);
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

// The free >= reserved invariant means a promised block can always be
// materialized, no matter how other traffic churns the pool in between.
TEST(BlockAllocatorTest, AllocReservedNeverFailsWithinReservation) {
  BlockAllocator allocator(64); // usable 63
  allocator.Reserve(10);        // sequence A's promise

  // B reserves and materializes every remaining block.
  allocator.Reserve(53);
  std::vector<BlockId> b_blocks = AllocN(allocator, 53);
  EXPECT_EQ(allocator.free_blocks(), 10); // exactly A's promise survives
  EXPECT_FALSE(allocator.CanReserve(1));

  std::set<BlockId> a_ids;
  const auto take_for_a = [&](int n) {
    for (int i = 0; i < n; ++i) {
      const BlockId id = allocator.AllocReserved();
      EXPECT_NE(id, kDummyBlockId);
      EXPECT_NE(id, kInvalidBlockId);
      EXPECT_GE(id, 1);
      EXPECT_LT(id, allocator.total_blocks());
      EXPECT_TRUE(a_ids.insert(id).second) << "block " << id << " handed out twice";
    }
  };

  take_for_a(5); // free 5, reserved 5 - the invariant is tight

  // B churns: frees 20 blocks, then re-reserves and re-allocates them while
  // A's remaining promises are still outstanding.
  for (int i = 0; i < 20; ++i) {
    allocator.FreeBlock(b_blocks.back());
    b_blocks.pop_back();
  }
  EXPECT_TRUE(allocator.CanReserve(20)); // 25 free - 5 promised to A = 20
  EXPECT_FALSE(allocator.CanReserve(21));
  allocator.Reserve(20);
  const std::vector<BlockId> more_b = AllocN(allocator, 20);
  b_blocks.insert(b_blocks.end(), more_b.begin(), more_b.end());

  take_for_a(5); // A's remaining promises still materialize
  EXPECT_EQ(a_ids.size(), 10u);
  EXPECT_EQ(allocator.free_blocks(), 0);
  EXPECT_EQ(allocator.reserved_blocks(), 0);

  FreeAll(allocator, b_blocks);
  for (const BlockId id : a_ids) {
    allocator.FreeBlock(id);
  }
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
}

// Interleaved allocation of reservations that together cover the whole
// usable pool: exhaustion is exact, every id in [1, total_blocks) is
// produced exactly once, and block 0 never appears.
TEST(BlockAllocatorTest, AllocUntilExhaustionHonorsReservations) {
  BlockAllocator allocator(64);
  allocator.Reserve(40);
  allocator.Reserve(23);
  // The whole pool is promised before a single block moves: new admissions
  // are refused even though free_blocks() is still 63.
  EXPECT_EQ(allocator.free_blocks(), 63);
  EXPECT_FALSE(allocator.CanReserve(1));

  std::vector<BlockId> ids;
  for (int i = 0; i < 63; ++i) {
    ids.push_back(allocator.AllocReserved());
    EXPECT_FALSE(allocator.CanReserve(1)); // headroom never reappears mid-drain
  }
  EXPECT_EQ(allocator.free_blocks(), 0);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(0));

  // Exactly {1 .. 63}: no dummy, no sentinel, no duplicates, full coverage.
  std::sort(ids.begin(), ids.end());
  for (int i = 0; i < 63; ++i) {
    EXPECT_EQ(ids[static_cast<std::size_t>(i)], i + 1);
  }

  FreeAll(allocator, ids);
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
}

TEST(BlockAllocatorTest, UnreserveReleasesUnusedRemainder) {
  BlockAllocator allocator(64);
  allocator.Reserve(10);
  const std::vector<BlockId> owned = AllocN(allocator, 4); // early finish after 4 blocks
  EXPECT_EQ(allocator.free_blocks(), 59);
  EXPECT_EQ(allocator.reserved_blocks(), 6);

  FreeAll(allocator, owned);
  allocator.Unreserve(6);
  EXPECT_EQ(allocator.free_blocks(), 63);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(63)); // full headroom restored
}

TEST(BlockAllocatorTest, FreedBlocksBecomeAllocatableAgain) {
  BlockAllocator allocator(64);
  allocator.Reserve(63);
  const std::vector<BlockId> all = AllocN(allocator, 63);
  EXPECT_EQ(allocator.free_blocks(), 0);

  const std::set<BlockId> returned = {all[4], all[16], all[41]};
  for (const BlockId id : returned) {
    allocator.FreeBlock(id);
  }
  EXPECT_EQ(allocator.free_blocks(), 3);
  EXPECT_TRUE(allocator.CanReserve(3));
  EXPECT_FALSE(allocator.CanReserve(4));

  allocator.Reserve(3);
  std::set<BlockId> again;
  for (int i = 0; i < 3; ++i) {
    again.insert(allocator.AllocReserved());
  }
  EXPECT_EQ(again, returned); // exactly the returned ids were available
  EXPECT_EQ(allocator.free_blocks(), 0);
}

// Deterministic interleaved reserve/alloc/free cycles: the outstanding set
// stays disjoint and never contains the dummy block or the sentinel.
TEST(BlockAllocatorTest, NeverHandsOutABlockTwiceNorTheDummy) {
  BlockAllocator allocator(20); // usable 19 - small, so churn hits capacity
  std::set<BlockId> outstanding;

  const auto alloc_checked = [&]() {
    const BlockId id = allocator.AllocReserved();
    EXPECT_NE(id, kDummyBlockId);
    EXPECT_NE(id, kInvalidBlockId);
    EXPECT_GE(id, 1);
    EXPECT_LT(id, allocator.total_blocks());
    EXPECT_TRUE(outstanding.insert(id).second) << "block " << id << " handed out twice";
  };

  for (int round = 0; round < 8; ++round) {
    const std::int64_t take = 7 + (round % 3); // 7..9 blocks per round
    if (!allocator.CanReserve(take)) {
      // Finish roughly half of the live sequences to make room.
      std::vector<BlockId> to_free(outstanding.begin(), outstanding.end());
      to_free.resize(to_free.size() / 2 + 1);
      for (const BlockId id : to_free) {
        allocator.FreeBlock(id);
        outstanding.erase(id);
      }
    }
    ASSERT_TRUE(allocator.CanReserve(take));
    allocator.Reserve(take);
    for (std::int64_t i = 0; i < take; ++i) {
      alloc_checked();
    }
    // Free every third outstanding id so the stack order gets scrambled.
    std::vector<BlockId> round_free;
    int k = 0;
    for (const BlockId id : outstanding) {
      if (k++ % 3 == 0) {
        round_free.push_back(id);
      }
    }
    for (const BlockId id : round_free) {
      allocator.FreeBlock(id);
      outstanding.erase(id);
    }
    EXPECT_EQ(allocator.free_blocks() + static_cast<std::int32_t>(outstanding.size()),
              allocator.usable_blocks());
  }

  for (const BlockId id : outstanding) {
    allocator.FreeBlock(id);
  }
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

// Property test: random admit / lazy-alloc / finish churn with request sizes
// pinned to the block-size boundary (prompt_len + max_new == 0 or 1 mod 16).
// After every single operation: free + outstanding == usable, reserved
// equals the sum of live unmaterialized promises, and free >= reserved.
TEST(BlockAllocatorTest, RandomizedChurnConservesBlocks) {
  constexpr std::int32_t kTotalBlocks = 64;
  BlockAllocator allocator(kTotalBlocks);

  struct LiveSeq {
    std::int64_t promised = 0; // reserved, not yet materialized
    std::vector<BlockId> owned;
  };
  std::vector<LiveSeq> live;
  std::set<BlockId> outstanding_ids;
  std::mt19937 rng(20260709u); // fixed seed: failures reproduce

  const auto check = [&]() {
    std::int64_t outstanding = 0;
    std::int64_t promised = 0;
    for (const LiveSeq& s : live) {
      outstanding += static_cast<std::int64_t>(s.owned.size());
      promised += s.promised;
    }
    ASSERT_EQ(allocator.free_blocks() + outstanding, allocator.usable_blocks());
    ASSERT_EQ(allocator.reserved_blocks(), promised);
    ASSERT_GE(allocator.free_blocks(), allocator.reserved_blocks()); // free >= reserved
    ASSERT_EQ(static_cast<std::int64_t>(outstanding_ids.size()), outstanding);
  };

  for (int step = 0; step < 5000; ++step) {
    const std::uint32_t op = rng() % 4u;
    if (op == 0u) { // admission attempt (reserve_full sizing)
      const std::int32_t full_blocks = static_cast<std::int32_t>(1u + rng() % 6u); // 1..6
      const std::int32_t extra = static_cast<std::int32_t>(rng() % 2u);            // 0 or 1 mod 16
      const std::int32_t total_tokens = 16 * full_blocks + extra;
      const std::int64_t need = BlocksForTokens(total_tokens);
      ASSERT_EQ(need, full_blocks + (extra != 0 ? 1 : 0)); // ceil boundary arithmetic
      const std::int64_t headroom = allocator.free_blocks() - allocator.reserved_blocks();
      ASSERT_EQ(allocator.CanReserve(need), headroom >= need); // admission mirror
      if (allocator.CanReserve(need)) {
        allocator.Reserve(need);
        live.push_back(LiveSeq{need, {}});
      }
    } else if (op == 1u || op == 2u) { // lazy materialization (weighted 2x)
      std::vector<std::size_t> candidates;
      for (std::size_t i = 0; i < live.size(); ++i) {
        if (live[i].promised > 0) {
          candidates.push_back(i);
        }
      }
      if (!candidates.empty()) {
        LiveSeq& s = live[candidates[rng() % candidates.size()]];
        const BlockId id = allocator.AllocReserved();
        ASSERT_NE(id, kDummyBlockId);
        ASSERT_NE(id, kInvalidBlockId);
        ASSERT_GE(id, 1);
        ASSERT_LT(id, kTotalBlocks);
        ASSERT_TRUE(outstanding_ids.insert(id).second) << "block " << id << " double-handed";
        s.owned.push_back(id);
        --s.promised;
      }
    } else if (!live.empty()) { // finish/abort: free owned, unreserve remainder
      const std::size_t idx = rng() % live.size();
      LiveSeq& s = live[idx];
      for (const BlockId id : s.owned) {
        allocator.FreeBlock(id);
        outstanding_ids.erase(id);
      }
      allocator.Unreserve(s.promised);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(idx));
    }
    check();
    if (::testing::Test::HasFatalFailure()) {
      return; // stop at the first broken step so the log points at it
    }
  }

  // Drain everything; the pool must be fully recovered.
  for (LiveSeq& s : live) {
    for (const BlockId id : s.owned) {
      allocator.FreeBlock(id);
      outstanding_ids.erase(id);
    }
    allocator.Unreserve(s.promised);
  }
  live.clear();
  check();
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
  EXPECT_EQ(allocator.free_blocks(), kTotalBlocks - 1);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(allocator.CanReserve(allocator.usable_blocks()));
}

// ---------------------------------------------------------------------------
// Debug-assert death tests. assert() compiles out under NDEBUG, so these run
// only in Debug builds (and only where googletest supports death tests);
// otherwise they skip with an explanatory message.

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
#define REDLINE_DEBUG_DEATH_TESTS 1
#else
#define REDLINE_DEBUG_DEATH_TESTS 0
#endif

[[maybe_unused]] constexpr char kNoDebugAssertsMsg[] =
    "assert() compiled out (NDEBUG) or no death-test support; build Debug to exercise";

TEST(BlockAllocatorTest, DoubleFreeAssertsInDebug) {
#if REDLINE_DEBUG_DEATH_TESTS
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  BlockAllocator allocator(8);
  allocator.Reserve(1);
  const BlockId block = allocator.AllocReserved();
  allocator.FreeBlock(block);
  EXPECT_DEATH(allocator.FreeBlock(block), "");
#else
  GTEST_SKIP() << kNoDebugAssertsMsg;
#endif
}

TEST(BlockAllocatorTest, FreeingDummyOrSentinelAssertsInDebug) {
#if REDLINE_DEBUG_DEATH_TESTS
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  BlockAllocator allocator(8);
  EXPECT_DEATH(allocator.FreeBlock(kDummyBlockId), "");
  EXPECT_DEATH(allocator.FreeBlock(kInvalidBlockId), "");
  EXPECT_DEATH(allocator.FreeBlock(8), ""); // out of pool range
  EXPECT_DEATH(allocator.FreeBlock(3), ""); // in range but never allocated
#else
  GTEST_SKIP() << kNoDebugAssertsMsg;
#endif
}

TEST(BlockAllocatorTest, ReserveBeyondHeadroomAssertsInDebug) {
#if REDLINE_DEBUG_DEATH_TESTS
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  BlockAllocator allocator(8); // usable 7
  EXPECT_DEATH(allocator.Reserve(8), "");
  allocator.Reserve(7);
  EXPECT_DEATH(allocator.Reserve(1), "");
#else
  GTEST_SKIP() << kNoDebugAssertsMsg;
#endif
}

TEST(BlockAllocatorTest, UnreserveUnderflowAssertsInDebug) {
#if REDLINE_DEBUG_DEATH_TESTS
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  BlockAllocator allocator(8);
  allocator.Reserve(2);
  EXPECT_DEATH(allocator.Unreserve(3), "");
#else
  GTEST_SKIP() << kNoDebugAssertsMsg;
#endif
}

TEST(BlockAllocatorTest, AllocWithoutReservationAssertsInDebug) {
#if REDLINE_DEBUG_DEATH_TESTS
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  BlockAllocator allocator(8);
  EXPECT_DEATH(allocator.AllocReserved(), "");
#else
  GTEST_SKIP() << kNoDebugAssertsMsg;
#endif
}

} // namespace
} // namespace redline
