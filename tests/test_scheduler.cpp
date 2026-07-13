// Scheduler unit tests: pure CPU, no GPU or model required. Step execution is
// simulated by handing PlanStep()'s output straight back to CompleteStep()
// with scripted sampled tokens (mock executor pattern), including scripted
// EOS positions. Written against the reservation contract of
// docs/DESIGN.md sections 7-8 and the test list of section 12(b). These
// tests exercise the real BlockAllocator contract, so free/reserved
// counters are asserted exactly.

#include "core/scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/block_allocator.hpp"
#include "core/request.hpp"

namespace redline {
namespace {

// Qwen2.5-1.5B-Instruct EOS set (docs/MODEL_SPEC.md); any TokenId works for
// the scheduler, these keep the fixtures realistic.
constexpr TokenId kEosPrimary = 151645;
constexpr TokenId kEosSecondary = 151643;
// Generic non-EOS "argmax" token the mock executor returns by default.
constexpr TokenId kTok = 7;

SchedulerOptions MakeOptions(AdmissionPolicy policy = AdmissionPolicy::kReserveFull,
                             std::int32_t max_batch = 8, std::int32_t prefill_chunk = 1024,
                             std::int32_t max_seq_len = 2048) {
  SchedulerOptions options;
  options.max_batch = max_batch;
  options.prefill_chunk_tokens = prefill_chunk;
  options.max_seq_len = max_seq_len;
  options.admission_policy = policy;
  options.eos_token_ids = {kEosPrimary, kEosSecondary};
  return options;
}

RequestPtr MakeRequest(RequestId id, std::int32_t prompt_len, std::int32_t max_new_tokens,
                       bool ignore_eos = false, std::vector<TokenId> forced_tokens = {}) {
  auto request = std::make_shared<Request>();
  request->id = id;
  request->prompt_tokens.assign(static_cast<std::size_t>(prompt_len), TokenId{1});
  request->max_new_tokens = max_new_tokens;
  request->ignore_eos = ignore_eos;
  request->forced_tokens = std::move(forced_tokens);
  return request;
}

// Mock executor: produce the sampled-token vector CompleteStep expects for
// `plan` - one token per decode row, one for a final prefill chunk, none
// otherwise. `sample` maps a Request to its scripted argmax.
template <typename SampleFn>
std::vector<TokenId> SampledFor(const StepPlan& plan, SampleFn&& sample) {
  std::vector<TokenId> tokens;
  if (plan.is_prefill()) {
    if (plan.final_prefill_chunk()) {
      tokens.push_back(sample(*plan.prefill));
    }
    return tokens;
  }
  tokens.reserve(plan.decode.size());
  for (const RequestPtr& request : plan.decode) {
    tokens.push_back(sample(*request));
  }
  return tokens;
}

std::vector<TokenId> SampledFor(const StepPlan& plan, TokenId token) {
  return SampledFor(plan, [token](const Request&) { return token; });
}

// One full scheduler iteration with a constant scripted argmax.
std::vector<StepResult> StepOnce(Scheduler& scheduler, TokenId token = kTok) {
  const StepPlan plan = scheduler.PlanStep();
  return scheduler.CompleteStep(plan, SampledFor(plan, token));
}

// Pump with a non-EOS token until every request drains via its length stop.
void DrainAll(Scheduler& scheduler, TokenId token = kTok, int max_steps = 100000) {
  for (int i = 0; i < max_steps; ++i) {
    if (scheduler.num_waiting() + scheduler.num_running() == 0) {
      return;
    }
    StepOnce(scheduler, token);
  }
  FAIL() << "scheduler failed to drain";
}

TEST(SchedulerTest, EnqueueLandsInWaitingQueue) {
  BlockAllocator allocator(128);
  Scheduler scheduler(&allocator, MakeOptions());
  scheduler.Enqueue(MakeRequest(1, 100, 32));
  EXPECT_EQ(scheduler.num_waiting(), 1u);
  EXPECT_EQ(scheduler.num_running(), 0u);
  // Enqueue must not touch the pool: admission happens inside PlanStep.
  EXPECT_EQ(allocator.free_blocks(), allocator.usable_blocks());
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(SchedulerTest, RejectsMalformedAndNeverFittingRequests) {
  BlockAllocator allocator(9); // usable 8 blocks = 128 tokens
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024,
                                              /*max_seq_len=*/128));
  EXPECT_THROW(scheduler.Enqueue(MakeRequest(1, 0, 4)), std::invalid_argument);    // empty prompt
  EXPECT_THROW(scheduler.Enqueue(MakeRequest(2, 4, 0)), std::invalid_argument);    // max_new == 0
  EXPECT_THROW(scheduler.Enqueue(MakeRequest(3, 4, -3)), std::invalid_argument);   // max_new < 0
  EXPECT_THROW(scheduler.Enqueue(MakeRequest(4, 100, 29)), std::invalid_argument); // 129 > 128
  // Exactly max_seq_len and exactly the usable pool is legal.
  EXPECT_NO_THROW(scheduler.Enqueue(MakeRequest(5, 100, 28)));

  // reserve_full never-fits: worst-case blocks exceed the whole usable pool.
  BlockAllocator small(5); // usable 4 blocks = 64 tokens
  Scheduler tight(&small, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 128));
  EXPECT_THROW(tight.Enqueue(MakeRequest(6, 60, 8)), std::invalid_argument); // 68 tok -> 5 > 4
  EXPECT_NO_THROW(tight.Enqueue(MakeRequest(7, 60, 4)));                     // 64 tok -> 4 == 4

  // watermark never-fits: even an idle pool minus the prompt breaks the
  // 10% free floor (max_new is irrelevant under this policy).
  BlockAllocator wm(101); // usable 100, floor 10
  Scheduler watermark(&wm, MakeOptions(AdmissionPolicy::kWatermark, 8, 1024, 2048));
  EXPECT_THROW(watermark.Enqueue(MakeRequest(8, 91 * 16, 8)), std::invalid_argument); // 100-91 < 10
  EXPECT_NO_THROW(watermark.Enqueue(MakeRequest(9, 90 * 16, 8))); // 100-90 >= 10
}

TEST(SchedulerTest, ReserveFullAdmissionReservesWorstCase) {
  BlockAllocator allocator(11); // usable 10
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 256));

  // 33 + 31 = 64 tokens -> worst case 4 blocks reserved at admission.
  scheduler.Enqueue(MakeRequest(1, 33, 31));
  const StepPlan plan1 = scheduler.PlanStep();
  ASSERT_TRUE(plan1.is_prefill());
  EXPECT_EQ(plan1.prefill->id, 1u);
  EXPECT_EQ(plan1.prefill_begin, 0);
  EXPECT_EQ(plan1.prefill_end, 33);
  EXPECT_TRUE(plan1.final_prefill_chunk());
  // Reservation 4; the prefill chunk lazily materialized ceil(33/16) = 3 of
  // them. No bulk allocation of the remainder happened.
  EXPECT_EQ(plan1.prefill->block_table.num_blocks(), 3);
  EXPECT_EQ(plan1.prefill->reserved_blocks, 1);
  EXPECT_EQ(plan1.prefill->slot, 0);
  EXPECT_EQ(allocator.free_blocks(), 7);
  EXPECT_EQ(allocator.reserved_blocks(), 1);
  scheduler.CompleteStep(plan1, {kTok});

  // Exact CanReserve boundary: request 2 needs 6 blocks and free - reserved
  // is exactly 6; request 3 (1 block) then finds 0 uncommitted and stalls.
  scheduler.Enqueue(MakeRequest(2, 81, 15)); // 96 tokens -> 6 blocks
  scheduler.Enqueue(MakeRequest(3, 1, 15));  // 16 tokens -> 1 block
  const StepPlan plan2 = scheduler.PlanStep();
  ASSERT_TRUE(plan2.is_prefill());
  EXPECT_EQ(plan2.prefill->id, 2u);
  EXPECT_EQ(scheduler.num_waiting(), 1u); // request 3 stalled, FIFO head intact
  EXPECT_EQ(scheduler.num_running(), 2u);
  EXPECT_EQ(allocator.free_blocks(), 1);
  EXPECT_EQ(allocator.reserved_blocks(), 1);
  scheduler.CompleteStep(plan2, {kTok});

  // Finish request 1 (EOS): its 3 blocks and 1-block remainder return, which
  // is exactly what request 3 needs.
  const StepPlan plan3 = scheduler.PlanStep();
  ASSERT_EQ(plan3.decode.size(), 2u);
  const auto emissions = scheduler.CompleteStep(plan3, {kEosPrimary, kTok});
  ASSERT_EQ(emissions.size(), 2u);
  EXPECT_TRUE(emissions[0].finished);
  EXPECT_EQ(emissions[0].finish_reason, FinishReason::kEos);
  EXPECT_EQ(allocator.free_blocks(), 4);
  EXPECT_EQ(allocator.reserved_blocks(), 0);

  const StepPlan plan4 = scheduler.PlanStep();
  ASSERT_TRUE(plan4.is_prefill());
  EXPECT_EQ(plan4.prefill->id, 3u);
  EXPECT_EQ(scheduler.num_waiting(), 0u);
}

TEST(SchedulerTest, WatermarkAdmissionUsesFreeFloorOnly) {
  BlockAllocator allocator(101); // usable 100, floor = 10.0
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kWatermark, 8, 1024, 4096));

  // 89 prompt blocks with a huge max_new: watermark ignores decode growth.
  // (Contrast: reserve_full would need 217 blocks and reject it outright.)
  scheduler.Enqueue(MakeRequest(1, 1424, 2048));
  {
    BlockAllocator full(101);
    Scheduler reserve_full(&full, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 4096));
    EXPECT_THROW(reserve_full.Enqueue(MakeRequest(1, 1424, 2048)), std::invalid_argument);
  }
  scheduler.Enqueue(MakeRequest(2, 16, 8)); // 1 prompt block
  scheduler.Enqueue(MakeRequest(3, 16, 8)); // 1 prompt block

  // Admission floor: r1 leaves 11 uncommitted (>= 10), r2 leaves exactly 10,
  // r3 would leave 9 and stalls.
  const StepPlan plan1 = scheduler.PlanStep();
  ASSERT_TRUE(plan1.is_prefill());
  EXPECT_EQ(plan1.prefill->id, 1u);
  EXPECT_FALSE(plan1.final_prefill_chunk()); // 1024 of 1424 prompt tokens
  EXPECT_EQ(scheduler.num_waiting(), 1u);
  EXPECT_EQ(scheduler.num_running(), 2u);
  // Reserved: 89 (r1 prompt) + 1 (r2 prompt) - 64 materialized this chunk.
  EXPECT_EQ(allocator.free_blocks(), 36);
  EXPECT_EQ(allocator.reserved_blocks(), 26);
  EXPECT_TRUE(scheduler.CompleteStep(plan1, {}).empty()); // intermediate chunk emits nothing

  const StepPlan plan2 = scheduler.PlanStep(); // [1024, 1424): final chunk
  ASSERT_TRUE(plan2.is_prefill());
  EXPECT_EQ(plan2.prefill_begin, 1024);
  EXPECT_EQ(plan2.prefill_end, 1424);
  EXPECT_TRUE(plan2.final_prefill_chunk());
  scheduler.CompleteStep(plan2, {kTok});
  EXPECT_EQ(allocator.free_blocks(), 11);
  EXPECT_EQ(allocator.reserved_blocks(), 1);

  StepOnce(scheduler);                    // prefill r2
  EXPECT_EQ(scheduler.num_waiting(), 1u); // r3 still cannot clear the floor

  // Decode both to EOS; the freed blocks finally admit r3.
  const StepPlan plan4 = scheduler.PlanStep();
  ASSERT_EQ(plan4.decode.size(), 2u);
  scheduler.CompleteStep(plan4, {kEosPrimary, kEosPrimary});
  EXPECT_EQ(allocator.free_blocks(), 100);
  EXPECT_EQ(allocator.reserved_blocks(), 0);

  const StepPlan plan5 = scheduler.PlanStep();
  ASSERT_TRUE(plan5.is_prefill());
  EXPECT_EQ(plan5.prefill->id, 3u);
  EXPECT_EQ(scheduler.num_waiting(), 0u);
}

TEST(SchedulerTest, WatermarkExhaustionAbortsYoungestWithKAborted) {
  BlockAllocator allocator(7); // usable 6
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kWatermark, 8, 1024, 64));
  auto r1 = MakeRequest(1, 16, 40);
  auto r2 = MakeRequest(2, 16, 40);
  scheduler.Enqueue(r1);
  scheduler.Enqueue(r2);

  // Prefill both (1 block each), then decode until the pool cannot cover
  // decode growth: block boundaries hit at output counts 1, 17, 33 per
  // sequence, and the pool holds only 4 spare blocks.
  bool saw_abort = false;
  for (int i = 0; i < 100 && !saw_abort; ++i) {
    const StepPlan plan = scheduler.PlanStep();
    const auto emissions = scheduler.CompleteStep(plan, SampledFor(plan, kTok));
    if (!plan.aborted.empty()) {
      saw_abort = true;
      // The YOUNGEST running sequence (request 2) is the victim; request 1
      // keeps running and got the freed block it needed.
      ASSERT_EQ(plan.aborted.size(), 1u);
      EXPECT_EQ(plan.aborted[0]->id, 2u);
      ASSERT_EQ(plan.decode.size(), 1u);
      EXPECT_EQ(plan.decode[0]->id, 1u);
      // Abort emission first, then this step's decode token.
      ASSERT_EQ(emissions.size(), 2u);
      EXPECT_EQ(emissions[0].request_id, 2u);
      EXPECT_EQ(emissions[0].token, 0);
      EXPECT_TRUE(emissions[0].finished);
      EXPECT_EQ(emissions[0].finish_reason, FinishReason::kAborted);
      EXPECT_EQ(emissions[1].request_id, 1u);
      EXPECT_FALSE(emissions[1].finished);
      // Victim fully released.
      EXPECT_EQ(r2->state, RequestState::kFinished);
      EXPECT_EQ(r2->finish_reason, FinishReason::kAborted);
      EXPECT_TRUE(r2->block_table.empty());
      EXPECT_EQ(r2->reserved_blocks, 0);
      EXPECT_EQ(r2->slot, -1);
      // Pool conservation: free + survivor's blocks == usable.
      EXPECT_EQ(allocator.free_blocks() + r1->block_table.num_blocks(), 6);
    }
  }
  ASSERT_TRUE(saw_abort);

  // The survivor finishes normally and the pool is whole again.
  StepOnce(scheduler, kEosPrimary);
  EXPECT_EQ(scheduler.num_running(), 0u);
  EXPECT_EQ(allocator.free_blocks(), 6);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(SchedulerTest, LazyAllocationAtBlockBoundaries) {
  // seq lens {15, 16, 17}: a fresh block materializes exactly when the fed
  // position crosses a 16-token boundary, never earlier.
  struct Case {
    std::int32_t prompt_len;
    std::vector<std::int32_t> free_after_step; // after each of 4 iterations
  };
  const std::vector<Case> cases = {
      {15, {15, 15, 14, 16}}, // prefill 1 block; decode allocates at fed pos 16 (step 3)
      {16, {15, 14, 14, 16}}, // prefill 1 block; first decode feeds pos 16 -> new block
      {17, {14, 14, 14, 16}}, // prefill 2 blocks; positions 17..19 stay inside them
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.prompt_len);
    BlockAllocator allocator(17); // usable 16
    Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
    auto request = MakeRequest(1, c.prompt_len, 4);
    scheduler.Enqueue(request);
    for (std::size_t step = 0; step < c.free_after_step.size(); ++step) {
      StepOnce(scheduler);
      EXPECT_EQ(allocator.free_blocks(), c.free_after_step[step]) << "step " << step;
    }
    EXPECT_EQ(request->state, RequestState::kFinished); // max_new = 4 reached
    EXPECT_EQ(allocator.reserved_blocks(), 0);
  }
}

TEST(SchedulerTest, PrefillRunsAsDedicatedChunkedStepsBeforeDecode) {
  BlockAllocator allocator(402); // usable 401
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8,
                                              /*prefill_chunk=*/2048, /*max_seq_len=*/8192));
  auto short_req = MakeRequest(1, 32, 8);
  auto long_req = MakeRequest(2, 5000, 8);
  scheduler.Enqueue(short_req);
  scheduler.Enqueue(long_req);

  // Plan 1: both admit; the short prompt prefills first (FIFO) and joins the
  // decode set.
  StepOnce(scheduler);
  EXPECT_EQ(short_req->num_output_tokens(), 1);

  // Plans 2..4: the 5000-token prompt runs as dedicated chunks
  // [0,2048) [2048,4096) [4096,5000); the running sequence stalls meanwhile.
  const std::vector<std::pair<std::int32_t, std::int32_t>> expected_chunks = {
      {0, 2048}, {2048, 4096}, {4096, 5000}};
  const std::vector<std::int32_t> expected_blocks = {128, 256, 313};
  for (std::size_t i = 0; i < expected_chunks.size(); ++i) {
    const StepPlan plan = scheduler.PlanStep();
    ASSERT_TRUE(plan.is_prefill()) << "chunk " << i;
    EXPECT_EQ(plan.prefill->id, 2u);
    EXPECT_EQ(plan.prefill_begin, expected_chunks[i].first);
    EXPECT_EQ(plan.prefill_end, expected_chunks[i].second);
    EXPECT_TRUE(plan.decode.empty());
    EXPECT_EQ(long_req->block_table.num_blocks(), expected_blocks[i]);
    EXPECT_EQ(plan.final_prefill_chunk(), i + 1 == expected_chunks.size());
    scheduler.CompleteStep(plan, SampledFor(plan, kTok));
    // The decode-side sequence did not advance during dedicated prefill.
    EXPECT_EQ(short_req->num_output_tokens(), 1);
  }
  EXPECT_EQ(long_req->state, RequestState::kRunning);

  // Next step is a decode batch over both, in admission order.
  const StepPlan decode_plan = scheduler.PlanStep();
  ASSERT_EQ(decode_plan.decode.size(), 2u);
  EXPECT_EQ(decode_plan.decode[0]->id, 1u);
  EXPECT_EQ(decode_plan.decode[1]->id, 2u);
  scheduler.CompleteStep(decode_plan, SampledFor(decode_plan, kTok));
  DrainAll(scheduler);
  EXPECT_EQ(allocator.free_blocks(), 401);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(SchedulerTest, DecodeBatchNeverExceedsMaxBatch) {
  BlockAllocator allocator(33); // usable 32: room for all four requests
  Scheduler scheduler(&allocator,
                      MakeOptions(AdmissionPolicy::kReserveFull, /*max_batch=*/3, 1024, 64));
  for (RequestId id = 1; id <= 4; ++id) {
    scheduler.Enqueue(MakeRequest(id, 16, 32)); // 3 blocks each
  }

  // The batch cap (not the pool) limits admission to 3; the 4th waits.
  StepOnce(scheduler); // admits 1..3, prefills request 1
  EXPECT_EQ(scheduler.num_waiting(), 1u);
  EXPECT_EQ(scheduler.num_running(), 3u);
  StepOnce(scheduler); // prefill request 2
  StepOnce(scheduler); // prefill request 3

  const StepPlan plan4 = scheduler.PlanStep();
  EXPECT_EQ(plan4.decode.size(), 3u); // exactly max_batch rows
  // Finish request 1 so a slot frees up for request 4.
  scheduler.CompleteStep(
      plan4, SampledFor(plan4, [](const Request& r) { return r.id == 1 ? kEosPrimary : kTok; }));

  const StepPlan plan5 = scheduler.PlanStep(); // admits + prefills request 4
  ASSERT_TRUE(plan5.is_prefill());
  EXPECT_EQ(plan5.prefill->id, 4u);
  EXPECT_EQ(scheduler.num_waiting(), 0u);
  scheduler.CompleteStep(plan5, SampledFor(plan5, kTok));

  const StepPlan plan6 = scheduler.PlanStep();
  ASSERT_EQ(plan6.decode.size(), 3u);
  EXPECT_EQ(plan6.decode[0]->id, 2u);
  EXPECT_EQ(plan6.decode[1]->id, 3u);
  EXPECT_EQ(plan6.decode[2]->id, 4u);
  scheduler.CompleteStep(plan6, SampledFor(plan6, kTok));
}

TEST(SchedulerTest, FinishedRequestsReleaseBlocksAndUnreserve) {
  BlockAllocator allocator(33); // usable 32
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 128));
  auto r1 = MakeRequest(1, 20, 4);  // 24 tokens -> reserve 2
  auto r2 = MakeRequest(2, 40, 20); // 60 tokens -> reserve 4
  scheduler.Enqueue(r1);
  scheduler.Enqueue(r2);

  StepOnce(scheduler); // prefill r1 (2 blocks)
  StepOnce(scheduler); // prefill r2 (3 blocks; 1 still reserved)
  EXPECT_EQ(allocator.free_blocks(), 27);
  EXPECT_EQ(allocator.reserved_blocks(), 1);

  // r1 reaches its length stop after 3 decode steps; every owned block AND
  // the unmaterialized remainder must return while r2 keeps its state.
  StepOnce(scheduler);
  StepOnce(scheduler);
  const auto emissions = StepOnce(scheduler);
  ASSERT_EQ(emissions.size(), 2u);
  EXPECT_TRUE(emissions[0].finished);
  EXPECT_EQ(emissions[0].finish_reason, FinishReason::kLength);
  EXPECT_EQ(r1->state, RequestState::kFinished);
  EXPECT_TRUE(r1->block_table.empty());
  EXPECT_EQ(r1->reserved_blocks, 0);
  EXPECT_EQ(r1->slot, -1);
  EXPECT_EQ(allocator.free_blocks(), 29);
  EXPECT_EQ(allocator.reserved_blocks(), 1);
  EXPECT_EQ(r2->block_table.num_blocks(), 3);

  // Full drain conserves the free list exactly.
  StepOnce(scheduler, kEosPrimary);
  EXPECT_EQ(scheduler.num_running(), 0u);
  EXPECT_EQ(allocator.free_blocks(), 32);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(SchedulerTest, EosOnFirstTokenFinishesFromPrefillState) {
  BlockAllocator allocator(17); // usable 16
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  auto request = MakeRequest(1, 20, 20); // 40 tokens -> reserve 3; prefill materializes 2
  scheduler.Enqueue(request);

  const StepPlan plan = scheduler.PlanStep();
  ASSERT_TRUE(plan.final_prefill_chunk());
  EXPECT_EQ(request->state, RequestState::kPrefill);
  // A nonzero reservation remainder is held across the prefill-exit finish,
  // so the release path below must both free blocks AND unreserve.
  EXPECT_EQ(request->reserved_blocks, 1);
  EXPECT_EQ(allocator.reserved_blocks(), 1);
  EXPECT_EQ(allocator.free_blocks(), 14);
  const auto emissions = scheduler.CompleteStep(plan, {kEosPrimary});
  ASSERT_EQ(emissions.size(), 1u);
  EXPECT_EQ(emissions[0].token, kEosPrimary);
  EXPECT_TRUE(emissions[0].finished);
  EXPECT_EQ(emissions[0].finish_reason, FinishReason::kEos);
  // Finished straight out of kPrefill: prompt blocks freed, remainder
  // unreserved, slot returned, never entered the decode set.
  EXPECT_EQ(request->state, RequestState::kFinished);
  EXPECT_TRUE(request->block_table.empty());
  EXPECT_EQ(request->reserved_blocks, 0);
  EXPECT_EQ(request->slot, -1);
  EXPECT_EQ(scheduler.num_running(), 0u);
  EXPECT_EQ(allocator.free_blocks(), 16);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  EXPECT_TRUE(scheduler.PlanStep().empty());
}

TEST(SchedulerTest, MaxNewOneFinishesFromPrefillStateWithLength) {
  BlockAllocator allocator(17);
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  auto request = MakeRequest(1, 16, 1); // 17 tokens -> reserve 2; prefill materializes 1
  scheduler.Enqueue(request);

  const StepPlan plan = scheduler.PlanStep();
  ASSERT_TRUE(plan.final_prefill_chunk());
  EXPECT_EQ(request->reserved_blocks, 1); // remainder to unreserve at the finish
  EXPECT_EQ(allocator.reserved_blocks(), 1);
  const auto emissions = scheduler.CompleteStep(plan, {kTok}); // non-EOS token
  ASSERT_EQ(emissions.size(), 1u);
  EXPECT_EQ(emissions[0].token, kTok);
  EXPECT_TRUE(emissions[0].finished);
  EXPECT_EQ(emissions[0].finish_reason, FinishReason::kLength);
  EXPECT_EQ(request->state, RequestState::kFinished);
  EXPECT_TRUE(request->block_table.empty());
  EXPECT_EQ(request->reserved_blocks, 0);
  EXPECT_EQ(allocator.free_blocks(), 16);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
}

TEST(SchedulerTest, IgnoreEosSuppressesEosStopButNotLengthStop) {
  BlockAllocator allocator(17);
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  auto request = MakeRequest(1, 16, 4, /*ignore_eos=*/true);
  scheduler.Enqueue(request);

  // The mock emits an EOS-set token at EVERY step; the sequence must run to
  // its length stop regardless, and the EOS tokens still land in the output.
  std::vector<StepResult> last;
  for (int i = 0; i < 4; ++i) {
    last = StepOnce(scheduler, kEosPrimary);
  }
  ASSERT_EQ(last.size(), 1u);
  EXPECT_TRUE(last[0].finished);
  EXPECT_EQ(last[0].finish_reason, FinishReason::kLength);
  EXPECT_EQ(request->output_tokens,
            (std::vector<TokenId>{kEosPrimary, kEosPrimary, kEosPrimary, kEosPrimary}));

  // Control: without ignore_eos the same script stops immediately with kEos
  // (both members of the EOS set stop).
  auto control = MakeRequest(2, 16, 4);
  scheduler.Enqueue(control);
  const auto emissions = StepOnce(scheduler, kEosSecondary);
  ASSERT_EQ(emissions.size(), 1u);
  EXPECT_EQ(emissions[0].finish_reason, FinishReason::kEos);
  EXPECT_EQ(control->output_tokens, (std::vector<TokenId>{kEosSecondary}));
}

TEST(SchedulerTest, NoPreemptionRunningSequencesKeepTheirBlocks) {
  BlockAllocator allocator(9); // usable 8 - exactly request 1's worst case
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 128));
  auto r1 = MakeRequest(1, 100, 28); // 128 tokens -> reserve 8
  auto r2 = MakeRequest(2, 16, 16);  // 2 blocks; legal, but must wait
  scheduler.Enqueue(r1);
  scheduler.Enqueue(r2);

  StepOnce(scheduler); // admit + prefill r1; r2 stalls (admission, not eviction)
  EXPECT_EQ(scheduler.num_waiting(), 1u);

  // Under reserve_full a running sequence is never evicted: r1 decodes to its
  // full length while r2 waits, and r1's block count never shrinks mid-run.
  std::int32_t max_blocks_seen = r1->block_table.num_blocks();
  for (int i = 0; i < 100 && r1->state != RequestState::kFinished; ++i) {
    StepOnce(scheduler);
    if (r1->state != RequestState::kFinished) {
      EXPECT_GE(r1->block_table.num_blocks(), max_blocks_seen);
      max_blocks_seen = r1->block_table.num_blocks();
      EXPECT_EQ(scheduler.num_waiting(), 1u);
    }
  }
  EXPECT_EQ(r1->state, RequestState::kFinished);
  EXPECT_EQ(r1->finish_reason, FinishReason::kLength);
  EXPECT_EQ(r1->num_output_tokens(), 28);

  // Only after the natural finish does the waiting request admit.
  const StepPlan plan = scheduler.PlanStep();
  ASSERT_TRUE(plan.is_prefill());
  EXPECT_EQ(plan.prefill->id, 2u);
  EXPECT_EQ(scheduler.num_waiting(), 0u);
}

TEST(SchedulerTest, ForcedTokensAppendForcedButEmitArgmax) {
  BlockAllocator allocator(17);
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  auto request = MakeRequest(1, 16, 3, /*ignore_eos=*/false,
                             /*forced_tokens=*/{100, 101, 102});
  scheduler.Enqueue(request);

  // The mock "argmax" is kTok (7) at every position; the emissions must
  // report it while the sequence itself follows the forced continuation
  // (emitted argmax != appended token).
  for (int step = 0; step < 3; ++step) {
    const auto emissions = StepOnce(scheduler, kTok);
    ASSERT_EQ(emissions.size(), 1u) << "step " << step;
    EXPECT_EQ(emissions[0].token, kTok);
    if (step < 2) {
      EXPECT_FALSE(emissions[0].finished);
    } else {
      EXPECT_TRUE(emissions[0].finished);
      EXPECT_EQ(emissions[0].finish_reason, FinishReason::kLength);
    }
  }
  EXPECT_EQ(request->output_tokens, (std::vector<TokenId>{100, 101, 102}));

  // A forced EOS finishes the sequence (stop checks run on the APPENDED
  // token) even though the emitted argmax is not an EOS token.
  auto forced_eos = MakeRequest(2, 16, 5, false, {kEosPrimary});
  scheduler.Enqueue(forced_eos);
  const auto eos_emissions = StepOnce(scheduler, kTok);
  ASSERT_EQ(eos_emissions.size(), 1u);
  EXPECT_EQ(eos_emissions[0].token, kTok);
  EXPECT_TRUE(eos_emissions[0].finished);
  EXPECT_EQ(eos_emissions[0].finish_reason, FinishReason::kEos);
  EXPECT_EQ(forced_eos->output_tokens, (std::vector<TokenId>{kEosPrimary}));

  // Past the end of forced_tokens the scheduler falls back to the engine's
  // own token.
  auto partial = MakeRequest(3, 16, 2, false, {100});
  scheduler.Enqueue(partial);
  StepOnce(scheduler, kTok); // prefill: appends forced 100
  StepOnce(scheduler, kTok); // decode: forced exhausted -> appends argmax
  EXPECT_EQ(partial->output_tokens, (std::vector<TokenId>{100, kTok}));
  EXPECT_EQ(partial->finish_reason, FinishReason::kLength);
}

TEST(SchedulerTest, EmissionOrderingFollowsPlanOrder) {
  BlockAllocator allocator(33);
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  for (RequestId id = 1; id <= 3; ++id) {
    scheduler.Enqueue(MakeRequest(id, 16, 8));
  }
  StepOnce(scheduler); // admit all, prefill 1
  StepOnce(scheduler); // prefill 2
  StepOnce(scheduler); // prefill 3

  // Decode rows and their emissions stay in admission (plan) order.
  const StepPlan plan = scheduler.PlanStep();
  ASSERT_EQ(plan.decode.size(), 3u);
  const std::vector<TokenId> sampled = {11, 22, 33};
  const auto emissions = scheduler.CompleteStep(plan, sampled);
  ASSERT_EQ(emissions.size(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(emissions[i].request_id, plan.decode[i]->id);
    EXPECT_EQ(emissions[i].request_id, i + 1);
    EXPECT_EQ(emissions[i].token, sampled[i]);
  }

  // A finish in the middle keeps the relative order of the survivors.
  const StepPlan plan2 = scheduler.PlanStep();
  scheduler.CompleteStep(
      plan2, SampledFor(plan2, [](const Request& r) { return r.id == 2 ? kEosPrimary : kTok; }));
  const StepPlan plan3 = scheduler.PlanStep();
  ASSERT_EQ(plan3.decode.size(), 2u);
  EXPECT_EQ(plan3.decode[0]->id, 1u);
  EXPECT_EQ(plan3.decode[1]->id, 3u);
  scheduler.CompleteStep(plan3, SampledFor(plan3, kTok));
}

TEST(SchedulerTest, BucketSelectionSequence) {
  // Ladder construction (docs/DESIGN.md section 9).
  EXPECT_EQ(MakeDecodeBuckets(8), (std::vector<std::int32_t>{1, 2, 4, 8}));
  EXPECT_EQ(MakeDecodeBuckets(64), (std::vector<std::int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
  EXPECT_EQ(MakeDecodeBuckets(6), (std::vector<std::int32_t>{1, 2, 4}));

  // Smallest bucket >= batch; -1 (eager) when none fits.
  const std::vector<std::int32_t> buckets = MakeDecodeBuckets(8);
  std::vector<std::int32_t> selected;
  for (const std::int32_t batch : {8, 7, 6, 5, 4, 3, 2, 1}) {
    selected.push_back(SelectBucket(batch, buckets));
  }
  EXPECT_EQ(selected, (std::vector<std::int32_t>{8, 8, 8, 8, 4, 4, 2, 1}));
  EXPECT_EQ(SelectBucket(9, buckets), -1);
  EXPECT_EQ(SelectBucket(0, buckets), -1);
  EXPECT_EQ(SelectBucket(5, MakeDecodeBuckets(6)), -1); // above every dev bucket

  // Driven through the scheduler: the batch shrinks 3 -> 2 -> 1 as sequences
  // finish, and the selected bucket follows {4, 2, 1}.
  BlockAllocator allocator(33);
  Scheduler scheduler(&allocator, MakeOptions(AdmissionPolicy::kReserveFull, 8, 1024, 64));
  for (RequestId id = 1; id <= 3; ++id) {
    scheduler.Enqueue(MakeRequest(id, 16, 8));
  }
  StepOnce(scheduler);
  StepOnce(scheduler);
  StepOnce(scheduler); // all three running
  std::vector<std::int32_t> bucket_sequence;
  for (const RequestId finish_id : {1, 2, 3}) {
    const StepPlan plan = scheduler.PlanStep();
    bucket_sequence.push_back(SelectBucket(static_cast<std::int32_t>(plan.decode.size()), buckets));
    scheduler.CompleteStep(plan, SampledFor(plan, [finish_id](const Request& r) {
                             return r.id == finish_id ? kEosPrimary : kTok;
                           }));
  }
  EXPECT_EQ(bucket_sequence, (std::vector<std::int32_t>{4, 2, 1}));
  EXPECT_EQ(scheduler.num_running(), 0u);
}

// Property test: random admit/decode/finish/abort churn with request sizes
// at prompt_len + max_new ≡ 0, 1 (mod 16) boundaries. After every step:
// free + outstanding == usable, and the allocator's reserved count equals
// the sum of live requests' unmaterialized reservations.
void ExpectPoolConsistent(const BlockAllocator& allocator, const std::vector<RequestPtr>& all,
                          std::int32_t usable) {
  std::int64_t outstanding = 0;
  std::int64_t reserved = 0;
  for (const RequestPtr& request : all) {
    outstanding += request->block_table.num_blocks();
    reserved += request->reserved_blocks;
    if (request->state == RequestState::kFinished) {
      EXPECT_TRUE(request->block_table.empty());
      EXPECT_EQ(request->reserved_blocks, 0);
      EXPECT_EQ(request->slot, -1);
    }
  }
  EXPECT_EQ(allocator.free_blocks() + outstanding, usable);
  EXPECT_EQ(allocator.reserved_blocks(), reserved);
  EXPECT_GE(static_cast<std::int64_t>(allocator.free_blocks()), allocator.reserved_blocks());
}

void RunChurn(AdmissionPolicy policy, std::uint32_t seed) {
  constexpr std::int32_t kUsable = 40;
  BlockAllocator allocator(kUsable + 1);
  Scheduler scheduler(&allocator, MakeOptions(policy, 8, /*prefill_chunk=*/64,
                                              /*max_seq_len=*/256));
  std::mt19937 rng(seed);
  std::vector<RequestPtr> all;
  RequestId next_id = 1;
  std::uint64_t aborts_seen = 0;

  // Scripted "argmax": occasional EOS so finishes are scattered (rare enough
  // that many sequences grow deep into decode).
  const auto sample = [&rng](const Request&) { return rng() % 50 == 0 ? kEosPrimary : kTok; };
  const auto pump = [&]() {
    const StepPlan plan = scheduler.PlanStep();
    aborts_seen += plan.aborted.size();
    const auto emissions = scheduler.CompleteStep(plan, SampledFor(plan, sample));
    for (const StepResult& emission : emissions) {
      EXPECT_EQ(emission.finished, emission.finish_reason != FinishReason::kNone);
    }
    ExpectPoolConsistent(allocator, all, kUsable);
  };

  for (int step = 0; step < 600; ++step) {
    if (scheduler.num_waiting() < 4 && rng() % 3 == 0) {
      // total tokens in {64, 65, 80, 81, ..., 128, 129}: 0 or 1 (mod 16).
      const std::int32_t total =
          16 * (4 + static_cast<std::int32_t>(rng() % 5)) + static_cast<std::int32_t>(rng() % 2);
      // Bias toward short prompts with large max_new - the shape whose decode
      // growth the watermark policy deliberately leaves uncovered.
      const std::int32_t prompt_len = rng() % 2 == 0
                                          ? 1 + static_cast<std::int32_t>(rng() % 16)
                                          : 1 + static_cast<std::int32_t>(rng() % (total - 1));
      const std::int32_t max_new = total - prompt_len;
      auto request = MakeRequest(next_id++, prompt_len, max_new,
                                 /*ignore_eos=*/rng() % 2 == 0);
      ASSERT_NO_THROW(scheduler.Enqueue(request)); // sizes always legal for this pool
      all.push_back(std::move(request));
    }
    pump();
  }

  // Drain: no new arrivals; everything queued or running must complete.
  for (int step = 0; step < 20000 && scheduler.num_waiting() + scheduler.num_running() > 0;
       ++step) {
    pump();
  }
  EXPECT_EQ(scheduler.num_waiting() + scheduler.num_running(), 0u);
  EXPECT_EQ(allocator.free_blocks(), kUsable);
  EXPECT_EQ(allocator.reserved_blocks(), 0);
  if (policy == AdmissionPolicy::kWatermark) {
    // 8 sequences growing toward up to 9 blocks each against 40 usable
    // blocks: decode growth must have hit the abort path.
    EXPECT_GT(aborts_seen, 0u);
  } else {
    // reserve_full makes decode-time exhaustion impossible by construction.
    EXPECT_EQ(aborts_seen, 0u);
  }
}

TEST(SchedulerTest, RandomizedChurnKeepsReservationArithmeticConsistent) {
  RunChurn(AdmissionPolicy::kReserveFull, 0xC0FFEEu);
  RunChurn(AdmissionPolicy::kWatermark, 0xBADF00Du);
}

} // namespace
} // namespace redline
