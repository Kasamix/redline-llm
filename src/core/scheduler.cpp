#include "core/scheduler.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

namespace redline {

std::vector<std::int32_t> MakeDecodeBuckets(std::int32_t max_batch) {
  // Canonical ladder of docs/DESIGN.md section 9, capped at max_batch.
  static constexpr std::int32_t kLadder[] = {1, 2, 4, 8, 16, 32, 48, 64};
  std::vector<std::int32_t> buckets;
  for (const std::int32_t bucket : kLadder) {
    if (bucket <= max_batch) {
      buckets.push_back(bucket);
    }
  }
  return buckets;
}

std::int32_t SelectBucket(std::int32_t batch_size, const std::vector<std::int32_t>& buckets) {
  if (batch_size <= 0) {
    return -1;
  }
  // Ascending list: the first bucket that fits is the smallest one.
  for (const std::int32_t bucket : buckets) {
    if (bucket >= batch_size) {
      return bucket;
    }
  }
  return -1; // larger than every configured bucket: eager path
}

Scheduler::Scheduler(BlockAllocator* allocator, SchedulerOptions options)
    : allocator_(allocator), options_(std::move(options)) {
  assert(allocator_ != nullptr);
  assert(options_.max_batch >= 1);
  assert(options_.prefill_chunk_tokens >= 1);
  assert(options_.max_seq_len >= 2); // one prompt token + one generated token
  assert(options_.watermark_free_fraction >= 0.0 && options_.watermark_free_fraction < 1.0);
  free_slots_.reserve(static_cast<std::size_t>(options_.max_batch));
  // Descending push so the back of the stack is slot 0: lowest slots are
  // assigned first (cosmetic determinism; any free slot would be correct).
  for (std::int32_t slot = options_.max_batch - 1; slot >= 0; --slot) {
    free_slots_.push_back(slot);
  }
}

void Scheduler::ValidateRequest(const Request& request) const {
  const std::int64_t prompt_len = static_cast<std::int64_t>(request.prompt_tokens.size());
  if (prompt_len == 0) {
    throw std::invalid_argument("redline: request " + std::to_string(request.id) +
                                ": prompt is empty");
  }
  if (request.max_new_tokens <= 0) {
    throw std::invalid_argument("redline: request " + std::to_string(request.id) +
                                ": max_new_tokens must be positive, got " +
                                std::to_string(request.max_new_tokens));
  }
  if (prompt_len + request.max_new_tokens > options_.max_seq_len) {
    throw std::invalid_argument("redline: request " + std::to_string(request.id) +
                                ": prompt_len (" + std::to_string(prompt_len) +
                                ") + max_new_tokens (" + std::to_string(request.max_new_tokens) +
                                ") exceeds max_seq_len (" + std::to_string(options_.max_seq_len) +
                                ")");
  }
  // Reject a request that could never be admitted even by an idle pool, so
  // it cannot sit in the waiting queue forever (docs/DESIGN.md section 8).
  if (options_.admission_policy == AdmissionPolicy::kReserveFull) {
    const std::int32_t worst_case =
        BlocksForTokens(request.num_prompt_tokens() + request.max_new_tokens);
    if (worst_case > allocator_->usable_blocks()) {
      throw std::invalid_argument("redline: request " + std::to_string(request.id) + ": needs " +
                                  std::to_string(worst_case) +
                                  " KV blocks worst-case but the pool has " +
                                  std::to_string(allocator_->usable_blocks()) + " usable blocks");
    }
  } else {
    const std::int32_t prompt_blocks = BlocksForTokens(request.num_prompt_tokens());
    const std::int32_t usable = allocator_->usable_blocks();
    // Same floor base as CanAdmit: usable, not total (see scheduler.hpp).
    const double floor = options_.watermark_free_fraction * usable;
    if (static_cast<double>(usable - prompt_blocks) < floor) {
      throw std::invalid_argument(
          "redline: request " + std::to_string(request.id) + ": prompt needs " +
          std::to_string(prompt_blocks) +
          " KV blocks; even an idle pool cannot admit it above the free floor (" +
          std::to_string(usable) + " usable, floor " + std::to_string(floor) + ")");
    }
  }
}

void Scheduler::Enqueue(RequestPtr request) {
  assert(request != nullptr);
  assert(request->state == RequestState::kWaiting);
  ValidateRequest(*request); // throws std::invalid_argument; nothing queued on failure
  waiting_.push_back(std::move(request));
}

bool Scheduler::CanAdmit(const Request& request) const {
  if (options_.admission_policy == AdmissionPolicy::kReserveFull) {
    return allocator_->CanReserve(
        BlocksForTokens(request.num_prompt_tokens() + request.max_new_tokens));
  }
  // kWatermark free floor. Outstanding reservations are subtracted because
  // materialization is lazy: the free list still contains blocks already
  // promised to admitted prompts, and Reserve() below requires
  // CanReserve(prompt_blocks) to hold (free - reserved >= n). With a
  // non-negative floor this check subsumes that precondition. Floor base is
  // usable, not section 8's total num_blocks (deviations documented on the
  // scheduler.hpp declaration).
  const std::int64_t uncommitted =
      static_cast<std::int64_t>(allocator_->free_blocks()) - allocator_->reserved_blocks();
  const std::int64_t prompt_blocks = BlocksForTokens(request.num_prompt_tokens());
  const double floor = options_.watermark_free_fraction * allocator_->usable_blocks();
  return static_cast<double>(uncommitted - prompt_blocks) >= floor;
}

std::int64_t Scheduler::ReservationFor(const Request& request) const {
  if (options_.admission_policy == AdmissionPolicy::kReserveFull) {
    // Worst case: every KV position the request could ever occupy. This is
    // what makes decode-time exhaustion impossible by construction.
    return BlocksForTokens(request.num_prompt_tokens() + request.max_new_tokens);
  }
  // kWatermark promises the prompt only; decode growth is taken from the
  // shared pool one block at a time (abort-youngest on exhaustion).
  return BlocksForTokens(request.num_prompt_tokens());
}

void Scheduler::Admit() {
  RequestPtr request = std::move(waiting_.front());
  waiting_.pop_front();
  const std::int64_t reservation = ReservationFor(*request);
  allocator_->Reserve(reservation); // CanAdmit established CanReserve
  request->reserved_blocks = reservation;
  assert(!free_slots_.empty()); // guaranteed by the max_batch admission cap
  request->slot = free_slots_.back();
  free_slots_.pop_back();
  request->state = RequestState::kPrefill;
  prefill_queue_.push_back(std::move(request));
}

StepPlan Scheduler::PlanStep() {
  StepPlan plan;

  // 1) FCFS admission: strictly in arrival order, stopping at the first head
  //    that cannot be admitted (no reordering, docs/DESIGN.md section 8).
  while (!waiting_.empty() && num_running() < static_cast<std::size_t>(options_.max_batch) &&
         CanAdmit(*waiting_.front())) {
    Admit();
  }

  // 2) Dedicated prefill step: one chunk of the oldest admitted prefill.
  //    Running decodes stall for this step (v1 tradeoff, bounded by the
  //    prefill_chunk_tokens knob).
  if (!prefill_queue_.empty()) {
    const RequestPtr& request = prefill_queue_.front();
    assert(request->state == RequestState::kPrefill);
    plan.prefill = request;
    plan.prefill_begin = request->prefill_progress;
    plan.prefill_end = std::min(request->prefill_progress + options_.prefill_chunk_tokens,
                                request->num_prompt_tokens());
    assert(plan.prefill_begin < plan.prefill_end);
    // Materialize the blocks covering every position this chunk scatters -
    // lazily, out of the admission reservation (docs/DESIGN.md section 7;
    // the prompt is covered under both policies).
    while (request->block_table.token_capacity() < plan.prefill_end) {
      assert(request->reserved_blocks > 0);
      request->block_table.Append(allocator_->AllocReserved());
      --request->reserved_blocks;
    }
    return plan;
  }

  // 3) Decode batch: running sequences, oldest first, capped at max_batch
  //    rows (admission also caps the running set at max_batch, so the cap
  //    here is belt-and-suspenders).
  for (std::size_t i = 0;
       i < running_.size() && plan.decode.size() < static_cast<std::size_t>(options_.max_batch);) {
    const RequestPtr request = running_[i];
    // The position fed this step is seq_len()-1 (the last appended token);
    // crossing a 16-token block boundary means its KV needs a fresh block.
    if (request->seq_len() - 1 >= request->block_table.token_capacity()) {
      assert(request->seq_len() - 1 == request->block_table.token_capacity());
      if (!EnsureDecodeBlockCapacity(request, plan)) {
        // `request` itself was the abort victim; it was running_.back(), so
        // the loop bound already shrank past index i. Re-test the condition.
        continue;
      }
    }
    plan.decode.push_back(request);
    ++i;
  }
  return plan;
}

bool Scheduler::EnsureDecodeBlockCapacity(const RequestPtr& request, StepPlan& plan) {
  if (options_.admission_policy == AdmissionPolicy::kReserveFull) {
    // Covered by the admission reservation; cannot fail (section 7
    // invariant: free >= reserved guarantees the block exists).
    assert(request->reserved_blocks > 0);
    request->block_table.Append(allocator_->AllocReserved());
    --request->reserved_blocks;
    return true;
  }
  // kWatermark: decode growth is not covered by any reservation. Claim one
  // block through the reservation API (Reserve + AllocReserved nets to a
  // plain pop); on exhaustion abort the youngest running sequence and retry
  // - bounded recovery instead of preemption (docs/DESIGN.md section 8).
  // Each abort returns at least one block (a running sequence owns at least
  // its prompt blocks), so this loop terminates.
  while (!allocator_->CanReserve(1)) {
    assert(!running_.empty()); // `request` itself is still in running_
    const bool victim_is_self = running_.back() == request;
    AbortYoungestRunning(plan);
    if (victim_is_self) {
      return false;
    }
  }
  allocator_->Reserve(1);
  request->block_table.Append(allocator_->AllocReserved());
  return true;
}

void Scheduler::AbortYoungestRunning(StepPlan& plan) {
  RequestPtr victim = std::move(running_.back());
  running_.pop_back();
  victim->state = RequestState::kFinished;
  victim->finish_reason = FinishReason::kAborted;
  ReleaseResources(*victim);
  plan.aborted.push_back(std::move(victim));
}

void Scheduler::ReleaseResources(Request& request) {
  for (const BlockId block : request.block_table.Release()) {
    allocator_->FreeBlock(block);
  }
  allocator_->Unreserve(request.reserved_blocks);
  request.reserved_blocks = 0;
  if (request.slot >= 0) {
    free_slots_.push_back(request.slot);
    request.slot = -1;
  }
}

std::vector<StepResult> Scheduler::CompleteStep(const StepPlan& plan,
                                                const std::vector<TokenId>& sampled_tokens) {
  std::vector<StepResult> emissions;
  emissions.reserve(plan.aborted.size() + plan.decode.size() + 1);

  // Watermark aborts were finalized during PlanStep; report them first so a
  // consumer sees the eviction ahead of this step's tokens.
  for (const RequestPtr& request : plan.aborted) {
    assert(request->state == RequestState::kFinished &&
           request->finish_reason == FinishReason::kAborted);
    emissions.push_back(StepResult{request->id, TokenId{0}, true, FinishReason::kAborted});
  }

  if (plan.is_prefill()) {
    const RequestPtr& request = plan.prefill;
    assert(!prefill_queue_.empty() && prefill_queue_.front() == request);
    assert(request->prefill_progress == plan.prefill_begin);
    request->prefill_progress = plan.prefill_end;
    if (!request->prefill_done()) {
      // Intermediate chunk: no logits were computed, nothing to emit.
      assert(sampled_tokens.empty());
      return emissions;
    }
    // Final chunk: the executor sampled the first generated token from the
    // last prompt position's logits.
    assert(sampled_tokens.size() == 1);
    prefill_queue_.pop_front();
    emissions.push_back(ApplySampledToken(request, sampled_tokens.front()));
    return emissions;
  }

  assert(sampled_tokens.size() == plan.decode.size());
  for (std::size_t i = 0; i < plan.decode.size(); ++i) {
    emissions.push_back(ApplySampledToken(plan.decode[i], sampled_tokens[i]));
  }
  return emissions;
}

StepResult Scheduler::ApplySampledToken(const RequestPtr& request, TokenId sampled_token) {
  assert(request->state == RequestState::kPrefill || request->state == RequestState::kRunning);
  const bool from_prefill = request->state == RequestState::kPrefill;

  // Teacher forcing (docs/DESIGN.md section 12(c)): append the scripted
  // token so the next step is conditioned on the reference continuation,
  // while the emission still reports the engine's own choice.
  const std::size_t position = request->output_tokens.size();
  const TokenId appended =
      position < request->forced_tokens.size() ? request->forced_tokens[position] : sampled_token;
  request->output_tokens.push_back(appended);

  // Stop checks on the APPENDED token. kEos wins a simultaneous length stop;
  // ignore_eos suppresses only the EOS check, never the length check.
  FinishReason reason = FinishReason::kNone;
  if (!request->ignore_eos && IsEosToken(appended)) {
    reason = FinishReason::kEos;
  } else if (request->num_output_tokens() >= request->max_new_tokens) {
    reason = FinishReason::kLength;
  }

  StepResult result{request->id, sampled_token, false, FinishReason::kNone};
  if (reason == FinishReason::kNone) {
    if (from_prefill) {
      // First generated token: join the decode batch. Prefill completion is
      // FIFO over admissions, so running_ stays admission-ordered.
      request->state = RequestState::kRunning;
      running_.push_back(request);
    }
    return result;
  }

  // Finish - possibly straight out of kPrefill (EOS on the first generated
  // token, or max_new_tokens == 1): the release path is identical (prompt
  // blocks freed, reservation remainder unreserved, slot returned).
  request->state = RequestState::kFinished;
  request->finish_reason = reason;
  ReleaseResources(*request);
  if (!from_prefill) {
    const auto it = std::find(running_.begin(), running_.end(), request);
    assert(it != running_.end());
    running_.erase(it); // order-preserving erase keeps back() == youngest
  }
  result.finished = true;
  result.finish_reason = reason;
  return result;
}

bool Scheduler::IsEosToken(TokenId token) const {
  return std::find(options_.eos_token_ids.begin(), options_.eos_token_ids.end(), token) !=
         options_.eos_token_ids.end();
}

} // namespace redline
