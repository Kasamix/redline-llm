#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "core/block_allocator.hpp"
#include "core/request.hpp"

namespace redline {

// Admission control policies (docs/DESIGN.md section 8).
enum class AdmissionPolicy : std::uint8_t {
  // Default. Admit only if CanReserve(BlocksForTokens(prompt_len + max_new));
  // worst-case KV is reserved up front, so decode can NEVER exhaust the pool
  // mid-flight - the reservation invariant, not admission-time arithmetic
  // alone, is what makes "no preemption in v1" sound. Cost: conservative
  // admission when callers over-state max_new.
  kReserveFull,
  // Higher occupancy: admit while a free-block floor holds, reserving only
  // the prompt. Decode growth is NOT covered, so a step may find the pool
  // empty; recovery is to abort the YOUNGEST running sequence (free its
  // blocks, emit FinishReason::kAborted). Init warns when
  // max_batch * BlocksForTokens(max_seq_len) exceeds usable blocks - the
  // configs in which the abort path is load-bearing.
  kWatermark,
};

struct SchedulerOptions {
  // Decode batch cap; also the largest CUDA-graph bucket the engine captures.
  // Dev RTX 2060 preset uses 8, bench RTX 4090 preset uses 64.
  std::int32_t max_batch = 8;

  // Prompt tokens processed per dedicated prefill step (dev preset 1024).
  std::int32_t prefill_chunk_tokens = 1024;

  // Hard cap on prompt_len + max_new_tokens per request; mirrors
  // EngineOptions::max_seq_len (the device block-table width bound, dev
  // preset 2048). Requests exceeding it are rejected at Enqueue with
  // std::invalid_argument.
  std::int32_t max_seq_len = 2048;

  AdmissionPolicy admission_policy = AdmissionPolicy::kReserveFull;

  // kWatermark only: admission keeps this fraction of usable blocks
  // uncommitted (see Scheduler::CanAdmit).
  double watermark_free_fraction = 0.10;

  // EOS stop set from generation_config.json ({151645, 151643} for
  // Qwen2.5-1.5B-Instruct, docs/MODEL_SPEC.md). An empty set disables the
  // EOS stop; the length stop always applies.
  std::vector<TokenId> eos_token_ids;
};

// What the engine should execute this iteration: exactly one chunked-prefill
// slice (prefill steps are dedicated) or a decode batch of running sequences.
struct StepPlan {
  RequestPtr prefill;             // nullptr for decode steps
  std::int32_t prefill_begin = 0; // prompt token range [begin, end) this step
  std::int32_t prefill_end = 0;
  std::vector<RequestPtr> decode; // sequences generating one token each

  // kWatermark only: sequences aborted while planning this step (in abort
  // order, youngest at the time of each abort) because the pool could not
  // cover decode growth. Their blocks, reservation remainder, and batch slot
  // are already released; CompleteStep turns each into one StepResult with
  // FinishReason::kAborted.
  std::vector<RequestPtr> aborted;

  bool empty() const { return prefill == nullptr && decode.empty() && aborted.empty(); }
  bool is_prefill() const { return prefill != nullptr; }

  // True when this prefill slice reaches the end of the prompt: the executor
  // computes logits for the last prompt token only and samples the first
  // generated token (docs/DESIGN.md section 6.3), which CompleteStep then
  // expects in sampled_tokens.
  bool final_prefill_chunk() const {
    return prefill != nullptr && prefill_end == prefill->num_prompt_tokens();
  }
};

// CUDA-graph decode bucket ladder for a given batch cap: the canonical set
// {1, 2, 4, 8, 16, 32, 48, 64} filtered to <= max_batch, ascending
// (docs/DESIGN.md section 9; dev preset yields {1, 2, 4, 8}).
std::vector<std::int32_t> MakeDecodeBuckets(std::int32_t max_batch);

// Smallest bucket >= batch_size from an ascending bucket list, or -1 when no
// bucket fits - the engine then runs that decode step eagerly. batch_size <=
// 0 also yields -1 (nothing to launch).
std::int32_t SelectBucket(std::int32_t batch_size, const std::vector<std::int32_t>& buckets);

// Iteration-level continuous-batching scheduler: FCFS admission against the
// reservation allocator (docs/DESIGN.md sections 7-8), dedicated
// chunked-prefill steps, no preemption in v1. Blocks are reserved at
// admission and materialized lazily (prompt blocks during prefill planning,
// decode blocks at 16-token boundaries via AllocReserved) - never bulk-
// allocated at admission. Pure CPU logic with no CUDA dependency, so it
// unit-tests against a mock executor. Continuous batching over a paged KV
// cache follows the system design space described by Kwon et al., 2023; this
// implementation is original.
//
// Not thread-safe; the engine serializes PlanStep/CompleteStep/Enqueue.
class Scheduler {
 public:
  Scheduler(BlockAllocator* allocator, SchedulerOptions options);

  // Validate and queue a new request (state stays kWaiting; admission
  // happens inside PlanStep). Throws std::invalid_argument - and does not
  // queue - on: empty prompt, max_new_tokens <= 0, prompt_len +
  // max_new_tokens > max_seq_len, or a request that could never be admitted
  // even by an idle pool (see ValidateRequest). Engine::AddRequest surfaces
  // these to the client instead of queueing the request forever.
  void Enqueue(RequestPtr request);

  // Decide the next iteration (docs/DESIGN.md section 8 step algorithm):
  //   1) admit waiting requests strictly FCFS while admitted-and-unfinished
  //      < max_batch and CanAdmit(front) holds (stop at the first head that
  //      does not fit - no reordering). Admission reserves blocks
  //      (policy-dependent amount) and assigns a batch slot.
  //   2) if any admitted request is still prefilling, return the next chunk
  //      [prefill_progress, +prefill_chunk_tokens) of the OLDEST one as a
  //      dedicated prefill step, materializing the blocks that cover it.
  //   3) otherwise return up to max_batch running sequences (oldest first)
  //      as one decode batch, materializing one block for each sequence
  //      whose next position crosses a 16-token boundary. Under kWatermark
  //      an exhausted pool aborts the youngest running sequence (released
  //      immediately, recorded in StepPlan::aborted) and retries.
  StepPlan PlanStep();

  // Commit one executed step and return everything to report for it, in
  // order: one kAborted result per plan.aborted entry first (token 0 -
  // nothing was produced), then one result per produced token in plan order
  // (final prefill chunk: exactly one; decode: one per plan.decode row).
  //
  // sampled_tokens carries the executor's greedy argmax per produced row:
  //   - prefill plan, non-final chunk: empty (no logits are computed);
  //   - prefill plan, final chunk:     exactly one token;
  //   - decode plan:                   plan.decode.size() tokens, row-aligned.
  //
  // Produced tokens are appended to their sequences; a request with
  // forced_tokens appends forced_tokens[n] instead while the emission still
  // carries the executor's token (teacher-forcing hook, docs/DESIGN.md
  // section 12(c)). Stop checks run on the APPENDED token: EOS-set
  // membership (skipped when ignore_eos; kEos wins a simultaneous length
  // stop), then max_new_tokens (kLength). Finishing - including straight out
  // of kPrefill (EOS on the first generated token, or max_new_tokens == 1) -
  // frees every owned block, releases the unused reservation remainder and
  // the batch slot.
  //
  // Contract: called exactly once with the StepPlan returned by the
  // immediately preceding PlanStep (single-threaded engine loop).
  std::vector<StepResult> CompleteStep(const StepPlan& plan,
                                       const std::vector<TokenId>& sampled_tokens);

  // Admission predicate for a validated waiting request:
  //   kReserveFull -> allocator->CanReserve(BlocksForTokens(prompt + max_new));
  //   kWatermark   -> uncommitted - BlocksForTokens(prompt) >=
  //                   watermark_free_fraction * usable, where uncommitted =
  //                   free - reserved. Outstanding reservations are counted
  //                   because materialization is lazy: the raw free-list size
  //                   still contains blocks already promised to admitted
  //                   prompts, and Reserve()'s CanReserve precondition must
  //                   hold at admission time.
  // Two deliberate deviations from the section 8 formula ("free -
  // prompt_blocks >= 0.10 * num_blocks"): free is tightened to uncommitted
  // (rationale above), and the floor base is usable_blocks()
  // (= num_blocks - 1) rather than total num_blocks - block 0 is the
  // permanently-held dummy block, so the floor is measured against blocks
  // that can actually be free. ValidateRequest's never-fits check uses the
  // same base.
  // Exposed for unit tests; PlanStep is the only production caller.
  bool CanAdmit(const Request& request) const;

  // The Enqueue rejection rules (docs/DESIGN.md section 8): empty prompt,
  // non-positive max_new_tokens, prompt_len + max_new_tokens > max_seq_len,
  // and never-fits-pool - a request whose admission predicate is false even
  // with every usable block free (kReserveFull: worst-case blocks > usable;
  // kWatermark: an idle pool minus the prompt still breaks the free floor).
  // Throws std::invalid_argument naming the failing limit.
  void ValidateRequest(const Request& request) const;

  std::size_t num_waiting() const { return waiting_.size(); }
  // Admitted and not yet finished: in-flight prefills plus decoding
  // sequences (surfaced as the "running" stat).
  std::size_t num_running() const { return prefill_queue_.size() + running_.size(); }

 private:
  // Pop the waiting head into the prefill queue: reserve (policy-dependent),
  // assign a batch slot, state kWaiting -> kPrefill.
  void Admit();
  // Blocks to promise at admission: kReserveFull covers prompt + max_new,
  // kWatermark covers the prompt only.
  std::int64_t ReservationFor(const Request& request) const;
  // Make the sequence's block table cover its next decode position. Under
  // kWatermark this may abort youngest sequences into `plan`; returns false
  // when `request` itself became the victim.
  bool EnsureDecodeBlockCapacity(const RequestPtr& request, StepPlan& plan);
  // Release + record the most recently admitted running sequence as aborted.
  void AbortYoungestRunning(StepPlan& plan);
  // FreeBlock every owned block, Unreserve the unmaterialized remainder,
  // return the batch slot. Idempotent per finished request.
  void ReleaseResources(Request& request);
  // Append one produced token (honoring forced_tokens), run the stop checks,
  // handle the kPrefill -> kRunning / -> kFinished transitions, and build
  // the StepResult for it.
  StepResult ApplySampledToken(const RequestPtr& request, TokenId sampled_token);
  bool IsEosToken(TokenId token) const;

  BlockAllocator* allocator_; // not owned
  SchedulerOptions options_;
  std::deque<RequestPtr> waiting_;       // FCFS arrival order
  std::deque<RequestPtr> prefill_queue_; // admitted, prompt not fully in KV; FIFO
  std::vector<RequestPtr> running_;      // decoding, admission order (back = youngest)
  std::vector<std::int32_t> free_slots_; // batch slots not held by an admitted request
};

} // namespace redline
