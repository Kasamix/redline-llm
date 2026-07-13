#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/block_table.hpp"
#include "core/types.hpp"

namespace redline {

enum class RequestState : std::uint8_t {
  kWaiting,  // queued; no KV blocks reserved yet
  kPrefill,  // admitted (blocks reserved); prompt entering the KV cache
  kRunning,  // in the decode batch; one token per step
  kFinished, // EOS, length, or abort; blocks freed, reservation released
};

// One generation request. Owned by the Engine and shared with the Scheduler;
// not thread-safe - mutated only inside Engine::Step and AddRequest, which
// callers serialize.
struct Request {
  RequestId id = 0;
  std::vector<TokenId> prompt_tokens;
  std::vector<TokenId> output_tokens;
  std::int32_t max_new_tokens = 0;

  // Skip the EOS-set stop check (length stop still applies). Exists so the
  // benchmark harness can force exact output lengths (docs/DESIGN.md
  // section 13); generated EOS tokens still appear in output_tokens.
  bool ignore_eos = false;

  // Teacher-forcing hook (debug/eval only; docs/DESIGN.md section 12(c)).
  // When non-empty, the scheduler appends forced_tokens[n] as the sequence's
  // n-th generated token while the StepResult still reports the engine's own
  // sampled token, so parity suites can compare the model's choice at every
  // position along a fixed reference continuation without divergence
  // compounding. Positions beyond the end of forced_tokens fall back to the
  // engine's token. Stop checks (EOS set, max_new_tokens) apply to the
  // APPENDED token. Empty (and therefore inert) in production serving.
  std::vector<TokenId> forced_tokens;

  RequestState state = RequestState::kWaiting;
  FinishReason finish_reason = FinishReason::kNone;

  // Stable scheduler-side batch slot in [0, max_batch), assigned at
  // admission and held until finish: the fixed-size slot pool caps
  // concurrent admissions and gives each admitted request a persistent
  // identity. NOT the per-step device row - per-step device inputs and the
  // device block-table mirror are uploaded as compacted bucket prefixes
  // (docs/DESIGN.md sections 8-9: bucket = smallest b >= |batch|, block
  // table as one [bucket, max_blocks_per_seq] slab copy, only trailing rows
  // padded), so a decode sequence's device row is its plan-order index in
  // [0, plan.decode.size()) and may change from step to step as other
  // sequences finish; prefill rows are chunk-token indices. -1 while
  // kWaiting and again after kFinished (returned to the scheduler's
  // free-slot pool).
  std::int32_t slot = -1;

  // Blocks promised to this request at admission and not yet materialized
  // (reserve_full: worst case prompt + max_new; watermark: prompt only);
  // returned via BlockAllocator::Unreserve on finish.
  std::int64_t reserved_blocks = 0;

  // Prompt tokens already written into the KV cache; chunked prefill advances
  // this by at most prefill_chunk_tokens per dedicated prefill step.
  std::int32_t prefill_progress = 0;

  // Physical KV blocks owned by this request (returned to the allocator when
  // the request finishes).
  BlockTable block_table;

  std::int32_t num_prompt_tokens() const { return static_cast<std::int32_t>(prompt_tokens.size()); }
  std::int32_t num_output_tokens() const { return static_cast<std::int32_t>(output_tokens.size()); }
  // Tokens with KV state once prefill has completed, plus generation so far.
  std::int32_t seq_len() const { return num_prompt_tokens() + num_output_tokens(); }
  bool prefill_done() const { return prefill_progress >= num_prompt_tokens(); }
};

using RequestPtr = std::shared_ptr<Request>;

} // namespace redline
