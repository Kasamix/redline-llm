#pragma once

#include <cstddef>
#include <cstdint>

// Core scalar types and small PODs shared across the engine. This header is
// CUDA-free so CPU-only translation units (scheduler/allocator unit tests)
// never pull in toolkit headers.

namespace redline {

// Token id as produced by the client-side tokenizer (int32 comfortably holds
// the Qwen2.5 vocab of 151,936 entries).
using TokenId = std::int32_t;

// Engine-assigned request identifier: monotonically increasing, never reused.
using RequestId = std::uint64_t;

// Index of a physical block inside the paged KV pool (shared index space
// across all layers; device block tables store these as int32).
using BlockId = std::int32_t;

// Host-side-only debug sentinel. It must never be written into a device
// block table (debug-asserted at mirror-fill time): device rows/slots are
// padded with kDummyBlockId instead, so any address a kernel might compute
// from padding stays in bounds (docs/DESIGN.md section 9 sentinel policy).
inline constexpr BlockId kInvalidBlockId = -1;

// Physical block 0 is reserved at init as the dummy block: it never enters
// the allocator's free list and is the padding target for unused device
// block-table entries and warmup/capture inputs.
inline constexpr BlockId kDummyBlockId = 0;

// Tokens per KV-cache block. Fixed at compile time: the paged attention kernel
// walks the cache block-by-block, and 16 balances internal fragmentation
// against block-table length. Paging concept from the PagedAttention paper
// (Kwon et al., 2023, arXiv:2309.06180); implementation here is original.
inline constexpr std::int32_t kKvBlockSize = 16;

// Blocks required to hold `num_tokens` tokens of KV state.
inline constexpr std::int32_t BlocksForTokens(std::int32_t num_tokens) {
  return (num_tokens + kKvBlockSize - 1) / kKvBlockSize;
}

// Element types encountered in checkpoints and runtime buffers.
enum class Dtype : std::uint8_t { kF16, kBF16, kF32, kF64, kI32, kI64, kU8, kUnknown };

inline constexpr std::size_t DtypeSize(Dtype dtype) {
  switch (dtype) {
  case Dtype::kF16:
  case Dtype::kBF16:
    return 2;
  case Dtype::kF32:
  case Dtype::kI32:
    return 4;
  case Dtype::kF64:
  case Dtype::kI64:
    return 8;
  case Dtype::kU8:
    return 1;
  case Dtype::kUnknown:
    return 0;
  }
  return 0;
}

enum class FinishReason : std::uint8_t {
  kNone,    // still generating
  kEos,     // model emitted an end-of-sequence token (and ignore_eos unset)
  kLength,  // reached max_new_tokens
  kAborted, // evicted by the watermark policy's abort-youngest recovery path;
            // first-class so an abort is never mistaken for a natural finish
};

// One generated token, as reported by Engine::Step(). Under teacher forcing
// (Request::forced_tokens) `token` carries the engine's own sampled token
// even though the forced token is what was appended to the sequence. A
// watermark-policy abort is reported as token 0 with finished=true and
// finish_reason kAborted - no token was produced for that request this step.
struct StepResult {
  RequestId request_id = 0;
  TokenId token = 0;
  bool finished = false;
  FinishReason finish_reason = FinishReason::kNone;
};

} // namespace redline
