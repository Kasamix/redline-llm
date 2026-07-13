#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/block_allocator.hpp"
#include "core/config.hpp"
#include "core/request.hpp"
#include "core/scheduler.hpp"
#include "core/types.hpp"

namespace redline {

class DeviceWeights; // loader/weights.hpp
class ModelExecutor; // core/model_executor.hpp

// Backpressure cap for Engine::HasCapacity - the waiting queue depth below
// which add_request is welcome. KV admission control is separate and happens
// inside Step (docs/DESIGN.md section 10).
inline constexpr std::size_t kMaxWaitingRequests = 1024;

// Runtime knobs. Everything that differs between the dev GPU (RTX 2060, 6 GB,
// sm_75) and the bench GPU (RTX 4090, 24 GB, sm_89) must be settable here -
// no hardware assumptions are compiled in. Defaults are the DEV preset of
// docs/DESIGN.md section 11 (the smallest supported GPU must be safe by
// default); the bench harness passes its full config explicitly.
struct EngineOptions {
  std::string model_dir; // directory containing config.json + *.safetensors

  // Paged KV pool budget in GiB (FP16). Qwen2.5-1.5B needs 448 KiB per
  // 16-token block, so 1.0 GiB holds 2,340 blocks = ~37K tokens of cache
  // (one block is reserved as the dummy block, docs/DESIGN.md section 9).
  double kv_pool_gb = 1.0;

  // Decode batch cap; also the largest CUDA-graph bucket in use.
  std::int32_t max_batch = 8;

  // Capture decode-only CUDA graphs per batch bucket at init and replay
  // them per decode step (docs/DESIGN.md section 9); eager fallback is
  // always available for prefill and unbucketed shapes, and a capture
  // failure degrades the whole engine to eager with a one-time warning
  // instead of failing init. false bypasses warmup and capture entirely.
  bool enable_cuda_graphs = true;

  // Hard cap on prompt_len + max_new_tokens per request; also sets the
  // device block-table width (dev preset 2048; bench preset 4096).
  std::int32_t max_seq_len = 2048;

  // Prompt tokens processed per dedicated prefill step (dev preset 1024;
  // bench preset 2048).
  std::int32_t prefill_chunk_tokens = 1024;

  AdmissionPolicy admission_policy = AdmissionPolicy::kReserveFull;

  // Debug/eval only (docs/DESIGN.md section 12d/e same-shape protocol);
  // 0 = off (default). Value b: every eager decode batch is padded to the
  // smallest configured bucket >= max(live_batch, b) with dummy rows
  // (seq_len=0, position 0, token 0, dummy-block table rows), so a solo
  // request can be forced through e.g. the bucket-8 GEMM shapes.
  // Integer-valued on purpose: `true` (== 1) means "smallest bucket covering
  // the live batch" while 8 forces the solo-vs-batch-of-8 comparison. Values
  // above the largest configured bucket are clamped to it; a live batch no
  // bucket covers runs unpadded. Forwarded verbatim to
  // ExecutorOptions::pad_eager_to_bucket.
  std::int32_t pad_eager_to_bucket = 0;

  // Debug/eval only (docs/DESIGN.md section 12f layer-0 parity fixture);
  // empty (default) = off. When set, the first prefill chunk writes its
  // layer-0 activations into this directory as raw FP16 binaries plus a JSON
  // shape descriptor (details on ExecutorOptions::debug_dump_dir, which this
  // is forwarded to verbatim). Consumed by tests/e2e/test_layer0.py.
  std::string debug_dump_dir;

  // Device block-table width, derived - never set independently of
  // max_seq_len (docs/DESIGN.md section 5).
  std::int32_t max_blocks_per_seq() const { return BlocksForTokens(max_seq_len); }
};

// Counters surfaced through Engine::Stats / redline.Engine.stats()
// (docs/DESIGN.md section 10).
struct EngineStats {
  // Engine iterations that executed a non-empty plan (a prefill chunk, a
  // decode batch, or watermark aborts). Step calls that found nothing to do
  // are not counted.
  std::uint64_t steps = 0;
  std::uint64_t prefill_tokens = 0;
  std::uint64_t decode_tokens = 0;
  std::uint64_t num_waiting = 0;
  std::uint64_t num_running = 0;
  std::int64_t free_kv_blocks = 0;
  std::int64_t reserved_kv_blocks = 0;
  std::int64_t total_kv_blocks = 0;
  std::uint64_t graph_replays = 0;
  std::uint64_t eager_decodes = 0;
  std::uint64_t aborts = 0;
  // (bucket size, decode steps executed at exactly that bucket's shape) per
  // configured bucket, in bucket order. Counts graph replays (which always
  // execute at their bucket's shape) and eager decode steps whose executed
  // row count - after any pad_eager_to_bucket padding - equals a configured
  // bucket. Off-ladder eager shapes appear in eager_decodes only.
  std::vector<std::pair<std::int32_t, std::uint64_t>> bucket_histogram;
};

// Single-GPU inference engine: the composition root that owns the loaded
// weight store, the ModelExecutor (the single paged KV pool, activation
// scratch, pinned mirrors, stream s0, and the cuBLASLt wrapper), the KV
// BlockAllocator, and the Scheduler, glued into the docs/DESIGN.md section 8
// step loop. Construction runs the full section 2 init sequence, in order:
//
//   1. parse + validate <model_dir>/config.json and generation_config.json
//      (ConfigError on any mismatch with docs/MODEL_SPEC.md);
//   2. compute the full device-memory budget - weights + KV pool + scratch +
//      cuBLASLt workspace + CUDA-graph budget + context/kernel-image slack -
//      and preflight it against cudaMemGetInfo, failing fast with the
//      itemized breakdown in the exception message when the configuration
//      cannot fit (docs/DESIGN.md section 11);
//   3. load the weights (safetensors mmap -> checked BF16->FP16 conversion ->
//      single device slab) on a temporary stream that is destroyed once the
//      store is resident;
//   4. construct the executor - every remaining device/pinned allocation and
//      every init-time cuBLASLt probe happens in its constructor, so nothing
//      allocates after this constructor returns;
//   5. size the BlockAllocator from the executor's pool (block 0 is the
//      reserved dummy block, so usable capacity = num_kv_blocks - 1);
//   6. create the Scheduler with the EOS stop set from generation_config.
//
// Under AdmissionPolicy::kWatermark, init warns when max_batch *
// BlocksForTokens(max_seq_len) exceeds the usable pool - the configurations
// in which the abort-youngest recovery path is load-bearing (section 8).
//
// Tokenization is client-side - the engine consumes and produces token ids
// only. Not thread-safe: AddRequest/Step/HasCapacity/Stats must be called
// from the constructing thread (debug-asserted; the Python binding releases
// the GIL around Step but drives the engine from one thread).
class Engine {
 public:
  // Runs the init sequence above. Throws ConfigError / SafetensorsError /
  // WeightsError / ExecutorError / GemmError from the failing component,
  // std::invalid_argument on out-of-range EngineOptions, and
  // std::runtime_error (with the computed budget breakdown) when the memory
  // preflight fails.
  explicit Engine(EngineOptions options);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Queue a request and return its id immediately; admission into the running
  // batch happens inside Step (scheduler admission control). Throws
  // std::invalid_argument - recording nothing - on an empty prompt,
  // max_new_tokens <= 0, prompt_len + max_new_tokens > max_seq_len, a request
  // that could never fit the KV pool even when idle, or any prompt/forced
  // token outside the vocabulary (an out-of-range id would otherwise be
  // admitted and then fail inside Step on the executor's mirror-fill checks,
  // wedging the queue). `ignore_eos` skips the EOS stop check so benchmark
  // runs can force exact output lengths. `forced_tokens` is the teacher-
  // forcing hook of docs/DESIGN.md section 12(c) (see Request::forced_tokens);
  // leave empty in production serving.
  RequestId AddRequest(std::vector<TokenId> prompt_tokens, std::int32_t max_new_tokens,
                       bool ignore_eos = false, std::vector<TokenId> forced_tokens = {});

  // Run one scheduler iteration - one chunked-prefill slice or one decode
  // batch (docs/DESIGN.md section 8) - and report every token it produced
  // (plus one kAborted result per watermark-policy abort, first). A decode
  // batch replays the captured CUDA graph of the smallest configured bucket
  // >= |batch| when decode graphs are active (docs/DESIGN.md section 9);
  // prefill, batches no bucket covers, graphs-off, and the degraded
  // capture-failure state run eagerly. Returns empty when nothing is
  // waiting or running. Executor failures (ExecutorError) propagate. NVTX
  // range `step` wraps the whole iteration; the executor emits the inner
  // taxonomy ranges.
  std::vector<StepResult> Step();

  // Backpressure check for clients: waiting queue below kMaxWaitingRequests.
  // Pure queue-depth check - KV admission happens inside Step.
  bool HasCapacity() const;

  // Fills every EngineStats field; bucket_histogram is paired with the
  // executor's configured bucket ladder, in bucket order.
  EngineStats Stats() const;

  // Debug probe (docs/DESIGN.md section 12d fragmentation audit): physical KV
  // block ids currently owned by a live request, in allocation order. Returns
  // empty for an unknown or finished id (finished requests have released
  // their blocks). Host-side scheduler state; zero device cost.
  std::vector<BlockId> DebugBlockTable(RequestId request_id) const;

  // Selected-algorithm report of the executor's cuBLASLt wrapper (JSON text,
  // docs/DESIGN.md section 12d) - surfaced through bench/test output.
  std::string AlgoReportJson() const;

 private:
  // Debug-asserts the single-thread contract (docs/DESIGN.md section 2);
  // compiles to nothing under NDEBUG. The owning thread is the one that ran
  // the constructor.
  void DebugAssertOwnerThread() const;

  EngineOptions options_;
  ModelConfig model_config_;

  // Declaration order is load-bearing: the executor keeps a reference to
  // *weights_, and the scheduler holds a raw pointer to *block_allocator_,
  // so reverse-order member destruction tears consumers down first.
  std::unique_ptr<DeviceWeights> weights_;
  std::unique_ptr<ModelExecutor> executor_;
  std::unique_ptr<BlockAllocator> block_allocator_;
  std::unique_ptr<Scheduler> scheduler_;

  // Live (waiting or admitted, not yet finished) requests by id - the
  // DebugBlockTable lookup path. Entries are erased as their finish emission
  // is returned from Step, so the map stays bounded across a long-serving
  // process (the scheduler drops its own references on finish too).
  std::unordered_map<RequestId, RequestPtr> requests_;
  RequestId next_request_id_ = 1;

  // Thread that constructed the engine; stored in every build type so the
  // layout does not depend on NDEBUG, consulted only by the debug assert.
  std::thread::id owner_thread_;

  std::uint64_t steps_ = 0;
  std::uint64_t prefill_tokens_ = 0;
  std::uint64_t decode_tokens_ = 0;
  std::uint64_t graph_replays_ = 0; // decode steps served by CUDA-graph replay
  std::uint64_t eager_decodes_ = 0;
  std::uint64_t aborts_ = 0;
  // Decode steps executed at each configured bucket's exact shape, indexed
  // like the executor's decode_buckets(); Stats pairs them up.
  std::vector<std::uint64_t> bucket_decode_steps_;
};

} // namespace redline
