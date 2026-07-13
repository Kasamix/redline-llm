#include "core/engine.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include "core/model_executor.hpp"
#include "loader/convert.hpp"
#include "loader/safetensors.hpp"
#include "loader/weights.hpp"

// Engine wiring (docs/DESIGN.md sections 2, 8, 9, 10, 11).
//
// The constructor is the ONLY place this file touches CUDA directly, and only
// for two things: the cudaMemGetInfo preflight and the temporary stream the
// weight upload runs on. Every persistent device/pinned resource belongs to
// DeviceWeights or ModelExecutor, both of which allocate exclusively in their
// constructors - so once Engine's constructor returns, the process performs
// no further CUDA allocations (the section 5 rule that makes graph capture
// legal later).
//
// Step() is the section 8 algorithm verbatim, split across components:
//   Scheduler::PlanStep      admission, dedicated prefill chunking, decode
//                            batch assembly, lazy block materialization, and
//                            (kWatermark) abort-youngest recovery;
//   ModelExecutor            the planned prefill chunk, or the decode batch -
//                            graph replay at the covering bucket when decode
//                            graphs were captured (docs/DESIGN.md section 9),
//                            eager otherwise;
//   Scheduler::CompleteStep  token append (teacher-forcing aware), stop
//                            checks, resource release, emission assembly.
// The engine adds bucket selection (smallest configured b >= |batch|) and
// the counters Stats() reports.

namespace redline {

namespace {

constexpr std::int64_t kMiB = std::int64_t{1} << 20;

// docs/DESIGN.md section 11 budget rows that live outside the executor's
// arithmetic: the CUDA context + cuBLASLt kernel images line (budgeted
// conservatively - part is consumed before cudaMemGetInfo runs, the Lt
// images load after), and the per-bucket CUDA-graph budget (64 MiB across
// the 4 dev buckets, 128 MiB across the 8 bench buckets).
constexpr std::int64_t kContextSlackBytes = 512 * kMiB;
constexpr std::int64_t kGraphBudgetPerBucketBytes = 16 * kMiB;

void CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("engine: ") + what + ": " + cudaGetErrorName(err) + ": " +
                             cudaGetErrorString(err));
  }
}

// NVTX RAII wrapper for the docs/DESIGN.md section 14 taxonomy. The `step`
// range wraps the whole iteration; the executor emits the inner ranges
// (h2d_inputs / prefill / decode / sample / d2h_tokens) and no NVTX ever
// sits inside a capturable sequence (section 9).
class NvtxRange {
 public:
  explicit NvtxRange(const char* label) { nvtxRangePushA(label); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

double ToMiB(std::int64_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(kMiB);
}

const char* AdmissionPolicyName(AdmissionPolicy policy) {
  switch (policy) {
  case AdmissionPolicy::kReserveFull:
    return "reserve_full";
  case AdmissionPolicy::kWatermark:
    return "watermark";
  }
  return "unknown";
}

// EngineOptions -> ExecutorOptions, 1:1 (the executor wants bytes, not GiB).
// kv_pool_gb must be validated before calling (a NaN/negative double cast to
// int64 is undefined).
ExecutorOptions MakeExecutorOptions(const EngineOptions& options) {
  ExecutorOptions exec;
  exec.kv_pool_bytes =
      static_cast<std::int64_t>(options.kv_pool_gb * static_cast<double>(std::int64_t{1} << 30));
  exec.max_batch = options.max_batch;
  exec.max_seq_len = options.max_seq_len;
  exec.prefill_chunk_tokens = options.prefill_chunk_tokens;
  exec.enable_cuda_graphs = options.enable_cuda_graphs;
  exec.pad_eager_to_bucket = options.pad_eager_to_bucket;
  exec.debug_dump_dir = options.debug_dump_dir;
  return exec;
}

} // namespace

Engine::Engine(EngineOptions options)
    : options_(std::move(options)), owner_thread_(std::this_thread::get_id()) {
  const auto init_begin = std::chrono::steady_clock::now();

  // -- 1. Parse + validate the model configuration (docs/DESIGN.md section 2
  //       ordering; hard ConfigError on any docs/MODEL_SPEC.md mismatch).
  model_config_ = ModelConfig::FromModelDir(options_.model_dir);

  // The one option the executor cannot range-check itself (it receives
  // bytes): guard the GiB -> bytes conversion. Everything else is validated
  // by ModelExecutor::ComputeBudget below, before any allocation.
  if (!(options_.kv_pool_gb > 0.0) || options_.kv_pool_gb > 1024.0) {
    throw std::invalid_argument("engine: kv_pool_gb must be in (0, 1024], got " +
                                std::to_string(options_.kv_pool_gb));
  }
  const ExecutorOptions exec_options = MakeExecutorOptions(options_);

  // -- 2. Full memory budget + cudaMemGetInfo preflight (docs/DESIGN.md
  //       section 11): pure arithmetic first - ComputeBudget validates the
  //       geometry and derives the pool/scratch/workspace plan, TotalDeviceBytes
  //       prices the weight slab - then fail fast with the itemized breakdown
  //       if the configuration cannot fit the device.
  SafetensorsModel model = SafetensorsModel::OpenDir(options_.model_dir);
  const bool has_separate_lm_head = model.Contains("lm_head.weight");
  const std::int64_t weight_bytes =
      DeviceWeights::TotalDeviceBytes(model_config_, has_separate_lm_head);
  const ExecutorMemoryBudget budget = ModelExecutor::ComputeBudget(model_config_, exec_options);
  const std::int64_t num_buckets =
      static_cast<std::int64_t>(MakeDecodeBuckets(options_.max_batch).size());
  const std::int64_t graph_bytes =
      options_.enable_cuda_graphs ? num_buckets * kGraphBudgetPerBucketBytes : 0;
  const std::int64_t required_bytes =
      weight_bytes + budget.total_device_bytes + graph_bytes + kContextSlackBytes;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo preflight");
  if (required_bytes > static_cast<std::int64_t>(free_bytes)) {
    std::string message = "engine preflight: configuration does not fit device memory "
                          "(docs/DESIGN.md section 11 budget):\n";
    const auto append = [&message](const std::string& label, double mib) {
      char line[128];
      std::snprintf(line, sizeof(line), "  %-30s %9.1f MiB\n", label.c_str(), mib);
      message += line;
    };
    append("weights (FP16)", ToMiB(weight_bytes));
    append("KV pool (" + std::to_string(budget.num_kv_blocks) + " blocks)",
           ToMiB(budget.kv_pool_bytes));
    append("activation + input scratch", ToMiB(budget.scratch_bytes));
    append("cuBLASLt workspace", ToMiB(budget.workspace_bytes));
    const std::string graphs_label =
        options_.enable_cuda_graphs ? "CUDA graphs (" + std::to_string(num_buckets) + " buckets)"
                                    : "CUDA graphs (disabled)";
    append(graphs_label, ToMiB(graph_bytes));
    append("context + kernel images", ToMiB(kContextSlackBytes));
    append("required total", ToMiB(required_bytes));
    char tail[160];
    std::snprintf(tail, sizeof(tail), "  %-30s %9.1f / %.1f MiB (cudaMemGetInfo)\n",
                  "device free / total", ToMiB(static_cast<std::int64_t>(free_bytes)),
                  ToMiB(static_cast<std::int64_t>(total_bytes)));
    message += tail;
    message += "Reduce kv_pool_gb, max_batch, max_seq_len, or prefill_chunk_tokens, "
               "or run on a larger GPU.";
    throw std::runtime_error(message);
  }

  // -- 3. Load the weights (docs/DESIGN.md section 4). The upload runs on a
  //       temporary non-default stream: the executor (and its stream s0) does
  //       not exist yet, and the DeviceWeights constructor drains every
  //       upload before returning, so the stream is idle - and immediately
  //       destroyable - on success and failure alike.
  const auto load_begin = std::chrono::steady_clock::now();
  cudaStream_t load_stream = nullptr;
  CudaCheck(cudaStreamCreateWithFlags(&load_stream, cudaStreamNonBlocking),
            "create weight-load stream");
  try {
    StagedUploader uploader; // pinned staging, freed at scope exit
    weights_ = std::make_unique<DeviceWeights>(model, model_config_, uploader, load_stream);
  } catch (...) {
    cudaStreamDestroy(load_stream); // best-effort; the load error is what matters
    throw;
  }
  CudaCheck(cudaStreamDestroy(load_stream), "destroy weight-load stream");
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin).count();

  // -- 4. Executor: every remaining device/pinned allocation and all
  //       init-time cuBLASLt probes happen inside this constructor.
  executor_ = std::make_unique<ModelExecutor>(model_config_, *weights_, exec_options);

  // -- 5. Allocator sized from the executor's pool: block 0 is the reserved
  //       dummy block, so usable capacity is num_kv_blocks() - 1.
  block_allocator_ = std::make_unique<BlockAllocator>(executor_->num_kv_blocks());

  // -- 6. Scheduler, with the EOS stop set from generation_config.json
  //       (docs/MODEL_SPEC.md F4: {151645, 151643} for this checkpoint).
  SchedulerOptions scheduler_options;
  scheduler_options.max_batch = options_.max_batch;
  scheduler_options.prefill_chunk_tokens = options_.prefill_chunk_tokens;
  scheduler_options.max_seq_len = options_.max_seq_len;
  scheduler_options.admission_policy = options_.admission_policy;
  scheduler_options.eos_token_ids = model_config_.eos_token_ids;
  scheduler_ = std::make_unique<Scheduler>(block_allocator_.get(), scheduler_options);

  bucket_decode_steps_.assign(executor_->decode_buckets().size(), 0);

  // Watermark configurations where decode growth can exhaust the pool make
  // the abort-youngest recovery path load-bearing rather than theoretical -
  // warn once at init (docs/DESIGN.md section 8).
  if (options_.admission_policy == AdmissionPolicy::kWatermark) {
    const std::int64_t worst_case_blocks =
        static_cast<std::int64_t>(options_.max_batch) * options_.max_blocks_per_seq();
    if (worst_case_blocks > block_allocator_->usable_blocks()) {
      std::fprintf(stderr,
                   "[redline] warning: watermark admission with max_batch (%d) x "
                   "blocks-per-sequence (%d) = %lld worst-case blocks against %d usable KV "
                   "blocks: decode growth can exhaust the pool, so the abort-youngest "
                   "recovery path is load-bearing in this configuration\n",
                   options_.max_batch, options_.max_blocks_per_seq(),
                   static_cast<long long>(worst_case_blocks), block_allocator_->usable_blocks());
    }
  }

  // CUDA-graph outcome (docs/DESIGN.md section 9): capture runs inside the
  // executor constructor and degrades to eager on failure, so report what
  // actually happened, not just the requested flag. The executor logged the
  // per-bucket capture and its graph-memory delta already.
  std::string graphs_state = "off";
  if (options_.enable_cuda_graphs) {
    graphs_state = executor_->has_decode_graphs()
                       ? "on (" + std::to_string(executor_->decode_buckets().size()) +
                             " decode buckets captured)"
                       : "on (capture failed; degraded to eager)";
  }

  const double init_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - init_begin).count();
  std::fprintf(stderr,
               "[redline] engine init: weights %.1f MiB in %.2f s (init total %.2f s); "
               "preflight %.1f MiB required, %.1f MiB free; %d KV blocks (%d usable); "
               "policy=%s; cuda_graphs=%s\n",
               ToMiB(weight_bytes), load_seconds, init_seconds, ToMiB(required_bytes),
               ToMiB(static_cast<std::int64_t>(free_bytes)), executor_->num_kv_blocks(),
               block_allocator_->usable_blocks(), AdmissionPolicyName(options_.admission_policy),
               graphs_state.c_str());
}

Engine::~Engine() = default;

RequestId Engine::AddRequest(std::vector<TokenId> prompt_tokens, std::int32_t max_new_tokens,
                             bool ignore_eos, std::vector<TokenId> forced_tokens) {
  DebugAssertOwnerThread();

  // Reject out-of-vocabulary ids at the API edge. The executor hard-checks
  // token range at mirror-fill time, but by then the request is admitted and
  // every Step would rethrow - a wedged queue instead of a clean reject.
  // Forced tokens are checked too: they are APPENDED to the sequence and fed
  // back as decode inputs (docs/DESIGN.md section 12(c)).
  const auto check_tokens = [this](const std::vector<TokenId>& tokens, const char* what) {
    for (const TokenId token : tokens) {
      if (token < 0 || token >= model_config_.vocab_size) {
        throw std::invalid_argument(std::string("engine: ") + what + " contains token id " +
                                    std::to_string(token) + " outside the vocabulary [0, " +
                                    std::to_string(model_config_.vocab_size) + ")");
      }
    }
  };
  check_tokens(prompt_tokens, "prompt_tokens");
  check_tokens(forced_tokens, "forced_tokens");

  auto request = std::make_shared<Request>();
  request->id = next_request_id_;
  request->prompt_tokens = std::move(prompt_tokens);
  request->max_new_tokens = max_new_tokens;
  request->ignore_eos = ignore_eos;
  request->forced_tokens = std::move(forced_tokens);

  // Enqueue validates (empty prompt, max_new_tokens <= 0, oversize prompt,
  // never-fits-pool) and throws std::invalid_argument WITHOUT queueing, so a
  // rejected request leaves no engine state behind - the id is not consumed
  // and nothing lands in requests_.
  scheduler_->Enqueue(request);

  ++next_request_id_;
  requests_.emplace(request->id, request);
  return request->id;
}

std::vector<StepResult> Engine::Step() {
  DebugAssertOwnerThread();
  const NvtxRange step_range("step");

  StepPlan plan = scheduler_->PlanStep();
  if (plan.empty()) {
    return {};
  }

  // Execute the plan: exactly one prefill chunk OR one decode batch
  // (docs/DESIGN.md section 8; prefill steps are dedicated). A plan can also
  // be abort-only under kWatermark (every survivor evicted) - nothing to
  // execute, but CompleteStep still emits the kAborted results.
  std::vector<TokenId> sampled_tokens;
  std::int32_t prefill_rows = 0;
  std::int32_t decode_rows = 0;
  std::int32_t executed_bucket = -1;
  bool replayed_graph = false;

  if (plan.is_prefill()) {
    const PrefillOutput out =
        executor_->PrefillChunk(plan.prefill, plan.prefill_begin, plan.prefill_end);
    prefill_rows = out.rows;
    if (out.final_chunk) {
      // The final chunk samples the sequence's first generated token
      // (docs/DESIGN.md section 6.3); earlier chunks compute no logits.
      sampled_tokens.push_back(out.first_token);
    }
  } else if (!plan.decode.empty()) {
    decode_rows = static_cast<std::int32_t>(plan.decode.size());
    // Decode dispatch (docs/DESIGN.md section 9): replay the captured graph
    // of the smallest bucket covering the live batch. Prefill (above), a
    // batch no bucket covers, graphs disabled, and the degraded-to-eager
    // capture-failure state all run the permanently maintained eager path.
    // has_decode_graphs() already implies enable_cuda_graphs was set.
    const std::int32_t bucket = SelectBucket(decode_rows, executor_->decode_buckets());
    replayed_graph = bucket >= 0 && executor_->has_decode_graphs();
    DecodeOutput out =
        replayed_graph ? executor_->DecodeGraph(plan.decode) : executor_->DecodeEager(plan.decode);
    // Histogram attribution comes from the rows the step EXECUTED at (the
    // replayed bucket, or what pad_eager_to_bucket may have padded to), not
    // the live batch size.
    executed_bucket = out.bucket;
    sampled_tokens = std::move(out.tokens);
  }

  std::vector<StepResult> results = scheduler_->CompleteStep(plan, sampled_tokens);

  // Counters update only after the whole plan-execute-complete cycle
  // succeeded; an executor throw leaves them untouched.
  ++steps_;
  aborts_ += plan.aborted.size();
  prefill_tokens_ += static_cast<std::uint64_t>(prefill_rows);
  if (decode_rows > 0) {
    decode_tokens_ += static_cast<std::uint64_t>(decode_rows); // live rows produce tokens
    if (replayed_graph) {
      ++graph_replays_;
    } else {
      ++eager_decodes_;
    }
    if (executed_bucket >= 0) {
      const std::vector<std::int32_t>& buckets = executor_->decode_buckets();
      const auto it = std::find(buckets.begin(), buckets.end(), executed_bucket);
      if (it != buckets.end()) {
        ++bucket_decode_steps_[static_cast<std::size_t>(it - buckets.begin())];
      }
    }
  }

  // Drop finished requests from the live map (their blocks/reservation/slot
  // were already released by CompleteStep).
  for (const StepResult& result : results) {
    if (result.finished) {
      requests_.erase(result.request_id);
    }
  }
  return results;
}

bool Engine::HasCapacity() const {
  DebugAssertOwnerThread();
  // Pure queue-depth backpressure (docs/DESIGN.md section 10); KV admission
  // happens inside Step.
  return scheduler_->num_waiting() < kMaxWaitingRequests;
}

EngineStats Engine::Stats() const {
  DebugAssertOwnerThread();
  EngineStats stats;
  stats.steps = steps_;
  stats.prefill_tokens = prefill_tokens_;
  stats.decode_tokens = decode_tokens_;
  stats.num_waiting = scheduler_->num_waiting();
  stats.num_running = scheduler_->num_running();
  stats.free_kv_blocks = block_allocator_->free_blocks();
  stats.reserved_kv_blocks = block_allocator_->reserved_blocks();
  stats.total_kv_blocks = block_allocator_->total_blocks();
  stats.graph_replays = graph_replays_;
  stats.eager_decodes = eager_decodes_;
  stats.aborts = aborts_;
  const std::vector<std::int32_t>& buckets = executor_->decode_buckets();
  stats.bucket_histogram.reserve(buckets.size());
  for (std::size_t i = 0; i < buckets.size(); ++i) {
    stats.bucket_histogram.emplace_back(buckets[i], bucket_decode_steps_[i]);
  }
  return stats;
}

std::vector<BlockId> Engine::DebugBlockTable(RequestId request_id) const {
  DebugAssertOwnerThread();
  const auto it = requests_.find(request_id);
  if (it == requests_.end()) {
    return {};
  }
  return it->second->block_table.blocks();
}

std::string Engine::AlgoReportJson() const {
  DebugAssertOwnerThread();
  return executor_->AlgoReportJson();
}

void Engine::DebugAssertOwnerThread() const {
  assert(std::this_thread::get_id() == owner_thread_ &&
         "Engine is single-threaded by contract: AddRequest/Step/HasCapacity/Stats must be "
         "called from the thread that constructed the engine (docs/DESIGN.md section 2)");
}

} // namespace redline
