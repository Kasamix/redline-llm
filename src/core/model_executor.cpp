#include "core/model_executor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvToolsExt.h>

#include "core/scheduler.hpp" // MakeDecodeBuckets / SelectBucket
#include "kernels/kernels.cuh"
#include "loader/weights.hpp"

// Implementation notes (docs/DESIGN.md sections 5, 6.3, 9, 11):
//
// Allocation discipline - this file performs exactly THREE device
// allocations (KV pool, scratch slab, and the cuBLASLt workspace inside
// CublasLtGemm::Init) plus ONE pinned host allocation, all from the
// constructor path. Nothing below the init helpers calls
// cudaMalloc/cudaHostAlloc, so the steady state is allocation-free and
// graph-capture-legal by construction.
//
// Scratch slab layout (one cudaMalloc, regions 256-byte aligned; R =
// max(prefill_chunk_tokens, max_batch) rows, dev preset values in
// parentheses):
//
//   region        elements                              dev preset bytes
//   ------------  ------------------------------------  ----------------
//   x             R * hidden                     (FP16)     3 MiB
//   normed        R * hidden                     (FP16)     3 MiB
//   qkv_out       R * (q_heads + 2 kv_heads) * D (FP16)     4 MiB
//   attn_out      R * hidden                     (FP16)     3 MiB
//   gateup        R * 2 * intermediate           (FP16)    35 MiB
//   mlp_out       R * hidden                     (FP16)     3 MiB
//   logits        max_batch * vocab              (FP16)   2.3 MiB
//   scores        q_heads * chunk * max_seq_len  (FP32)    96 MiB
//   probs         q_heads * chunk * max_seq_len  (FP16)    48 MiB
//   khat          kv_heads * max_seq_len * D     (FP16)     1 MiB
//   vhat          kv_heads * max_seq_len * D     (FP16)     1 MiB
//   input_ids     R                              (i32)      4 KiB
//   positions     R                              (i32)      4 KiB
//   seq_lens      max_batch                      (i32)     32 B
//   block_tables  max_batch * max_blocks_per_seq (i32)      4 KiB
//   argmax_out    max_batch                      (i32)     32 B
//
// scores/probs are sized as capacity ([heads, chunk, max_seq_len]); the
// prefill composite packs them at the chunk's CURRENT kv extent
// (ldd = ctx, batch stride T*ctx - the section 6.3 descriptor recipes), so
// the region is a flat pool, not a strided 3-D tensor.
//
// Both device allocations are zero-filled once at init: the device block
// tables thereby start as all-dummy rows (kDummyBlockId == 0), and padded
// decode rows - whose attention CTA exits at seq_len == 0 without writing
// attn_out - read deterministic finite values instead of uninitialized
// memory on the first step. Padded-row garbage can never reach a live row
// (every stage is row-wise), but deterministic contents keep runs
// reproducible.

namespace redline {

namespace {

constexpr std::int64_t kSlabAlign = 256; // matches cudaMalloc's base guarantee

void CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw ExecutorError(std::string("model_executor: ") + what + ": " + cudaGetErrorName(err) +
                        ": " + cudaGetErrorString(err));
  }
}

// NVTX RAII wrapper for the docs/DESIGN.md section 14 taxonomy. Used only
// OUTSIDE any capturable sequence (docs/DESIGN.md section 9: no NVTX inside
// a captured region - ranges wrap the enqueue/replay calls instead).
class NvtxRange {
 public:
  explicit NvtxRange(const char* label) { nvtxRangePushA(label); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

std::int64_t AlignUp(std::int64_t bytes, std::int64_t align) {
  return (bytes + align - 1) / align * align;
}

// Byte offsets of every region inside the device scratch slab (table above).
struct ScratchLayout {
  std::int64_t x = 0;
  std::int64_t normed = 0;
  std::int64_t qkv = 0;
  std::int64_t attn_out = 0;
  std::int64_t gateup = 0;
  std::int64_t mlp_out = 0;
  std::int64_t logits = 0;
  std::int64_t scores = 0;
  std::int64_t probs = 0;
  std::int64_t khat = 0;
  std::int64_t vhat = 0;
  std::int64_t input_ids = 0;
  std::int64_t positions = 0;
  std::int64_t seq_lens = 0;
  std::int64_t block_tables = 0;
  std::int64_t argmax = 0;
  std::int64_t total_bytes = 0;
};

// Byte offsets inside the pinned host mirror slab.
struct PinnedLayout {
  std::int64_t input_ids = 0;
  std::int64_t positions = 0;
  std::int64_t seq_lens = 0;
  std::int64_t block_tables = 0;
  std::int64_t argmax = 0;
  std::int64_t total_bytes = 0;
};

// Sequential 256-byte-aligned packer shared by both slab plans.
class SlabPlanner {
 public:
  std::int64_t Take(std::int64_t bytes) {
    const std::int64_t at = offset_;
    offset_ += AlignUp(bytes, kSlabAlign);
    return at;
  }
  std::int64_t total() const { return offset_; }

 private:
  std::int64_t offset_ = 0;
};

ScratchLayout ComputeScratchLayout(const ModelConfig& config, const ExecutorOptions& options) {
  const std::int64_t rows = std::max(options.prefill_chunk_tokens, options.max_batch);
  const std::int64_t hidden = config.hidden_size;
  const std::int64_t qkv_width =
      (static_cast<std::int64_t>(config.num_q_heads) + 2 * config.num_kv_heads) * config.head_dim;
  const std::int64_t gateup_width = 2 * static_cast<std::int64_t>(config.intermediate_size);
  const std::int64_t score_elems = static_cast<std::int64_t>(config.num_q_heads) *
                                   options.prefill_chunk_tokens * options.max_seq_len;
  const std::int64_t gather_elems =
      static_cast<std::int64_t>(config.num_kv_heads) * options.max_seq_len * config.head_dim;
  const std::int64_t table_elems =
      static_cast<std::int64_t>(options.max_batch) * options.max_blocks_per_seq();

  SlabPlanner plan;
  ScratchLayout layout;
  layout.x = plan.Take(rows * hidden * 2);
  layout.normed = plan.Take(rows * hidden * 2);
  layout.qkv = plan.Take(rows * qkv_width * 2);
  layout.attn_out = plan.Take(rows * hidden * 2);
  layout.gateup = plan.Take(rows * gateup_width * 2);
  layout.mlp_out = plan.Take(rows * hidden * 2);
  layout.logits = plan.Take(static_cast<std::int64_t>(options.max_batch) * config.vocab_size * 2);
  layout.scores = plan.Take(score_elems * 4);
  layout.probs = plan.Take(score_elems * 2);
  layout.khat = plan.Take(gather_elems * 2);
  layout.vhat = plan.Take(gather_elems * 2);
  layout.input_ids = plan.Take(rows * 4);
  layout.positions = plan.Take(rows * 4);
  layout.seq_lens = plan.Take(static_cast<std::int64_t>(options.max_batch) * 4);
  layout.block_tables = plan.Take(table_elems * 4);
  layout.argmax = plan.Take(static_cast<std::int64_t>(options.max_batch) * 4);
  layout.total_bytes = plan.total();
  return layout;
}

PinnedLayout ComputePinnedLayout(const ModelConfig& /*config*/, const ExecutorOptions& options) {
  const std::int64_t rows = std::max(options.prefill_chunk_tokens, options.max_batch);
  const std::int64_t table_elems =
      static_cast<std::int64_t>(options.max_batch) * options.max_blocks_per_seq();

  SlabPlanner plan;
  PinnedLayout layout;
  layout.input_ids = plan.Take(rows * 4);
  layout.positions = plan.Take(rows * 4);
  layout.seq_lens = plan.Take(static_cast<std::int64_t>(options.max_batch) * 4);
  layout.block_tables = plan.Take(table_elems * 4);
  layout.argmax = plan.Take(static_cast<std::int64_t>(options.max_batch) * 4);
  layout.total_bytes = plan.total();
  return layout;
}

void ValidateGeometry(const ModelConfig& config, const ExecutorOptions& options) {
  const auto require = [](bool ok, const char* what) {
    if (!ok) {
      throw ExecutorError(std::string("model_executor: invalid configuration: ") + what);
    }
  };
  require(config.hidden_size > 0 && config.num_layers > 0 && config.num_q_heads > 0 &&
              config.num_kv_heads > 0 && config.head_dim > 0 && config.intermediate_size > 0 &&
              config.vocab_size > 0,
          "every ModelConfig dimension must be positive");
  require(config.num_q_heads * config.head_dim == config.hidden_size,
          "num_q_heads * head_dim must equal hidden_size");
  require(config.num_q_heads % config.num_kv_heads == 0,
          "num_q_heads must be a multiple of num_kv_heads (grouped-query attention)");
  require(options.max_batch >= 1, "max_batch must be >= 1");
  require(options.max_seq_len >= 1, "max_seq_len must be >= 1");
  require(options.prefill_chunk_tokens >= 1, "prefill_chunk_tokens must be >= 1");
  require(options.pad_eager_to_bucket >= 0, "pad_eager_to_bucket must be >= 0");
  require(options.kv_pool_bytes > 0, "kv_pool_bytes must be positive");
}

template <typename T> T* SlabAt(void* base, std::int64_t byte_offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + byte_offset);
}

} // namespace

const char* ToString(PrefillGemmMode mode) {
  switch (mode) {
  case PrefillGemmMode::kStridedBatched:
    return "strided_batched";
  case PrefillGemmMode::kPerHead:
    return "per_head";
  }
  return "unknown";
}

ExecutorMemoryBudget ModelExecutor::ComputeBudget(const ModelConfig& config,
                                                  const ExecutorOptions& options) {
  ValidateGeometry(config, options);

  // FP16 bytes of one 16-token block across ALL layers (458,752 B for
  // Qwen2.5-1.5B: 16 tokens * 28,672 B/token - docs/DESIGN.md section 1).
  const std::int64_t block_bytes = kKvBlockSize * config.KvBytesPerToken();
  const std::int64_t num_blocks = options.kv_pool_bytes / block_bytes;
  if (num_blocks < 2) {
    throw ExecutorError("model_executor: kv_pool_bytes " + std::to_string(options.kv_pool_bytes) +
                        " holds fewer than 2 KV blocks (block = " + std::to_string(block_bytes) +
                        " B); block 0 is the reserved dummy block, so at least 2 are required");
  }

  ExecutorMemoryBudget budget;
  budget.num_kv_blocks = static_cast<std::int32_t>(num_blocks);
  budget.kv_pool_bytes = num_blocks * block_bytes;
  budget.scratch_bytes = ComputeScratchLayout(config, options).total_bytes;
  budget.workspace_bytes = static_cast<std::int64_t>(CublasLtGemm::kDefaultWorkspaceBytes);
  budget.total_device_bytes = budget.kv_pool_bytes + budget.scratch_bytes + budget.workspace_bytes;
  budget.pinned_host_bytes = ComputePinnedLayout(config, options).total_bytes;
  return budget;
}

ModelExecutor::ModelExecutor(const ModelConfig& config, const DeviceWeights& weights,
                             ExecutorOptions options)
    : config_(config), weights_(&weights), options_(options) {
  budget_ = ComputeBudget(config_, options_); // validates config + options
  if (weights.num_layers() != config_.num_layers) {
    throw ExecutorError("model_executor: DeviceWeights has " +
                        std::to_string(weights.num_layers()) + " layers but the config declares " +
                        std::to_string(config_.num_layers));
  }

  buckets_ = MakeDecodeBuckets(options_.max_batch);
  max_step_rows_ = std::max(options_.prefill_chunk_tokens, options_.max_batch);
  max_blocks_per_seq_ = options_.max_blocks_per_seq();
  num_kv_blocks_ = budget_.num_kv_blocks;
  pool_layer_stride_elems_ = static_cast<std::int64_t>(num_kv_blocks_) * 2 * config_.num_kv_heads *
                             kKvBlockSize * config_.head_dim;
  qkv_width_ = (config_.num_q_heads + 2 * config_.num_kv_heads) * config_.head_dim;
  k_view_offset_ = config_.num_q_heads * config_.head_dim;
  v_view_offset_ = k_view_offset_ + config_.num_kv_heads * config_.head_dim;
  gateup_width_ = 2 * config_.intermediate_size;
  heads_per_group_ = config_.num_q_heads / config_.num_kv_heads; // divisibility validated
  gather_plane_stride_ = static_cast<std::int64_t>(options_.max_seq_len) * config_.head_dim;
  attn_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
  // Layer-0 parity dump (docs/DESIGN.md section 12f): armed only when a dump
  // directory is configured; the first prefill chunk disarms it.
  debug_dump_pending_ = !options_.debug_dump_dir.empty();

  try {
    CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream s0");
    gemm_.Init(); // handle + the shared 32 MiB workspace + shapes self-check
    AllocateDeviceMemory();
    AllocatePinnedMirrors();
    ProbeDecodeGemmConfigs();
    ProbePrefillGemmConfigs();
    if (options_.enable_cuda_graphs) {
      // docs/DESIGN.md section 9: warmup + capture at init, after model
      // load. Degrades to eager on any failure (one-time warning inside);
      // only a truly unexpected exception reaches the release path below.
      CaptureDecodeGraphs();
    }
  } catch (...) {
    ReleaseAll();
    throw;
  }

  std::fprintf(stderr,
               "[redline] executor init: %d KV blocks (%.1f MiB pool), scratch %.1f MiB, "
               "%zu decode buckets (max %d), qkv bias epilogue available, "
               "prefill attention GEMMs %s\n",
               num_kv_blocks_, static_cast<double>(budget_.kv_pool_bytes) / (1 << 20),
               static_cast<double>(budget_.scratch_bytes) / (1 << 20), buckets_.size(),
               buckets_.empty() ? 0 : buckets_.back(), ToString(prefill_gemm_mode_));
}

ModelExecutor::~ModelExecutor() {
  ReleaseAll();
}

void ModelExecutor::AllocateDeviceMemory() {
  // 1/2: the single layer-outer paged KV pool (docs/DESIGN.md section 5).
  void* pool = nullptr;
  CudaCheck(cudaMalloc(&pool, static_cast<std::size_t>(budget_.kv_pool_bytes)),
            "cudaMalloc KV pool");
  kv_pool_ = static_cast<half*>(pool);

  // 2/2: the activation + per-step-input scratch slab (layout table above).
  const ScratchLayout layout = ComputeScratchLayout(config_, options_);
  CudaCheck(cudaMalloc(&scratch_slab_, static_cast<std::size_t>(layout.total_bytes)),
            "cudaMalloc scratch slab");

  d_x_ = SlabAt<half>(scratch_slab_, layout.x);
  d_normed_ = SlabAt<half>(scratch_slab_, layout.normed);
  d_qkv_ = SlabAt<half>(scratch_slab_, layout.qkv);
  d_attn_out_ = SlabAt<half>(scratch_slab_, layout.attn_out);
  d_gateup_ = SlabAt<half>(scratch_slab_, layout.gateup);
  d_mlp_out_ = SlabAt<half>(scratch_slab_, layout.mlp_out);
  d_logits_ = SlabAt<half>(scratch_slab_, layout.logits);
  d_scores_ = SlabAt<float>(scratch_slab_, layout.scores);
  d_probs_ = SlabAt<half>(scratch_slab_, layout.probs);
  d_khat_ = SlabAt<half>(scratch_slab_, layout.khat);
  d_vhat_ = SlabAt<half>(scratch_slab_, layout.vhat);
  d_input_ids_ = SlabAt<std::int32_t>(scratch_slab_, layout.input_ids);
  d_positions_ = SlabAt<std::int32_t>(scratch_slab_, layout.positions);
  d_seq_lens_ = SlabAt<std::int32_t>(scratch_slab_, layout.seq_lens);
  d_block_tables_ = SlabAt<std::int32_t>(scratch_slab_, layout.block_tables);
  d_argmax_ = SlabAt<std::int32_t>(scratch_slab_, layout.argmax);

  // Zero-fill once (rationale in the file header). Zeroed block tables are
  // all-dummy rows because kDummyBlockId == 0.
  CudaCheck(cudaMemsetAsync(kv_pool_, 0, static_cast<std::size_t>(budget_.kv_pool_bytes), stream_),
            "zero KV pool");
  CudaCheck(
      cudaMemsetAsync(scratch_slab_, 0, static_cast<std::size_t>(layout.total_bytes), stream_),
      "zero scratch slab");
  CudaCheck(cudaStreamSynchronize(stream_), "drain init memsets");
}

void ModelExecutor::AllocatePinnedMirrors() {
  const PinnedLayout layout = ComputePinnedLayout(config_, options_);
  CudaCheck(cudaHostAlloc(&pinned_slab_, static_cast<std::size_t>(layout.total_bytes),
                          cudaHostAllocDefault),
            "cudaHostAlloc mirror slab");
  std::memset(pinned_slab_, 0, static_cast<std::size_t>(layout.total_bytes));

  h_input_ids_ = SlabAt<std::int32_t>(pinned_slab_, layout.input_ids);
  h_positions_ = SlabAt<std::int32_t>(pinned_slab_, layout.positions);
  h_seq_lens_ = SlabAt<std::int32_t>(pinned_slab_, layout.seq_lens);
  h_block_tables_ = SlabAt<std::int32_t>(pinned_slab_, layout.block_tables);
  h_argmax_ = SlabAt<std::int32_t>(pinned_slab_, layout.argmax);
}

void ModelExecutor::ProbeDecodeGemmConfigs() {
  // Every decode-path (shape, epilogue) combination is probed - and its plan
  // cached - at init, per bucket plus max_batch (an off-ladder max_batch
  // still runs eagerly unpadded). This keeps heuristic queries out of the
  // step path for every bucketed shape, satisfies the query-once-at-init
  // rule of docs/DESIGN.md section 6.1, and records availability in the algo
  // report BEFORE the first step. Off-ladder eager shapes (live batches
  // between buckets when padding is off) fill the plan cache lazily on first
  // use, which is legal outside graph capture.
  std::vector<std::int32_t> row_shapes = buckets_;
  if (row_shapes.empty() || row_shapes.back() != options_.max_batch) {
    row_shapes.push_back(options_.max_batch);
  }

  const std::int32_t hidden = config_.hidden_size;
  std::string unavailable;
  for (const std::int32_t rows : row_shapes) {
    const std::string prefix = "decode[b=" + std::to_string(rows) + "]:";
    const struct {
      const char* name;
      GemmConfig config;
    } probes[] = {
        {"qkv_bias",
         GemmConfig::Linear(rows, qkv_width_, hidden, hidden, qkv_width_, GemmEpilogue::kBias)},
        {"o", GemmConfig::Linear(rows, hidden, hidden, hidden, hidden)},
        {"gateup", GemmConfig::Linear(rows, gateup_width_, hidden, hidden, gateup_width_)},
        {"down",
         GemmConfig::Linear(rows, hidden, config_.intermediate_size, gateup_width_, hidden)},
        {"lm_head",
         GemmConfig::Linear(rows, config_.vocab_size, hidden, hidden, config_.vocab_size)},
    };
    for (const auto& probe : probes) {
      if (!gemm_.Probe(probe.config, prefix + probe.name)) {
        unavailable += (unavailable.empty() ? "" : ", ") + prefix + probe.name;
      }
    }
  }

  if (!unavailable.empty()) {
    // The documented fallbacks (docs/DESIGN.md section 16: qkv bias folded
    // into a following kernel; plain shapes have none) are not wired in v1,
    // so an unavailable decode configuration is fatal at init - with the
    // probe decision recorded in AlgoReportJson - rather than mid-serving.
    throw ExecutorError(
        "model_executor: cuBLASLt offers no algorithm for decode GEMM configuration(s) [" +
        unavailable + "] on this device; see AlgoReportJson() and docs/DESIGN.md section 16");
  }
}

void ModelExecutor::ProbePrefillGemmConfigs() {
  // Representative shape: the largest chunk any request can produce (T =
  // prefill_chunk_tokens, capped by max_seq_len) attending to the largest
  // possible context. Full chunks - every chunk of a prompt longer than the
  // chunk size - hit these exact linear plans; partial final chunks run at
  // other row counts and fill the plan cache lazily on the eager path
  // (run-path-legal per the wrapper contract; prefill is never captured).
  const std::int32_t rep_rows = std::min(options_.prefill_chunk_tokens, options_.max_seq_len);
  const std::int32_t rep_ctx = options_.max_seq_len;
  const std::int32_t hidden = config_.hidden_size;
  const std::int32_t head_dim = config_.head_dim;

  // (a) The linear-layer GEMMs at the full-chunk row count. Same fatality
  // policy and rationale as ProbeDecodeGemmConfigs: no fallback is wired, so
  // unavailability surfaces at init instead of on the first long prompt.
  // lm_head is deliberately absent - prefill computes logits at rows = 1
  // only (docs/DESIGN.md section 6.3), a configuration ProbeDecodeGemmConfigs
  // already covered via bucket 1.
  const std::string prefix = "prefill[T=" + std::to_string(rep_rows) + "]:";
  const struct {
    const char* name;
    GemmConfig config;
  } probes[] = {
      {"qkv_bias",
       GemmConfig::Linear(rep_rows, qkv_width_, hidden, hidden, qkv_width_, GemmEpilogue::kBias)},
      {"o", GemmConfig::Linear(rep_rows, hidden, hidden, hidden, hidden)},
      {"gateup", GemmConfig::Linear(rep_rows, gateup_width_, hidden, hidden, gateup_width_)},
      {"down",
       GemmConfig::Linear(rep_rows, hidden, config_.intermediate_size, gateup_width_, hidden)},
  };
  std::string unavailable;
  for (const auto& probe : probes) {
    if (!gemm_.Probe(probe.config, prefix + probe.name)) {
      unavailable += (unavailable.empty() ? "" : ", ") + prefix + probe.name;
    }
  }
  if (!unavailable.empty()) {
    throw ExecutorError(
        "model_executor: cuBLASLt offers no algorithm for prefill GEMM configuration(s) [" +
        unavailable + "] on this device; see AlgoReportJson() and docs/DESIGN.md section 16");
  }

  // (b) The scores/PV attention products, probed in BOTH fallback-chain
  // forms regardless of the outcome so the algo report records the per-arch
  // availability of each (docs/DESIGN.md section 16 "decision is probed and
  // recorded"). The selected form seeds ResolvePrefillGemmMode.
  const bool batched_scores = gemm_.Probe(
      GemmConfig::PrefillScores(rep_rows, rep_ctx, head_dim, heads_per_group_, qkv_width_),
      "prefill:scores_strided_batched");
  const bool batched_pv =
      gemm_.Probe(GemmConfig::PrefillPv(rep_rows, rep_ctx, head_dim, heads_per_group_, hidden),
                  "prefill:pv_strided_batched");
  const bool per_head_scores =
      gemm_.Probe(GemmConfig::PrefillScores(rep_rows, rep_ctx, head_dim, 1, qkv_width_),
                  "prefill:scores_per_head");
  const bool per_head_pv = gemm_.Probe(
      GemmConfig::PrefillPv(rep_rows, rep_ctx, head_dim, 1, hidden), "prefill:pv_per_head");

  if (batched_scores && batched_pv) {
    prefill_gemm_mode_ = PrefillGemmMode::kStridedBatched;
  } else if (per_head_scores && per_head_pv) {
    prefill_gemm_mode_ = PrefillGemmMode::kPerHead;
    std::fprintf(stderr,
                 "[redline] executor init: strided-batched prefill attention GEMMs unavailable "
                 "at T=%d ctx=%d (scores=%d, pv=%d); using per-head calls "
                 "(docs/DESIGN.md section 16 fallback chain)\n",
                 rep_rows, rep_ctx, batched_scores ? 1 : 0, batched_pv ? 1 : 0);
  } else {
    throw ExecutorError(
        "model_executor: cuBLASLt offers no algorithm for the prefill attention GEMMs at T=" +
        std::to_string(rep_rows) + ", ctx=" + std::to_string(rep_ctx) +
        " in either the strided-batched or the per-head form; the third fallback stage of "
        "docs/DESIGN.md section 16 (explicit repack kernels) is not implemented in v1 - "
        "see AlgoReportJson() for the probe record");
  }
}

void ModelExecutor::FillDummyDecodeMirrors(std::int32_t rows) {
  // Dummy-but-VALID warmup/capture inputs (docs/DESIGN.md section 9): token
  // 0 (a real embedding row), position 0, seq_len 1 - so the attention CTA
  // walks one real KV tile instead of exiting - and all-dummy block tables,
  // so this step's kv_scatter lands in the sacrificial dummy block. Every
  // row writes the same dummy-block slot; the racing values are finite and
  // the dummy block is never read by a live sequence (padded rows exit at
  // seq_len == 0 before touching the pool).
  for (std::int32_t r = 0; r < rows; ++r) {
    h_input_ids_[r] = 0;
    h_positions_[r] = 0;
    h_seq_lens_[r] = 1;
    std::int32_t* table_row = h_block_tables_ + static_cast<std::int64_t>(r) * max_blocks_per_seq_;
    std::fill(table_row, table_row + max_blocks_per_seq_, kDummyBlockId);
  }
}

void ModelExecutor::WarmupDecodeBucket(std::int32_t rows) {
  // One complete eager decode step at the bucket shape, drained. This is
  // the docs/DESIGN.md section 9 pre-capture warmup: it executes (not just
  // plans) every cuBLASLt matmul of the step - binding the workspace and
  // loading the Lt kernel images - plus every custom kernel (module load
  // under lazy loading) and the RoPE inv-freq constant-table upload, so the
  // capture below records the sequence without any first-use side effects.
  FillDummyDecodeMirrors(rows);
  UploadDecodeInputs(rows);
  EnqueueDecodeForward(rows);
  EnqueueArgmax(rows);
  CudaCheck(cudaMemcpyAsync(h_argmax_, d_argmax_,
                            static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, stream_),
            "warmup D2H argmax_out");
  CudaCheck(cudaStreamSynchronize(stream_), "warmup decode step sync");
  CudaCheck(cudaGetLastError(), "warmup decode step launches");
}

void ModelExecutor::CaptureDecodeGraphs() {
  // docs/DESIGN.md section 9 capture procedure, at init after model load.
  // Phase 1 warms up EVERY bucket eagerly before ANY capture. Phase 2
  // captures the byte-stable EnqueueDecodeForward + EnqueueArgmax sequence
  // per bucket on stream_ (ThreadLocal mode), instantiates, and uploads the
  // executable so no launch state materializes lazily on the serving path.
  // The H2D input copies and the argmax D2H stay OUTSIDE the graph - they
  // are re-issued per replay (section 9 replay procedure).
  //
  // ANY failure in either phase degrades to eager with a one-time warning
  // and never aborts init (section 9): the eager path is permanently
  // maintained, and an error that also breaks eager resurfaces on the first
  // real step exactly as it would with graphs disabled.
  assert(graph_execs_.empty() && "decode graphs are captured exactly once, at init");

  std::size_t free_before = 0;
  std::size_t free_after = 0;
  std::size_t total = 0;
  std::vector<cudaGraphExec_t> execs;
  execs.reserve(buckets_.size());
  bool capturing = false;
  const auto degrade = [&](const char* what) {
    if (capturing) {
      // A capture region is still open (the failure struck between Begin
      // and EndCapture): terminate it so the stream leaves capture mode.
      cudaGraph_t dead = nullptr;
      cudaStreamEndCapture(stream_, &dead);
      if (dead != nullptr) {
        cudaGraphDestroy(dead);
      }
    }
    for (cudaGraphExec_t exec : execs) {
      cudaGraphExecDestroy(exec);
    }
    cudaGetLastError();             // clear any sticky launch status
    cudaStreamSynchronize(stream_); // best effort: leave s0 quiet for eager
    std::fprintf(stderr,
                 "[redline] warning: decode CUDA-graph capture failed - running every decode "
                 "step eagerly (docs/DESIGN.md section 9 degrade rule): %s\n",
                 what);
  };

  try {
    for (const std::int32_t bucket : buckets_) {
      WarmupDecodeBucket(bucket);
    }
    // Graph-memory accounting starts AFTER warmup so kernel-module loads are
    // not attributed to the graphs (the section 11 budget row is graphs
    // only: 16 MiB per bucket).
    CudaCheck(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo before capture");

    for (const std::int32_t bucket : buckets_) {
      // Refresh the dummy inputs at this bucket's shape (section 9 step 2).
      // The captured work reads persistent device buffers, so contents do
      // not shape the graph; refreshing them keeps instantiation-time
      // validation on defined values of the right extent.
      FillDummyDecodeMirrors(bucket);
      UploadDecodeInputs(bucket);
      CudaCheck(cudaStreamSynchronize(stream_), "drain dummy inputs before capture");

      CudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture");
      capturing = true;
      EnqueueDecodeForward(bucket);
      EnqueueArgmax(bucket);
      cudaGraph_t graph = nullptr;
      const cudaError_t end_status = cudaStreamEndCapture(stream_, &graph);
      capturing = false;
      CudaCheck(end_status, "cudaStreamEndCapture");

      cudaGraphExec_t exec = nullptr;
      const cudaError_t instantiate_status = cudaGraphInstantiate(&exec, graph, 0);
      cudaGraphDestroy(graph); // the template is not needed once instantiated
      CudaCheck(instantiate_status, "cudaGraphInstantiate");
      execs.push_back(exec);

      CudaCheck(cudaGraphUpload(exec, stream_), "cudaGraphUpload");
      CudaCheck(cudaStreamSynchronize(stream_), "drain graph upload");
    }

    CudaCheck(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo after capture");
  } catch (const std::exception& e) {
    degrade(e.what());
    return;
  } catch (...) {
    degrade("unknown exception");
    return;
  }

  graph_execs_ = std::move(execs);

  // The init-time graph-memory log line (docs/DESIGN.md sections 9, 11):
  // the cudaMemGetInfo delta across capture + instantiate + upload of every
  // bucket, to be read against the section 11 graph budget. On a shared
  // (display-attached / WDDM-virtualized) GPU, other processes can move
  // free memory between the two samples, so the raw before/after values are
  // logged alongside the delta.
  const std::int64_t delta_bytes =
      static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
  const double delta_mib = static_cast<double>(delta_bytes) / static_cast<double>(1 << 20);
  std::string bucket_list;
  for (const std::int32_t bucket : buckets_) {
    bucket_list += (bucket_list.empty() ? "" : ",") + std::to_string(bucket);
  }
  std::fprintf(stderr,
               "[redline] decode CUDA graphs: %zu buckets captured {%s}; graph memory delta "
               "%+.1f MiB (cudaMemGetInfo free %.1f -> %.1f MiB across capture)\n",
               graph_execs_.size(), bucket_list.c_str(), delta_mib,
               static_cast<double>(free_before) / (1 << 20),
               static_cast<double>(free_after) / (1 << 20));
}

void ModelExecutor::ReleaseAll() noexcept {
  // Graph executables go first: they were built from (and launch onto) the
  // stream destroyed below. Every step ends with a stream sync, so no
  // executable can still be running here.
  for (cudaGraphExec_t exec : graph_execs_) {
    if (exec != nullptr) {
      cudaGraphExecDestroy(exec);
    }
  }
  graph_execs_.clear();
  if (scratch_slab_ != nullptr) {
    cudaFree(scratch_slab_);
    scratch_slab_ = nullptr;
  }
  if (kv_pool_ != nullptr) {
    cudaFree(kv_pool_);
    kv_pool_ = nullptr;
  }
  if (pinned_slab_ != nullptr) {
    cudaFreeHost(pinned_slab_);
    pinned_slab_ = nullptr;
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  d_x_ = d_normed_ = d_qkv_ = d_attn_out_ = d_gateup_ = d_mlp_out_ = d_logits_ = nullptr;
  d_probs_ = d_khat_ = d_vhat_ = nullptr;
  d_scores_ = nullptr;
  d_input_ids_ = d_positions_ = d_seq_lens_ = d_block_tables_ = d_argmax_ = nullptr;
  h_input_ids_ = h_positions_ = h_seq_lens_ = h_block_tables_ = h_argmax_ = nullptr;
}

std::int32_t ModelExecutor::BucketForRows(std::int32_t rows) const {
  for (const std::int32_t bucket : buckets_) {
    if (bucket == rows) {
      return bucket;
    }
  }
  return -1;
}

DecodeOutput ModelExecutor::DecodeEager(const std::vector<RequestPtr>& batch) {
  DecodeOutput output;
  if (batch.empty()) {
    return output;
  }
  if (static_cast<std::int32_t>(batch.size()) > options_.max_batch) {
    throw ExecutorError("model_executor: decode batch of " + std::to_string(batch.size()) +
                        " exceeds max_batch " + std::to_string(options_.max_batch));
  }

  const std::int32_t rows = FillDecodeMirrors(batch, /*pad_to_rows=*/0);

  {
    NvtxRange range("h2d_inputs");
    UploadDecodeInputs(rows);
  }
  {
    char label[32];
    std::snprintf(label, sizeof(label), "decode[b=%d]", rows);
    NvtxRange range(label);
    EnqueueDecodeForward(rows);
  }
  {
    NvtxRange range("sample");
    EnqueueArgmax(rows);
  }
  {
    NvtxRange range("d2h_tokens");
    CudaCheck(cudaMemcpyAsync(h_argmax_, d_argmax_,
                              static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H argmax_out");
    // The step boundary (docs/DESIGN.md section 9): the scheduler needs the
    // sampled tokens on host before it can plan the next iteration.
    CudaCheck(cudaStreamSynchronize(stream_), "decode step sync");
    CudaCheck(cudaGetLastError(), "decode step launches");
  }

  output.rows = rows;
  output.bucket = BucketForRows(rows);
  output.tokens.assign(h_argmax_, h_argmax_ + batch.size());
  return output;
}

DecodeOutput ModelExecutor::DecodeGraph(const std::vector<RequestPtr>& batch) {
  // The docs/DESIGN.md section 9 replay procedure: identical to DecodeEager
  // except the batch is padded to its covering bucket and the captured
  // executable replaces the EnqueueDecodeForward + EnqueueArgmax enqueue
  // calls. The engine dispatches here only when has_decode_graphs() and a
  // bucket covers the live batch; both are hard-checked anyway because a
  // wrong dispatch would otherwise launch a graph at the wrong shape.
  DecodeOutput output;
  if (batch.empty()) {
    return output;
  }
  if (!has_decode_graphs()) {
    throw ExecutorError("model_executor: DecodeGraph called with no captured decode graphs "
                        "(enable_cuda_graphs off, or capture degraded to eager at init)");
  }
  const std::int32_t live = static_cast<std::int32_t>(batch.size());
  if (live > options_.max_batch) {
    throw ExecutorError("model_executor: decode batch of " + std::to_string(batch.size()) +
                        " exceeds max_batch " + std::to_string(options_.max_batch));
  }
  const std::int32_t bucket = SelectBucket(live, buckets_);
  if (bucket < 0) {
    throw ExecutorError("model_executor: no decode bucket covers a batch of " +
                        std::to_string(batch.size()) +
                        "; the engine must run this step eagerly (docs/DESIGN.md section 9)");
  }
  const std::size_t bucket_index = static_cast<std::size_t>(
      std::find(buckets_.begin(), buckets_.end(), bucket) - buckets_.begin());

  const std::int32_t rows = FillDecodeMirrors(batch, /*pad_to_rows=*/bucket);
  assert(rows == bucket && "graph replay must execute at exactly the captured bucket shape");

  {
    NvtxRange range("h2d_inputs");
    UploadDecodeInputs(rows);
  }
  {
    // NVTX wraps the replay CALL - never the captured region itself
    // (docs/DESIGN.md section 9). The graph contains the forward sequence
    // plus argmax, so no separate `sample` range exists on this path.
    char label[32];
    std::snprintf(label, sizeof(label), "graph_replay[b=%d]", rows);
    NvtxRange range(label);
    CudaCheck(cudaGraphLaunch(graph_execs_[bucket_index], stream_), "cudaGraphLaunch decode");
  }
  {
    NvtxRange range("d2h_tokens");
    CudaCheck(cudaMemcpyAsync(h_argmax_, d_argmax_,
                              static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H argmax_out");
    // The step boundary (docs/DESIGN.md section 9), exactly as in
    // DecodeEager: tokens must be on host before the next plan.
    CudaCheck(cudaStreamSynchronize(stream_), "graph replay sync");
    CudaCheck(cudaGetLastError(), "graph replay launches");
  }

  output.rows = rows;
  output.bucket = bucket;
  output.tokens.assign(h_argmax_, h_argmax_ + batch.size());
  return output;
}

std::int32_t ModelExecutor::FillDecodeMirrors(const std::vector<RequestPtr>& batch,
                                              std::int32_t pad_to_rows) {
  const std::int32_t live = static_cast<std::int32_t>(batch.size());

  // Row-count selection. pad_to_rows > 0 is the graph-replay path: pad to
  // exactly the caller's bucket (>= live by the SelectBucket contract;
  // asserted - not user-reachable). Otherwise eager sizing applies: live
  // rows, plus the pad_eager_to_bucket debug option (header doc) - smallest
  // configured bucket >= max(live, requested), requests clamped to the
  // largest bucket; no covering bucket -> run unpadded at live size.
  std::int32_t rows = live;
  if (pad_to_rows > 0) {
    assert(pad_to_rows >= live && pad_to_rows <= options_.max_batch &&
           "pad_to_rows must cover the live batch within max_batch");
    rows = pad_to_rows;
  } else if (options_.pad_eager_to_bucket > 0 && !buckets_.empty()) {
    const std::int32_t want =
        std::max(live, std::min(options_.pad_eager_to_bucket, buckets_.back()));
    const std::int32_t bucket = SelectBucket(want, buckets_);
    if (bucket > 0) {
      rows = bucket;
    }
  }

  for (std::int32_t r = 0; r < live; ++r) {
    const Request& request = *batch[r];
    const auto fail = [&](const std::string& what) {
      throw ExecutorError("model_executor: decode row " + std::to_string(r) + " (request " +
                          std::to_string(request.id) + "): " + what);
    };

    // A decoding sequence has produced at least its first token (emitted by
    // the final prefill chunk); the input token of this step is the last
    // APPENDED token (the forced token under teacher forcing).
    if (request.output_tokens.empty()) {
      fail("decode scheduled before the first generated token exists");
    }
    const std::int32_t seq_len = request.seq_len();
    if (seq_len < 1 || seq_len > options_.max_seq_len) {
      fail("seq_len " + std::to_string(seq_len) + " outside [1, max_seq_len " +
           std::to_string(options_.max_seq_len) + "]");
    }
    const TokenId token = request.output_tokens.back();
    if (token < 0 || token >= config_.vocab_size) {
      fail("input token " + std::to_string(token) + " outside the vocab");
    }
    const std::vector<BlockId>& blocks = request.block_table.blocks();
    const std::int32_t needed = BlocksForTokens(seq_len);
    if (static_cast<std::int32_t>(blocks.size()) < needed) {
      fail("block table holds " + std::to_string(blocks.size()) + " blocks but position " +
           std::to_string(seq_len - 1) + " needs " + std::to_string(needed));
    }
    if (static_cast<std::int32_t>(blocks.size()) > max_blocks_per_seq_) {
      fail("block table exceeds max_blocks_per_seq " + std::to_string(max_blocks_per_seq_));
    }

    h_input_ids_[r] = token;
    h_positions_[r] = seq_len - 1; // the position being scattered this step
    h_seq_lens_[r] = seq_len;      // context length including that token

    std::int32_t* table_row = h_block_tables_ + static_cast<std::int64_t>(r) * max_blocks_per_seq_;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      const BlockId block = blocks[i];
      // Sentinel policy (docs/DESIGN.md section 9): kInvalidBlockId is a
      // host-only debug sentinel and must never reach a device table.
      assert(block != kInvalidBlockId &&
             "kInvalidBlockId must never be written into a device block table");
      // A live sequence can never own the dummy block (the allocator never
      // hands out block 0), and every id must address the pool.
      if (block <= kDummyBlockId || block >= num_kv_blocks_) {
        fail("block table entry " + std::to_string(i) + " holds invalid block id " +
             std::to_string(block));
      }
      table_row[i] = block;
    }
    std::fill(table_row + blocks.size(), table_row + max_blocks_per_seq_, kDummyBlockId);
  }

  // Padded rows (docs/DESIGN.md section 9): token 0, position 0, seq_len 0
  // (every kernel that consumes rows either exits on seq_len == 0 or
  // produces isolated row-local values), block-table rows all dummy.
  for (std::int32_t r = live; r < rows; ++r) {
    h_input_ids_[r] = 0;
    h_positions_[r] = 0;
    h_seq_lens_[r] = 0;
    std::int32_t* table_row = h_block_tables_ + static_cast<std::int64_t>(r) * max_blocks_per_seq_;
    std::fill(table_row, table_row + max_blocks_per_seq_, kDummyBlockId);
  }

  return rows;
}

void ModelExecutor::UploadDecodeInputs(std::int32_t rows) {
  // Exactly FOUR contiguous H2D copies from pinned mirrors, with the
  // block-table slab as ONE [rows, max_blocks_per_seq] copy - never per-row
  // copies (docs/DESIGN.md section 9 upload rule).
  const std::size_t row_bytes = static_cast<std::size_t>(rows) * sizeof(std::int32_t);
  CudaCheck(cudaMemcpyAsync(d_input_ids_, h_input_ids_, row_bytes, cudaMemcpyHostToDevice, stream_),
            "H2D input_ids");
  CudaCheck(cudaMemcpyAsync(d_positions_, h_positions_, row_bytes, cudaMemcpyHostToDevice, stream_),
            "H2D positions");
  CudaCheck(cudaMemcpyAsync(d_seq_lens_, h_seq_lens_, row_bytes, cudaMemcpyHostToDevice, stream_),
            "H2D seq_lens");
  CudaCheck(
      cudaMemcpyAsync(d_block_tables_, h_block_tables_,
                      static_cast<std::size_t>(rows) * max_blocks_per_seq_ * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice, stream_),
      "H2D block-table slab");
}

void ModelExecutor::EnqueueDecodeForward(std::int32_t rows) {
  // The exact docs/DESIGN.md section 6.3 decode sequence. Every argument is
  // a pointer into a persistent buffer and every activation view carries its
  // explicit row stride (section 5 strided-view rule): Q/K/V are read in
  // place from qkv_out at +0 / +k_view_offset_ / +v_view_offset_ with row
  // stride qkv_width_ - no split/repack kernel exists. No NVTX and no host
  // work in here: the graph path captures this sequence verbatim.
  const DeviceWeights& w = *weights_;
  const std::int32_t hidden = config_.hidden_size;
  const std::int64_t qkv_stride = qkv_width_;
  const float eps = config_.rms_norm_eps;

  kernels::LaunchEmbedGather(d_x_, hidden, w.embed(), d_input_ids_, rows, hidden, stream_);

  for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
    // Input norm: layer 0 reads the fresh embedding residual; later layers
    // first fold the previous layer's block output into the residual.
    if (layer == 0) {
      kernels::LaunchRmsNorm(d_normed_, d_x_, w.input_norm(0), rows, hidden, eps, stream_);
    } else {
      kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.input_norm(layer), rows, hidden,
                                     eps, stream_);
    }

    // Attention: fused QKV projection (per-output-feature bias epilogue),
    // RoPE in place on the Q/K views, scatter this step's K/V into the
    // layer's pool slice, paged GQA attention over the block table, then the
    // output projection into the block-output buffer.
    gemm_.GemmBiasFp16(d_qkv_, qkv_stride, d_normed_, hidden, w.w_qkv(layer), w.b_qkv(layer), rows,
                       qkv_width_, hidden, stream_);
    kernels::LaunchRopeInplace(d_qkv_, d_qkv_ + k_view_offset_, qkv_stride, qkv_stride,
                               d_positions_, rows, config_.num_q_heads, config_.num_kv_heads,
                               config_.head_dim, config_.rope_theta, stream_);
    kernels::LaunchKvScatter(PoolLayerSlice(layer), d_qkv_ + k_view_offset_,
                             d_qkv_ + v_view_offset_, qkv_stride, qkv_stride, d_block_tables_,
                             max_blocks_per_seq_, d_positions_, rows, config_.num_kv_heads,
                             config_.head_dim, stream_);
    kernels::LaunchPagedAttentionDecode(
        d_attn_out_, hidden, d_qkv_, qkv_stride, PoolLayerSlice(layer), d_block_tables_,
        d_seq_lens_, rows, max_blocks_per_seq_, config_.num_q_heads, config_.num_kv_heads,
        config_.head_dim, attn_scale_, stream_);
    gemm_.GemmFp16(d_mlp_out_, hidden, d_attn_out_, hidden, w.w_o(layer), rows, hidden, hidden,
                   stream_);

    // MLP: fold the attention block output into the residual, project to the
    // fused gate|up buffer, SwiGLU in place over the gate half, then the
    // down projection reads that [rows, intermediate] slice with the full
    // gate|up row stride (docs/DESIGN.md section 6.2 #8).
    kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.post_attn_norm(layer), rows,
                                   hidden, eps, stream_);
    gemm_.GemmFp16(d_gateup_, gateup_width_, d_normed_, hidden, w.w_gateup(layer), rows,
                   gateup_width_, hidden, stream_);
    kernels::LaunchSiluMul(d_gateup_, d_gateup_, rows, config_.intermediate_size, stream_);
    gemm_.GemmFp16(d_mlp_out_, hidden, d_gateup_, gateup_width_, w.w_down(layer), rows, hidden,
                   config_.intermediate_size, stream_);
  }

  // Final norm folds the last block output, then the tied lm_head projects
  // every row to vocab logits (decode computes logits for all batch rows).
  kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.final_norm(), rows, hidden, eps,
                                 stream_);
  gemm_.GemmFp16(d_logits_, config_.vocab_size, d_normed_, hidden, w.lm_head(), rows,
                 config_.vocab_size, hidden, stream_);
}

void ModelExecutor::EnqueueArgmax(std::int32_t rows) {
  kernels::LaunchGreedyArgmax(d_argmax_, d_logits_, rows, config_.vocab_size, stream_);
}

PrefillOutput ModelExecutor::PrefillChunk(const RequestPtr& request, std::int32_t chunk_begin,
                                          std::int32_t chunk_end) {
  if (request == nullptr) {
    throw ExecutorError("model_executor: PrefillChunk called with a null request");
  }

  const std::int32_t rows = FillPrefillMirrors(*request, chunk_begin, chunk_end);
  const std::int32_t ctx = chunk_end; // gather runs after this chunk's scatter
  const bool final_chunk = chunk_end == request->num_prompt_tokens();
  // Host-only cuBLASLt plan resolution happens BEFORE anything is enqueued:
  // a throw here leaves no async copies in flight against the mirrors.
  const PrefillGemmMode mode = ResolvePrefillGemmMode(rows, ctx);

  {
    NvtxRange range("h2d_inputs");
    UploadPrefillInputs(rows);
  }
  {
    // Section 14 taxonomy `prefill[len,chunk]`: len = the context extent the
    // chunk attends to (chunk_end - the kv_len of every composite in the
    // step), chunk = the rows pushed through the layer loop; together they
    // parameterize the section 6.3 prefill cost model an nsys trace is
    // checked against.
    char label[48];
    std::snprintf(label, sizeof(label), "prefill[len=%d,chunk=%d]", ctx, rows);
    NvtxRange range(label);
    EnqueuePrefillForward(rows, chunk_begin, ctx, final_chunk, mode);
  }

  PrefillOutput output;
  output.rows = rows;
  output.final_chunk = final_chunk;
  if (final_chunk) {
    {
      NvtxRange range("sample");
      EnqueueArgmax(1); // the rows=1 lm_head GEMM wrote logits row 0
    }
    NvtxRange range("d2h_tokens");
    CudaCheck(cudaMemcpyAsync(h_argmax_, d_argmax_, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                              stream_),
              "D2H first generated token");
    CudaCheck(cudaStreamSynchronize(stream_), "prefill chunk sync");
    CudaCheck(cudaGetLastError(), "prefill chunk launches");
    output.first_token = h_argmax_[0];
  } else {
    // No logits and nothing to download for an intermediate chunk, but the
    // step still drains: the async H2D copies read pinned mirrors that the
    // next step's fill overwrites, and the sync also surfaces this chunk's
    // async launch errors at the step that caused them (the step boundary,
    // docs/DESIGN.md section 9).
    CudaCheck(cudaStreamSynchronize(stream_), "prefill chunk sync");
    CudaCheck(cudaGetLastError(), "prefill chunk launches");
  }

  // Layer-0 parity dump (docs/DESIGN.md section 12f), first prefill chunk
  // only. The disabled path costs exactly this one branch - the acceptance
  // rule - because the capture re-derives layer 0 AFTER the chunk drained,
  // leaving the hot sequence above untouched (rationale on the declaration).
  if (debug_dump_pending_) {
    debug_dump_pending_ = false;
    DumpLayer0Activations(rows, chunk_begin, ctx, mode);
  }
  return output;
}

std::int32_t ModelExecutor::FillPrefillMirrors(const Request& request, std::int32_t chunk_begin,
                                               std::int32_t chunk_end) {
  const auto fail = [&](const std::string& what) {
    throw ExecutorError("model_executor: prefill chunk (request " + std::to_string(request.id) +
                        "): " + what);
  };

  const std::int32_t prompt_len = request.num_prompt_tokens();
  if (chunk_begin < 0 || chunk_begin >= chunk_end || chunk_end > prompt_len) {
    fail("chunk [" + std::to_string(chunk_begin) + ", " + std::to_string(chunk_end) +
         ") is not a non-empty slice of the " + std::to_string(prompt_len) + "-token prompt");
  }
  const std::int32_t rows = chunk_end - chunk_begin;
  if (rows > options_.prefill_chunk_tokens) {
    fail("chunk of " + std::to_string(rows) + " tokens exceeds prefill_chunk_tokens " +
         std::to_string(options_.prefill_chunk_tokens) +
         " (the per-step input and scores-scratch row capacity)");
  }
  if (chunk_end > options_.max_seq_len) {
    fail("chunk end " + std::to_string(chunk_end) + " exceeds max_seq_len " +
         std::to_string(options_.max_seq_len) + " (the scores/khat kv-extent capacity)");
  }

  const std::vector<BlockId>& blocks = request.block_table.blocks();
  const std::int32_t needed = BlocksForTokens(chunk_end);
  if (static_cast<std::int32_t>(blocks.size()) < needed) {
    fail("block table holds " + std::to_string(blocks.size()) +
         " blocks but the chunk's context end " + std::to_string(chunk_end) + " needs " +
         std::to_string(needed));
  }
  if (static_cast<std::int32_t>(blocks.size()) > max_blocks_per_seq_) {
    fail("block table exceeds max_blocks_per_seq " + std::to_string(max_blocks_per_seq_));
  }

  for (std::int32_t i = 0; i < rows; ++i) {
    const TokenId token = request.prompt_tokens[chunk_begin + i];
    if (token < 0 || token >= config_.vocab_size) {
      fail("prompt token at position " + std::to_string(chunk_begin + i) + " (" +
           std::to_string(token) + ") outside the vocab");
    }
    h_input_ids_[i] = token;
    // Absolute positions: RoPE angles and the in-kernel scatter slot math
    // (p >> 4 / p & 15) both key off them.
    h_positions_[i] = chunk_begin + i;
  }

  // The ONE sequence's block-table row lands in mirror row 0: kv_scatter
  // reads it with row stride 0 (all chunk rows share it) and kv_gather takes
  // it as its block_table_row. Clobbering row 0 never leaks into decode -
  // every decode step re-uploads all the rows it reads.
  std::int32_t* table_row = h_block_tables_;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const BlockId block = blocks[i];
    // Sentinel policy (docs/DESIGN.md section 9): kInvalidBlockId is a
    // host-only debug sentinel and must never reach a device table.
    assert(block != kInvalidBlockId &&
           "kInvalidBlockId must never be written into a device block table");
    if (block <= kDummyBlockId || block >= num_kv_blocks_) {
      fail("block table entry " + std::to_string(i) + " holds invalid block id " +
           std::to_string(block));
    }
    table_row[i] = block;
  }
  std::fill(table_row + blocks.size(), table_row + max_blocks_per_seq_, kDummyBlockId);

  return rows;
}

void ModelExecutor::UploadPrefillInputs(std::int32_t rows) {
  // Exactly THREE contiguous H2D copies from pinned mirrors - the prefill
  // analogue of the decode four-copy rule (docs/DESIGN.md sections 5, 9):
  // chunk token ids, absolute positions, and the one block-table row.
  // seq_lens stays untouched (no prefill kernel consumes it).
  const std::size_t row_bytes = static_cast<std::size_t>(rows) * sizeof(std::int32_t);
  CudaCheck(cudaMemcpyAsync(d_input_ids_, h_input_ids_, row_bytes, cudaMemcpyHostToDevice, stream_),
            "H2D input_ids");
  CudaCheck(cudaMemcpyAsync(d_positions_, h_positions_, row_bytes, cudaMemcpyHostToDevice, stream_),
            "H2D positions");
  CudaCheck(cudaMemcpyAsync(d_block_tables_, h_block_tables_,
                            static_cast<std::size_t>(max_blocks_per_seq_) * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream_),
            "H2D block-table row");
}

PrefillGemmMode ModelExecutor::ResolvePrefillGemmMode(std::int32_t rows, std::int32_t ctx) {
  const std::int32_t head_dim = config_.head_dim;
  const std::int32_t hidden = config_.hidden_size;
  const std::string shape =
      "prefill[T=" + std::to_string(rows) + ",ctx=" + std::to_string(ctx) + "]:";

  if (prefill_gemm_mode_ == PrefillGemmMode::kStridedBatched) {
    const bool scores_ok =
        gemm_.Probe(GemmConfig::PrefillScores(rows, ctx, head_dim, heads_per_group_, qkv_width_),
                    shape + "scores_strided_batched");
    const bool pv_ok =
        gemm_.Probe(GemmConfig::PrefillPv(rows, ctx, head_dim, heads_per_group_, hidden),
                    shape + "pv_strided_batched");
    if (scores_ok && pv_ok) {
      return PrefillGemmMode::kStridedBatched;
    }
    if (!prefill_demotion_warned_) {
      prefill_demotion_warned_ = true;
      std::fprintf(stderr,
                   "[redline] prefill chunk shape T=%d ctx=%d: strided-batched attention GEMMs "
                   "unavailable (scores=%d, pv=%d); demoting this shape to per-head calls "
                   "(docs/DESIGN.md section 16; logged once, all demotions recorded in the "
                   "algo report)\n",
                   rows, ctx, scores_ok ? 1 : 0, pv_ok ? 1 : 0);
    }
  }

  const bool per_head_scores_ok = gemm_.Probe(
      GemmConfig::PrefillScores(rows, ctx, head_dim, 1, qkv_width_), shape + "scores_per_head");
  const bool per_head_pv_ok =
      gemm_.Probe(GemmConfig::PrefillPv(rows, ctx, head_dim, 1, hidden), shape + "pv_per_head");
  if (per_head_scores_ok && per_head_pv_ok) {
    return PrefillGemmMode::kPerHead;
  }
  throw ExecutorError(
      "model_executor: cuBLASLt offers no algorithm for the prefill attention GEMMs at chunk "
      "shape T=" +
      std::to_string(rows) + ", ctx=" + std::to_string(ctx) +
      " in either the strided-batched or the per-head form; the third fallback stage of "
      "docs/DESIGN.md section 16 (explicit repack kernels) is not implemented in v1");
}

void ModelExecutor::EnqueuePrefillForward(std::int32_t rows, std::int32_t chunk_start,
                                          std::int32_t ctx, bool final_chunk,
                                          PrefillGemmMode mode) {
  // The docs/DESIGN.md section 6.3 prefill sequence: the decode layer loop
  // with the attention composite in place of the decode kernel. Deliberately
  // NOT shared with EnqueueDecodeForward - that body is the graph-capture
  // target and stays byte-stable (rationale on the declaration).
  const DeviceWeights& w = *weights_;
  const std::int32_t hidden = config_.hidden_size;
  const std::int64_t qkv_stride = qkv_width_;
  const float eps = config_.rms_norm_eps;

  kernels::LaunchEmbedGather(d_x_, hidden, w.embed(), d_input_ids_, rows, hidden, stream_);

  for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
    if (layer == 0) {
      kernels::LaunchRmsNorm(d_normed_, d_x_, w.input_norm(0), rows, hidden, eps, stream_);
    } else {
      kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.input_norm(layer), rows, hidden,
                                     eps, stream_);
    }

    // Attention: fused QKV projection, RoPE on the Q/K views, scatter the
    // chunk's K/V into the layer's pool slice - all chunk rows share the ONE
    // sequence's block-table row (row stride 0, kernels.cuh contract) - then
    // the gather/GEMM/softmax composite and the output projection.
    gemm_.GemmBiasFp16(d_qkv_, qkv_stride, d_normed_, hidden, w.w_qkv(layer), w.b_qkv(layer), rows,
                       qkv_width_, hidden, stream_);
    kernels::LaunchRopeInplace(d_qkv_, d_qkv_ + k_view_offset_, qkv_stride, qkv_stride,
                               d_positions_, rows, config_.num_q_heads, config_.num_kv_heads,
                               config_.head_dim, config_.rope_theta, stream_);
    kernels::LaunchKvScatter(PoolLayerSlice(layer), d_qkv_ + k_view_offset_,
                             d_qkv_ + v_view_offset_, qkv_stride, qkv_stride, d_block_tables_,
                             /*block_table_row_stride=*/0, d_positions_, rows, config_.num_kv_heads,
                             config_.head_dim, stream_);
    EnqueuePrefillAttention(layer, rows, chunk_start, ctx, mode);
    gemm_.GemmFp16(d_mlp_out_, hidden, d_attn_out_, hidden, w.w_o(layer), rows, hidden, hidden,
                   stream_);

    // MLP: identical to the decode step (docs/DESIGN.md section 6.2 #8).
    kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.post_attn_norm(layer), rows,
                                   hidden, eps, stream_);
    gemm_.GemmFp16(d_gateup_, gateup_width_, d_normed_, hidden, w.w_gateup(layer), rows,
                   gateup_width_, hidden, stream_);
    kernels::LaunchSiluMul(d_gateup_, d_gateup_, rows, config_.intermediate_size, stream_);
    gemm_.GemmFp16(d_mlp_out_, hidden, d_gateup_, gateup_width_, w.w_down(layer), rows, hidden,
                   config_.intermediate_size, stream_);
  }

  if (final_chunk) {
    // Logits for the LAST prompt token only (docs/DESIGN.md section 6.3:
    // computing them for all chunk rows would waste a [chunk x vocab] GEMM):
    // final norm and lm_head run at rows = 1 on the last chunk row's
    // pointers - a configuration ProbeDecodeGemmConfigs warmed via bucket 1.
    // The result lands in logits row 0 for EnqueueArgmax(1).
    const std::int64_t last_row = static_cast<std::int64_t>(rows - 1) * hidden;
    kernels::LaunchRmsNormResidual(d_normed_ + last_row, d_x_ + last_row, d_mlp_out_ + last_row,
                                   w.final_norm(), 1, hidden, eps, stream_);
    gemm_.GemmFp16(d_logits_, config_.vocab_size, d_normed_ + last_row, hidden, w.lm_head(), 1,
                   config_.vocab_size, hidden, stream_);
  }
  // Non-final chunks stop after the last down GEMM: the scattered KV pool
  // slices are the only state the next chunk reads.
}

void ModelExecutor::EnqueuePrefillAttention(std::int32_t layer, std::int32_t rows,
                                            std::int32_t chunk_start, std::int32_t ctx,
                                            PrefillGemmMode mode) {
  // One layer's composite (docs/DESIGN.md section 6.3 descriptor recipes):
  //   kv_gather -> scores GEMMs -> prefill softmax -> PV GEMMs.
  // scores/probs are packed dense [num_q_heads, rows, ctx] at the CURRENT
  // ctx (head plane stride rows*ctx, query-row stride ctx) - exactly what
  // the scores GEMMs write (ldd = ctx, batch stride rows*ctx), the softmax
  // contract expects, and the PV GEMMs read; the [heads, chunk, max_seq_len]
  // scratch regions are max-size backing only.
  const std::int32_t head_dim = config_.head_dim;
  const std::int64_t score_plane = static_cast<std::int64_t>(rows) * ctx;

  // Dense staging of this sequence's cached K/V, positions [0, ctx): rows
  // dense at head_dim (the lda = head_dim the GEMM recipes require), head
  // planes spaced by the allocated max_seq_len extent. Runs AFTER the
  // chunk's own scatter, so the chunk attends to itself (causal-across-
  // chunks acceptance: ctx == chunk_start + rows covers all previously
  // scattered positions plus this chunk).
  kernels::LaunchKvGather(d_khat_, d_vhat_, gather_plane_stride_, head_dim, gather_plane_stride_,
                          head_dim, PoolLayerSlice(layer), d_block_tables_, ctx,
                          config_.num_kv_heads, head_dim, stream_);

  if (mode == PrefillGemmMode::kStridedBatched) {
    // One call per KV head g, batched over its heads_per_group_ Q heads
    // h = g*group .. g*group+group-1: the gathered K/V tile broadcasts with
    // batch stride 0, Q heads step head_dim elements within the qkv_out row,
    // score planes step rows*ctx, and the PV output lands in attn_out at
    // column offset h*head_dim (batch stride head_dim) - no repack.
    const GemmConfig scores =
        GemmConfig::PrefillScores(rows, ctx, head_dim, heads_per_group_, qkv_width_);
    for (std::int32_t g = 0; g < config_.num_kv_heads; ++g) {
      const std::int64_t h0 = static_cast<std::int64_t>(g) * heads_per_group_;
      gemm_.GemmStridedBatched(d_scores_ + h0 * score_plane, d_khat_ + g * gather_plane_stride_,
                               d_qkv_ + h0 * head_dim, attn_scale_, scores, stream_);
    }
  } else {
    const GemmConfig scores = GemmConfig::PrefillScores(rows, ctx, head_dim, 1, qkv_width_);
    for (std::int32_t h = 0; h < config_.num_q_heads; ++h) {
      const std::int32_t g = h / heads_per_group_;
      gemm_.GemmStridedBatched(d_scores_ + h * score_plane, d_khat_ + g * gather_plane_stride_,
                               d_qkv_ + static_cast<std::int64_t>(h) * head_dim, attn_scale_,
                               scores, stream_);
    }
  }

  // Causal row softmax over all heads in one launch: query row i has global
  // position chunk_start + i; the scores GEMMs already carried the
  // 1/sqrt(head_dim) scale (the kernel applies none); masked probs are
  // written as exact zeros so the PV contraction over the full ctx drops
  // them (the prefill-softmax kernel's exact-zeros contract).
  kernels::LaunchPrefillSoftmax(d_probs_, d_scores_, config_.num_q_heads, rows, ctx, chunk_start,
                                stream_);

  if (mode == PrefillGemmMode::kStridedBatched) {
    const GemmConfig pv =
        GemmConfig::PrefillPv(rows, ctx, head_dim, heads_per_group_, config_.hidden_size);
    for (std::int32_t g = 0; g < config_.num_kv_heads; ++g) {
      const std::int64_t h0 = static_cast<std::int64_t>(g) * heads_per_group_;
      gemm_.GemmStridedBatched(d_attn_out_ + h0 * head_dim, d_vhat_ + g * gather_plane_stride_,
                               d_probs_ + h0 * score_plane, /*alpha=*/1.0f, pv, stream_);
    }
  } else {
    const GemmConfig pv = GemmConfig::PrefillPv(rows, ctx, head_dim, 1, config_.hidden_size);
    for (std::int32_t h = 0; h < config_.num_q_heads; ++h) {
      const std::int32_t g = h / heads_per_group_;
      gemm_.GemmStridedBatched(d_attn_out_ + static_cast<std::int64_t>(h) * head_dim,
                               d_vhat_ + g * gather_plane_stride_, d_probs_ + h * score_plane,
                               /*alpha=*/1.0f, pv, stream_);
    }
  }
}

namespace {

// Layer-0 parity dump artifacts (docs/DESIGN.md section 12f). The file names
// and the meta schema are a cross-language contract with
// tests/e2e/test_layer0.py (comparison) and tests/e2e/gen_layer0_fixture.py
// (the HF capture the comparison runs against).
constexpr int kDumpSchemaVersion = 1;
constexpr const char kDumpMetaFile[] = "layer0_meta.json";
constexpr const char kDumpInputIdsFile[] = "layer0_input_ids.bin";
constexpr const char kDumpNormedFile[] = "layer0_normed.bin";
constexpr const char kDumpQkvFile[] = "layer0_qkv.bin";
constexpr const char kDumpAttnOutFile[] = "layer0_attn_out.bin";
constexpr const char kDumpMlpOutFile[] = "layer0_mlp_out.bin";

// Raw little-endian write of host bytes; hard error on any I/O failure (the
// dump is debug-only and a silently truncated file would fail the parity
// comparison with a misleading shape/NaN story instead of the real cause).
void WriteDumpBytes(const std::filesystem::path& path, const void* data, std::int64_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw ExecutorError("model_executor: layer-0 dump: cannot open " + path.string() +
                        " for writing");
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) {
    throw ExecutorError("model_executor: layer-0 dump: short write to " + path.string());
  }
}

// D2H + raw write of one device buffer. The caller drained the stream, so a
// plain synchronous cudaMemcpy is ordered even against the non-blocking s0
// (which does NOT synchronize with the legacy null stream by itself), and a
// pageable staging vector is fine on this debug-only path.
void WriteDeviceDump(const std::filesystem::path& path, const void* device_ptr,
                     std::int64_t bytes) {
  std::vector<char> host(static_cast<std::size_t>(bytes));
  CudaCheck(cudaMemcpy(host.data(), device_ptr, host.size(), cudaMemcpyDeviceToHost),
            "layer-0 dump D2H");
  WriteDumpBytes(path, host.data(), bytes);
}

} // namespace

void ModelExecutor::DumpLayer0Activations(std::int32_t rows, std::int32_t chunk_start,
                                          std::int32_t ctx, PrefillGemmMode mode) {
  namespace fs = std::filesystem;
  const DeviceWeights& w = *weights_;
  const std::int32_t hidden = config_.hidden_size;
  const std::int64_t qkv_stride = qkv_width_;
  const float eps = config_.rms_norm_eps;

  const fs::path dir(options_.debug_dump_dir);
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw ExecutorError("model_executor: layer-0 dump: cannot create directory " + dir.string() +
                        ": " + ec.message());
  }

  // Drain the stage just enqueued on s0, then stage it to a file. Draining
  // per stage is what makes each captured buffer the stage's OUTPUT rather
  // than a racing snapshot; the cost is irrelevant here (debug path, run
  // once per process).
  const auto capture_fp16 = [&](const char* file, const half* buffer, std::int64_t elems) {
    CudaCheck(cudaStreamSynchronize(stream_), "layer-0 dump stage sync");
    CudaCheck(cudaGetLastError(), "layer-0 dump stage launches");
    WriteDeviceDump(dir / file, buffer, elems * static_cast<std::int64_t>(sizeof(half)));
  };

  // Re-derivation of embed + the full layer-0 block (the exact
  // EnqueuePrefillForward layer-0 sequence). The chunk's device inputs are
  // still resident - nothing between upload and here writes input_ids /
  // positions / the block-table row - and the replay repeats the chunk's
  // arithmetic: the kernels use fixed-tree reductions (no atomics) and
  // cuBLASLt re-runs the SAME cached plan on the same operands. Caveat: a
  // cached cuBLASLt plan repeats bitwise only if its algorithm does not use
  // the atomics-based split-K reduction scheme (the wrapper records each
  // config's reduction_scheme in the algo report, so which one was selected
  // is auditable from the emitted report). Under such an algorithm the captured stages -
  // and the K/V bytes the re-scatter below rewrites into the pool - could
  // differ from the chunk's values by reduction-order ULPs: immaterial
  // against the section 12f 2e-2 HF tolerance, and confined to this
  // debug-only path (the chunk's first_token was already read above). The
  // layer-0 kv_scatter re-write is otherwise idempotent: positions
  // [chunk_start, chunk_start + rows) rewrite their own pool slots, and for
  // the first chunk (chunk_start == 0) those positions are exactly the
  // [0, ctx) extent the re-run gather reads.
  kernels::LaunchEmbedGather(d_x_, hidden, w.embed(), d_input_ids_, rows, hidden, stream_);

  // Stage 1: post-input_layernorm (HF layers[0].input_layernorm output).
  kernels::LaunchRmsNorm(d_normed_, d_x_, w.input_norm(0), rows, hidden, eps, stream_);
  capture_fp16(kDumpNormedFile, d_normed_, static_cast<std::int64_t>(rows) * hidden);

  // Stage 2: fused QKV projection with the bias epilogue, captured PRE-RoPE
  // so it compares against HF's q/k/v_proj outputs (rotary is applied later
  // inside HF attention). Column layout [q 0:1536 | k 1536:1792 | v
  // 1792:2048] - a K/V row-range swap inside w_qkv or a bias applied on the
  // wrong axis shows up here directly (docs/DESIGN.md section 12f).
  gemm_.GemmBiasFp16(d_qkv_, qkv_stride, d_normed_, hidden, w.w_qkv(0), w.b_qkv(0), rows,
                     qkv_width_, hidden, stream_);
  capture_fp16(kDumpQkvFile, d_qkv_, static_cast<std::int64_t>(rows) * qkv_width_);

  // Stage 3: attention block output - RoPE, scatter, the prefill attention
  // composite, then the o GEMM whose output lands in the block-output buffer
  // d_mlp_out_ (deliberate buffer reuse). Compares against HF layers[0].self_attn
  // output (post o_proj, pre residual add).
  kernels::LaunchRopeInplace(d_qkv_, d_qkv_ + k_view_offset_, qkv_stride, qkv_stride, d_positions_,
                             rows, config_.num_q_heads, config_.num_kv_heads, config_.head_dim,
                             config_.rope_theta, stream_);
  kernels::LaunchKvScatter(PoolLayerSlice(0), d_qkv_ + k_view_offset_, d_qkv_ + v_view_offset_,
                           qkv_stride, qkv_stride, d_block_tables_,
                           /*block_table_row_stride=*/0, d_positions_, rows, config_.num_kv_heads,
                           config_.head_dim, stream_);
  EnqueuePrefillAttention(/*layer=*/0, rows, chunk_start, ctx, mode);
  gemm_.GemmFp16(d_mlp_out_, hidden, d_attn_out_, hidden, w.w_o(0), rows, hidden, hidden, stream_);
  capture_fp16(kDumpAttnOutFile, d_mlp_out_, static_cast<std::int64_t>(rows) * hidden);

  // Stage 4: MLP block output - post-attention norm (folds the attention
  // block output into the residual), gate|up GEMM, SwiGLU, down GEMM.
  // Compares against HF layers[0].mlp output (pre residual add).
  kernels::LaunchRmsNormResidual(d_normed_, d_x_, d_mlp_out_, w.post_attn_norm(0), rows, hidden,
                                 eps, stream_);
  gemm_.GemmFp16(d_gateup_, gateup_width_, d_normed_, hidden, w.w_gateup(0), rows, gateup_width_,
                 hidden, stream_);
  kernels::LaunchSiluMul(d_gateup_, d_gateup_, rows, config_.intermediate_size, stream_);
  gemm_.GemmFp16(d_mlp_out_, hidden, d_gateup_, gateup_width_, w.w_down(0), rows, hidden,
                 config_.intermediate_size, stream_);
  capture_fp16(kDumpMlpOutFile, d_mlp_out_, static_cast<std::int64_t>(rows) * hidden);

  // The consumed token ids, straight from the pinned mirror (still holding
  // this chunk): lets the comparison refuse to diff activations of two
  // different prompts.
  WriteDumpBytes(dir / kDumpInputIdsFile, h_input_ids_,
                 static_cast<std::int64_t>(rows) * sizeof(std::int32_t));

  // Shape descriptor. Raw binaries carry no header, so the reader takes
  // every shape from here rather than trusting its own arithmetic.
  nlohmann::json meta;
  meta["schema_version"] = kDumpSchemaVersion;
  meta["dtype"] = "float16";
  meta["byte_order"] = "little";
  meta["rows"] = rows;
  meta["chunk_start"] = chunk_start;
  meta["ctx"] = ctx;
  meta["hidden_size"] = hidden;
  meta["qkv_width"] = qkv_width_;
  meta["num_q_heads"] = config_.num_q_heads;
  meta["num_kv_heads"] = config_.num_kv_heads;
  meta["head_dim"] = config_.head_dim;
  meta["qkv_rope_applied"] = false; // captured before LaunchRopeInplace
  meta["prefill_gemm_mode"] = ToString(mode);
  meta["files"] = {
      {"input_ids", {{"file", kDumpInputIdsFile}, {"dtype", "int32"}, {"shape", {rows}}}},
      {"normed", {{"file", kDumpNormedFile}, {"dtype", "float16"}, {"shape", {rows, hidden}}}},
      {"qkv", {{"file", kDumpQkvFile}, {"dtype", "float16"}, {"shape", {rows, qkv_width_}}}},
      {"attn_out", {{"file", kDumpAttnOutFile}, {"dtype", "float16"}, {"shape", {rows, hidden}}}},
      {"mlp_out", {{"file", kDumpMlpOutFile}, {"dtype", "float16"}, {"shape", {rows, hidden}}}},
  };
  const std::string meta_text = meta.dump(2);
  WriteDumpBytes(dir / kDumpMetaFile, meta_text.data(),
                 static_cast<std::int64_t>(meta_text.size()));

  std::fprintf(stderr,
               "[redline] layer-0 parity dump (docs/DESIGN.md section 12f): %d rows "
               "(chunk_start=%d, ctx=%d) -> %s\n",
               rows, chunk_start, ctx, dir.string().c_str());
}

} // namespace redline
