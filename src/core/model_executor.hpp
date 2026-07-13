#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "core/config.hpp"
#include "core/request.hpp"
#include "core/types.hpp"
#include "gemm/cublaslt_gemm.hpp"

// ModelExecutor - owns every persistent device/host resource of the forward
// pass and runs both eager step flavors: the decode step and the chunked
// prefill step (docs/DESIGN.md sections 5, 6.3).
//
// Ownership at init (every CUDA allocation this component ever performs
// happens in the constructor; none later - the no-allocation-after-init rule
// of docs/DESIGN.md section 5 is what makes CUDA-graph capture legal):
//
//   - the single layer-outer paged KV pool (`cudaMalloc`), logically
//     pool[L][num_blocks][2 (0=K,1=V)][num_kv_heads][16][head_dim] FP16, with
//     num_blocks = floor(kv_pool_bytes / (16 * KvBytesPerToken)); physical
//     block 0 is the reserved dummy block (docs/DESIGN.md section 9);
//   - one device scratch slab carved into the section 5 activation and
//     per-step-input regions (layout table in model_executor.cpp): residual
//     `x`, `normed`, `qkv_out [rows, 2048]`, `attn_out`, `gateup
//     [rows, 17920]`, `mlp_out`, `logits [max_batch, vocab]`, prefill-only
//     `scores` FP32 / `probs` FP16 [heads, chunk, max_seq_len] and
//     `khat`/`vhat` [kv_heads, max_seq_len, head_dim], plus the int32
//     per-step inputs `input_ids`/`positions` (max(prefill_chunk, max_batch)
//     rows - prefill pushes chunk rows through the same buffers decode uses),
//     `seq_lens` [max_batch], the device block tables
//     [max_batch, max_blocks_per_seq], and `argmax_out` [max_batch];
//   - one pinned host slab (`cudaHostAlloc`) holding the persistent mirrors
//     of every per-step input and of argmax_out (docs/DESIGN.md section 5:
//     pageable sources silently degrade cudaMemcpyAsync to staged
//     synchronous copies);
//   - the single non-default stream s0 all device work runs on;
//   - the cuBLASLt wrapper (handle + 32 MiB workspace), with every decode
//     GEMM configuration probed per bucket - and the prefill configurations
//     probed at their largest shape, selecting the docs/DESIGN.md section 16
//     fallback stage - at init, so the plan cache is warm and availability
//     is recorded before the first step.
//
// Eager decode step (DecodeEager) - the exact section 6.3 sequence:
//
//   fill pinned mirrors (batch-compacted rows; padded rows seq_len=0/
//                        position 0/token 0/dummy-block table rows)
//   4 contiguous H2D cudaMemcpyAsync: input_ids, positions, seq_lens,
//                                     block-table slab as ONE copy
//   embed_gather
//   for l in 0..L-1:
//     (l==0 ? rmsnorm : fused_add_rmsnorm(mlp_out))    # normed; updates x
//     qkv GEMM+bias -> rope -> kv_scatter(l) -> decode_attn(l) -> o GEMM
//     fused_add_rmsnorm(attn block out) -> gateup GEMM -> silu_mul -> down GEMM
//   fused_add_rmsnorm(mlp_out, final_norm) -> lm_head GEMM -> argmax
//   1 D2H cudaMemcpyAsync of argmax_out -> cudaStreamSynchronize(s0)
//
// Q/K/V are consumed as strided views into qkv_out (q = +0, k = +1536,
// v = +1792 for this model, all with row stride 2048) - no split/repack
// kernel exists anywhere on the path (docs/DESIGN.md section 5 strided-view
// rule), and every kernel launcher call passes its strides explicitly.
//
// Chunked-prefill step (PrefillChunk) - positions [chunk_begin, chunk_end)
// of ONE sequence's prompt through the same layer loop and the same per-step
// input buffers (sized for prefill_chunk_tokens rows at init), with decode
// attention replaced by the docs/DESIGN.md section 6.3 composite:
//
//   fill pinned mirrors (chunk token ids, absolute positions, the ONE
//                        sequence's block-table row)
//   3 contiguous H2D cudaMemcpyAsync: input_ids, positions, block-table row
//   embed_gather
//   for l in 0..L-1:   # as decode, but attention is the prefill composite
//     norm -> qkv GEMM+bias -> rope
//     kv_scatter(l, block-table row stride 0: all chunk rows share the row)
//     kv_gather(l, ctx = chunk_end) -> scores GEMMs -> prefill softmax
//                                   -> PV GEMMs
//     o GEMM -> norm -> gateup GEMM -> silu_mul -> down GEMM
//   final chunk only: rows=1 final norm + lm_head GEMM over the LAST prompt
//                     row, argmax, 1 D2H of the first generated token
//   cudaStreamSynchronize(s0)
//
// The scores and PV products run as ONE strided-batched cuBLASLt call per KV
// head (batch = the group's Q heads, K/V operand broadcast with batch stride
// 0, scores packed [heads, chunk, ctx] at the CURRENT ctx, PV written
// straight into attn_out at column offset h*head_dim - no repack kernel),
// falling back to per-head non-batched calls when the heuristic does not
// cover the batched layouts. The third fallback stage of docs/DESIGN.md
// section 16 (explicit repack kernels) is not implemented in v1 and
// hard-errors with a clear message if ever required; the probed decision is
// recorded at init and queryable through prefill_gemm_mode(). Prefill is
// eager-only and never graph-captured, so its per-(chunk, ctx) GEMM plans
// legally fill the wrapper's cache on first use. All prefill scratch
// (`scores`, `probs`, `khat`, `vhat`) is allocated by the constructor, so
// the composite adds no allocations.
//
// CUDA-graph decode (docs/DESIGN.md section 9) - when
// ExecutorOptions::enable_cuda_graphs is set, the constructor finishes by
// (1) running one full eager decode step per configured bucket with
// dummy-but-valid inputs (seq_len=1, all-dummy block tables), which forces
// every cuBLASLt plan execution, workspace binding, kernel-module load, and
// the RoPE constant-table upload to happen BEFORE any capture; then
// (2) capturing EnqueueDecodeForward + EnqueueArgmax verbatim per bucket on
// s0 in ThreadLocal mode and instantiating one executable graph per bucket.
// Replay (DecodeGraph) is the eager step with the enqueue calls replaced by
// a single cudaGraphLaunch: mirrors padded to the bucket -> the same 4 H2D
// copies -> launch -> argmax D2H -> stream sync. Every captured argument is
// a pointer to a persistent buffer and only buffer contents change between
// replays, so graphs are never re-instantiated. Any warmup/capture failure
// degrades to eager with a one-time warning and never aborts init.
//
// Not thread-safe; matches the engine's single-thread contract
// (docs/DESIGN.md section 10).

namespace redline {

class DeviceWeights; // loader/weights.hpp

// Typed error for executor failures: invalid options, a CUDA failure while
// building or running the step, or a decode-batch row that violates the
// scheduler contract (sequence longer than max_seq_len, block table not
// covering the sequence, block id outside the pool). Derives from
// std::runtime_error so generic catch sites keep working.
class ExecutorError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Runtime geometry, mapped 1:1 from EngineOptions by the engine.
struct ExecutorOptions {
  // Paged KV pool budget in BYTES; the pool holds
  // floor(kv_pool_bytes / (kKvBlockSize * KvBytesPerToken)) blocks.
  std::int64_t kv_pool_bytes = std::int64_t{1} << 30;

  // Decode batch cap; also the largest CUDA-graph bucket in use and the row
  // count of the seq_lens / logits / argmax buffers.
  std::int32_t max_batch = 8;

  // Hard cap on prompt_len + max_new_tokens; sets the device block-table
  // width max_blocks_per_seq = BlocksForTokens(max_seq_len) and the
  // prefill-scratch kv extent.
  std::int32_t max_seq_len = 2048;

  // Prompt tokens per dedicated prefill step; sizes input_ids/positions and
  // the activation rows at max(prefill_chunk_tokens, max_batch).
  std::int32_t prefill_chunk_tokens = 1024;

  // Capture the decode step as one CUDA graph per configured bucket at the
  // end of construction and replay it per decode step (docs/DESIGN.md
  // section 9). false skips warmup and capture entirely - the engine then
  // routes every decode step through DecodeEager. Capture failure is never
  // fatal: the executor warns once, releases anything partially captured,
  // and degrades to eager (has_decode_graphs() reports the outcome).
  bool enable_cuda_graphs = false;

  // Debug/eval only (docs/DESIGN.md section 12d/e same-shape protocol);
  // 0 = off (default). Value b: every eager decode batch is padded to the
  // smallest configured bucket >= max(live_batch, b) with dummy rows
  // (seq_len=0, position 0, token 0, dummy-block block-table rows), so a
  // solo request can be forced through e.g. the bucket-8 GEMM shapes.
  // Integer-valued on purpose: `true` (== 1) means "smallest bucket covering
  // the live batch" - the graph-equivalence protocol - while 8 forces the
  // solo-vs-batch-of-8 comparison. Values above the largest configured
  // bucket are clamped to it; a live batch no bucket covers runs unpadded.
  std::int32_t pad_eager_to_bucket = 0;

  // Debug/eval only (docs/DESIGN.md section 12f layer-0 parity fixture);
  // empty (default) = off. When set, the FIRST prefill chunk this executor
  // runs writes its layer-0 activations - post-input_layernorm `normed`,
  // pre-RoPE `qkv_out`, the attention block output (post o-GEMM), and the
  // MLP block output - into this directory as raw little-endian FP16 files
  // plus a JSON shape descriptor (file names in model_executor.cpp;
  // consumed by tests/e2e/test_layer0.py). The disabled path costs exactly
  // one branch at prefill-chunk end and nothing else.
  std::string debug_dump_dir;

  // Device block-table width, derived - never set independently of
  // max_seq_len (docs/DESIGN.md section 5).
  std::int32_t max_blocks_per_seq() const { return BlocksForTokens(max_seq_len); }
};

// The executor's device-byte plan, computable BEFORE any allocation (pure
// arithmetic) so the engine's cudaMemGetInfo preflight can fail fast with a
// per-item breakdown (docs/DESIGN.md section 11). Weights are separate
// (DeviceWeights::TotalDeviceBytes); the engine sums the two.
struct ExecutorMemoryBudget {
  std::int32_t num_kv_blocks = 0;      // floor(kv_pool_bytes / block bytes)
  std::int64_t kv_pool_bytes = 0;      // num_kv_blocks * 16 * KvBytesPerToken
  std::int64_t scratch_bytes = 0;      // activation + per-step-input device slab
  std::int64_t workspace_bytes = 0;    // shared cuBLASLt workspace
  std::int64_t total_device_bytes = 0; // pool + scratch + workspace
  std::int64_t pinned_host_bytes = 0;  // host-side mirrors (not device memory)
};

// One executed decode step, as reported back to the engine.
struct DecodeOutput {
  // Greedy argmax token per LIVE batch row, aligned with the input batch
  // order (row r of the device batch is plan-order index r - never
  // Request::slot, which is scheduler-side identity only).
  std::vector<TokenId> tokens;

  // Device rows the step executed at, live rows plus padding. This is the
  // GEMM n-dimension of every launch in the step.
  std::int32_t rows = 0;

  // The configured decode bucket whose size equals `rows`, or -1 when the
  // step ran at an off-ladder shape. When pad_eager_to_bucket engaged, rows
  // equals the padding target bucket by construction. The engine attributes
  // its bucket histogram from this.
  std::int32_t bucket = -1;
};

// Execution form of the prefill scores/PV attention GEMMs - stages one and
// two of the docs/DESIGN.md section 16 fallback chain. Stage three (explicit
// repack kernels) is not implemented in v1: a chunk shape neither form
// covers throws ExecutorError from PrefillChunk.
enum class PrefillGemmMode : std::int32_t {
  kStridedBatched = 0, // one call per KV head, batched over its Q-head group
  kPerHead = 1,        // one non-batched call per Q head
};

// Log/report name of a PrefillGemmMode ("strided_batched" / "per_head").
const char* ToString(PrefillGemmMode mode);

// One executed prefill chunk, as reported back to the engine.
struct PrefillOutput {
  // Greedy argmax over the LAST prompt token's logits - the sequence's first
  // generated token. Valid only when final_chunk; -1 otherwise (no logits
  // are computed for non-final chunks, docs/DESIGN.md section 6.3).
  TokenId first_token = -1;

  // True when the chunk reached the end of the prompt (chunk_end == prompt
  // length; mirrors StepPlan::final_prefill_chunk()).
  bool final_chunk = false;

  // Chunk rows executed (chunk_end - chunk_begin) - the GEMM n-dimension of
  // every linear launch in the step and the T of the attention composite.
  std::int32_t rows = 0;
};

class ModelExecutor {
 public:
  // Pure arithmetic (no CUDA): the allocation plan the constructor executes.
  // Throws ExecutorError on invalid options (non-positive dimensions, or a
  // pool too small for the dummy block plus one usable block).
  static ExecutorMemoryBudget ComputeBudget(const ModelConfig& config,
                                            const ExecutorOptions& options);

  // Performs every allocation and init-time probe described above. `weights`
  // must already be fully resident (the DeviceWeights constructor drains its
  // uploads) and must outlive this executor; the executor keeps a reference.
  // Throws ExecutorError on invalid options or any CUDA failure, and
  // GemmError if the cuBLASLt wrapper cannot initialize. A decode or prefill
  // LINEAR GEMM configuration for which the heuristic offers no algorithm is
  // also fatal here - the documented qkv-bias fallback (plain GEMM + bias
  // folded into a kernel, docs/DESIGN.md section 16) is not wired in v1, so
  // this fails at init with the probe decision recorded rather than
  // mid-serving. The prefill scores/PV products are the exception with a
  // wired fallback: strided-batched demotes to per-head per section 16, and
  // only both forms missing is fatal (the stage-three repack kernels are a
  // deliberate v1 stub).
  ModelExecutor(const ModelConfig& config, const DeviceWeights& weights, ExecutorOptions options);
  ~ModelExecutor();
  ModelExecutor(const ModelExecutor&) = delete;
  ModelExecutor& operator=(const ModelExecutor&) = delete;

  // Run one eager decode step for `batch` (1..max_batch running sequences,
  // in plan order). For each live row the executor derives the device inputs
  // from the request: input token = the last appended token
  // (output_tokens.back()), position = seq_len()-1, seq_lens = seq_len()
  // (context length including the token being scattered this step), block
  // table = the request's materialized blocks. The scheduler guarantees the
  // table covers the sequence (lazy boundary allocation happens at plan
  // time); violations throw ExecutorError before anything is uploaded.
  // Mirror-fill additionally debug-asserts that kInvalidBlockId never
  // reaches the device tables - padding is always the dummy block
  // (docs/DESIGN.md section 9 sentinel policy).
  //
  // Blocking: returns after cudaStreamSynchronize(s0) with the sampled
  // tokens on host (this sync is the step boundary, docs/DESIGN.md
  // section 9). NVTX ranges: h2d_inputs, decode[b=rows], sample, d2h_tokens.
  DecodeOutput DecodeEager(const std::vector<RequestPtr>& batch);

  // Run one decode step for `batch` by replaying the captured graph of the
  // smallest configured bucket covering the batch (docs/DESIGN.md section 9
  // replay procedure): fill the pinned mirrors padded to that bucket
  // (padded slots: token 0, position 0, seq_len 0, all-dummy block-table
  // rows) -> the same exactly-4 H2D copies as DecodeEager ->
  // cudaGraphLaunch(exec[bucket], s0) -> D2H of argmax_out[0..bucket) ->
  // cudaStreamSynchronize(s0). Per-row derivation, hard validation, and the
  // returned DecodeOutput semantics are identical to DecodeEager (tokens
  // carry live rows only; rows == bucket by construction). Requires
  // has_decode_graphs() and a covering bucket - the engine's dispatch
  // conditions; a violation throws ExecutorError before anything is
  // uploaded. NVTX ranges: h2d_inputs, graph_replay[b=rows], d2h_tokens
  // (sample is inside the captured graph, so no separate range exists).
  DecodeOutput DecodeGraph(const std::vector<RequestPtr>& batch);

  // True when a decode graph was captured for EVERY configured bucket
  // (enable_cuda_graphs was set and init-time capture succeeded); false
  // when disabled or degraded to eager after a capture failure. Capture is
  // all-or-nothing - a partial bucket ladder never serves.
  bool has_decode_graphs() const { return !graph_execs_.empty(); }

  // Run one chunked-prefill step for `request`, pushing prompt positions
  // [chunk_begin, chunk_end) through the layer loop and scattering their K/V
  // into the paged pool (docs/DESIGN.md section 6.3). The scheduler plan
  // guarantees positions [0, chunk_begin) were scattered by earlier chunks
  // of the same request (uncheckable here - the causal-across-chunks
  // premise) and that the block table covers BlocksForTokens(chunk_end);
  // every checkable violation - empty/negative range, chunk longer than
  // prefill_chunk_tokens, chunk_end beyond the prompt or max_seq_len, block
  // table not covering chunk_end, invalid block ids, out-of-vocab prompt
  // tokens - throws ExecutorError before anything is uploaded.
  //
  // The final chunk (chunk_end == prompt length) computes logits for the
  // LAST prompt token only and returns the first generated token in the
  // output; earlier chunks compute no logits at all. Blocking like
  // DecodeEager: every chunk ends with cudaStreamSynchronize(s0) - the
  // step's H2D copies read pinned mirrors that the next step's fill
  // overwrites, so the step boundary applies to prefill too. NVTX ranges:
  // h2d_inputs, prefill[len=chunk_end,chunk=rows], and (final chunk only)
  // sample + d2h_tokens.
  PrefillOutput PrefillChunk(const RequestPtr& request, std::int32_t chunk_begin,
                             std::int32_t chunk_end);

  // Pool geometry for the engine's BlockAllocator (usable = num_kv_blocks-1
  // after the reserved dummy block).
  std::int32_t num_kv_blocks() const { return num_kv_blocks_; }

  // The single non-default stream s0 every launch of this executor uses.
  cudaStream_t stream() const { return stream_; }

  // Ascending CUDA-graph bucket ladder for this max_batch (the canonical
  // {1,2,4,8,16,32,48,64} capped; docs/DESIGN.md section 9). Decode padding
  // and the graph path share this list.
  const std::vector<std::int32_t>& decode_buckets() const { return buckets_; }

  // The prefill scores/PV GEMM form selected by the init-time probes at the
  // largest legal chunk shape (docs/DESIGN.md section 16 fallback chain;
  // also recorded in AlgoReportJson and the init log line). Individual chunk
  // shapes the batched form does not cover demote themselves to per-head at
  // run time (probed lazily, first demotion logged); a shape neither form
  // covers throws from PrefillChunk because the stage-three repack fallback
  // is not implemented in v1.
  PrefillGemmMode prefill_gemm_mode() const { return prefill_gemm_mode_; }

  // Selected-algorithm report of the owned cuBLASLt wrapper (JSON string,
  // docs/DESIGN.md section 12d) - surfaced through bench/test output.
  std::string AlgoReportJson() const { return gemm_.AlgoReportJson(); }

  const ExecutorOptions& options() const { return options_; }
  const ExecutorMemoryBudget& budget() const { return budget_; }

 private:
  // --- init helpers (the only methods that allocate) ---------------------
  void AllocateDeviceMemory(); // KV pool + scratch slab, zero-filled
  void AllocatePinnedMirrors();
  // Builds + caches the plan for every decode GEMM configuration at every
  // bucket size (and max_batch when off-ladder) and throws ExecutorError,
  // with the failing tag, if any is unavailable. Run before the first step
  // so the run path never misses the cache on a bucketed shape.
  void ProbeDecodeGemmConfigs();
  // Probes the prefill GEMM configurations at the largest legal chunk shape
  // (T = full chunk, ctx = max_seq_len): the linear layers (fatal when
  // unavailable, same policy as decode - no fallback is wired) and the
  // scores/PV attention products in BOTH fallback-chain forms, recording
  // each in the algo report and selecting prefill_gemm_mode_. Throws
  // ExecutorError when neither form is available (the stage-three repack
  // fallback is a v1 stub by design). Partial final chunks run at other
  // shapes and fill the plan cache lazily on the eager path.
  void ProbePrefillGemmConfigs();
  // The docs/DESIGN.md section 9 capture procedure (ctor-only, after the
  // probes): warm up EVERY bucket eagerly, then per bucket capture
  // EnqueueDecodeForward + EnqueueArgmax on stream_ in ThreadLocal mode,
  // instantiate, and upload the executable graph. On success fills
  // graph_execs_ (parallel to buckets_) and logs the cudaMemGetInfo delta
  // across the capture phase. ANY failure - warmup included - degrades to
  // eager with a one-time warning instead of throwing: the eager path is
  // permanently maintained, and an error that also breaks eager resurfaces
  // on the first real step exactly as it would with graphs disabled.
  void CaptureDecodeGraphs();
  // Dummy-but-valid decode inputs for warmup/capture (docs/DESIGN.md
  // section 9): every row token 0, position 0, seq_len 1 (the attention CTA
  // walks one real KV tile instead of exiting), all-dummy block-table rows
  // (the step's K/V scatter lands in the sacrificial dummy block, which no
  // live sequence ever reads).
  void FillDummyDecodeMirrors(std::int32_t rows);
  // One full eager decode step at `rows` from the dummy mirrors, drained.
  // Forces cuBLASLt plan execution + workspace binding, kernel-module load,
  // and the RoPE constant-table upload before any capture region opens -
  // heuristic queries, allocations, or cudaMemcpyToSymbol inside capture
  // are bugs by design (docs/DESIGN.md section 9).
  void WarmupDecodeBucket(std::int32_t rows);
  void ReleaseAll() noexcept;

  // --- step helpers (allocation-free) ------------------------------------
  // Fills the pinned mirrors for `batch` plus padding rows; returns the
  // executed row count. pad_to_rows == 0 selects eager sizing (live rows
  // plus pad_eager_to_bucket padding); pad_to_rows >= live is the exact row
  // count to pad to (the graph-replay path passes its bucket).
  std::int32_t FillDecodeMirrors(const std::vector<RequestPtr>& batch, std::int32_t pad_to_rows);
  // The exactly-4 H2D copies of docs/DESIGN.md section 9 (ids, positions,
  // seq_lens, block-table slab as ONE contiguous copy), async on s0.
  void UploadDecodeInputs(std::int32_t rows);
  // Enqueues the section 6.3 decode sequence at `rows` device rows, reading
  // only persistent buffers: embed_gather through the lm_head GEMM. Contains
  // no NVTX and no host work, so the graph path can capture
  // EnqueueDecodeForward + EnqueueArgmax verbatim.
  void EnqueueDecodeForward(std::int32_t rows);
  void EnqueueArgmax(std::int32_t rows);

  // Validates the chunk against the request (hard checks listed on
  // PrefillChunk) and fills the pinned mirrors: chunk token ids, absolute
  // positions [chunk_begin, chunk_end), and the ONE sequence's block-table
  // row into mirror row 0 (dummy-padded). Returns the chunk row count.
  std::int32_t FillPrefillMirrors(const Request& request, std::int32_t chunk_begin,
                                  std::int32_t chunk_end);
  // The exactly-3 contiguous H2D copies of a prefill step (ids, positions,
  // one block-table row), async on s0. seq_lens has no prefill consumer -
  // the composite's kv extent travels as launch/GEMM scalars instead.
  void UploadPrefillInputs(std::int32_t rows);
  // Resolves which GEMM form serves THIS chunk shape, starting from the
  // init-selected prefill_gemm_mode_: probes (plan-cache hits after the
  // first occurrence of a shape; heuristic queries are legal here because
  // prefill is never captured) the mode's scores/PV configs at (rows, ctx),
  // demoting strided-batched to per-head per shape when needed and throwing
  // ExecutorError when neither form is available. Host-only; called before
  // any of the chunk's work is enqueued.
  PrefillGemmMode ResolvePrefillGemmMode(std::int32_t rows, std::int32_t ctx);
  // Enqueues the section 6.3 prefill sequence for one chunk: embed_gather,
  // the layer loop with the attention composite, and - final chunk only -
  // the rows=1 final norm + lm_head GEMM over the last prompt row. Reads
  // only persistent buffers; NVTX-free like its decode counterpart (prefill
  // is never captured, but the taxonomy ranges wrap whole spans from
  // PrefillChunk). Deliberately a separate body from EnqueueDecodeForward:
  // that sequence is the graph-capture target and stays byte-stable
  // rather than sharing a parameterized helper with the eager-only path.
  void EnqueuePrefillForward(std::int32_t rows, std::int32_t chunk_start, std::int32_t ctx,
                             bool final_chunk, PrefillGemmMode mode);
  // One layer's attention composite: kv_gather -> scores GEMMs -> prefill
  // softmax -> PV GEMMs, in `mode`'s form (docs/DESIGN.md section 6.3
  // descriptor recipes).
  void EnqueuePrefillAttention(std::int32_t layer, std::int32_t rows, std::int32_t chunk_start,
                               std::int32_t ctx, PrefillGemmMode mode);

  // The docs/DESIGN.md section 12f layer-0 parity dump, called at most once
  // (first prefill chunk) from PrefillChunk's chunk-end branch when
  // ExecutorOptions::debug_dump_dir is set. The activation buffers are
  // reused by all 28 layers, so the layer-0 values no longer exist at chunk
  // end; instead this re-enqueues embed_gather plus the full layer-0 block
  // from the chunk's still-resident device inputs (input_ids / positions /
  // block-table row are untouched since upload; the replay repeats the
  // chunk's arithmetic bitwise short of one caveat - a cuBLASLt algorithm
  // using the atomics-based split-K reduction, see the definition's comment
  // - and the layer-0 re-scatter rewrites only its own pool slots),
  // draining the stream after each captured stage for a pageable D2H + raw
  // file write. Debug path only: never called on the serving path, file I/O
  // errors throw ExecutorError.
  void DumpLayer0Activations(std::int32_t rows, std::int32_t chunk_start, std::int32_t ctx,
                             PrefillGemmMode mode);

  // Base of layer `layer`'s slice of the paged pool (section 5 layout).
  half* PoolLayerSlice(std::int32_t layer) const {
    return kv_pool_ + static_cast<std::int64_t>(layer) * pool_layer_stride_elems_;
  }
  // The configured bucket whose size equals `rows`, or -1.
  std::int32_t BucketForRows(std::int32_t rows) const;

  ModelConfig config_;
  const DeviceWeights* weights_ = nullptr; // not owned; outlives the executor
  ExecutorOptions options_;
  ExecutorMemoryBudget budget_;
  std::vector<std::int32_t> buckets_;

  // Derived geometry (elements unless noted), fixed at init.
  std::int32_t max_step_rows_ = 0;      // max(prefill_chunk_tokens, max_batch)
  std::int32_t max_blocks_per_seq_ = 0; // BlocksForTokens(max_seq_len)
  std::int32_t num_kv_blocks_ = 0;
  std::int64_t pool_layer_stride_elems_ = 0; // num_blocks * 2 * kv_heads * 16 * head_dim
  std::int32_t qkv_width_ = 0;               // (num_q_heads + 2*num_kv_heads) * head_dim
  std::int32_t k_view_offset_ = 0;           // num_q_heads * head_dim (1536)
  std::int32_t v_view_offset_ = 0;           // k_view_offset_ + num_kv_heads * head_dim
  std::int32_t gateup_width_ = 0;            // 2 * intermediate_size
  std::int32_t heads_per_group_ = 0;         // num_q_heads / num_kv_heads (GQA group, 6)
  std::int64_t gather_plane_stride_ = 0;     // khat/vhat head-plane spacing:
                                             // max_seq_len * head_dim (allocated extent)
  float attn_scale_ = 0.0f;                  // 1 / sqrt(head_dim)

  // Init-selected prefill attention GEMM form (see prefill_gemm_mode()).
  PrefillGemmMode prefill_gemm_mode_ = PrefillGemmMode::kStridedBatched;
  // First run-time strided-batched -> per-head demotion warns once; later
  // demotions are silent (availability is per shape, cached by the wrapper).
  bool prefill_demotion_warned_ = false;

  // True until the layer-0 parity dump has run; initialized in the
  // constructor to whether options_.debug_dump_dir is set, cleared by the
  // first prefill chunk. Always false in production configurations, so the
  // dump costs one predictable branch per prefill chunk (section 12f
  // zero-overhead-when-disabled rule).
  bool debug_dump_pending_ = false;

  CublasLtGemm gemm_;
  cudaStream_t stream_ = nullptr;

  // Instantiated decode graphs, parallel to buckets_ (docs/DESIGN.md
  // section 9); empty when enable_cuda_graphs is off or capture degraded to
  // eager. The executables are persistent for the executor's lifetime -
  // contents-only updates between replays, never re-instantiated - and are
  // destroyed first in ReleaseAll.
  std::vector<cudaGraphExec_t> graph_execs_;

  // Device memory: two allocations (pool + scratch slab); the named pointers
  // below are carved offsets into scratch_slab_, never separately freed.
  half* kv_pool_ = nullptr;
  void* scratch_slab_ = nullptr;
  half* d_x_ = nullptr;        // residual stream [rows, hidden]
  half* d_normed_ = nullptr;   // norm output [rows, hidden]
  half* d_qkv_ = nullptr;      // fused QKV GEMM out [rows, qkv_width_]
  half* d_attn_out_ = nullptr; // attention head concat [rows, hidden]
  half* d_gateup_ = nullptr;   // fused gate|up GEMM out [rows, 2*I]
  // Block output added to the residual: carries the o-projection output
  // between the o GEMM and the post-attention norm, then the down-projection
  // output between the down GEMM and the next input norm (the section 5
  // scratch list's `mlp_out`; both block outputs stream through it).
  half* d_mlp_out_ = nullptr;
  half* d_logits_ = nullptr;               // [max_batch, vocab]
  float* d_scores_ = nullptr;              // prefill scratch [heads, chunk, max_seq_len]
  half* d_probs_ = nullptr;                // prefill scratch, same extent
  half* d_khat_ = nullptr;                 // prefill K gather [kv_heads, max_seq_len, head_dim]
  half* d_vhat_ = nullptr;                 // prefill V gather, same extent
  std::int32_t* d_input_ids_ = nullptr;    // [max_step_rows_]
  std::int32_t* d_positions_ = nullptr;    // [max_step_rows_]
  std::int32_t* d_seq_lens_ = nullptr;     // [max_batch]
  std::int32_t* d_block_tables_ = nullptr; // [max_batch, max_blocks_per_seq]
  std::int32_t* d_argmax_ = nullptr;       // [max_batch]

  // Pinned host mirrors: one cudaHostAlloc slab, carved like the above.
  void* pinned_slab_ = nullptr;
  std::int32_t* h_input_ids_ = nullptr;
  std::int32_t* h_positions_ = nullptr;
  std::int32_t* h_seq_lens_ = nullptr;
  std::int32_t* h_block_tables_ = nullptr;
  std::int32_t* h_argmax_ = nullptr;
};

} // namespace redline
