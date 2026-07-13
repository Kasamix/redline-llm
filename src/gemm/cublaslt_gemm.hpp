#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace redline {

// Typed error for every failure raised by the GEMM wrapper (cuBLASLt status
// failures, CUDA failures, contract violations such as a heuristic query
// attempted inside a CUDA-graph capture region). Derives from
// std::runtime_error so generic catch sites keep working.
class GemmError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Epilogue selector. kBias maps to CUBLASLT_EPILOGUE_BIAS with an FP16
// per-output-feature vector: in the column-major dual used throughout this
// wrapper the bias length is m = out_features, broadcast across the token
// columns - the intended per-feature semantics (docs/DESIGN.md section 6.1).
enum class GemmEpilogue : std::int32_t {
  kNone = 0,
  kBias = 1,
};

// The complete tuple that determines a cuBLASLt matmul configuration: every
// field participates in descriptor construction, in the heuristic query, and
// in the plan-cache key (two configs differing in any field get independent
// heuristic selections). All problems are expressed in the COLUMN-MAJOR DUAL
// of docs/DESIGN.md section 6.1: a row-major buffer M[r, c] with leading
// dimension c *is* the column-major matrix M^T[c, r] with ld = c, so the
// logical row-major linear layer
//   out[rows, n_out] = act[rows, in] @ weight[n_out, in]^T (+ bias[n_out])
// is computed as the plain TN GEMM
//   D[m = n_out, n = rows] = op_T(A = weight as col-major [in, n_out])
//                          * (B = act as col-major [in, rows]).
// A and B are always FP16; D is FP16 for the linear layers and the prefill
// PV product, FP32 for the prefill scores product. Compute type is always
// CUBLAS_COMPUTE_32F with FP32 alpha/beta.
struct GemmConfig {
  // Problem dimensions of the column-major dual: D is m x n, k is the
  // contracted dimension.
  std::int32_t m = 0;
  std::int32_t n = 0;
  std::int32_t k = 0;

  cublasOperation_t op_a = CUBLAS_OP_T;
  cublasOperation_t op_b = CUBLAS_OP_N;

  // Leading dimensions (elements) of the matrices AS STORED (before op_*).
  std::int64_t lda = 0;
  std::int64_t ldb = 0;
  std::int64_t ldd = 0; // C shares D's pointer and layout (beta is always 0)

  // Output element type: CUDA_R_16F or CUDA_R_32F. A/B are always CUDA_R_16F.
  cudaDataType_t type_d = CUDA_R_16F;

  GemmEpilogue epilogue = GemmEpilogue::kNone;

  // Strided-batch description. batch == 1 means a plain GEMM (strides
  // ignored, kept 0 so equal problems share one cache entry). For batch > 1
  // the strides are element offsets between consecutive batch matrices;
  // stride 0 on A or B broadcasts one operand across the whole batch (the
  // shared-K/V trick of docs/DESIGN.md section 6.3). stride_d must be
  // nonzero when batch > 1.
  std::int32_t batch = 1;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_d = 0;

  // Guaranteed minimum byte alignment of the A/B/D pointers, including every
  // strided-batch element (base + i*stride). Fed to the heuristic through
  // CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES so it never selects an
  // algorithm whose vectorized loads assume more alignment than the run-time
  // pointers deliver; cuBLASLt's default assumption is 256 bytes, which the
  // prefill batch strides (arbitrary T*ctx element offsets) do not satisfy.
  // The factory functions below derive conservative worst cases.
  std::int32_t min_align_a_bytes = 16;
  std::int32_t min_align_b_bytes = 16;
  std::int32_t min_align_d_bytes = 16;

  bool operator==(const GemmConfig&) const = default;

  // Linear-layer config (the section 6.1 shapes table): m = out_features,
  // n = rows, k = in_features; A = weight, row-major [out_features,
  // in_features] read as col-major with lda = in_features (weights are always
  // dense); B = activation with ldb = act_row_stride (dense: in_features;
  // the down projection reads the [rows, 8960] gate slice of the gate_up
  // buffer with act_row_stride 17920, docs/DESIGN.md section 6.2 #8);
  // D with ldd = out_row_stride (dense: out_features). Alignment guarantee
  // 16 bytes on every pointer - all engine buffers are cudaMalloc-derived at
  // offsets that are multiples of a full FP16 row, far above 16.
  static GemmConfig Linear(std::int32_t rows, std::int32_t out_features, std::int32_t in_features,
                           std::int64_t act_row_stride, std::int64_t out_row_stride,
                           GemmEpilogue epilogue = GemmEpilogue::kNone);

  // Prefill scores product for one KV head group g, encoding the
  // docs/DESIGN.md section 6.3 recipe verbatim. Logical, batched over the
  // heads_per_group Q heads h of the group:
  //   S_h[T, ctx] = alpha * Q_h[T, 128] @ K_g[ctx, 128]^T
  // Dual: D[ctx, T] = op_T(A) * B with
  //   A = khat[g] (dense [ctx, head_dim] row-major = col-major [head_dim,
  //       ctx]), op_A = T, lda = head_dim, batch stride 0 - all heads of the
  //       group share the one gathered K tile;
  //   B = Q head h as a view into qkv_out: col-major [head_dim, T],
  //       op_B = N, ldb = q_row_stride (2048), batch stride head_dim (128) -
  //       the next Q head starts 128 elements into the same row;
  //   D = scores + h*T*ctx, FP32, col-major [ctx, T], ldd = ctx, batch
  //       stride T*ctx (rows packed at the current ctx, not max_seq_len).
  // alpha (1/sqrt(head_dim)) is a run-time argument, not part of the config.
  // Alignment: A is a 256B-aligned gather buffer at head offsets that are
  // multiples of ctx*128 halves; B batch elements step 128 halves = 256 B;
  // D batch elements step T*ctx floats - worst case 4 B, hence min_align_d 4.
  // heads_per_group = 1 yields the non-batched per-head fallback config
  // (docs/DESIGN.md section 16 fallback chain).
  static GemmConfig PrefillScores(std::int32_t chunk_rows, std::int32_t ctx, std::int32_t head_dim,
                                  std::int32_t heads_per_group, std::int64_t q_row_stride);

  // Prefill PV product for one KV head group g (docs/DESIGN.md section 6.3):
  //   O_h[T, 128] = P_h[T, ctx] @ V_g[ctx, 128]
  // Dual: D[128, T] = A * B with
  //   A = vhat[g] (col-major [head_dim, ctx]), op_A = N, lda = head_dim,
  //       batch stride 0 (shared V tile);
  //   B = probs + h*T*ctx, FP16, col-major [ctx, T], op_B = N, ldb = ctx,
  //       batch stride T*ctx - worst-case element offset alignment is one
  //       FP16 element, hence min_align_b 2;
  //   D written straight into attn_out at column offset h*head_dim:
  //       col-major [head_dim, T], ldd = out_row_stride (1536), batch stride
  //       head_dim - no head-to-row repack kernel exists in the design.
  static GemmConfig PrefillPv(std::int32_t chunk_rows, std::int32_t ctx, std::int32_t head_dim,
                              std::int32_t heads_per_group, std::int64_t out_row_stride);
};

// Hash for the plan-cache key (all fields mixed; padding never read).
struct GemmConfigHash {
  std::size_t operator()(const GemmConfig& config) const noexcept;
};

// Owner of the cuBLASLt handle plus the single shared device workspace,
// serving every matmul in the model: QKV projection (fused bias epilogue),
// O projection, fused gate+up projection, down projection, lm_head, and the
// prefill scores/PV strided-batched products. FP16 inputs, FP32 compute.
//
// API surface:
//   - Init            - create the handle, allocate the shared workspace
//                       (default 32 MiB, docs/DESIGN.md section 6.1), and run
//                       the section 6.1 shapes-table self-check, which builds
//                       every descriptor the model uses and records heuristic
//                       availability for each.
//   - Probe           - build + cache the plan for one configuration and
//                       report availability (heuristic returned at least one
//                       algorithm), so the engine can select the documented
//                       fallbacks (qkv: plain GEMM with the bias folded into
//                       a following kernel; prefill: per-head non-batched
//                       calls, then repack kernels - docs/DESIGN.md section
//                       16) and record the decision.
//   - GemmFp16        - run, plain.
//   - GemmBiasFp16    - run, CUBLASLT_EPILOGUE_BIAS (per-output-feature FP16
//                       vector).
//   - GemmStridedBatched (half* / float* D overloads) - run, strided-batched
//                       with stride-0 operand broadcast and caller alpha.
//   - AlgoReportJson  - the selected-algorithm report (device, cuBLASLt
//                       version, every cached configuration with its
//                       availability and algo IDs) as a JSON string, consumed
//                       by the test/bench result JSON (docs/DESIGN.md
//                       section 12d).
//
// Plan cache & heuristic timing (docs/DESIGN.md sections 6.1, 9): heuristics
// run once per configuration and are cached together with fully built
// descriptors. A run call whose configuration is already cached performs NO
// allocation and NO heuristic query - it is a hash lookup plus
// cublasLtMatmul, and is therefore safe inside CUDA-graph capture. A run
// call on an uncached configuration fills the cache once; it first verifies
// the stream is NOT capturing and throws GemmError otherwise, because a
// heuristic query inside a capture region is a bug by design. The engine's
// init-time per-bucket eager warmup (docs/DESIGN.md section 9) drives every
// decode-path configuration through that fill before any capture; eager
// prefill configurations (whose ctx varies per chunk and cannot be
// enumerated at init) fill lazily on first use and are never captured.
//
// Not thread-safe; matches the engine's single-thread contract
// (docs/DESIGN.md section 10). All run calls are asynchronous on `stream`.
class CublasLtGemm {
 public:
  // Shared workspace size (docs/DESIGN.md section 6.1). The heuristic query
  // is capped to this, so a cached algorithm never needs more at run time.
  static constexpr std::size_t kDefaultWorkspaceBytes = 32ull << 20;

  CublasLtGemm(); // trivial; safe to construct without a CUDA device
  ~CublasLtGemm();
  CublasLtGemm(const CublasLtGemm&) = delete;
  CublasLtGemm& operator=(const CublasLtGemm&) = delete;

  // Creates the cuBLASLt handle, allocates the shared device workspace, and
  // runs the section 6.1 shapes-table self-check: descriptors are built (a
  // build failure throws GemmError - the wrapper itself is broken) and the
  // heuristic is probed (zero algorithms is NOT an init failure; it is
  // recorded, surfaces through Probe/AlgoReportJson, and the affected run
  // call throws if the caller ignores it) for
  //   qkv 2048x1536 (+BIAS and the plain fallback), o 1536x1536,
  //   gate_up 17920x1536, down 1536x8960 (act stride 17920),
  //   lm_head 151936x1536, and representative prefill scores/PV layouts in
  //   both batched (6) and per-head fallback (1) forms.
  // Throws GemmError on handle/workspace/descriptor failure or double Init.
  void Init(std::size_t workspace_bytes = kDefaultWorkspaceBytes);

  bool initialized() const { return handle_ != nullptr; }

  // Builds (or reuses) the cached plan for `config` and returns availability:
  // true iff the heuristic offers at least one algorithm within the shared
  // workspace budget. Init-time API (also legal from any non-captured
  // context); `tag` labels the configuration in the algo report. Throws
  // GemmError before Init or on descriptor-construction failure.
  bool Probe(const GemmConfig& config, std::string_view tag = "probe");

  // Plain linear layer (run path):
  //   out[rows, out_features] = act[rows, in_features]
  //                             @ weight[out_features, in_features]^T
  // `weight` keeps the safetensors row-major [out_features, in_features]
  // layout. Row strides are in elements; dense callers pass the row widths
  // (act_row_stride = in_features, out_row_stride = out_features). The down
  // projection passes act_row_stride 17920 to read the gate slice in place.
  void GemmFp16(half* out, std::int64_t out_row_stride, const half* act,
                std::int64_t act_row_stride, const half* weight, std::int32_t rows,
                std::int32_t out_features, std::int32_t in_features, cudaStream_t stream);

  // Linear layer with the fused bias epilogue (the QKV projection):
  // bias is an FP16 [out_features] vector added per output feature
  // (CUBLASLT_EPILOGUE_BIAS; length m in the column-major dual).
  void GemmBiasFp16(half* out, std::int64_t out_row_stride, const half* act,
                    std::int64_t act_row_stride, const half* weight, const half* bias,
                    std::int32_t rows, std::int32_t out_features, std::int32_t in_features,
                    cudaStream_t stream);

  // Strided-batched matmul, D = alpha * op(A) * op(B), FP16 D (prefill PV;
  // config from GemmConfig::PrefillPv or caller-built). `a`/`b`/`d` are the
  // batch-0 base pointers; per-element offsets come from config strides
  // (stride 0 broadcasts A or B). config.type_d must be CUDA_R_16F and
  // config.epilogue kNone.
  void GemmStridedBatched(half* d, const half* a, const half* b, float alpha,
                          const GemmConfig& config, cudaStream_t stream);

  // Strided-batched matmul with FP32 D (prefill scores; config from
  // GemmConfig::PrefillScores). config.type_d must be CUDA_R_32F and
  // config.epilogue kNone.
  void GemmStridedBatched(float* d, const half* a, const half* b, float alpha,
                          const GemmConfig& config, cudaStream_t stream);

  // Selected-algorithm report as a JSON string: device name, compute
  // capability, cuBLASLt version, workspace size, and one entry per cached
  // configuration (tag, full config tuple, availability, algo id / tile /
  // stages / split-k / reduction scheme / swizzle / custom option, the
  // algorithm's workspace requirement, waves count). Consumed by the
  // test/bench result JSON (docs/DESIGN.md section 12d). Valid before Init
  // (reports initialized=false, empty config list).
  std::string AlgoReportJson() const;

 private:
  struct Plan; // descriptors + heuristic result for one GemmConfig

  // Zeroes the (meaningless) batch strides of non-batched configs so equal
  // problems share one cache entry.
  static GemmConfig Normalized(const GemmConfig& config);
  Plan* FindPlan(const GemmConfig& normalized);
  // Validates the config, builds all descriptors, runs the one-time
  // heuristic query, and caches the result (also when unavailable).
  Plan& BuildPlan(const GemmConfig& normalized, std::string_view tag);
  // Run-path lookup; on miss, verifies `stream` is not capturing and builds
  // the plan. Never allocates and never touches a CUDA API on a hit.
  Plan& PlanFor(const GemmConfig& normalized, cudaStream_t stream, std::string_view tag);
  void RunImpl(void* d, const void* a, const void* b, const void* bias, float alpha,
               const GemmConfig& config, std::string_view tag, cudaStream_t stream);
  // The section 6.1 shapes-table + section 6.3 prefill-layout probes run by
  // Init; entries appear in the algo report tagged "self_check:*".
  void SelfCheckModelShapes();
  void RequireInit() const;

  cublasLtHandle_t handle_ = nullptr;
  void* workspace_ = nullptr;
  std::size_t workspace_bytes_ = 0;

  // Report metadata captured at Init.
  std::string device_name_;
  std::int32_t cc_major_ = 0;
  std::int32_t cc_minor_ = 0;
  std::size_t cublaslt_version_ = 0;

  // Plans owned behind unique_ptr so cached references stay stable across
  // cache growth; insertion order preserved for the report.
  std::vector<std::unique_ptr<Plan>> plans_;
  std::unordered_map<GemmConfig, std::size_t, GemmConfigHash> plan_index_;
};

} // namespace redline
