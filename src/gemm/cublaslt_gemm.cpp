#include "gemm/cublaslt_gemm.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string>
#include <system_error>
#include <utility>

namespace redline {

namespace {

// How many candidates to request per heuristic query. The first result is
// the heuristic's preferred choice; the rest are fallbacks in case a
// candidate reports a failed state or an over-budget workspace.
constexpr int kRequestedAlgos = 8;

// Local status-name table so this translation unit needs no symbol from the
// full cuBLAS library (redline_core links CUDA::cublasLt only).
const char* CublasStatusName(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_<unknown>";
  }
}

const char* OpName(cublasOperation_t op) {
  switch (op) {
  case CUBLAS_OP_N:
    return "N";
  case CUBLAS_OP_T:
    return "T";
  case CUBLAS_OP_C:
    return "C";
  default:
    return "?";
  }
}

const char* TypeName(cudaDataType_t type) {
  switch (type) {
  case CUDA_R_16F:
    return "f16";
  case CUDA_R_32F:
    return "f32";
  default:
    return "?";
  }
}

const char* EpilogueName(GemmEpilogue epilogue) {
  return epilogue == GemmEpilogue::kBias ? "bias" : "none";
}

std::string DescribeConfig(const GemmConfig& c) {
  std::string s = "{m=" + std::to_string(c.m) + ", n=" + std::to_string(c.n) +
                  ", k=" + std::to_string(c.k) + ", op_a=" + OpName(c.op_a) +
                  ", op_b=" + OpName(c.op_b) + ", lda=" + std::to_string(c.lda) +
                  ", ldb=" + std::to_string(c.ldb) + ", ldd=" + std::to_string(c.ldd) +
                  ", type_d=" + TypeName(c.type_d) + ", epilogue=" + EpilogueName(c.epilogue) +
                  ", batch=" + std::to_string(c.batch);
  if (c.batch > 1) {
    s += ", stride_a=" + std::to_string(c.stride_a) + ", stride_b=" + std::to_string(c.stride_b) +
         ", stride_d=" + std::to_string(c.stride_d);
  }
  s += "}";
  return s;
}

void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw GemmError(std::string("cublaslt_gemm: ") + what + " failed: " + CublasStatusName(status));
  }
}

void CheckCublas(cublasStatus_t status, const char* what, const GemmConfig& config) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw GemmError(std::string("cublaslt_gemm: ") + what + " failed: " + CublasStatusName(status) +
                    " for " + DescribeConfig(config));
  }
}

void CheckCuda(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw GemmError(std::string("cublaslt_gemm: ") + what + " failed: " + cudaGetErrorName(error) +
                    " (" + cudaGetErrorString(error) + ")");
  }
}

bool IsPowerOfTwo(std::int32_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

void ValidateConfig(const GemmConfig& c) {
  const auto fail = [&c](const char* why) {
    throw GemmError(std::string("cublaslt_gemm: invalid GemmConfig (") + why +
                    "): " + DescribeConfig(c));
  };
  if (c.m < 1 || c.n < 1 || c.k < 1)
    fail("m, n, k must all be >= 1");
  if (c.op_a != CUBLAS_OP_N && c.op_a != CUBLAS_OP_T)
    fail("op_a must be N or T");
  if (c.op_b != CUBLAS_OP_N && c.op_b != CUBLAS_OP_T)
    fail("op_b must be N or T");
  // ld of a column-major matrix must cover its stored row count.
  const std::int64_t stored_rows_a = (c.op_a == CUBLAS_OP_N) ? c.m : c.k;
  const std::int64_t stored_rows_b = (c.op_b == CUBLAS_OP_N) ? c.k : c.n;
  if (c.lda < stored_rows_a)
    fail("lda smaller than the stored A row count");
  if (c.ldb < stored_rows_b)
    fail("ldb smaller than the stored B row count");
  if (c.ldd < c.m)
    fail("ldd smaller than m");
  if (c.type_d != CUDA_R_16F && c.type_d != CUDA_R_32F) {
    fail("type_d must be CUDA_R_16F or CUDA_R_32F");
  }
  if (c.batch < 1)
    fail("batch must be >= 1");
  if (c.batch > 1) {
    if (c.stride_a < 0 || c.stride_b < 0 || c.stride_d < 0) {
      fail("batch strides must be non-negative");
    }
    if (c.stride_d == 0)
      fail("stride_d of 0 would overlap batched outputs");
  }
  if (c.epilogue == GemmEpilogue::kBias) {
    if (c.batch != 1)
      fail("the bias epilogue is only used non-batched (qkv projection)");
    if (c.type_d != CUDA_R_16F)
      fail("the bias epilogue requires FP16 D");
  } else if (c.epilogue != GemmEpilogue::kNone) {
    fail("unknown epilogue");
  }
  if (!IsPowerOfTwo(c.min_align_a_bytes) || !IsPowerOfTwo(c.min_align_b_bytes) ||
      !IsPowerOfTwo(c.min_align_d_bytes)) {
    fail("minimum alignments must be powers of two");
  }
}

// Destroys a matmul preference on scope exit, including on the throw paths.
struct PreferenceGuard {
  cublasLtMatmulPreference_t pref = nullptr;
  ~PreferenceGuard() {
    if (pref != nullptr)
      cublasLtMatmulPreferenceDestroy(pref);
  }
};

std::string EscapeJson(std::string_view in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (const char ch : in) {
    switch (ch) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(ch) & 0xff);
        out += buf;
      } else {
        out += ch;
      }
    }
  }
  return out;
}

// Locale-independent float formatting (std::to_chars); the report may be
// generated inside a Python process whose locale is not "C".
void AppendJsonFloat(std::string& json, float value) {
  if (!std::isfinite(value)) {
    json += "0";
    return;
  }
  char buf[48];
  const auto result = std::to_chars(buf, buf + sizeof(buf), static_cast<double>(value));
  if (result.ec == std::errc()) {
    json.append(buf, result.ptr);
  } else {
    json += "0";
  }
}

} // namespace

// ---------------------------------------------------------------- GemmConfig

GemmConfig GemmConfig::Linear(std::int32_t rows, std::int32_t out_features,
                              std::int32_t in_features, std::int64_t act_row_stride,
                              std::int64_t out_row_stride, GemmEpilogue epilogue) {
  GemmConfig config;
  config.m = out_features;
  config.n = rows;
  config.k = in_features;
  config.op_a = CUBLAS_OP_T; // weight row-major [n_out, in] read col-major [in, n_out]
  config.op_b = CUBLAS_OP_N; // activation row-major [rows, in] read col-major [in, rows]
  config.lda = in_features;  // weights are always dense
  config.ldb = act_row_stride;
  config.ldd = out_row_stride;
  config.type_d = CUDA_R_16F;
  config.epilogue = epilogue;
  config.batch = 1;
  config.stride_a = 0;
  config.stride_b = 0;
  config.stride_d = 0;
  // Every linear-layer pointer the engine passes is a cudaMalloc-derived
  // buffer at an offset that is a multiple of a full FP16 row (>= 1536
  // elements = 3072 bytes); 16 bytes is a conservative floor that still
  // admits 8-wide half vectorized algorithms.
  config.min_align_a_bytes = 16;
  config.min_align_b_bytes = 16;
  config.min_align_d_bytes = 16;
  return config;
}

GemmConfig GemmConfig::PrefillScores(std::int32_t chunk_rows, std::int32_t ctx,
                                     std::int32_t head_dim, std::int32_t heads_per_group,
                                     std::int64_t q_row_stride) {
  // docs/DESIGN.md section 6.3, scores recipe: D[ctx, T] = op_T(A) * B,
  // batched over the group's Q heads with the K tile broadcast (stride 0).
  GemmConfig config;
  config.m = ctx;
  config.n = chunk_rows;
  config.k = head_dim;
  config.op_a = CUBLAS_OP_T; // khat[g]: dense [ctx, head_dim] row-major
                             //   == col-major [head_dim, ctx], transposed
  config.op_b = CUBLAS_OP_N; // Q head view into qkv_out: col-major [head_dim, T]
  config.lda = head_dim;
  config.ldb = q_row_stride;  // canonical 2048: next token's row in qkv_out
  config.ldd = ctx;           // scores packed [T, ctx] row-major per head
  config.type_d = CUDA_R_32F; // FP32 scores feed the prefill softmax
  config.epilogue = GemmEpilogue::kNone;
  config.batch = heads_per_group; // 6 for the batched path, 1 for the per-head fallback
  if (heads_per_group > 1) {
    config.stride_a = 0;        // all heads of the group share one gathered K tile
    config.stride_b = head_dim; // next Q head starts 128 elements into the same row
    config.stride_d = static_cast<std::int64_t>(chunk_rows) * ctx;
  } else {
    config.stride_a = 0;
    config.stride_b = 0;
    config.stride_d = 0;
  }
  // A: khat head slices sit at multiples of ctx*head_dim halves (>= 256 B).
  // B: Q head offsets are multiples of head_dim = 128 halves = 256 B.
  // D: batch elements step chunk_rows*ctx floats - worst case one float.
  config.min_align_a_bytes = 16;
  config.min_align_b_bytes = 16;
  config.min_align_d_bytes = 4;
  return config;
}

GemmConfig GemmConfig::PrefillPv(std::int32_t chunk_rows, std::int32_t ctx, std::int32_t head_dim,
                                 std::int32_t heads_per_group, std::int64_t out_row_stride) {
  // docs/DESIGN.md section 6.3, PV recipe: D[head_dim, T] = A * B with the V
  // tile broadcast (stride 0) and D written straight into attn_out at column
  // offset h*head_dim - no head-to-row repack kernel exists in the design.
  GemmConfig config;
  config.m = head_dim;
  config.n = chunk_rows;
  config.k = ctx;
  config.op_a = CUBLAS_OP_N; // vhat[g]: dense [ctx, head_dim] row-major
                             //   == col-major [head_dim, ctx]
  config.op_b = CUBLAS_OP_N; // probs head slice: col-major [ctx, T]
  config.lda = head_dim;
  config.ldb = ctx;
  config.ldd = out_row_stride; // canonical 1536: attn_out row width
  config.type_d = CUDA_R_16F;
  config.epilogue = GemmEpilogue::kNone;
  config.batch = heads_per_group;
  if (heads_per_group > 1) {
    config.stride_a = 0; // shared V tile
    config.stride_b = static_cast<std::int64_t>(chunk_rows) * ctx;
    config.stride_d = head_dim; // next head's 128 columns of attn_out
  } else {
    config.stride_a = 0;
    config.stride_b = 0;
    config.stride_d = 0;
  }
  // A: vhat head slices, >= 256 B. B: probs batch elements step
  // chunk_rows*ctx halves - worst case one half (2 bytes). D: attn_out head
  // offsets are multiples of 128 halves = 256 B.
  config.min_align_a_bytes = 16;
  config.min_align_b_bytes = 2;
  config.min_align_d_bytes = 16;
  return config;
}

std::size_t GemmConfigHash::operator()(const GemmConfig& config) const noexcept {
  std::uint64_t h = 0x9e3779b97f4a7c15ull;
  const auto mix = [&h](std::uint64_t value) {
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 31;
    h = (h ^ value) * 0x94d049bb133111ebull;
  };
  mix(static_cast<std::uint32_t>(config.m));
  mix(static_cast<std::uint32_t>(config.n));
  mix(static_cast<std::uint32_t>(config.k));
  mix(static_cast<std::uint32_t>(config.op_a));
  mix(static_cast<std::uint32_t>(config.op_b));
  mix(static_cast<std::uint64_t>(config.lda));
  mix(static_cast<std::uint64_t>(config.ldb));
  mix(static_cast<std::uint64_t>(config.ldd));
  mix(static_cast<std::uint32_t>(config.type_d));
  mix(static_cast<std::uint32_t>(config.epilogue));
  mix(static_cast<std::uint32_t>(config.batch));
  mix(static_cast<std::uint64_t>(config.stride_a));
  mix(static_cast<std::uint64_t>(config.stride_b));
  mix(static_cast<std::uint64_t>(config.stride_d));
  mix(static_cast<std::uint32_t>(config.min_align_a_bytes));
  mix(static_cast<std::uint32_t>(config.min_align_b_bytes));
  mix(static_cast<std::uint32_t>(config.min_align_d_bytes));
  return static_cast<std::size_t>(h ^ (h >> 32));
}

// ---------------------------------------------------------------------- Plan

// Fully built descriptor set plus the heuristic selection for one
// configuration. Owned behind unique_ptr in the cache; the destructor
// releases the cuBLASLt objects (also on partially-built error paths).
struct CublasLtGemm::Plan {
  GemmConfig config;
  std::string tag;

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr; // C shares D's pointer/layout (beta = 0)
  cublasLtMatrixLayout_t layout_d = nullptr;

  bool available = false;      // heuristic returned a usable algorithm
  cublasLtMatmulAlgo_t algo{}; // valid only when available
  std::size_t algo_workspace_bytes = 0;
  float waves_count = 0.0f;
  // Algorithm identity for the report (docs/DESIGN.md section 12d); -1 when
  // an attribute cannot be read.
  std::int64_t algo_id = -1;
  std::int64_t tile_id = -1;
  std::int64_t stages_id = -1;
  std::int64_t splitk_num = -1;
  std::int64_t reduction_scheme = -1;
  std::int64_t cta_swizzling = -1;
  std::int64_t custom_option = -1;

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (layout_d != nullptr)
      cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != nullptr)
      cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != nullptr)
      cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != nullptr)
      cublasLtMatrixLayoutDestroy(layout_a);
    if (op_desc != nullptr)
      cublasLtMatmulDescDestroy(op_desc);
  }
};

// -------------------------------------------------------------- CublasLtGemm

CublasLtGemm::CublasLtGemm() = default;

CublasLtGemm::~CublasLtGemm() {
  plans_.clear(); // Plan destructors release the cuBLASLt descriptors
  if (workspace_ != nullptr) {
    cudaFree(workspace_); // best effort; destructors must not throw
    workspace_ = nullptr;
  }
  if (handle_ != nullptr) {
    cublasLtDestroy(handle_);
    handle_ = nullptr;
  }
}

void CublasLtGemm::Init(std::size_t workspace_bytes) {
  if (initialized()) {
    throw GemmError("cublaslt_gemm: Init() called twice");
  }
  if (workspace_bytes == 0) {
    throw GemmError("cublaslt_gemm: workspace size must be nonzero");
  }
  cublasLtHandle_t handle = nullptr;
  CheckCublas(cublasLtCreate(&handle), "cublasLtCreate");
  handle_ = handle; // owned from here; the destructor cleans up on any throw below

  workspace_bytes_ = workspace_bytes;
  const cudaError_t alloc_status = cudaMalloc(&workspace_, workspace_bytes_);
  if (alloc_status != cudaSuccess) {
    workspace_ = nullptr;
    throw GemmError("cublaslt_gemm: cudaMalloc of the " + std::to_string(workspace_bytes_) +
                    "-byte shared workspace failed: " + cudaGetErrorName(alloc_status));
  }

  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  CheckCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  device_name_ = prop.name;
  cc_major_ = prop.major;
  cc_minor_ = prop.minor;
  cublaslt_version_ = cublasLtGetVersion();

  SelfCheckModelShapes();
}

void CublasLtGemm::RequireInit() const {
  if (!initialized()) {
    throw GemmError("cublaslt_gemm: Init() must be called before use");
  }
}

GemmConfig CublasLtGemm::Normalized(const GemmConfig& config) {
  GemmConfig normalized = config;
  if (normalized.batch == 1) {
    // Strides are meaningless for a non-batched problem; zero them so equal
    // problems share one cache entry regardless of how the caller filled them.
    normalized.stride_a = 0;
    normalized.stride_b = 0;
    normalized.stride_d = 0;
  }
  return normalized;
}

CublasLtGemm::Plan* CublasLtGemm::FindPlan(const GemmConfig& normalized) {
  const auto it = plan_index_.find(normalized);
  return it == plan_index_.end() ? nullptr : plans_[it->second].get();
}

bool CublasLtGemm::Probe(const GemmConfig& config, std::string_view tag) {
  RequireInit();
  const GemmConfig normalized = Normalized(config);
  if (Plan* plan = FindPlan(normalized)) {
    return plan->available;
  }
  return BuildPlan(normalized, tag).available;
}

CublasLtGemm::Plan& CublasLtGemm::BuildPlan(const GemmConfig& config, std::string_view tag) {
  ValidateConfig(config);

  auto plan = std::make_unique<Plan>();
  plan->config = config;
  plan->tag = std::string(tag);

  // Operation descriptor: FP32 compute and FP32 alpha/beta for every problem
  // (docs/DESIGN.md section 6.1).
  CheckCublas(cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
              "cublasLtMatmulDescCreate", config);
  const std::int32_t transa = static_cast<std::int32_t>(config.op_a);
  const std::int32_t transb = static_cast<std::int32_t>(config.op_b);
  CheckCublas(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                             sizeof(transa)),
              "set CUBLASLT_MATMUL_DESC_TRANSA", config);
  CheckCublas(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                             sizeof(transb)),
              "set CUBLASLT_MATMUL_DESC_TRANSB", config);
  if (config.epilogue == GemmEpilogue::kBias) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CheckCublas(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue)),
                "set CUBLASLT_MATMUL_DESC_EPILOGUE", config);
    // FP16 bias vector for FP16-D / FP32-compute, matching the stored b_qkv;
    // set explicitly rather than relying on the default-to-D-type rule.
    const std::int32_t bias_type = static_cast<std::int32_t>(CUDA_R_16F);
    CheckCublas(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                               &bias_type, sizeof(bias_type)),
                "set CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE", config);
    // The bias pointer itself is a per-call value, set in RunImpl.
  }

  // Matrix layouts describe the matrices AS STORED (column-major); op_a/op_b
  // apply on top. A and B are always FP16; C mirrors D because beta is 0 and
  // they share one pointer.
  const auto stored_rows_a =
      static_cast<std::uint64_t>(config.op_a == CUBLAS_OP_N ? config.m : config.k);
  const auto stored_cols_a =
      static_cast<std::uint64_t>(config.op_a == CUBLAS_OP_N ? config.k : config.m);
  const auto stored_rows_b =
      static_cast<std::uint64_t>(config.op_b == CUBLAS_OP_N ? config.k : config.n);
  const auto stored_cols_b =
      static_cast<std::uint64_t>(config.op_b == CUBLAS_OP_N ? config.n : config.k);
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->layout_a, CUDA_R_16F, stored_rows_a, stored_cols_a,
                                         config.lda),
              "create A layout", config);
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->layout_b, CUDA_R_16F, stored_rows_b, stored_cols_b,
                                         config.ldb),
              "create B layout", config);
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->layout_c, config.type_d,
                                         static_cast<std::uint64_t>(config.m),
                                         static_cast<std::uint64_t>(config.n), config.ldd),
              "create C layout", config);
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->layout_d, config.type_d,
                                         static_cast<std::uint64_t>(config.m),
                                         static_cast<std::uint64_t>(config.n), config.ldd),
              "create D layout", config);

  if (config.batch > 1) {
    const auto set_batch = [&](cublasLtMatrixLayout_t layout, std::int64_t stride,
                               const char* what) {
      CheckCublas(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                   &config.batch, sizeof(config.batch)),
                  what, config);
      CheckCublas(cublasLtMatrixLayoutSetAttribute(
                      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
                  what, config);
    };
    set_batch(plan->layout_a, config.stride_a, "set A batch attributes");
    set_batch(plan->layout_b, config.stride_b, "set B batch attributes");
    set_batch(plan->layout_c, config.stride_d, "set C batch attributes");
    set_batch(plan->layout_d, config.stride_d, "set D batch attributes");
  }

  // Heuristic query, bounded by the shared workspace and by the alignment the
  // run-time pointers actually guarantee (cuBLASLt otherwise assumes 256 B,
  // which the prefill batch strides do not satisfy).
  PreferenceGuard pref;
  CheckCublas(cublasLtMatmulPreferenceCreate(&pref.pref), "cublasLtMatmulPreferenceCreate", config);
  const std::uint64_t max_workspace = workspace_bytes_;
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(pref.pref,
                                                   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &max_workspace, sizeof(max_workspace)),
              "set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", config);
  const auto align_a = static_cast<std::uint32_t>(config.min_align_a_bytes);
  const auto align_b = static_cast<std::uint32_t>(config.min_align_b_bytes);
  const auto align_d = static_cast<std::uint32_t>(config.min_align_d_bytes);
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_a, sizeof(align_a)),
              "set CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES", config);
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_b, sizeof(align_b)),
              "set CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES", config);
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_d, sizeof(align_d)),
              "set CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES", config);
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_d, sizeof(align_d)),
              "set CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES", config);

  cublasLtMatmulHeuristicResult_t results[kRequestedAlgos] = {};
  int returned = 0;
  const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle_, plan->op_desc, plan->layout_a, plan->layout_b, plan->layout_c, plan->layout_d,
      pref.pref, kRequestedAlgos, results, &returned);
  if (heuristic_status == CUBLAS_STATUS_SUCCESS) {
    for (int i = 0; i < returned; ++i) {
      if (results[i].state != CUBLAS_STATUS_SUCCESS)
        continue;
      if (results[i].workspaceSize > workspace_bytes_)
        continue;
      plan->algo = results[i].algo;
      plan->algo_workspace_bytes = results[i].workspaceSize;
      plan->waves_count = results[i].wavesCount;
      plan->available = true;
      break;
    }
  } else if (heuristic_status != CUBLAS_STATUS_NOT_SUPPORTED) {
    // NOT_SUPPORTED means "no algorithm for this configuration" - a probe
    // result, not an error. Anything else is a wrapper bug (bad descriptor).
    CheckCublas(heuristic_status, "cublasLtMatmulAlgoGetHeuristic", config);
  }

  if (plan->available) {
    // Algorithm identity for the report; tolerate unreadable attributes.
    const auto read_algo_attr = [&plan](cublasLtMatmulAlgoConfigAttributes_t attr) -> std::int64_t {
      std::int32_t value = 0;
      std::size_t written = 0;
      const cublasStatus_t status =
          cublasLtMatmulAlgoConfigGetAttribute(&plan->algo, attr, &value, sizeof(value), &written);
      if (status != CUBLAS_STATUS_SUCCESS || written != sizeof(value))
        return -1;
      return value;
    };
    plan->algo_id = read_algo_attr(CUBLASLT_ALGO_CONFIG_ID);
    plan->tile_id = read_algo_attr(CUBLASLT_ALGO_CONFIG_TILE_ID);
    plan->stages_id = read_algo_attr(CUBLASLT_ALGO_CONFIG_STAGES_ID);
    plan->splitk_num = read_algo_attr(CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    plan->reduction_scheme = read_algo_attr(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    plan->cta_swizzling = read_algo_attr(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
    plan->custom_option = read_algo_attr(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  }

  plans_.push_back(std::move(plan));
  plan_index_.emplace(config, plans_.size() - 1);
  return *plans_.back();
}

CublasLtGemm::Plan& CublasLtGemm::PlanFor(const GemmConfig& normalized, cudaStream_t stream,
                                          std::string_view tag) {
  if (Plan* plan = FindPlan(normalized)) {
    return *plan; // hot path: no allocation, no heuristic, no CUDA API call
  }
  RequireInit();
  // Cache miss. A heuristic query inside a CUDA-graph capture region is a bug
  // by design (docs/DESIGN.md sections 6.1, 9): decode shapes must have been
  // filled by the init-time eager warmup before any capture. Eager-only
  // prefill shapes (per-chunk ctx) legitimately reach this once per shape.
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  const cudaError_t capture_query = cudaStreamIsCapturing(stream, &capture);
  if (capture_query != cudaSuccess) {
    throw GemmError(std::string("cublaslt_gemm: cudaStreamIsCapturing failed: ") +
                    cudaGetErrorName(capture_query) + " for " + DescribeConfig(normalized));
  }
  if (capture != cudaStreamCaptureStatusNone) {
    throw GemmError("cublaslt_gemm: configuration missing from the plan cache while the stream is "
                    "capturing - heuristic queries inside capture are bugs; probe or warm up every "
                    "captured shape at init (docs/DESIGN.md sections 6.1, 9): " +
                    DescribeConfig(normalized));
  }
  return BuildPlan(normalized, tag);
}

void CublasLtGemm::RunImpl(void* d, const void* a, const void* b, const void* bias, float alpha,
                           const GemmConfig& config, std::string_view tag, cudaStream_t stream) {
  RequireInit();
  const GemmConfig normalized = Normalized(config);
  Plan& plan = PlanFor(normalized, stream, tag);
  if (!plan.available) {
    throw GemmError("cublaslt_gemm: no cuBLASLt algorithm available for " +
                    DescribeConfig(normalized) +
                    "; Probe() reported this configuration unavailable and the caller must "
                    "take its recorded fallback (docs/DESIGN.md section 16)");
  }
  if (normalized.epilogue == GemmEpilogue::kBias) {
    // Per-call pointer value on the cached descriptor: a host-side attribute
    // write, no allocation, capture-safe (each captured call bakes its own
    // bias pointer at capture time).
    CheckCublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &bias, sizeof(bias)),
                "set CUBLASLT_MATMUL_DESC_BIAS_POINTER", normalized);
  }
  const float beta = 0.0f; // D is never accumulated into; C aliases D
  CheckCublas(cublasLtMatmul(handle_, plan.op_desc, &alpha, a, plan.layout_a, b, plan.layout_b,
                             &beta, d, plan.layout_c, d, plan.layout_d, &plan.algo, workspace_,
                             workspace_bytes_, stream),
              "cublasLtMatmul", normalized);
}

void CublasLtGemm::GemmFp16(half* out, std::int64_t out_row_stride, const half* act,
                            std::int64_t act_row_stride, const half* weight, std::int32_t rows,
                            std::int32_t out_features, std::int32_t in_features,
                            cudaStream_t stream) {
  const GemmConfig config = GemmConfig::Linear(rows, out_features, in_features, act_row_stride,
                                               out_row_stride, GemmEpilogue::kNone);
  RunImpl(out, weight, act, /*bias=*/nullptr, /*alpha=*/1.0f, config, "linear", stream);
}

void CublasLtGemm::GemmBiasFp16(half* out, std::int64_t out_row_stride, const half* act,
                                std::int64_t act_row_stride, const half* weight, const half* bias,
                                std::int32_t rows, std::int32_t out_features,
                                std::int32_t in_features, cudaStream_t stream) {
  if (bias == nullptr) {
    throw GemmError("cublaslt_gemm: GemmBiasFp16 requires a non-null bias vector");
  }
  const GemmConfig config = GemmConfig::Linear(rows, out_features, in_features, act_row_stride,
                                               out_row_stride, GemmEpilogue::kBias);
  RunImpl(out, weight, act, bias, /*alpha=*/1.0f, config, "linear_bias", stream);
}

void CublasLtGemm::GemmStridedBatched(half* d, const half* a, const half* b, float alpha,
                                      const GemmConfig& config, cudaStream_t stream) {
  if (config.type_d != CUDA_R_16F) {
    throw GemmError("cublaslt_gemm: FP16-D overload called with config.type_d != CUDA_R_16F: " +
                    DescribeConfig(config));
  }
  if (config.epilogue != GemmEpilogue::kNone) {
    throw GemmError("cublaslt_gemm: strided-batched runs do not support epilogues: " +
                    DescribeConfig(config));
  }
  RunImpl(d, a, b, /*bias=*/nullptr, alpha, config, "strided_batched_f16", stream);
}

void CublasLtGemm::GemmStridedBatched(float* d, const half* a, const half* b, float alpha,
                                      const GemmConfig& config, cudaStream_t stream) {
  if (config.type_d != CUDA_R_32F) {
    throw GemmError("cublaslt_gemm: FP32-D overload called with config.type_d != CUDA_R_32F: " +
                    DescribeConfig(config));
  }
  if (config.epilogue != GemmEpilogue::kNone) {
    throw GemmError("cublaslt_gemm: strided-batched runs do not support epilogues: " +
                    DescribeConfig(config));
  }
  RunImpl(d, a, b, /*bias=*/nullptr, alpha, config, "strided_batched_f32", stream);
}

void CublasLtGemm::SelfCheckModelShapes() {
  // The docs/DESIGN.md section 6.1 shapes table plus the section 6.3 prefill
  // layouts (batched and per-head fallback forms). Building these descriptors
  // is the init self-check: a construction failure throws and fails Init; a
  // heuristic returning zero algorithms is recorded (Probe/AlgoReportJson)
  // for the engine's fallback decision, not an init failure. Real decode
  // bucket shapes are probed by the executor's eager warmup before graph
  // capture; per-ctx prefill shapes fill lazily on the eager path.
  // Constants mirror docs/MODEL_SPEC.md (Qwen2.5-1.5B-Instruct).
  constexpr std::int32_t kHidden = 1536;
  constexpr std::int32_t kQkvOut = 2048;     // (12 Q + 2 K + 2 V heads) x 128
  constexpr std::int32_t kGateUpOut = 17920; // gate 8960 | up 8960
  constexpr std::int32_t kIntermediate = 8960;
  constexpr std::int32_t kVocab = 151936;
  constexpr std::int32_t kHeadDim = 128;
  constexpr std::int32_t kGqaGroup = 6;        // 12 Q heads / 2 KV heads
  constexpr std::int64_t kQkvRowStride = 2048; // Q/K/V strided views into qkv_out
  constexpr std::int64_t kAttnOutStride = 1536;
  constexpr std::int32_t kProbeRows = 1;   // representative decode row count
  constexpr std::int32_t kProbeChunk = 64; // representative prefill chunk rows
  constexpr std::int32_t kProbeCtx = 256;  // representative gathered context

  // qkv: 2048 x T x 1536, BIAS epilogue - and its documented fallback form.
  Probe(GemmConfig::Linear(kProbeRows, kQkvOut, kHidden, kHidden, kQkvOut, GemmEpilogue::kBias),
        "self_check:qkv_bias");
  Probe(GemmConfig::Linear(kProbeRows, kQkvOut, kHidden, kHidden, kQkvOut),
        "self_check:qkv_plain_fallback");
  // o: 1536 x T x 1536.
  Probe(GemmConfig::Linear(kProbeRows, kHidden, kHidden, kHidden, kHidden), "self_check:o_proj");
  // gate_up: 17920 x T x 1536.
  Probe(GemmConfig::Linear(kProbeRows, kGateUpOut, kHidden, kHidden, kGateUpOut),
        "self_check:gate_up");
  // down: 1536 x T x 8960, reading the [rows, 8960] gate slice of the
  // gate_up buffer in place (act row stride 17920, docs/DESIGN.md 6.2 #8).
  Probe(GemmConfig::Linear(kProbeRows, kHidden, kIntermediate, kGateUpOut, kHidden),
        "self_check:down");
  // lm_head: 151936 x R x 1536 (weight aliases the embedding table).
  Probe(GemmConfig::Linear(kProbeRows, kVocab, kHidden, kHidden, kVocab), "self_check:lm_head");
  // Prefill scores/PV, batched over the 6 Q heads of a KV group with the
  // stride-0 K/V broadcast, plus the per-head non-batched fallback forms
  // (docs/DESIGN.md sections 6.3, 16).
  Probe(GemmConfig::PrefillScores(kProbeChunk, kProbeCtx, kHeadDim, kGqaGroup, kQkvRowStride),
        "self_check:prefill_scores_batched");
  Probe(GemmConfig::PrefillScores(kProbeChunk, kProbeCtx, kHeadDim, 1, kQkvRowStride),
        "self_check:prefill_scores_per_head");
  Probe(GemmConfig::PrefillPv(kProbeChunk, kProbeCtx, kHeadDim, kGqaGroup, kAttnOutStride),
        "self_check:prefill_pv_batched");
  Probe(GemmConfig::PrefillPv(kProbeChunk, kProbeCtx, kHeadDim, 1, kAttnOutStride),
        "self_check:prefill_pv_per_head");
}

std::string CublasLtGemm::AlgoReportJson() const {
  std::string json = "{\"initialized\": ";
  json += initialized() ? "true" : "false";
  json += ", \"device\": \"";
  json += EscapeJson(device_name_);
  json += "\", \"compute_capability\": \"";
  json += std::to_string(cc_major_);
  json += ".";
  json += std::to_string(cc_minor_);
  json += "\", \"cublaslt_version\": ";
  json += std::to_string(cublaslt_version_);
  json += ", \"workspace_bytes\": ";
  json += std::to_string(workspace_bytes_);
  json += ", \"configs\": [";
  bool first = true;
  for (const auto& plan : plans_) {
    if (!first)
      json += ", ";
    first = false;
    const GemmConfig& c = plan->config;
    json += "{\"tag\": \"";
    json += EscapeJson(plan->tag);
    json += "\", \"m\": ";
    json += std::to_string(c.m);
    json += ", \"n\": ";
    json += std::to_string(c.n);
    json += ", \"k\": ";
    json += std::to_string(c.k);
    json += ", \"op_a\": \"";
    json += OpName(c.op_a);
    json += "\", \"op_b\": \"";
    json += OpName(c.op_b);
    json += "\", \"lda\": ";
    json += std::to_string(c.lda);
    json += ", \"ldb\": ";
    json += std::to_string(c.ldb);
    json += ", \"ldd\": ";
    json += std::to_string(c.ldd);
    json += ", \"type_d\": \"";
    json += TypeName(c.type_d);
    json += "\", \"epilogue\": \"";
    json += EpilogueName(c.epilogue);
    json += "\", \"batch\": ";
    json += std::to_string(c.batch);
    json += ", \"stride_a\": ";
    json += std::to_string(c.stride_a);
    json += ", \"stride_b\": ";
    json += std::to_string(c.stride_b);
    json += ", \"stride_d\": ";
    json += std::to_string(c.stride_d);
    json += ", \"min_align_bytes\": [";
    json += std::to_string(c.min_align_a_bytes);
    json += ", ";
    json += std::to_string(c.min_align_b_bytes);
    json += ", ";
    json += std::to_string(c.min_align_d_bytes);
    json += "], \"available\": ";
    json += plan->available ? "true" : "false";
    if (plan->available) {
      json += ", \"algo\": {\"id\": ";
      json += std::to_string(plan->algo_id);
      json += ", \"tile_id\": ";
      json += std::to_string(plan->tile_id);
      json += ", \"stages_id\": ";
      json += std::to_string(plan->stages_id);
      json += ", \"splitk\": ";
      json += std::to_string(plan->splitk_num);
      json += ", \"reduction_scheme\": ";
      json += std::to_string(plan->reduction_scheme);
      json += ", \"cta_swizzling\": ";
      json += std::to_string(plan->cta_swizzling);
      json += ", \"custom_option\": ";
      json += std::to_string(plan->custom_option);
      json += ", \"workspace_bytes\": ";
      json += std::to_string(plan->algo_workspace_bytes);
      json += ", \"waves_count\": ";
      AppendJsonFloat(json, plan->waves_count);
      json += "}";
    } else {
      json += ", \"algo\": null";
    }
    json += "}";
  }
  json += "]}";
  return json;
}

} // namespace redline
