// Kernel unit suite (a) of docs/DESIGN.md section 12: every custom kernel is
// checked against double-precision CPU references computing the same math on
// FP16-quantized inputs. The references live in tests/reference/ (header-only,
// no dependency on the kernels under test); shapes sweep the block-boundary
// edges seq_len {1, 15, 16, 17, 63, 64, 65, 257} and batch {1, 3, 8} with
// padded rows present.
//
// Tolerances (section 12a): elementwise kernels max-abs 1e-2 / mean-abs 1e-3;
// attention (decode kernel, prefill softmax) max-abs 2e-2; argmax and every
// pure bit-move (embed_gather, kv_scatter/kv_gather) exact. Test inputs are
// unit-scale activations (the regime those absolute tolerances were set for);
// silu_mul inputs stay within [-3, 3] so a one-ulp FP16 flip of the largest
// products remains below the max-abs gate.
//
// Mandatory adversarial cases (section 12a), all present below:
//   (1) BF16->FP16 exhaustive conversion - lives in tests/test_bf16_convert.cpp
//       (Bf16ConvertTest), not here.
//   (2) fused_add_rmsnorm rounding order - half-ULP boundary fixture with
//       reference-derived expectations, margin-based column exclusion, and
//       mutant self-validation (flip counts are seed/weight-dependent, so the
//       fixture derives every expected value from the FP64 reference and
//       asserts its own discriminating power instead of hardcoding counts).
//   (3) NaN-poisoned pool decode attention.
//   (4) primary KV-head-centric decode kernel vs the per-Q-head oracle.
//   (5) RoPE vs the committed HF-captured fixture
//       (tests/data/rope_fixture.json, scripts/gen_rope_fixture.py).
// Every launcher that takes row strides runs with BOTH packed
// (stride == width) and qkv_out-style (stride == 2048) inputs; LaunchRmsNorm,
// LaunchRmsNormResidual, LaunchPrefillSoftmax, LaunchSiluMul and
// LaunchGreedyArgmax have strideless contracts (dense rows), so the
// both-stride item does not apply to them.
//
// GPU tests skip cleanly (GTEST_SKIP) when no CUDA device is present; the
// CPU-side tests (reference self-checks, fixture self-validation, committed
// fixture vs reference) run everywhere so CI stays meaningful without a GPU.

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.cuh"

#include "reference/attention_ref.hpp"
#include "reference/elementwise_ref.hpp"
#include "reference/fp16_ref.hpp"
#include "reference/pool_ref.hpp"
#include "reference/rmsnorm_ref.hpp"
#include "reference/rope_ref.hpp"

namespace redline::kernels {
namespace {

namespace ref = ::redline::reference;

// ------------------------------------------------------------------ plumbing

bool HasCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define REDLINE_REQUIRE_GPU()                                                                      \
  do {                                                                                             \
    if (!HasCudaDevice()) {                                                                        \
      GTEST_SKIP() << "no CUDA device available";                                                  \
    }                                                                                              \
  } while (0)

// CUDA failures throw; googletest reports the uncaught exception as a test
// failure with the message intact. Throwing (rather than ASSERT) keeps the
// helpers usable from value-returning functions.
void CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

class CudaStream {
 public:
  CudaStream() { CudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~CudaStream() { cudaStreamDestroy(stream_); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

template <typename T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CudaCheck(
        cudaMalloc(reinterpret_cast<void**>(&ptr_), std::max<std::size_t>(count, 1) * sizeof(T)),
        "cudaMalloc");
  }
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) { Upload(host); }
  ~DeviceBuffer() {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void Upload(const std::vector<T>& host) {
    if (host.size() != count_)
      throw std::runtime_error("DeviceBuffer::Upload size mismatch");
    CudaCheck(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D");
  }
  std::vector<T> Download() const {
    std::vector<T> host(count_);
    CudaCheck(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H");
    return host;
  }
  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// The device buffers hold FP16 payloads as uint16 bit patterns end to end;
// only the launcher calls reinterpret to half*. Bit-exact by construction.
// (Launch sites pass non-const buffers; const half* parameters bind through
// the implicit qualification conversion, so one overload suffices.)
half* AsHalf(std::uint16_t* p) {
  return reinterpret_cast<half*>(p);
}

void SyncStream(cudaStream_t stream) {
  CudaCheck(cudaGetLastError(), "kernel launch");
  CudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

std::string Hex4(std::uint16_t v) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "0x%04X", v);
  return buf;
}

std::string TestDataPath(const char* filename) {
  if (const char* dir = std::getenv("REDLINE_TEST_DATA_DIR")) {
    return std::string(dir) + "/" + filename;
  }
  return std::string("tests/data/") + filename; // repo-root fallback
}

// ------------------------------------------------------------- data helpers

std::vector<std::uint16_t> UniformHalfBits(ref::SplitMix64& rng, std::size_t n, double lo,
                                           double hi) {
  std::vector<std::uint16_t> v(n);
  for (auto& b : v)
    b = rng.NextHalfBitsUniform(lo, hi);
  return v;
}

std::uint16_t NanPattern(std::size_t i) {
  static constexpr std::uint16_t kPatterns[] = {0x7E00u, 0x7C01u, 0x7FFFu,
                                                0xFE00u, 0xFFFFu, 0xFDABu};
  return kPatterns[i % (sizeof(kPatterns) / sizeof(kPatterns[0]))];
}

std::vector<std::uint16_t> NanFilled(std::size_t n) {
  std::vector<std::uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = NanPattern(i);
  return v;
}

std::vector<double> BitsToDoubles(const std::vector<std::uint16_t>& bits) {
  std::vector<double> out(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i)
    out[i] = ref::HalfBitsToDouble(bits[i]);
  return out;
}

// Copies packed rows into a strided image at column `offset` of each row.
void PlaceRows(std::vector<std::uint16_t>& image, const std::vector<std::uint16_t>& packed,
               std::size_t rows, std::size_t width, std::size_t stride, std::size_t offset) {
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(&image[r * stride + offset], &packed[r * width], width * sizeof(std::uint16_t));
  }
}

std::vector<std::uint16_t> TakeRows(const std::vector<std::uint16_t>& image, std::size_t rows,
                                    std::size_t width, std::size_t stride, std::size_t offset) {
  std::vector<std::uint16_t> packed(rows * width);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(&packed[r * width], &image[r * stride + offset], width * sizeof(std::uint16_t));
  }
  return packed;
}

// ------------------------------------------------------------- comparators

void ExpectBitsEqual(const std::vector<std::uint16_t>& actual,
                     const std::vector<std::uint16_t>& expected, const std::string& label) {
  ASSERT_EQ(actual.size(), expected.size()) << label;
  std::size_t diffs = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      if (diffs == 0)
        first = i;
      ++diffs;
    }
  }
  EXPECT_EQ(diffs, 0u) << label << ": " << diffs << " differing bit patterns; first at flat "
                       << first << " actual " << Hex4(actual[first]) << " expected "
                       << Hex4(expected[first]);
}

void ExpectHalfNearDouble(const std::vector<std::uint16_t>& actual_bits,
                          const std::vector<double>& expected, double max_abs, double mean_abs,
                          const std::string& label) {
  ASSERT_EQ(actual_bits.size(), expected.size()) << label;
  const ref::ToleranceReport rep =
      ref::CompareHalfAgainstDouble(actual_bits.data(), expected.data(), actual_bits.size());
  EXPECT_LE(rep.max_abs, max_abs) << label << ": worst at flat index " << rep.worst_index
                                  << " actual " << rep.worst_actual << " expected "
                                  << rep.worst_expected;
  EXPECT_LE(rep.mean_abs, mean_abs) << label;
}

void ExpectHalfNearBits(const std::vector<std::uint16_t>& actual_bits,
                        const std::vector<std::uint16_t>& expected_bits, double max_abs,
                        double mean_abs, const std::string& label) {
  ExpectHalfNearDouble(actual_bits, BitsToDoubles(expected_bits), max_abs, mean_abs, label);
}

// Everything OUTSIDE the per-row views [r*stride + offset, +width), r < rows,
// must be bit-identical between `before` and `after` (strided-gap and
// tail-region protection).
void ExpectOutsideViewsUntouched(const std::vector<std::uint16_t>& before,
                                 const std::vector<std::uint16_t>& after, std::size_t rows,
                                 std::size_t width, std::size_t stride, std::size_t offset,
                                 const std::string& label) {
  ASSERT_EQ(before.size(), after.size()) << label;
  std::size_t diffs = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    const std::size_t r = i / stride;
    const std::size_t c = i % stride;
    const bool in_view = r < rows && c >= offset && c < offset + width;
    if (!in_view && before[i] != after[i]) {
      if (diffs == 0)
        first = i;
      ++diffs;
    }
  }
  EXPECT_EQ(diffs, 0u) << label << ": " << diffs
                       << " elements outside the strided views changed; first at flat " << first;
}

// Section 12a tolerance constants.
constexpr double kElementwiseMaxAbs = 1e-2;
constexpr double kElementwiseMeanAbs = 1e-3;
constexpr double kAttentionMaxAbs = 2e-2;

constexpr std::int32_t kHidden = 1536;
constexpr std::int32_t kQkvStride = 2048;
constexpr std::int32_t kNumQHeads = 12;
constexpr std::int32_t kNumKvHeads = 2;
constexpr std::int32_t kHeadDim = 128;
constexpr float kAttnScale = 0.088388347648318447f; // 1/sqrt(128)

// ===========================================================================
// KernelReferenceSelfTest - pins the header-only FP16 helpers to the CUDA
// host conversions and to the landmark roundings the boundary fixture is
// built on. CPU-only (runs on GPU-less CI).
// ===========================================================================

TEST(KernelReferenceSelfTest, HalfBitsRoundTripAllPatterns) {
  for (std::uint32_t b = 0; b < 0x10000u; ++b) {
    const auto bits = static_cast<std::uint16_t>(b);
    const double v = ref::HalfBitsToDouble(bits);
    if (ref::IsNanHalfBits(bits)) {
      EXPECT_TRUE(std::isnan(v)) << Hex4(bits);
      EXPECT_TRUE(ref::IsNanHalfBits(ref::QuantizeDoubleToHalfBits(v))) << Hex4(bits);
    } else {
      ASSERT_EQ(ref::QuantizeDoubleToHalfBits(v), bits) << Hex4(bits);
    }
  }
}

TEST(KernelReferenceSelfTest, QuantizeMatchesCudaHostConversions) {
  // Floats first (__float2half_rn): a float-valued double quantizes in one
  // step to the same half the CUDA conversion produces.
  ref::SplitMix64 rng(0xF16F16u);
  for (int i = 0; i < 200000; ++i) {
    const int exponent = static_cast<int>(rng.Next() % 48) - 30; // 2^-30 .. 2^17
    const double mant = 1.0 + rng.NextUniform();
    const double sign = (rng.Next() & 1) ? -1.0 : 1.0;
    const float f = static_cast<float>(sign * std::ldexp(mant, exponent));
    const __half cuda_h = __float2half_rn(f);
    std::uint16_t cuda_bits;
    std::memcpy(&cuda_bits, &cuda_h, sizeof(cuda_bits));
    ASSERT_EQ(ref::QuantizeDoubleToHalfBits(static_cast<double>(f)), cuda_bits) << "float " << f;
  }
  // Doubles (__double2half), including ties that a float round trip would
  // double-round: 1 + 2^-11 (pair A) and 1 + 2^-10 + 2^-11 (pair B).
  const double crafted[] = {1.0 + 0x1.0p-11,
                            1.0 + 0x1.0p-10 + 0x1.0p-11,
                            -(1.0 + 0x1.0p-11),
                            65519.999,
                            65520.0,
                            0x1.0p-25,
                            1.5 * 0x1.0p-25,
                            0x1.0p-14,
                            0x1.0p-14 - 0x1.0p-26,
                            0.0,
                            -0.0};
  for (const double d : crafted) {
    const __half cuda_h = __double2half(d);
    std::uint16_t cuda_bits;
    std::memcpy(&cuda_bits, &cuda_h, sizeof(cuda_bits));
    EXPECT_EQ(ref::QuantizeDoubleToHalfBits(d), cuda_bits) << "double " << d;
  }
}

TEST(KernelReferenceSelfTest, BoundaryPairLandmarks) {
  // The two constructions the rmsnorm fixture rests on (both FP64 sums are
  // exact midpoints; RN-even picks the even mantissa).
  EXPECT_EQ(ref::HalfAddRn(ref::kPairAResidualBits, ref::kBoundaryInputBits),
            ref::kPairAExpectedResidualBits); // 1.0 + 2^-11 ties DOWN to 0x3C00
  EXPECT_EQ(ref::HalfAddRn(ref::kPairBResidualBits, ref::kBoundaryInputBits),
            ref::kPairBExpectedResidualBits); // (1 + 2^-10) + 2^-11 ties UP to 0x3C02
  // A value exactly on a boundary has relative distance 0; the representable
  // 1.0 sits half an FP16 ulp (2^-12 relative on the low side) away.
  EXPECT_EQ(ref::RelDistanceToHalfBoundary(1.0 + 0x1.0p-11), 0.0);
  EXPECT_NEAR(ref::RelDistanceToHalfBoundary(1.0), 0x1.0p-12, 1e-9);
}

// ===========================================================================
// KernelEmbedGatherTest - pure copy: bit-exact, both stride modes, scalar
// fallback shapes.
// ===========================================================================

void RunEmbedGatherCase(std::int32_t num_tokens, std::int32_t hidden, std::int64_t out_stride,
                        std::uint64_t seed) {
  const std::int32_t vocab = 512;
  ref::SplitMix64 rng(seed);
  std::vector<std::uint16_t> table = UniformHalfBits(rng, std::size_t(vocab) * hidden, -2.0, 2.0);
  // Plant NaN payloads in a few table rows: a copy must move them bit-exactly.
  for (std::int32_t d = 0; d < hidden; d += 7)
    table[std::size_t(3) * hidden + d] = NanPattern(d);

  std::vector<std::int32_t> ids(num_tokens);
  for (std::int32_t t = 0; t < num_tokens; ++t) {
    ids[t] = static_cast<std::int32_t>(rng.Next() % vocab);
  }
  if (num_tokens >= 4) {
    ids[0] = 0;                   // lowest id
    ids[1] = vocab - 1;           // highest id
    ids[2] = ids[num_tokens - 1]; // repeated id (same row gathered twice)
    ids[3] = 3;                   // the NaN-planted row: bits must move exactly
  }

  const std::vector<std::uint16_t> out_before =
      UniformHalfBits(rng, std::size_t(num_tokens) * out_stride, -1.0, 1.0);

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_table(table);
  DeviceBuffer<std::int32_t> d_ids(ids);
  DeviceBuffer<std::uint16_t> d_out(out_before);
  LaunchEmbedGather(AsHalf(d_out.get()), out_stride, AsHalf(d_table.get()), d_ids.get(), num_tokens,
                    hidden, stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> out_after = d_out.Download();

  std::vector<std::uint16_t> expected(std::size_t(num_tokens) * hidden);
  for (std::int32_t t = 0; t < num_tokens; ++t) {
    std::memcpy(&expected[std::size_t(t) * hidden], &table[std::size_t(ids[t]) * hidden],
                std::size_t(hidden) * sizeof(std::uint16_t));
  }
  const std::string label = "embed_gather tokens=" + std::to_string(num_tokens) +
                            " hidden=" + std::to_string(hidden) +
                            " stride=" + std::to_string(out_stride);
  ExpectBitsEqual(TakeRows(out_after, num_tokens, hidden, out_stride, 0), expected, label);
  ExpectOutsideViewsUntouched(out_before, out_after, num_tokens, hidden, out_stride, 0,
                              label + " gap");
}

TEST(KernelEmbedGatherTest, BitExactCopyPackedRows) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t tokens : {1, 3, 8, 63, 257}) {
    RunEmbedGatherCase(tokens, kHidden, kHidden, 0xE1u + tokens);
  }
}

TEST(KernelEmbedGatherTest, BitExactCopyQkvStridedRows) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t tokens : {1, 8, 257}) {
    RunEmbedGatherCase(tokens, kHidden, kQkvStride, 0xE2u + tokens);
  }
}

TEST(KernelEmbedGatherTest, OddShapesTakeScalarFallback) {
  REDLINE_REQUIRE_GPU();
  RunEmbedGatherCase(6, 129, 129, 0xE3u); // odd hidden: scalar kernel
  RunEmbedGatherCase(6, 130, 131, 0xE4u); // even hidden, odd stride: scalar kernel
}

// ===========================================================================
// KernelRmsNormTest - plain rmsnorm (layer-0 input norm). Strideless
// contract: rows are dense-packed, so the both-stride item does not apply.
// ===========================================================================

void RunRmsNormCase(std::int32_t rows, std::int32_t hidden, std::uint64_t seed) {
  ref::SplitMix64 rng(seed);
  const std::vector<std::uint16_t> input =
      UniformHalfBits(rng, std::size_t(rows) * hidden, -2.0, 2.0);
  const std::vector<std::uint16_t> weight = UniformHalfBits(rng, hidden, 0.5, 2.0);
  const float eps = 1e-6f;

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_in(input);
  DeviceBuffer<std::uint16_t> d_w(weight);
  DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * hidden);
  LaunchRmsNorm(AsHalf(d_out.get()), AsHalf(d_in.get()), AsHalf(d_w.get()), rows, hidden, eps,
                stream.get());
  SyncStream(stream.get());

  const std::string label =
      "rmsnorm rows=" + std::to_string(rows) + " hidden=" + std::to_string(hidden);
  ExpectHalfNearBits(d_out.Download(), ref::RmsNormReference(input, weight, rows, hidden, eps),
                     kElementwiseMaxAbs, kElementwiseMeanAbs, label);
  // The plain variant must not modify its input (it is re-read in pass 2).
  ExpectBitsEqual(d_in.Download(), input, label + " input unmodified");
}

TEST(KernelRmsNormTest, MatchesFp64ReferenceAcrossRowSweep) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t rows : {1, 3, 8, 15, 16, 17, 63, 64, 65, 257}) {
    RunRmsNormCase(rows, kHidden, 0xA0u + rows);
  }
}

TEST(KernelRmsNormTest, LoopTailAndIdleThreadHiddenSizes) {
  REDLINE_REQUIRE_GPU();
  // Below / not multiples of the 256-thread block: loop tail + idle reducers.
  for (const std::int32_t hidden : {8, 130, 300}) {
    RunRmsNormCase(3, hidden, 0xA100u + hidden);
  }
}

TEST(KernelRmsNormTest, EpsInsideRsqrtKeepsZeroRowFinite) {
  REDLINE_REQUIRE_GPU();
  // An all-zero row makes rsqrt(0 + eps) vs rsqrt(0) observable: with eps
  // inside, out = fp16(0 * rsqrt(eps)) * w = +0 exactly; with eps misplaced
  // the kernel would produce 0 * inf = NaN.
  const std::int32_t rows = 2;
  ref::SplitMix64 rng(0xE9Au);
  std::vector<std::uint16_t> input = UniformHalfBits(rng, std::size_t(rows) * kHidden, -2.0, 2.0);
  for (std::int32_t c = 0; c < kHidden; ++c)
    input[c] = 0x0000; // row 0 all +0
  const std::vector<std::uint16_t> weight = UniformHalfBits(rng, kHidden, 0.5, 2.0);

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_in(input);
  DeviceBuffer<std::uint16_t> d_w(weight);
  DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * kHidden);
  LaunchRmsNorm(AsHalf(d_out.get()), AsHalf(d_in.get()), AsHalf(d_w.get()), rows, kHidden, 1e-6f,
                stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> out = d_out.Download();
  for (std::int32_t c = 0; c < kHidden; ++c) {
    ASSERT_EQ(out[c], 0x0000) << "zero-row column " << c << " produced " << Hex4(out[c]);
  }
  ExpectHalfNearBits(out, ref::RmsNormReference(input, weight, rows, kHidden, 1e-6f),
                     kElementwiseMaxAbs, kElementwiseMeanAbs, "rmsnorm zero-row case");
}

// ===========================================================================
// KernelRmsNormResidualTest - fused residual-add + rmsnorm, including the
// section 12a mandatory rounding-order boundary fixture.
// ===========================================================================

void RunFusedRmsNormCase(std::int32_t rows, std::int32_t hidden, std::uint64_t seed) {
  ref::SplitMix64 rng(seed);
  const std::vector<std::uint16_t> residual_in =
      UniformHalfBits(rng, std::size_t(rows) * hidden, -2.0, 2.0);
  const std::vector<std::uint16_t> input =
      UniformHalfBits(rng, std::size_t(rows) * hidden, -2.0, 2.0);
  const std::vector<std::uint16_t> weight = UniformHalfBits(rng, hidden, 0.5, 2.0);
  const float eps = 1e-6f;

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_res(residual_in);
  DeviceBuffer<std::uint16_t> d_in(input);
  DeviceBuffer<std::uint16_t> d_w(weight);
  DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * hidden);
  LaunchRmsNormResidual(AsHalf(d_out.get()), AsHalf(d_res.get()), AsHalf(d_in.get()),
                        AsHalf(d_w.get()), rows, hidden, eps, stream.get());
  SyncStream(stream.get());

  const ref::FusedRmsNormReference expect =
      ref::ComputeFusedRmsNormReference(residual_in, input, weight, rows, hidden, eps);
  const std::string label =
      "fused_add_rmsnorm rows=" + std::to_string(rows) + " hidden=" + std::to_string(hidden);
  // The write-back h = fp16(r + x) is exactly determined (single rounding of
  // an exact sum on both sides): bit-exact, not tolerance.
  ExpectBitsEqual(d_res.Download(), expect.residual_bits, label + " residual write-back");
  ExpectHalfNearBits(d_out.Download(), expect.out_bits, kElementwiseMaxAbs, kElementwiseMeanAbs,
                     label);
  // The plain input operand is never written.
  ExpectBitsEqual(d_in.Download(), input, label + " input unmodified");
}

TEST(KernelRmsNormResidualTest, MatchesFp64ReferenceAcrossRowSweep) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t rows : {1, 3, 8, 15, 16, 17, 63, 64, 65, 257}) {
    RunFusedRmsNormCase(rows, kHidden, 0xB0u + rows);
  }
  for (const std::int32_t hidden : {8, 130, 300}) {
    RunFusedRmsNormCase(3, hidden, 0xB100u + hidden);
  }
}

// The pinned fixture seed. Chosen by a bit-exact numpy simulation of
// BuildRmsNormBoundaryFixture + the references (splitmix64 streams, IEEE
// float64 arithmetic, RN-even FP16 casts are language-independent): seed 1
// gives, per row, mutant-1 flips 209..227 and mutant-2 flips 296..384 on
// non-excluded columns (gate: >= 16), 28..55 excluded columns of 1536, every
// boundary column non-excluded, and zero non-excluded bit flips under
// kernel-noise emulation (FP32 mean-square in four summation orders x +-2ulp
// rsqrt perturbation x FP32-product double rounding). Seeds 2/4/6 were
// REJECTED by the same simulation: their shared filler stream (splitmix64
// seed+r == 6) drives a pair-A row down to 15 mutant-1 flips - the
// self-validation below exists precisely to catch such regressions if the
// construction or seed ever changes.
constexpr std::uint64_t kBoundaryFixtureSeed = 1;

TEST(KernelRmsNormResidualTest, BoundaryFixtureSelfValidatesAgainstMutants) {
  // CPU-only: guards the fixture against regression to vacuity (a fixture
  // that no longer distinguishes the wrong operation orders must fail HERE,
  // not silently pass the GPU gate). Runs on GPU-less CI too.
  const ref::RmsNormBoundaryFixture fx = ref::BuildRmsNormBoundaryFixture(kBoundaryFixtureSeed);
  const int rows = ref::kBoundaryFixtureRows;
  const int hidden = ref::kBoundaryFixtureHidden;
  const float eps = 1e-6f;
  const ref::FusedRmsNormReference correct = ref::ComputeFusedRmsNormReference(
      fx.residual, fx.input, fx.weight, rows, hidden, eps, ref::kBoundaryExclusionRelFloor);
  const std::vector<std::uint16_t> mutant1 =
      ref::FusedRmsNormMutantUnroundedSum(fx.residual, fx.input, fx.weight, rows, hidden, eps);
  const std::vector<std::uint16_t> mutant2 =
      ref::FusedRmsNormMutantSingleRounding(fx.residual, fx.input, fx.weight, rows, hidden, eps);

  for (int r = 0; r < rows; ++r) {
    const std::size_t base = std::size_t(r) * hidden;
    int flips1 = 0;
    int flips2 = 0;
    int excluded = 0;
    for (int c = 0; c < hidden; ++c) {
      const std::size_t i = base + c;
      if (fx.is_boundary[i]) {
        // Boundary columns must survive the margin exclusion (their h * inv
        // must not itself sit on a rounding boundary), and the reference's
        // write-back must land on the pinned RN-even results.
        EXPECT_FALSE(correct.excluded[i]) << "boundary column excluded: row " << r << " col " << c;
        EXPECT_EQ(correct.residual_bits[i], fx.is_pair_b[r] ? ref::kPairBExpectedResidualBits
                                                            : ref::kPairAExpectedResidualBits)
            << "row " << r << " col " << c;
      }
      if (correct.excluded[i]) {
        ++excluded;
        continue;
      }
      flips1 += (mutant1[i] != correct.out_bits[i]) ? 1 : 0;
      flips2 += (mutant2[i] != correct.out_bits[i]) ? 1 : 0;
    }
    EXPECT_GE(flips1, ref::kBoundaryMinMutantFlips)
        << "row " << r << ": unrounded-sum mutant is not discriminated (fixture went vacuous)";
    EXPECT_GE(flips2, ref::kBoundaryMinMutantFlips)
        << "row " << r << ": single-rounding mutant is not discriminated (fixture went vacuous)";
    EXPECT_LT(excluded, hidden / 4) << "row " << r << ": exclusion swallowed too many columns";
    // Every implementation writes back fp16(r + x) regardless of what its
    // mean-square consumed, so residual bits alone can NOT pin rounding
    // point 1 - the non-excluded out-bit gate above is what does. Kept as a
    // documented property of the fixture design.
  }
}

TEST(KernelRmsNormResidualTest, BoundaryFixtureRoundingOrderBitExactOnGpu) {
  REDLINE_REQUIRE_GPU();
  const ref::RmsNormBoundaryFixture fx = ref::BuildRmsNormBoundaryFixture(kBoundaryFixtureSeed);
  const int rows = ref::kBoundaryFixtureRows;
  const int hidden = ref::kBoundaryFixtureHidden;
  const float eps = 1e-6f;
  const ref::FusedRmsNormReference expect = ref::ComputeFusedRmsNormReference(
      fx.residual, fx.input, fx.weight, rows, hidden, eps, ref::kBoundaryExclusionRelFloor);

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_res(fx.residual);
  DeviceBuffer<std::uint16_t> d_in(fx.input);
  DeviceBuffer<std::uint16_t> d_w(fx.weight);
  DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * hidden);
  LaunchRmsNormResidual(AsHalf(d_out.get()), AsHalf(d_res.get()), AsHalf(d_in.get()),
                        AsHalf(d_w.get()), rows, hidden, eps, stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> out = d_out.Download();
  const std::vector<std::uint16_t> res = d_res.Download();

  // Residual write-back: bit-exact on ALL columns (exact-sum rounding), and
  // explicitly the pinned tie directions on the boundary columns - pair A
  // catches round-up/ties-away, pair B catches truncation/round-down.
  ExpectBitsEqual(res, expect.residual_bits, "boundary fixture residual write-back");
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < hidden; ++c) {
      const std::size_t i = std::size_t(r) * hidden + c;
      if (!fx.is_boundary[i])
        continue;
      ASSERT_EQ(res[i],
                fx.is_pair_b[r] ? ref::kPairBExpectedResidualBits : ref::kPairAExpectedResidualBits)
          << "row " << r << " col " << c << " (pair " << (fx.is_pair_b[r] ? "B" : "A") << ")";
    }
  }

  // Output: bit-exact against the HF-operation-order reference on every
  // non-excluded column. This is the gate that pins rounding point 1 (the
  // mean-square over the ROUNDED residual): the unrounded-sum order flips
  // hundreds of these bits (see the self-validation test), while legitimate
  // FP32-vs-FP64 noise is confined to the excluded columns by construction.
  std::size_t diffs = 0;
  std::size_t first = 0;
  std::size_t compared = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (expect.excluded[i])
      continue;
    ++compared;
    if (out[i] != expect.out_bits[i]) {
      if (diffs == 0)
        first = i;
      ++diffs;
    }
  }
  EXPECT_EQ(diffs, 0u) << "rounding-order gate: " << diffs << " of " << compared
                       << " non-excluded columns differ; first at flat " << first << " (row "
                       << first / hidden << " col " << first % hidden << ") actual "
                       << Hex4(out[first]) << " expected " << Hex4(expect.out_bits[first])
                       << " - an unrounded-sum or single-rounding implementation fails here";
}

TEST(KernelRmsNormResidualTest, DeterministicAcrossRepeatedLaunches) {
  REDLINE_REQUIRE_GPU();
  ref::SplitMix64 rng(0xDE7u);
  const std::int32_t rows = 8;
  const std::vector<std::uint16_t> residual_in =
      UniformHalfBits(rng, std::size_t(rows) * kHidden, -2.0, 2.0);
  const std::vector<std::uint16_t> input =
      UniformHalfBits(rng, std::size_t(rows) * kHidden, -2.0, 2.0);
  const std::vector<std::uint16_t> weight = UniformHalfBits(rng, kHidden, 0.5, 2.0);

  CudaStream stream;
  std::vector<std::uint16_t> outs[2];
  for (int pass = 0; pass < 2; ++pass) {
    DeviceBuffer<std::uint16_t> d_res(residual_in);
    DeviceBuffer<std::uint16_t> d_in(input);
    DeviceBuffer<std::uint16_t> d_w(weight);
    DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * kHidden);
    LaunchRmsNormResidual(AsHalf(d_out.get()), AsHalf(d_res.get()), AsHalf(d_in.get()),
                          AsHalf(d_w.get()), rows, kHidden, 1e-6f, stream.get());
    SyncStream(stream.get());
    outs[pass] = d_out.Download();
  }
  ExpectBitsEqual(outs[1], outs[0], "fused rmsnorm run-to-run determinism");
}

// ===========================================================================
// KernelRopeTest - NeoX half-rotation, FP64 reference + the committed
// HF-captured fixture, packed and qkv_out-strided modes.
// ===========================================================================

struct RopeFixture {
  double theta = 0.0;
  std::int32_t head_dim = 0;
  std::int32_t num_q_heads = 0;
  std::int32_t num_kv_heads = 0;
  std::int32_t num_tokens = 0;
  std::vector<std::int32_t> positions;
  std::vector<std::uint16_t> q_in, k_in, q_out, k_out;
};

std::vector<std::uint16_t> FlattenU16Rows(const nlohmann::json& j, std::size_t rows,
                                          std::size_t width, const char* name) {
  const auto nested = j.get<std::vector<std::vector<std::uint16_t>>>();
  if (nested.size() != rows) {
    throw std::runtime_error(std::string("rope fixture: ") + name + " row count mismatch");
  }
  std::vector<std::uint16_t> flat;
  flat.reserve(rows * width);
  for (const auto& row : nested) {
    if (row.size() != width) {
      throw std::runtime_error(std::string("rope fixture: ") + name + " row width mismatch");
    }
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

RopeFixture LoadRopeFixture() {
  const std::string path = TestDataPath("rope_fixture.json");
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error(
        "cannot open '" + path +
        "' - set REDLINE_TEST_DATA_DIR to <repo>/tests/data (ctest "
        "does) or run from the repo root; regenerate with scripts/gen_rope_fixture.py");
  }
  nlohmann::json j;
  file >> j;
  const nlohmann::json& meta = j.at("meta");
  RopeFixture fx;
  fx.theta = meta.at("rope_theta").get<double>();
  fx.head_dim = meta.at("head_dim").get<std::int32_t>();
  fx.num_q_heads = meta.at("num_q_heads").get<std::int32_t>();
  fx.num_kv_heads = meta.at("num_kv_heads").get<std::int32_t>();
  fx.num_tokens = meta.at("num_tokens").get<std::int32_t>();
  fx.positions = meta.at("positions").get<std::vector<std::int32_t>>();
  const auto t = static_cast<std::size_t>(fx.num_tokens);
  fx.q_in = FlattenU16Rows(j.at("q_in"), t, std::size_t(fx.num_q_heads) * fx.head_dim, "q_in");
  fx.k_in = FlattenU16Rows(j.at("k_in"), t, std::size_t(fx.num_kv_heads) * fx.head_dim, "k_in");
  fx.q_out = FlattenU16Rows(j.at("q_out"), t, std::size_t(fx.num_q_heads) * fx.head_dim, "q_out");
  fx.k_out = FlattenU16Rows(j.at("k_out"), t, std::size_t(fx.num_kv_heads) * fx.head_dim, "k_out");
  if (fx.positions.size() != t)
    throw std::runtime_error("rope fixture: positions mismatch");
  return fx;
}

TEST(KernelRopeTest, CommittedFixtureAgreesWithFp64ReferenceCpuOnly) {
  // CPU-only coherence check between the two independent oracles: the
  // committed HF capture (FP32 trig chain) and the FP64 reference must agree
  // within the elementwise tolerances, or one of them mis-encodes the
  // convention. Keeps GPU-less CI validating the committed data.
  const RopeFixture fx = LoadRopeFixture();
  ASSERT_EQ(fx.head_dim, kHeadDim);
  ASSERT_EQ(fx.num_q_heads, kNumQHeads);
  ASSERT_EQ(fx.num_kv_heads, kNumKvHeads);
  ASSERT_DOUBLE_EQ(fx.theta, 1e6);

  std::vector<std::uint16_t> q = fx.q_in;
  std::vector<std::uint16_t> k = fx.k_in;
  ref::RopeReferenceInplace(q, fx.positions, fx.num_tokens, fx.num_q_heads, fx.head_dim, fx.theta);
  ref::RopeReferenceInplace(k, fx.positions, fx.num_tokens, fx.num_kv_heads, fx.head_dim, fx.theta);
  ExpectHalfNearDouble(fx.q_out, BitsToDoubles(q), kElementwiseMaxAbs, kElementwiseMeanAbs,
                       "rope fixture q vs FP64 reference");
  ExpectHalfNearDouble(fx.k_out, BitsToDoubles(k), kElementwiseMaxAbs, kElementwiseMeanAbs,
                       "rope fixture k vs FP64 reference");
}

void RunRopeAgainstExpected(const RopeFixture& fx, bool strided,
                            const std::vector<std::uint16_t>& expected_q,
                            const std::vector<std::uint16_t>& expected_k,
                            const std::string& label) {
  const std::size_t t = fx.num_tokens;
  const std::size_t qw = std::size_t(fx.num_q_heads) * fx.head_dim;
  const std::size_t kw = std::size_t(fx.num_kv_heads) * fx.head_dim;

  CudaStream stream;
  DeviceBuffer<std::int32_t> d_pos(fx.positions);
  if (!strided) {
    DeviceBuffer<std::uint16_t> d_q(fx.q_in);
    DeviceBuffer<std::uint16_t> d_k(fx.k_in);
    LaunchRopeInplace(AsHalf(d_q.get()), AsHalf(d_k.get()), static_cast<std::int64_t>(qw),
                      static_cast<std::int64_t>(kw), d_pos.get(), fx.num_tokens, fx.num_q_heads,
                      fx.num_kv_heads, fx.head_dim, static_cast<float>(fx.theta), stream.get());
    SyncStream(stream.get());
    ExpectHalfNearBits(d_q.Download(), expected_q, kElementwiseMaxAbs, kElementwiseMeanAbs,
                       label + " q packed");
    ExpectHalfNearBits(d_k.Download(), expected_k, kElementwiseMaxAbs, kElementwiseMeanAbs,
                       label + " k packed");
    return;
  }
  // qkv_out-style: one [T, 2048] image, q view at +0, k view at +1536, v
  // region [1792, 2048) must come through untouched.
  ref::SplitMix64 rng(0x517AB1Eu);
  std::vector<std::uint16_t> image = UniformHalfBits(rng, t * kQkvStride, -1.0, 1.0);
  PlaceRows(image, fx.q_in, t, qw, kQkvStride, 0);
  PlaceRows(image, fx.k_in, t, kw, kQkvStride, qw);
  const std::vector<std::uint16_t> before = image;

  DeviceBuffer<std::uint16_t> d_image(image);
  LaunchRopeInplace(AsHalf(d_image.get()), AsHalf(d_image.get()) + qw, kQkvStride, kQkvStride,
                    d_pos.get(), fx.num_tokens, fx.num_q_heads, fx.num_kv_heads, fx.head_dim,
                    static_cast<float>(fx.theta), stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> after = d_image.Download();
  ExpectHalfNearBits(TakeRows(after, t, qw, kQkvStride, 0), expected_q, kElementwiseMaxAbs,
                     kElementwiseMeanAbs, label + " q strided");
  ExpectHalfNearBits(TakeRows(after, t, kw, kQkvStride, qw), expected_k, kElementwiseMaxAbs,
                     kElementwiseMeanAbs, label + " k strided");
  // Everything outside the q||k views - the V slice and any tail - is not
  // the kernel's to touch.
  ExpectOutsideViewsUntouched(before, after, t, qw + kw, kQkvStride, 0, label + " v-region");
}

TEST(KernelRopeTest, MatchesCommittedHfFixturePacked) {
  REDLINE_REQUIRE_GPU();
  const RopeFixture fx = LoadRopeFixture();
  RunRopeAgainstExpected(fx, /*strided=*/false, fx.q_out, fx.k_out, "hf fixture");
}

TEST(KernelRopeTest, MatchesCommittedHfFixtureQkvStrided) {
  REDLINE_REQUIRE_GPU();
  const RopeFixture fx = LoadRopeFixture();
  RunRopeAgainstExpected(fx, /*strided=*/true, fx.q_out, fx.k_out, "hf fixture");
}

RopeFixture MakeSyntheticRopeCase(std::uint64_t seed, std::int32_t num_tokens, double theta,
                                  const std::vector<std::int32_t>& positions) {
  RopeFixture fx;
  fx.theta = theta;
  fx.head_dim = kHeadDim;
  fx.num_q_heads = kNumQHeads;
  fx.num_kv_heads = kNumKvHeads;
  fx.num_tokens = num_tokens;
  fx.positions = positions;
  ref::SplitMix64 rng(seed);
  fx.q_in = UniformHalfBits(rng, std::size_t(num_tokens) * kNumQHeads * kHeadDim, -2.0, 2.0);
  fx.k_in = UniformHalfBits(rng, std::size_t(num_tokens) * kNumKvHeads * kHeadDim, -2.0, 2.0);
  fx.q_out = fx.q_in;
  fx.k_out = fx.k_in;
  ref::RopeReferenceInplace(fx.q_out, fx.positions, num_tokens, kNumQHeads, kHeadDim, theta);
  ref::RopeReferenceInplace(fx.k_out, fx.positions, num_tokens, kNumKvHeads, kHeadDim, theta);
  return fx;
}

TEST(KernelRopeTest, MatchesFp64ReferenceBothStrideModes) {
  REDLINE_REQUIRE_GPU();
  const RopeFixture fx = MakeSyntheticRopeCase(
      0x40FEu, 8, 1e6, std::vector<std::int32_t>{0, 1, 2, 15, 16, 17, 1023, 2047});
  RunRopeAgainstExpected(fx, /*strided=*/false, fx.q_out, fx.k_out, "fp64 reference");
  RunRopeAgainstExpected(fx, /*strided=*/true, fx.q_out, fx.k_out, "fp64 reference");
}

TEST(KernelRopeTest, ThetaSweepReuploadsFrequencyTable) {
  REDLINE_REQUIRE_GPU();
  // Alternating theta values across launches exercises the constant-table
  // re-upload path (test-only; the engine holds theta fixed).
  for (const double theta : {10000.0, 1e6, 10000.0}) {
    const RopeFixture fx = MakeSyntheticRopeCase(0x7E7Au + static_cast<std::uint64_t>(theta), 4,
                                                 theta, std::vector<std::int32_t>{1, 7, 100, 511});
    RunRopeAgainstExpected(fx, /*strided=*/false, fx.q_out, fx.k_out,
                           "theta=" + std::to_string(theta));
  }
}

TEST(KernelRopeTest, PositionZeroRowsPassThroughBitExact) {
  REDLINE_REQUIRE_GPU();
  // Padded rows carry position 0 (docs/DESIGN.md section 9): angle 0 gives
  // cos 1 / sin 0 exactly, an identity rotation even through the FP16 chain.
  const std::int32_t num_tokens = 3;
  RopeFixture fx =
      MakeSyntheticRopeCase(0x1DE4u, num_tokens, 1e6, std::vector<std::int32_t>{0, 0, 0});
  // The pass-through is a bit identity only for nonzero elements: cos_h == 1
  // and sin_h == +0 make every product and sum exact, EXCEPT that a -0 input
  // whose sin-term partner contributes +0 rounds to +0 (sign-bit flip). Seed
  // 0x1DE4 happens to draw no +/-0 halves, but scrub zeros anyway so a future
  // seed/generator change cannot surface as a baffling one-bit failure. (Only
  // q_in/k_in matter here; fx.q_out/k_out are unused by this test.)
  for (auto* buf : {&fx.q_in, &fx.k_in}) {
    for (std::uint16_t& b : *buf) {
      if (b == 0x0000u || b == 0x8000u)
        b = 0x3C00u; // 1.0
    }
  }
  CudaStream stream;
  DeviceBuffer<std::int32_t> d_pos(fx.positions);
  DeviceBuffer<std::uint16_t> d_q(fx.q_in);
  DeviceBuffer<std::uint16_t> d_k(fx.k_in);
  LaunchRopeInplace(AsHalf(d_q.get()), AsHalf(d_k.get()), std::int64_t(kNumQHeads) * kHeadDim,
                    std::int64_t(kNumKvHeads) * kHeadDim, d_pos.get(), num_tokens, kNumQHeads,
                    kNumKvHeads, kHeadDim, 1e6f, stream.get());
  SyncStream(stream.get());
  ExpectBitsEqual(d_q.Download(), fx.q_in, "rope position-0 q identity");
  ExpectBitsEqual(d_k.Download(), fx.k_in, "rope position-0 k identity");
}

// ===========================================================================
// KernelKvScatterTest / KernelKvGatherTest - paged pool I/O: pure bit moves,
// in-kernel slot derivation, both stride modes, scalar fallbacks,
// NaN-poisoned inertness. All comparisons bit-exact.
// ===========================================================================

// Builds a shuffled, non-contiguous physical-block id sequence (block 0 is
// the reserved dummy and never enters the live pool).
std::vector<std::int32_t> ShuffledLiveBlocks(ref::SplitMix64& rng, std::int32_t num_blocks) {
  std::vector<std::int32_t> ids;
  for (std::int32_t b = 1; b < num_blocks; ++b)
    ids.push_back(b);
  for (std::size_t i = ids.size(); i > 1; --i) {
    std::swap(ids[i - 1], ids[rng.Next() % i]);
  }
  return ids;
}

struct ScatterCase {
  std::int32_t num_blocks = 0;
  std::int32_t num_tokens = 0;
  std::int64_t k_row_stride = 0;
  std::int64_t v_row_stride = 0;
  std::int64_t table_row_stride = 0;
  std::size_t k_offset = 0; // view offset inside the source image
  std::size_t v_offset = 0;
  std::vector<std::uint16_t> source; // one image holding the k/v views
  std::vector<std::int32_t> positions;
  std::vector<std::int32_t> block_tables;
};

// Runs LaunchKvScatter on a NaN-poisoned pool and compares the ENTIRE pool
// bit-exactly against the CPU mirror - proving both that every written slot
// holds the exact source bits and that no other slot (NaN patterns included)
// was touched.
void RunScatterCase(const ScatterCase& sc, const std::string& label) {
  std::vector<std::uint16_t> pool_mirror(
      static_cast<std::size_t>(ref::PoolSliceElems(sc.num_blocks, kNumKvHeads, kHeadDim)));
  pool_mirror = NanFilled(pool_mirror.size());

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_pool(pool_mirror);
  DeviceBuffer<std::uint16_t> d_src(sc.source);
  DeviceBuffer<std::int32_t> d_pos(sc.positions);
  DeviceBuffer<std::int32_t> d_tables(sc.block_tables);
  LaunchKvScatter(AsHalf(d_pool.get()), AsHalf(d_src.get()) + sc.k_offset,
                  AsHalf(d_src.get()) + sc.v_offset, sc.k_row_stride, sc.v_row_stride,
                  d_tables.get(), sc.table_row_stride, d_pos.get(), sc.num_tokens, kNumKvHeads,
                  kHeadDim, stream.get());
  SyncStream(stream.get());

  ref::ScatterReference(pool_mirror, sc.source.data() + sc.k_offset, sc.source.data() + sc.v_offset,
                        sc.k_row_stride, sc.v_row_stride, sc.block_tables.data(),
                        sc.table_row_stride, sc.positions.data(), sc.num_tokens, kNumKvHeads,
                        kHeadDim);
  ExpectBitsEqual(d_pool.Download(), pool_mirror, label);
}

ScatterCase MakeDecodeScatterCase(std::uint64_t seed, bool qkv_strided) {
  ScatterCase sc;
  sc.num_tokens = 8;
  sc.num_blocks = 40;
  ref::SplitMix64 rng(seed);
  const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, sc.num_blocks);
  const std::int32_t max_blocks_per_seq = 16; // covers positions < 256
  sc.table_row_stride = max_blocks_per_seq;
  // Positions cross block boundaries: {0, 15, 16, 17, 31, 63, 200, 255}.
  sc.positions = {0, 15, 16, 17, 31, 63, 200, 255};
  // Decode: one table row per token. Only the entry positions[t] >> 4 is
  // ever dereferenced by the kernel, so exactly that entry maps to a
  // DISTINCT live block per token (no cross-token write collisions, which
  // would make the sequential CPU mirror order-dependent); every other
  // entry keeps the dummy block 0, matching the section 9 padding policy.
  sc.block_tables.assign(std::size_t(sc.num_tokens) * max_blocks_per_seq, 0);
  for (std::int32_t t = 0; t < sc.num_tokens; ++t) {
    sc.block_tables[std::size_t(t) * max_blocks_per_seq +
                    sc.positions[std::size_t(t)] / ref::kKvBlockSize] = live[std::size_t(t)];
  }
  const std::size_t row_width = std::size_t(kNumKvHeads) * kHeadDim; // 256
  if (qkv_strided) {
    sc.k_row_stride = kQkvStride;
    sc.v_row_stride = kQkvStride;
    sc.k_offset = 1536;
    sc.v_offset = 1792;
    sc.source = UniformHalfBits(rng, std::size_t(sc.num_tokens) * kQkvStride, -2.0, 2.0);
  } else {
    sc.k_row_stride = static_cast<std::int64_t>(row_width);
    sc.v_row_stride = static_cast<std::int64_t>(row_width);
    sc.k_offset = 0;
    sc.v_offset = std::size_t(sc.num_tokens) * row_width; // separate packed k and v planes
    sc.source = UniformHalfBits(rng, 2 * std::size_t(sc.num_tokens) * row_width, -2.0, 2.0);
  }
  // Plant NaN payload bits in the sources too: pure bit moves must carry them.
  for (std::size_t i = sc.k_offset; i < sc.k_offset + 8; ++i)
    sc.source[i] = NanPattern(i);
  return sc;
}

TEST(KernelKvScatterTest, DecodeModeWritesExactSlotsQkvStrided) {
  REDLINE_REQUIRE_GPU();
  RunScatterCase(MakeDecodeScatterCase(0x5CA7u, /*qkv_strided=*/true),
                 "kv_scatter decode qkv-strided");
}

TEST(KernelKvScatterTest, DecodeModeWritesExactSlotsPacked) {
  REDLINE_REQUIRE_GPU();
  RunScatterCase(MakeDecodeScatterCase(0x5CA8u, /*qkv_strided=*/false), "kv_scatter decode packed");
}

TEST(KernelKvScatterTest, PrefillModeSharesOneTableRowWithStrideZero) {
  REDLINE_REQUIRE_GPU();
  // One sequence's chunk: all rows share one block-table row via row-stride
  // 0; positions are the chunk's absolute range (chunk_start 33, len 17 -
  // spans a partial leading block and a partial trailing block).
  ScatterCase sc;
  sc.num_tokens = 17;
  sc.num_blocks = 12;
  ref::SplitMix64 rng(0xC0FEu);
  const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, sc.num_blocks);
  sc.table_row_stride = 0;
  sc.block_tables.assign(8, 0); // covers positions < 128
  for (std::size_t j = 0; j < sc.block_tables.size(); ++j)
    sc.block_tables[j] = live[j];
  sc.positions.resize(sc.num_tokens);
  for (std::int32_t i = 0; i < sc.num_tokens; ++i)
    sc.positions[i] = 33 + i;
  const std::size_t row_width = std::size_t(kNumKvHeads) * kHeadDim;
  sc.k_row_stride = static_cast<std::int64_t>(row_width);
  sc.v_row_stride = static_cast<std::int64_t>(row_width);
  sc.k_offset = 0;
  sc.v_offset = std::size_t(sc.num_tokens) * row_width;
  sc.source = UniformHalfBits(rng, 2 * std::size_t(sc.num_tokens) * row_width, -2.0, 2.0);
  RunScatterCase(sc, "kv_scatter prefill stride-0");
}

TEST(KernelKvScatterTest, PaddedDummyRowLandsInBlockZeroOnly) {
  REDLINE_REQUIRE_GPU();
  // A padded decode row (position 0, all-dummy table row) must write only
  // into reserved block 0; the full-pool mirror comparison proves every
  // other unwritten slot kept its NaN pattern.
  ScatterCase sc = MakeDecodeScatterCase(0x9ADu, /*qkv_strided=*/true);
  // Turn token 3 into a padded slot: dummy-block table row, position 0.
  for (std::int32_t j = 0; j < static_cast<std::int32_t>(sc.table_row_stride); ++j) {
    sc.block_tables[std::size_t(3) * sc.table_row_stride + j] = 0;
  }
  sc.positions[3] = 0;
  RunScatterCase(sc, "kv_scatter padded dummy row");
}

TEST(KernelKvScatterTest, OddStridesAndMisalignedBasesTakeScalarPath) {
  REDLINE_REQUIRE_GPU();
  ScatterCase sc;
  sc.num_tokens = 5;
  sc.num_blocks = 10;
  ref::SplitMix64 rng(0x0DDu);
  const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, sc.num_blocks);
  sc.table_row_stride = 4;
  sc.positions = {0, 15, 16, 33, 63};
  // Same no-collision invariant as MakeDecodeScatterCase: only the entry
  // positions[t] >> 4 is ever dereferenced, and it maps to a DISTINCT live
  // block per token; every other entry stays dummy 0 (section 9 padding
  // policy). Tokens 0 and 2 both land in slot 0 of their block, so sharing
  // one block between them would let two CTAs race on the same pool row
  // while the sequential CPU mirror resolves it last-writer-wins.
  sc.block_tables.assign(std::size_t(sc.num_tokens) * 4, 0);
  for (std::int32_t t = 0; t < sc.num_tokens; ++t) {
    sc.block_tables[std::size_t(t) * 4 + sc.positions[std::size_t(t)] / ref::kKvBlockSize] =
        live[std::size_t(t)];
  }
  // Odd row strides and 2-byte-misaligned view bases force the scalar
  // instantiation of the kernel (the uint4 path needs 16 B alignment and
  // stride % 8 == 0).
  sc.k_row_stride = 259;
  sc.v_row_stride = 261;
  sc.k_offset = 1;
  sc.v_offset = std::size_t(sc.num_tokens) * 261 + 3;
  sc.source = UniformHalfBits(
      rng, sc.v_offset + std::size_t(sc.num_tokens) * 261 + kNumKvHeads * kHeadDim, -2.0, 2.0);
  RunScatterCase(sc, "kv_scatter scalar fallback");
}

TEST(KernelKvScatterTest, BlockBoundaryPositionsHitExpectedSlots) {
  REDLINE_REQUIRE_GPU();
  // Positions 15/16/17 must land in (row[0], slot 15), (row[1], slot 0),
  // (row[1], slot 1) - probed directly by pool index, independent of the
  // mirror helper.
  const std::int32_t num_blocks = 6;
  ScatterCase sc;
  sc.num_tokens = 3;
  sc.num_blocks = num_blocks;
  sc.table_row_stride = 2;
  sc.block_tables = {4, 2, /* token 1 */ 4, 2, /* token 2 */ 4, 2};
  sc.positions = {15, 16, 17};
  const std::size_t row_width = std::size_t(kNumKvHeads) * kHeadDim;
  ref::SplitMix64 rng(0xB10Cu);
  sc.k_row_stride = static_cast<std::int64_t>(row_width);
  sc.v_row_stride = static_cast<std::int64_t>(row_width);
  sc.k_offset = 0;
  sc.v_offset = std::size_t(sc.num_tokens) * row_width;
  sc.source = UniformHalfBits(rng, 2 * std::size_t(sc.num_tokens) * row_width, -2.0, 2.0);
  RunScatterCase(sc, "kv_scatter boundary positions");

  // Re-run on a fresh pool and probe the three expected (block, slot) rows.
  std::vector<std::uint16_t> pool(
      static_cast<std::size_t>(ref::PoolSliceElems(num_blocks, kNumKvHeads, kHeadDim)));
  pool = NanFilled(pool.size());
  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_pool(pool);
  DeviceBuffer<std::uint16_t> d_src(sc.source);
  DeviceBuffer<std::int32_t> d_pos(sc.positions);
  DeviceBuffer<std::int32_t> d_tables(sc.block_tables);
  LaunchKvScatter(AsHalf(d_pool.get()), AsHalf(d_src.get()), AsHalf(d_src.get()) + sc.v_offset,
                  sc.k_row_stride, sc.v_row_stride, d_tables.get(), sc.table_row_stride,
                  d_pos.get(), sc.num_tokens, kNumKvHeads, kHeadDim, stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> got = d_pool.Download();
  const struct {
    std::int32_t token, block, slot;
  } expected[] = {{0, 4, 15}, {1, 2, 0}, {2, 2, 1}};
  for (const auto& e : expected) {
    for (std::int32_t h = 0; h < kNumKvHeads; ++h) {
      for (std::int32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t k_idx = static_cast<std::size_t>(
            ref::PoolElemIndex(e.block, 0, h, e.slot, d, kNumKvHeads, kHeadDim));
        const std::size_t src = std::size_t(e.token) * row_width + std::size_t(h) * kHeadDim + d;
        ASSERT_EQ(got[k_idx], sc.source[src])
            << "K token " << e.token << " head " << h << " dim " << d;
      }
    }
  }
}

TEST(KernelKvGatherTest, ScatterThenGatherRoundTripsBitExact) {
  REDLINE_REQUIRE_GPU();
  // Scatter positions [0, ctx) of one sequence (prefill pattern), gather
  // them back into dense khat/vhat, compare against the original packed
  // rows as raw bits - NaN payloads planted in the sources included. Runs
  // the ctx sweep incl. non-multiples of 16 (partial final block).
  for (const std::int32_t ctx : {1, 15, 16, 17, 63, 64, 65, 257}) {
    const std::int32_t num_blocks = ctx / ref::kKvBlockSize + 3;
    ref::SplitMix64 rng(0x6A7Bu + ctx);
    const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, num_blocks);
    std::vector<std::int32_t> table_row(std::size_t((ctx + 15) / 16) + 2, 0);
    for (std::size_t j = 0; j < table_row.size(); ++j) {
      table_row[j] = live[j % live.size()];
    }
    const std::size_t row_width = std::size_t(kNumKvHeads) * kHeadDim;
    std::vector<std::uint16_t> k_rows =
        UniformHalfBits(rng, std::size_t(ctx) * row_width, -2.0, 2.0);
    std::vector<std::uint16_t> v_rows =
        UniformHalfBits(rng, std::size_t(ctx) * row_width, -2.0, 2.0);
    for (std::size_t i = 0; i < std::min<std::size_t>(k_rows.size(), 16); ++i) {
      k_rows[i] = NanPattern(i);
    }
    std::vector<std::int32_t> positions(ctx);
    for (std::int32_t i = 0; i < ctx; ++i)
      positions[i] = i;

    std::vector<std::uint16_t> pool(
        static_cast<std::size_t>(ref::PoolSliceElems(num_blocks, kNumKvHeads, kHeadDim)));
    pool = NanFilled(pool.size());

    CudaStream stream;
    DeviceBuffer<std::uint16_t> d_pool(pool);
    DeviceBuffer<std::uint16_t> d_k(k_rows);
    DeviceBuffer<std::uint16_t> d_v(v_rows);
    DeviceBuffer<std::int32_t> d_pos(positions);
    DeviceBuffer<std::int32_t> d_table(table_row);
    LaunchKvScatter(AsHalf(d_pool.get()), AsHalf(d_k.get()), AsHalf(d_v.get()),
                    static_cast<std::int64_t>(row_width), static_cast<std::int64_t>(row_width),
                    d_table.get(), /*block_table_row_stride=*/0, d_pos.get(), ctx, kNumKvHeads,
                    kHeadDim, stream.get());
    SyncStream(stream.get());

    // Dense gather destination: head stride ctx * head_dim, rows dense.
    const std::size_t plane = std::size_t(ctx) * kHeadDim;
    DeviceBuffer<std::uint16_t> d_khat(std::size_t(kNumKvHeads) * plane);
    DeviceBuffer<std::uint16_t> d_vhat(std::size_t(kNumKvHeads) * plane);
    LaunchKvGather(AsHalf(d_khat.get()), AsHalf(d_vhat.get()), static_cast<std::int64_t>(plane),
                   kHeadDim, static_cast<std::int64_t>(plane), kHeadDim, AsHalf(d_pool.get()),
                   d_table.get(), ctx, kNumKvHeads, kHeadDim, stream.get());
    SyncStream(stream.get());

    // Reassemble [ctx, kv*hd] rows from the [kv, ctx, hd] planes.
    const std::vector<std::uint16_t> khat = d_khat.Download();
    const std::vector<std::uint16_t> vhat = d_vhat.Download();
    std::vector<std::uint16_t> k_round(std::size_t(ctx) * row_width);
    std::vector<std::uint16_t> v_round(std::size_t(ctx) * row_width);
    for (std::int32_t t = 0; t < ctx; ++t) {
      for (std::int32_t h = 0; h < kNumKvHeads; ++h) {
        std::memcpy(&k_round[std::size_t(t) * row_width + std::size_t(h) * kHeadDim],
                    &khat[std::size_t(h) * plane + std::size_t(t) * kHeadDim],
                    std::size_t(kHeadDim) * sizeof(std::uint16_t));
        std::memcpy(&v_round[std::size_t(t) * row_width + std::size_t(h) * kHeadDim],
                    &vhat[std::size_t(h) * plane + std::size_t(t) * kHeadDim],
                    std::size_t(kHeadDim) * sizeof(std::uint16_t));
      }
    }
    ExpectBitsEqual(k_round, k_rows, "kv round-trip K ctx=" + std::to_string(ctx));
    ExpectBitsEqual(v_round, v_rows, "kv round-trip V ctx=" + std::to_string(ctx));
  }
}

TEST(KernelKvGatherTest, MatchesCpuMirrorAcrossStrideModes) {
  REDLINE_REQUIRE_GPU();
  // Gather from a randomly filled pool through three destination stride
  // configurations: dense (ctx*hd planes), allocation-style planes
  // (max_seq_len*hd = 2048*128 spacing) with sentinel-gap checking, and an
  // arbitrary vec-hostile row stride (130 halves: stride % 8 != 0 -> scalar
  // instantiation) plus a misaligned variant via odd stride 133.
  const std::int32_t ctx = 63;
  const std::int32_t num_blocks = 9;
  ref::SplitMix64 rng(0x6A88u);
  std::vector<std::uint16_t> pool = UniformHalfBits(
      rng, static_cast<std::size_t>(ref::PoolSliceElems(num_blocks, kNumKvHeads, kHeadDim)), -2.0,
      2.0);
  const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, num_blocks);
  std::vector<std::int32_t> table_row(6, 0);
  for (std::size_t j = 0; j < table_row.size(); ++j)
    table_row[j] = live[j % live.size()];

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_pool(pool);
  DeviceBuffer<std::int32_t> d_table(table_row);

  struct StrideMode {
    std::int64_t head_stride, row_stride;
    const char* name;
  };
  const StrideMode modes[] = {
      {std::int64_t(ctx) * kHeadDim, kHeadDim, "dense"},
      {std::int64_t(2048) * kHeadDim, kHeadDim, "allocation-style"},
      {std::int64_t(ctx) * 130, 130, "scalar row stride 130"},
      {std::int64_t(ctx) * 133, 133, "scalar odd row stride 133"},
  };
  for (const StrideMode& m : modes) {
    const std::size_t total = std::size_t(kNumKvHeads) * m.head_stride;
    const std::vector<std::uint16_t> sentinel = UniformHalfBits(rng, total, -1.0, 1.0);
    DeviceBuffer<std::uint16_t> d_khat(sentinel);
    DeviceBuffer<std::uint16_t> d_vhat(sentinel);
    LaunchKvGather(AsHalf(d_khat.get()), AsHalf(d_vhat.get()), m.head_stride, m.row_stride,
                   m.head_stride, m.row_stride, AsHalf(d_pool.get()), d_table.get(), ctx,
                   kNumKvHeads, kHeadDim, stream.get());
    SyncStream(stream.get());

    std::vector<std::uint16_t> khat_mirror = sentinel;
    std::vector<std::uint16_t> vhat_mirror = sentinel;
    ref::GatherReference(khat_mirror.data(), vhat_mirror.data(), m.head_stride, m.row_stride,
                         m.head_stride, m.row_stride, pool, table_row.data(), ctx, kNumKvHeads,
                         kHeadDim);
    // Full-buffer equality: gathered rows exact AND everything outside them
    // (row gaps, plane tails) untouched.
    ExpectBitsEqual(d_khat.Download(), khat_mirror, std::string("kv_gather khat ") + m.name);
    ExpectBitsEqual(d_vhat.Download(), vhat_mirror, std::string("kv_gather vhat ") + m.name);
  }
}

// ===========================================================================
// KernelPagedAttentionDecodeTest - GQA decode attention: FP64 reference,
// NaN-poisoned pool, oracle cross-check, both stride modes, padded rows.
// ===========================================================================

struct DecodeCase {
  std::int32_t num_seqs = 0;
  std::int32_t max_blocks_per_seq = 0;
  std::int32_t num_q_heads = 0;
  std::int32_t num_kv_heads = 0;
  std::vector<std::int32_t> seq_lens;
  std::vector<std::int32_t> block_tables; // [num_seqs, max_blocks_per_seq], dummy-padded
  std::vector<std::uint16_t> pool;        // NaN-poisoned; only live slots written
  std::vector<std::uint16_t> q_packed;    // [num_seqs, q_heads * head_dim]
};

// NaN-poisons a pool sized for the case and CPU-scatters random K/V for
// exactly the live positions of every sequence - the section 12a
// NaN-poisoned-pool protocol. Block tables are shuffled (non-contiguous) and
// padded with dummy block 0.
DecodeCase BuildDecodeCase(std::uint64_t seed, const std::vector<std::int32_t>& seq_lens,
                           std::int32_t num_q_heads, std::int32_t num_kv_heads) {
  DecodeCase dc;
  dc.num_seqs = static_cast<std::int32_t>(seq_lens.size());
  dc.seq_lens = seq_lens;
  dc.num_q_heads = num_q_heads;
  dc.num_kv_heads = num_kv_heads;
  std::int32_t max_len = 1;
  std::int32_t blocks_needed = 0;
  for (const std::int32_t len : seq_lens) {
    max_len = std::max(max_len, len);
    blocks_needed += (len + ref::kKvBlockSize - 1) / ref::kKvBlockSize;
  }
  dc.max_blocks_per_seq = (max_len + ref::kKvBlockSize - 1) / ref::kKvBlockSize;
  const std::int32_t num_blocks = blocks_needed + 2; // + dummy + one spare

  ref::SplitMix64 rng(seed);
  const std::vector<std::int32_t> live = ShuffledLiveBlocks(rng, num_blocks);
  dc.block_tables.assign(std::size_t(dc.num_seqs) * dc.max_blocks_per_seq, 0);
  dc.pool =
      NanFilled(static_cast<std::size_t>(ref::PoolSliceElems(num_blocks, num_kv_heads, kHeadDim)));

  std::size_t next_live = 0;
  const std::size_t kv_row_width = std::size_t(num_kv_heads) * kHeadDim;
  for (std::int32_t s = 0; s < dc.num_seqs; ++s) {
    const std::int32_t len = seq_lens[s];
    if (len <= 0)
      continue; // padded slot: dummy-only table row
    const std::int32_t rows_blocks = (len + ref::kKvBlockSize - 1) / ref::kKvBlockSize;
    std::vector<std::int32_t> row(dc.max_blocks_per_seq, 0);
    for (std::int32_t j = 0; j < rows_blocks; ++j)
      row[j] = live[next_live++];
    std::copy(row.begin(), row.end(),
              dc.block_tables.begin() + std::size_t(s) * dc.max_blocks_per_seq);
    // Live K/V bits for positions [0, len).
    const std::vector<std::uint16_t> k_rows =
        UniformHalfBits(rng, std::size_t(len) * kv_row_width, -1.0, 1.0);
    const std::vector<std::uint16_t> v_rows =
        UniformHalfBits(rng, std::size_t(len) * kv_row_width, -1.0, 1.0);
    std::vector<std::int32_t> positions(len);
    for (std::int32_t i = 0; i < len; ++i)
      positions[i] = i;
    ref::ScatterReference(
        dc.pool, k_rows.data(), v_rows.data(), static_cast<std::int64_t>(kv_row_width),
        static_cast<std::int64_t>(kv_row_width), row.data(),
        /*block_table_row_stride=*/0, positions.data(), len, num_kv_heads, kHeadDim);
  }
  dc.q_packed = UniformHalfBits(rng, std::size_t(dc.num_seqs) * num_q_heads * kHeadDim, -1.0, 1.0);
  return dc;
}

enum class DecodeKernel { kPrimary, kOracle };

// Launches the requested decode kernel with optional qkv-style strides and
// returns the packed output rows. Verifies strided-gap integrity and that
// padded rows (seq_len 0) stay untouched.
std::vector<std::uint16_t> RunDecode(const DecodeCase& dc, DecodeKernel which, bool strided,
                                     const std::string& label) {
  const std::size_t q_width = std::size_t(dc.num_q_heads) * kHeadDim;
  const std::int64_t q_stride = strided ? kQkvStride : static_cast<std::int64_t>(q_width);
  const std::int64_t out_stride = strided ? kQkvStride : static_cast<std::int64_t>(q_width);

  ref::SplitMix64 rng(0x0A7Bu);
  std::vector<std::uint16_t> q_image(std::size_t(dc.num_seqs) * q_stride);
  q_image = UniformHalfBits(rng, q_image.size(), -1.0, 1.0); // garbage outside the view
  PlaceRows(q_image, dc.q_packed, dc.num_seqs, q_width, q_stride, 0);
  const std::vector<std::uint16_t> out_before =
      UniformHalfBits(rng, std::size_t(dc.num_seqs) * out_stride, -1.0, 1.0);

  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_pool(dc.pool);
  DeviceBuffer<std::uint16_t> d_q(q_image);
  DeviceBuffer<std::uint16_t> d_out(out_before);
  DeviceBuffer<std::int32_t> d_tables(dc.block_tables);
  DeviceBuffer<std::int32_t> d_lens(dc.seq_lens);
  if (which == DecodeKernel::kPrimary) {
    LaunchPagedAttentionDecode(AsHalf(d_out.get()), out_stride, AsHalf(d_q.get()), q_stride,
                               AsHalf(d_pool.get()), d_tables.get(), d_lens.get(), dc.num_seqs,
                               dc.max_blocks_per_seq, dc.num_q_heads, dc.num_kv_heads, kHeadDim,
                               kAttnScale, stream.get());
  } else {
    LaunchPagedAttentionDecodeOracle(AsHalf(d_out.get()), out_stride, AsHalf(d_q.get()), q_stride,
                                     AsHalf(d_pool.get()), d_tables.get(), d_lens.get(),
                                     dc.num_seqs, dc.max_blocks_per_seq, dc.num_q_heads,
                                     dc.num_kv_heads, kHeadDim, kAttnScale, stream.get());
  }
  SyncStream(stream.get());
  const std::vector<std::uint16_t> out_after = d_out.Download();

  ExpectOutsideViewsUntouched(out_before, out_after, dc.num_seqs, q_width, out_stride, 0,
                              label + " out gap");
  for (std::int32_t s = 0; s < dc.num_seqs; ++s) {
    if (dc.seq_lens[s] > 0)
      continue;
    // Padded slot: the kernel exits before writing anything. (Non-fatal
    // reporting: this helper returns a value, so fatal ASSERTs cannot be
    // used here.)
    for (std::size_t c = 0; c < q_width; ++c) {
      const std::size_t i = std::size_t(s) * out_stride + c;
      if (out_after[i] != out_before[i]) {
        ADD_FAILURE() << label << ": padded row " << s << " column " << c << " was written";
        break;
      }
    }
  }
  return TakeRows(out_after, dc.num_seqs, q_width, out_stride, 0);
}

// Compares live rows of `actual` against `expected` (packed, padded rows
// zeroed in the reference) at the attention tolerance and requires them
// NaN-free.
void ExpectDecodeMatches(const DecodeCase& dc, const std::vector<std::uint16_t>& actual,
                         const std::vector<std::uint16_t>& expected_bits,
                         const std::string& label) {
  const std::size_t q_width = std::size_t(dc.num_q_heads) * kHeadDim;
  std::vector<std::uint16_t> live_actual;
  std::vector<double> live_expected;
  for (std::int32_t s = 0; s < dc.num_seqs; ++s) {
    if (dc.seq_lens[s] <= 0)
      continue;
    for (std::size_t c = 0; c < q_width; ++c) {
      const std::size_t i = std::size_t(s) * q_width + c;
      ASSERT_FALSE(ref::IsNanHalfBits(actual[i]))
          << label << ": NaN leaked into live row " << s << " column " << c;
      live_actual.push_back(actual[i]);
      live_expected.push_back(ref::HalfBitsToDouble(expected_bits[i]));
    }
  }
  ExpectHalfNearDouble(live_actual, live_expected, kAttentionMaxAbs, kElementwiseMeanAbs, label);
}

TEST(KernelPagedAttentionDecodeTest, NanPoisonedPoolMatchesReferenceRaggedBatch8) {
  REDLINE_REQUIRE_GPU();
  // Batch 8 with padded rows and the full section 12a seq_len boundary set;
  // every unwritten pool slot holds a NaN pattern, so select-style masking
  // is load-bearing for this to pass at all. Strided q (qkv_out view) and
  // strided out.
  const DecodeCase dc = BuildDecodeCase(
      0xA77Eu, std::vector<std::int32_t>{257, 0, 64, 1, 17, 15, 63, 65}, kNumQHeads, kNumKvHeads);
  const std::vector<std::uint16_t> expected = ref::PagedAttentionDecodeReference(
      dc.q_packed, dc.pool, dc.block_tables, dc.seq_lens, dc.num_seqs, dc.max_blocks_per_seq,
      dc.num_q_heads, dc.num_kv_heads, kHeadDim, static_cast<double>(kAttnScale));
  const std::vector<std::uint16_t> got =
      RunDecode(dc, DecodeKernel::kPrimary, /*strided=*/true, "decode batch8 strided");
  ExpectDecodeMatches(dc, got, expected, "decode batch8 strided vs FP64 reference");
}

TEST(KernelPagedAttentionDecodeTest, PackedModeSmallBatchesWithPaddedRows) {
  REDLINE_REQUIRE_GPU();
  const std::vector<std::vector<std::int32_t>> cases = {{1}, {257}, {16, 0, 65}};
  std::uint64_t seed = 0xA801u;
  for (const auto& lens : cases) {
    const DecodeCase dc = BuildDecodeCase(seed++, lens, kNumQHeads, kNumKvHeads);
    const std::vector<std::uint16_t> expected = ref::PagedAttentionDecodeReference(
        dc.q_packed, dc.pool, dc.block_tables, dc.seq_lens, dc.num_seqs, dc.max_blocks_per_seq,
        dc.num_q_heads, dc.num_kv_heads, kHeadDim, static_cast<double>(kAttnScale));
    const std::string label = "decode batch" + std::to_string(dc.num_seqs) + " packed";
    const std::vector<std::uint16_t> got =
        RunDecode(dc, DecodeKernel::kPrimary, /*strided=*/false, label);
    ExpectDecodeMatches(dc, got, expected, label + " vs FP64 reference");
  }
}

TEST(KernelPagedAttentionDecodeTest, PrimaryMatchesPerQHeadOracleOnRandomizedShapes) {
  REDLINE_REQUIRE_GPU();
  // Section 12a mandatory case 4: the KV-head-centric primary kernel against
  // the structurally independent per-Q-head oracle on randomized shapes.
  // Both also stay within tolerance of the FP64 reference.
  ref::SplitMix64 shape_rng(0x0AC1Eu);
  for (int trial = 0; trial < 6; ++trial) {
    const bool small_gqa = (trial % 2) == 1; // alternate 12q/2kv and 6q/1kv
    const std::int32_t num_q = small_gqa ? 6 : kNumQHeads;
    const std::int32_t num_kv = small_gqa ? 1 : kNumKvHeads;
    const std::int32_t num_seqs = 1 + static_cast<std::int32_t>(shape_rng.Next() % 8);
    std::vector<std::int32_t> lens(num_seqs);
    for (auto& len : lens)
      len = static_cast<std::int32_t>(shape_rng.Next() % 301);
    lens[0] = std::max(lens[0], 1); // at least one live row
    const DecodeCase dc = BuildDecodeCase(0xF00Du + trial, lens, num_q, num_kv);
    const bool strided = (trial % 2) == 0;
    const std::string label = "oracle cross-check trial " + std::to_string(trial);

    const std::vector<std::uint16_t> primary =
        RunDecode(dc, DecodeKernel::kPrimary, strided, label + " primary");
    const std::vector<std::uint16_t> oracle =
        RunDecode(dc, DecodeKernel::kOracle, strided, label + " oracle");
    ExpectDecodeMatches(dc, primary, oracle, label + " primary vs oracle");

    const std::vector<std::uint16_t> expected = ref::PagedAttentionDecodeReference(
        dc.q_packed, dc.pool, dc.block_tables, dc.seq_lens, dc.num_seqs, dc.max_blocks_per_seq,
        dc.num_q_heads, dc.num_kv_heads, kHeadDim, static_cast<double>(kAttnScale));
    ExpectDecodeMatches(dc, primary, expected, label + " primary vs FP64 reference");
    ExpectDecodeMatches(dc, oracle, expected, label + " oracle vs FP64 reference");
  }
}

TEST(KernelPagedAttentionDecodeTest, DeterministicAcrossRepeatedLaunches) {
  REDLINE_REQUIRE_GPU();
  const DecodeCase dc =
      BuildDecodeCase(0xDE7E2u, std::vector<std::int32_t>{65, 0, 17, 257}, kNumQHeads, kNumKvHeads);
  const std::vector<std::uint16_t> first =
      RunDecode(dc, DecodeKernel::kPrimary, /*strided=*/false, "determinism pass 1");
  const std::vector<std::uint16_t> second =
      RunDecode(dc, DecodeKernel::kPrimary, /*strided=*/false, "determinism pass 2");
  // Padded rows carry launch-local garbage in the packed extraction, so
  // compare live rows only.
  const std::size_t q_width = std::size_t(dc.num_q_heads) * kHeadDim;
  for (std::int32_t s = 0; s < dc.num_seqs; ++s) {
    if (dc.seq_lens[s] <= 0)
      continue;
    for (std::size_t c = 0; c < q_width; ++c) {
      const std::size_t i = std::size_t(s) * q_width + c;
      ASSERT_EQ(first[i], second[i]) << "decode nondeterminism at row " << s << " col " << c;
    }
  }
}

// ===========================================================================
// KernelPrefillSoftmaxTest - causal row softmax over FP32 scores. Dense
// contract (no stride parameters).
// ===========================================================================

std::vector<float> UniformScores(ref::SplitMix64& rng, std::size_t n, double lo, double hi) {
  std::vector<float> v(n);
  for (auto& s : v)
    s = static_cast<float>(lo + rng.NextUniform() * (hi - lo));
  return v;
}

void RunPrefillSoftmaxCase(std::int32_t num_q_heads, std::int32_t chunk_len, std::int32_t kv_len,
                           std::int32_t chunk_start, std::uint64_t seed) {
  ref::SplitMix64 rng(seed);
  const std::size_t total = std::size_t(num_q_heads) * chunk_len * kv_len;
  const std::vector<float> scores = UniformScores(rng, total, -8.0, 8.0);

  CudaStream stream;
  DeviceBuffer<float> d_scores(scores);
  DeviceBuffer<std::uint16_t> d_probs(total);
  LaunchPrefillSoftmax(AsHalf(d_probs.get()), d_scores.get(), num_q_heads, chunk_len, kv_len,
                       chunk_start, stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> probs = d_probs.Download();

  const std::vector<std::uint16_t> expected =
      ref::PrefillSoftmaxReference(scores, num_q_heads, chunk_len, kv_len, chunk_start);
  const std::string label = "prefill softmax heads=" + std::to_string(num_q_heads) +
                            " T=" + std::to_string(chunk_len) + " kv=" + std::to_string(kv_len) +
                            " start=" + std::to_string(chunk_start);
  ExpectHalfNearBits(probs, expected, kElementwiseMaxAbs, kElementwiseMeanAbs, label);
  // Mask region: exact positive zeros (+0x0000), bitwise - the PV GEMMs
  // contract over the full kv_len extent, so a masked key must contribute
  // exactly nothing.
  std::size_t masked_bad = 0;
  for (std::int32_t h = 0; h < num_q_heads; ++h) {
    for (std::int32_t i = 0; i < chunk_len; ++i) {
      const std::size_t base = (std::size_t(h) * chunk_len + i) * kv_len;
      const std::int32_t live = std::min(chunk_start + i + 1, kv_len);
      for (std::int32_t j = live; j < kv_len; ++j) {
        masked_bad += (probs[base + j] != 0x0000u) ? 1 : 0;
      }
    }
  }
  EXPECT_EQ(masked_bad, 0u) << label << ": masked probabilities not exact +0";
}

TEST(KernelPrefillSoftmaxTest, FirstChunkMatchesReferenceAndRowZeroIsExact) {
  REDLINE_REQUIRE_GPU();
  RunPrefillSoftmaxCase(kNumQHeads, 17, 17, 0, 0x9E1u);
  // Row 0 of a first chunk is a single-key softmax: probability exactly 1.
  ref::SplitMix64 rng(0x9E2u);
  const std::int32_t chunk = 5;
  const std::vector<float> scores =
      UniformScores(rng, std::size_t(kNumQHeads) * chunk * chunk, -8.0, 8.0);
  CudaStream stream;
  DeviceBuffer<float> d_scores(scores);
  DeviceBuffer<std::uint16_t> d_probs(scores.size());
  LaunchPrefillSoftmax(AsHalf(d_probs.get()), d_scores.get(), kNumQHeads, chunk, chunk, 0,
                       stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> probs = d_probs.Download();
  for (std::int32_t h = 0; h < kNumQHeads; ++h) {
    const std::size_t base = std::size_t(h) * chunk * chunk;
    ASSERT_EQ(probs[base], ref::kHalfOneBits) << "head " << h << " row 0 self-probability";
    for (std::int32_t j = 1; j < chunk; ++j) {
      ASSERT_EQ(probs[base + j], 0x0000u) << "head " << h << " row 0 masked col " << j;
    }
  }
}

TEST(KernelPrefillSoftmaxTest, MaskBoundaryExactWithPoisonedMaskedRegion) {
  REDLINE_REQUIRE_GPU();
  // Key j == q is live, j == q + 1 is the first masked column. The masked
  // region is poisoned with +inf, NaN and huge values: none of it may
  // influence the live prefix, and its probabilities must be exact +0 -
  // the select-style analogue of the NaN-poisoned-pool decode test.
  const std::int32_t heads = 2;
  const std::int32_t chunk = 16;
  const std::int32_t kv = 40;
  const std::int32_t chunk_start = 8;
  ref::SplitMix64 rng(0xFACEu);
  std::vector<float> scores = UniformScores(rng, std::size_t(heads) * chunk * kv, -4.0, 4.0);
  for (std::int32_t h = 0; h < heads; ++h) {
    for (std::int32_t i = 0; i < chunk; ++i) {
      const std::size_t base = (std::size_t(h) * chunk + i) * kv;
      const std::int32_t q = chunk_start + i;
      for (std::int32_t j = q + 1; j < kv; ++j) {
        const std::uint64_t pick = rng.Next() % 3;
        scores[base + j] = (pick == 0)   ? std::numeric_limits<float>::infinity()
                           : (pick == 1) ? std::numeric_limits<float>::quiet_NaN()
                                         : 1000.0f; // huge finite: masked j == q+1 included
      }
    }
  }
  CudaStream stream;
  DeviceBuffer<float> d_scores(scores);
  DeviceBuffer<std::uint16_t> d_probs(scores.size());
  LaunchPrefillSoftmax(AsHalf(d_probs.get()), d_scores.get(), heads, chunk, kv, chunk_start,
                       stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> probs = d_probs.Download();
  const std::vector<std::uint16_t> expected =
      ref::PrefillSoftmaxReference(scores, heads, chunk, kv, chunk_start);
  ExpectHalfNearBits(probs, expected, kElementwiseMaxAbs, kElementwiseMeanAbs,
                     "poisoned mask region");
  for (std::int32_t h = 0; h < heads; ++h) {
    for (std::int32_t i = 0; i < chunk; ++i) {
      const std::size_t base = (std::size_t(h) * chunk + i) * kv;
      const std::int32_t q = chunk_start + i;
      for (std::int32_t j = q + 1; j < kv; ++j) {
        ASSERT_EQ(probs[base + j], 0x0000u)
            << "masked (h=" << h << ", i=" << i << ", j=" << j << ") leaked";
      }
      // No NaN anywhere in the live prefix either.
      for (std::int32_t j = 0; j <= std::min(q, kv - 1); ++j) {
        ASSERT_FALSE(ref::IsNanHalfBits(probs[base + j]))
            << "NaN in live prefix (h=" << h << ", i=" << i << ", j=" << j << ")";
      }
    }
  }
}

TEST(KernelPrefillSoftmaxTest, MidChunkShapeSweepKvGreaterThanChunk) {
  REDLINE_REQUIRE_GPU();
  // kv_len > chunk_len is the normal case for every chunk after the first.
  RunPrefillSoftmaxCase(kNumQHeads, 17, 50, 33, 0xC401u);
  RunPrefillSoftmaxCase(kNumQHeads, 15, 16, 1, 0xC402u);
  RunPrefillSoftmaxCase(kNumQHeads, 16, 32, 16, 0xC403u);
  RunPrefillSoftmaxCase(kNumQHeads, 1, 257, 256, 0xC404u); // single-row chunk
  RunPrefillSoftmaxCase(2, 8, 255, 247, 0xC405u);          // kv crossing the 256-thread stride
  RunPrefillSoftmaxCase(2, 8, 256, 248, 0xC406u);
  RunPrefillSoftmaxCase(2, 8, 257, 249, 0xC407u);
  RunPrefillSoftmaxCase(2, 64, 2112, 2048, 0xC408u); // bench-shaped tail chunk
}

TEST(KernelPrefillSoftmaxTest, DeterministicAcrossRepeatedLaunches) {
  REDLINE_REQUIRE_GPU();
  ref::SplitMix64 rng(0xD371u);
  const std::size_t total = std::size_t(kNumQHeads) * 33 * 65;
  const std::vector<float> scores = UniformScores(rng, total, -8.0, 8.0);
  CudaStream stream;
  DeviceBuffer<float> d_scores(scores);
  std::vector<std::uint16_t> outs[2];
  for (int pass = 0; pass < 2; ++pass) {
    DeviceBuffer<std::uint16_t> d_probs(total);
    LaunchPrefillSoftmax(AsHalf(d_probs.get()), d_scores.get(), kNumQHeads, 33, 65, 32,
                         stream.get());
    SyncStream(stream.get());
    outs[pass] = d_probs.Download();
  }
  ExpectBitsEqual(outs[1], outs[0], "prefill softmax run-to-run determinism");
}

// ===========================================================================
// KernelSiluMulTest - SwiGLU activation. Strideless contract; the aliased
// in-place mode (out == gate_up) is dispatched on pointer identity.
// ===========================================================================

void RunSiluMulSeparateCase(std::int32_t rows, std::int32_t inter, std::uint64_t seed) {
  ref::SplitMix64 rng(seed);
  const std::vector<std::uint16_t> gate_up =
      UniformHalfBits(rng, std::size_t(rows) * 2 * inter, -3.0, 3.0);
  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_in(gate_up);
  DeviceBuffer<std::uint16_t> d_out(std::size_t(rows) * inter);
  LaunchSiluMul(AsHalf(d_out.get()), AsHalf(d_in.get()), rows, inter, stream.get());
  SyncStream(stream.get());
  const std::string label = "silu_mul rows=" + std::to_string(rows) + " I=" + std::to_string(inter);
  ExpectHalfNearBits(d_out.Download(), ref::SiluMulReference(gate_up, rows, inter),
                     kElementwiseMaxAbs, kElementwiseMeanAbs, label);
  ExpectBitsEqual(d_in.Download(), gate_up, label + " input unmodified");
}

TEST(KernelSiluMulTest, SeparateOutMatchesFp64Reference) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t rows : {1, 3, 8, 257}) {
    RunSiluMulSeparateCase(rows, 8960, 0x51Cu + rows);
  }
}

TEST(KernelSiluMulTest, AliasedInPlaceOverGateHalfLeavesUpHalfUntouched) {
  REDLINE_REQUIRE_GPU();
  const std::int32_t rows = 8;
  const std::int32_t inter = 8960;
  ref::SplitMix64 rng(0xA11A5u);
  const std::vector<std::uint16_t> gate_up =
      UniformHalfBits(rng, std::size_t(rows) * 2 * inter, -3.0, 3.0);
  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_buf(gate_up);
  // out == gate_up: the launcher dispatches the in-place mode (out row
  // stride 2I over the gate half).
  LaunchSiluMul(AsHalf(d_buf.get()), AsHalf(d_buf.get()), rows, inter, stream.get());
  SyncStream(stream.get());
  const std::vector<std::uint16_t> after = d_buf.Download();

  const std::vector<std::uint16_t> expected = ref::SiluMulReference(gate_up, rows, inter);
  std::vector<std::uint16_t> gate_half(std::size_t(rows) * inter);
  for (std::int32_t r = 0; r < rows; ++r) {
    std::memcpy(&gate_half[std::size_t(r) * inter], &after[std::size_t(r) * 2 * inter],
                std::size_t(inter) * sizeof(std::uint16_t));
    // Up half must be byte-identical to the input.
    for (std::int32_t c = 0; c < inter; ++c) {
      const std::size_t i = std::size_t(r) * 2 * inter + inter + c;
      ASSERT_EQ(after[i], gate_up[i]) << "up half modified at row " << r << " col " << c;
    }
  }
  ExpectHalfNearBits(gate_half, expected, kElementwiseMaxAbs, kElementwiseMeanAbs,
                     "silu_mul aliased gate half");
}

TEST(KernelSiluMulTest, OddIntermediateSizesTakeScalarPath) {
  REDLINE_REQUIRE_GPU();
  RunSiluMulSeparateCase(3, 129, 0x51D1u); // odd I: scalar kernel
  RunSiluMulSeparateCase(3, 130, 0x51D2u); // even small I: half2 kernel tail
}

// ===========================================================================
// KernelArgmaxTest - greedy argmax: exact, including crafted ties, signed
// zeros, NaN policy and vocab-size tails. Strideless contract.
// ===========================================================================

std::vector<std::int32_t> RunArgmax(const std::vector<std::uint16_t>& logits, std::int32_t num_seqs,
                                    std::int32_t vocab) {
  CudaStream stream;
  DeviceBuffer<std::uint16_t> d_logits(logits);
  DeviceBuffer<std::int32_t> d_out(static_cast<std::size_t>(num_seqs));
  LaunchGreedyArgmax(d_out.get(), AsHalf(d_logits.get()), num_seqs, vocab, stream.get());
  SyncStream(stream.get());
  return d_out.Download();
}

TEST(KernelArgmaxTest, MatchesReferenceOnRandomLogitsFullVocab) {
  REDLINE_REQUIRE_GPU();
  const std::int32_t vocab = 151936;
  for (const std::int32_t rows : {1, 3, 8}) {
    ref::SplitMix64 rng(0xA46u + rows);
    std::vector<std::uint16_t> logits = UniformHalfBits(rng, std::size_t(rows) * vocab, -4.0, 4.0);
    // Plant one unambiguous winner per row at a random column.
    for (std::int32_t r = 0; r < rows; ++r) {
      logits[std::size_t(r) * vocab + rng.Next() % vocab] = 0x4880; // 9.0
    }
    const std::vector<std::int32_t> got = RunArgmax(logits, rows, vocab);
    const std::vector<std::int32_t> expected = ref::ArgmaxReference(logits, rows, vocab);
    for (std::int32_t r = 0; r < rows; ++r) {
      ASSERT_EQ(got[r], expected[r]) << "rows=" << rows << " row " << r;
    }
  }
}

TEST(KernelArgmaxTest, CraftedTiesResolveToLowestIndex) {
  REDLINE_REQUIRE_GPU();
  const std::int32_t vocab = 151936;
  const std::uint16_t tie = 0x4840; // 8.5, exact in FP16
  // Tie pairs straddle the extremes, thread boundaries (255|256), warp
  // boundaries (31|32, 127|128), a same-thread stride pair (c, c+256), and
  // the first reduction level (0, 128); plus a 3-way tie.
  const std::vector<std::vector<std::int32_t>> tie_sets = {
      {0, 151935}, {255, 256}, {31, 32}, {127, 128}, {1000, 1256}, {0, 128}, {5, 700, 151000}};
  const std::int32_t rows = static_cast<std::int32_t>(tie_sets.size());
  ref::SplitMix64 rng(0x7135u);
  std::vector<std::uint16_t> logits = UniformHalfBits(rng, std::size_t(rows) * vocab, -1.0, 1.0);
  for (std::int32_t r = 0; r < rows; ++r) {
    for (const std::int32_t idx : tie_sets[r]) {
      logits[std::size_t(r) * vocab + idx] = tie;
    }
  }
  const std::vector<std::int32_t> got = RunArgmax(logits, rows, vocab);
  const std::vector<std::int32_t> expected = ref::ArgmaxReference(logits, rows, vocab);
  for (std::int32_t r = 0; r < rows; ++r) {
    ASSERT_EQ(got[r], tie_sets[r].front()) << "tie set " << r << " must pick the lowest index";
    ASSERT_EQ(got[r], expected[r]) << "reference disagrees on tie set " << r;
  }
}

TEST(KernelArgmaxTest, SignedZerosCompareEqualAndResolveByIndex) {
  REDLINE_REQUIRE_GPU();
  const std::int32_t vocab = 1024;
  std::vector<std::uint16_t> logits(vocab, 0xBC00); // all -1.0
  logits[100] = 0x8000;                             // -0.0
  logits[200] = 0x0000;                             // +0.0 - equal compare, higher index
  const std::vector<std::int32_t> got = RunArgmax(logits, 1, vocab);
  ASSERT_EQ(got[0], 100) << "-0.0 == +0.0 must resolve to the lower index";
  ASSERT_EQ(got[0], ref::ArgmaxReference(logits, 1, vocab)[0]);
}

TEST(KernelArgmaxTest, NanPolicyNeverAdoptsNanAndAllNanYieldsTokenZero) {
  REDLINE_REQUIRE_GPU();
  const std::int32_t vocab = 4096;
  const std::int32_t rows = 4;
  ref::SplitMix64 rng(0x7A7Au);
  std::vector<std::uint16_t> logits = UniformHalfBits(rng, std::size_t(rows) * vocab, -1.0, 1.0);
  // Row 0: NaN at index 0, max 3.0 at 77 - torch.argmax would return 0 (NaN
  // propagates as the maximum); this kernel documents the deviation: NaN is
  // upstream corruption and must not steer token selection.
  logits[0] = NanPattern(0);
  logits[77] = 0x4200; // 3.0
  // Row 1: NaNs scattered among finite values, max 3.0 at 1234.
  for (std::int32_t i = 0; i < vocab; i += 97) {
    logits[std::size_t(1) * vocab + i] = NanPattern(i);
  }
  logits[std::size_t(1) * vocab + 1234] = 0x4200;
  // Row 2: all NaN -> token 0 (documented).
  for (std::int32_t i = 0; i < vocab; ++i)
    logits[std::size_t(2) * vocab + i] = NanPattern(i);
  // Row 3: all -inf -> token 0 through the ordinary index tiebreak.
  for (std::int32_t i = 0; i < vocab; ++i)
    logits[std::size_t(3) * vocab + i] = 0xFC00;

  const std::vector<std::int32_t> got = RunArgmax(logits, rows, vocab);
  const std::vector<std::int32_t> expected = ref::ArgmaxReference(logits, rows, vocab);
  EXPECT_EQ(got[0], 77) << "NaN at index 0 must not win";
  EXPECT_EQ(got[1], 1234);
  EXPECT_EQ(got[2], 0) << "all-NaN row must yield token 0";
  EXPECT_EQ(got[3], 0) << "all -inf row must yield token 0";
  for (std::int32_t r = 0; r < rows; ++r) {
    EXPECT_EQ(got[r], expected[r]) << "reference disagrees on row " << r;
  }
}

TEST(KernelArgmaxTest, VocabSizeTailsAndTinyRows) {
  REDLINE_REQUIRE_GPU();
  for (const std::int32_t vocab : {1, 255, 256, 257}) {
    const std::int32_t rows = 3;
    ref::SplitMix64 rng(0x70CABu + vocab);
    std::vector<std::uint16_t> logits = UniformHalfBits(rng, std::size_t(rows) * vocab, -2.0, 2.0);
    if (vocab >= 257) {
      // Tie straddling the 256-thread boundary within the tail shape.
      logits[std::size_t(2) * vocab + 255] = 0x4840;
      logits[std::size_t(2) * vocab + 256] = 0x4840;
    }
    const std::vector<std::int32_t> got = RunArgmax(logits, rows, vocab);
    const std::vector<std::int32_t> expected = ref::ArgmaxReference(logits, rows, vocab);
    for (std::int32_t r = 0; r < rows; ++r) {
      ASSERT_EQ(got[r], expected[r]) << "vocab=" << vocab << " row " << r;
    }
  }
}

} // namespace
} // namespace redline::kernels
