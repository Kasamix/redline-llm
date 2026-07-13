// BF16 -> FP16 conversion unit tests (docs/DESIGN.md sections 4 and 12a).
//
// The core check is exhaustive: every one of the 65,536 BF16 bit patterns is
// converted and compared bit-exactly against the committed torch-generated
// golden table (tests/data/bf16_fp16_golden.bin, produced by
// scripts/gen_bf16_golden.py). NaN entries are compared by "is NaN" - torch's
// NaN payload policy varies by code path, while the loader emits the
// canonical quiet NaN. Patterns beyond the FP16 finite range (|x| > 65504)
// appear in the table as +-inf (torch's IEEE rounding); the loader's checked
// path must hard-fail on them instead, naming tensor and element index.
//
// The golden file is located through the REDLINE_TEST_DATA_DIR environment
// variable (ctest sets it to <repo>/tests/data); running the binary directly
// from the repo root also works via the relative-path fallback.
//
// Staging-upload tests need a CUDA device and skip cleanly without one, so
// the suite stays green on CPU-only machines and CI.

#include "loader/convert.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/types.hpp"

namespace redline {
namespace {

constexpr std::size_t kNumPatterns = 1u << 16;
constexpr std::size_t kGoldenBytes = kNumPatterns * 2; // 128 KiB
// |x| > 65504 band: abs bits in [0x4780 (65536), 0x7F80 (inf)], both signs.
constexpr std::size_t kOverflowPatterns = 2 * (0x7F80 - 0x4780 + 1);

std::string Hex(std::uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%04X", v);
  return buf;
}

// Loads the golden table once; on failure returns nullptr and fills `error`.
const std::vector<std::uint16_t>* LoadGoldenTable(std::string* error) {
  static const auto* table = []() -> const std::vector<std::uint16_t>* {
    std::string path;
    if (const char* dir = std::getenv("REDLINE_TEST_DATA_DIR")) {
      path = std::string(dir) + "/bf16_fp16_golden.bin";
    } else {
      path = "tests/data/bf16_fp16_golden.bin"; // repo-root fallback
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
      return nullptr;
    std::vector<unsigned char> raw(kGoldenBytes);
    file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(file.gcount()) != kGoldenBytes || file.get() != EOF) {
      return nullptr; // truncated or oversized file
    }
    auto* golden = new std::vector<std::uint16_t>(kNumPatterns);
    for (std::size_t i = 0; i < kNumPatterns; ++i) { // explicit little-endian decode
      (*golden)[i] = static_cast<std::uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
    }
    return golden;
  }();
  if (table == nullptr && error != nullptr) {
    *error = "cannot load golden table 'bf16_fp16_golden.bin' (expected 131,072 bytes). "
             "Set REDLINE_TEST_DATA_DIR to <repo>/tests/data (ctest does) or run from the "
             "repo root. Regenerate with scripts/gen_bf16_golden.py if missing.";
  }
  return table;
}

#define REDLINE_REQUIRE_GOLDEN(golden)                                                             \
  std::string golden_error_;                                                                       \
  const std::vector<std::uint16_t>* golden = LoadGoldenTable(&golden_error_);                      \
  ASSERT_NE(golden, nullptr) << golden_error_

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

// Bytes of a u16 vector, for feeding the std::byte* conversion interface.
const std::byte* AsBytes(const std::vector<std::uint16_t>& v) {
  return reinterpret_cast<const std::byte*>(v.data());
}

TEST(Bf16ConvertTest, GoldenTableHasExpectedSizeAndLandmarks) {
  REDLINE_REQUIRE_GOLDEN(golden);
  ASSERT_EQ(golden->size(), kNumPatterns);
  // A few independently computed entries guard against a stale or corrupted
  // regeneration of the committed file.
  EXPECT_EQ((*golden)[0x0000], 0x0000); // +0
  EXPECT_EQ((*golden)[0x8000], 0x8000); // -0
  EXPECT_EQ((*golden)[0x3F80], 0x3C00); // 1.0
  EXPECT_EQ((*golden)[0x477F], 0x7BF8); // 65280: largest BF16 below FP16 max, exact
  EXPECT_EQ((*golden)[0x4780], 0x7C00); // 65536: torch rounds to inf (loader hard-fails)
  EXPECT_EQ((*golden)[0x7F80], 0x7C00); // +inf
  EXPECT_EQ((*golden)[0xFF80], 0xFC00); // -inf
  EXPECT_TRUE(IsNanFp16Bits((*golden)[0x7FC0]));
}

TEST(Bf16ConvertTest, ExhaustiveScalarMatchesTorchGoldenTable) {
  REDLINE_REQUIRE_GOLDEN(golden);
  for (std::uint32_t p = 0; p < kNumPatterns; ++p) {
    const std::uint16_t want = (*golden)[p];
    const std::uint16_t got = Bf16BitsToFp16BitsRne(static_cast<std::uint16_t>(p));
    if (IsNanFp16Bits(want)) {
      // NaN in, NaN out; payloads may differ (see file comment).
      ASSERT_TRUE(IsNanFp16Bits(got)) << "BF16 " << Hex(p) << ": golden is NaN (" << Hex(want)
                                      << ") but converter produced " << Hex(got);
      ASSERT_EQ(got & 0x8000u, want & 0x8000u) << "NaN sign lost for BF16 " << Hex(p);
    } else {
      ASSERT_EQ(got, want) << "BF16 " << Hex(p) << ": converter produced " << Hex(got)
                           << ", torch golden is " << Hex(want);
    }
  }
}

TEST(Bf16ConvertTest, OverflowBandMatchesGoldenInfinitiesAndHardFails) {
  REDLINE_REQUIRE_GOLDEN(golden);
  std::size_t band_count = 0;
  for (std::uint32_t p = 0; p < kNumPatterns; ++p) {
    const auto bits = static_cast<std::uint16_t>(p);
    // The hard-fail predicate must cover exactly the patterns torch rounds
    // to +-inf: finite |x| > 65504 plus the two infinities. NaN is outside
    // the band (|x| > 65504 is a magnitude test; NaN compares unordered).
    const bool in_band = Bf16OverflowsFp16(bits);
    ASSERT_EQ(in_band, IsInfFp16Bits((*golden)[p]))
        << "BF16 " << Hex(p) << ": overflow predicate disagrees with the golden table";
    if (!in_band)
      continue;
    ++band_count;

    std::uint16_t out = 0;
    std::byte src[2];
    std::memcpy(src, &bits, sizeof(bits));
    ASSERT_THROW(ConvertBf16ToFp16(src, 1, &out, "w", 0), Fp16OverflowError)
        << "BF16 " << Hex(p) << " (|x| > 65504) must hard-fail, never clamp";
  }
  EXPECT_EQ(band_count, kOverflowPatterns);
}

TEST(Bf16ConvertTest, CheckedConversionMatchesScalarOnAllInRangePatterns) {
  // Bulk-convert every non-overflow pattern in one call: exercises the
  // byte-stream interface and must agree bit-for-bit with the scalar core
  // (NaN payloads included - same code path).
  std::vector<std::uint16_t> src_bits;
  src_bits.reserve(kNumPatterns - kOverflowPatterns);
  for (std::uint32_t p = 0; p < kNumPatterns; ++p) {
    if (!Bf16OverflowsFp16(static_cast<std::uint16_t>(p))) {
      src_bits.push_back(static_cast<std::uint16_t>(p));
    }
  }
  ASSERT_EQ(src_bits.size(), kNumPatterns - kOverflowPatterns);

  std::vector<std::uint16_t> out(src_bits.size(), 0xDEAD);
  ASSERT_NO_THROW(ConvertBf16ToFp16(AsBytes(src_bits), src_bits.size(), out.data(), "bulk"));
  for (std::size_t i = 0; i < src_bits.size(); ++i) {
    ASSERT_EQ(out[i], Bf16BitsToFp16BitsRne(src_bits[i]))
        << "bulk conversion diverged from scalar at BF16 " << Hex(src_bits[i]);
  }
}

TEST(Bf16ConvertTest, OverflowReportsTensorNameAndAbsoluteElementIndex) {
  // Overflow mid-tensor with a nonzero index_base: reported index must be
  // tensor-absolute (chunked callers pass their chunk's element offset).
  const std::vector<std::uint16_t> src_bits = {0x3F80, 0x0000, 0xBF80, 0x4780, 0x3F80};
  std::vector<std::uint16_t> out(src_bits.size());
  try {
    ConvertBf16ToFp16(AsBytes(src_bits), src_bits.size(), out.data(),
                      "model.layers.0.mlp.up_proj.weight", 1000);
    FAIL() << "expected Fp16OverflowError";
  } catch (const Fp16OverflowError& e) {
    EXPECT_EQ(e.tensor_name(), "model.layers.0.mlp.up_proj.weight");
    EXPECT_EQ(e.element_index(), 1003u);
    EXPECT_EQ(e.bf16_bits(), 0x4780);
    const std::string what = e.what();
    EXPECT_NE(what.find("model.layers.0.mlp.up_proj.weight"), std::string::npos) << what;
    EXPECT_NE(what.find("1003"), std::string::npos) << what;
  }
  // Elements before the offending one were already converted.
  EXPECT_EQ(out[0], 0x3C00);
}

TEST(Bf16ConvertTest, SpotChecksDocumentConversionSemantics) {
  // Human-readable landmarks, independent of the golden file.
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x3F80), 0x3C00); // 1.0
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0xC000), 0xC000); // -2.0 (same bits in both formats)
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x8000), 0x8000); // -0.0 keeps its sign
  // Subnormals preserved, no flush-to-zero, round-to-nearest-even:
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x3300), 0x0000); // 2^-25: tie -> even mantissa (0)
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x3340), 0x0001); // 1.5*2^-25 -> min subnormal 2^-24
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0xB340), 0x8001); // negative subnormal keeps sign
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x387F), 0x03FC); // just below 2^-14 -> subnormal, exact
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x3880), 0x0400); // 2^-14 -> FP16 minimum normal
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x0001), 0x0000); // BF16 min subnormal (~9.2e-41) -> +0
  // Top of the finite range: 65280 converts exactly, 65536 is out of range.
  // (BF16's grid step is 256 here, so no BF16 value lands in (65280, 65536).)
  EXPECT_EQ(Bf16BitsToFp16BitsRne(0x477F), 0x7BF8); // 65280
  EXPECT_FALSE(Bf16OverflowsFp16(0x477F));
  EXPECT_TRUE(Bf16OverflowsFp16(0x4780));  // 65536
  EXPECT_TRUE(Bf16OverflowsFp16(0xFF80));  // -inf: |x| > 65504 under IEEE comparison
  EXPECT_FALSE(Bf16OverflowsFp16(0x7FC0)); // NaN: unordered, not gated by magnitude
}

TEST(Bf16ConvertTest, CheckedConversionPassesNanThrough) {
  // NaN weights are not range violations: they convert to FP16 NaN (and
  // would surface identically on both sides of the HF-parity suite).
  const std::vector<std::uint16_t> src_bits = {0x7FC0, 0xFFC0, 0x7F81, 0xFFFF};
  std::vector<std::uint16_t> out(src_bits.size(), 0);
  ASSERT_NO_THROW(ConvertBf16ToFp16(AsBytes(src_bits), src_bits.size(), out.data(), "nan"));
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_TRUE(IsNanFp16Bits(out[i])) << "index " << i << " produced " << Hex(out[i]);
    EXPECT_EQ(out[i] & 0x8000u, src_bits[i] & 0x8000u) << "NaN sign lost at index " << i;
  }
}

// --------------------------------------------------------- staging + upload
// These exercise the pinned-staging streaming path (bounded buffer, chunked
// convert + cudaMemcpyAsync on the caller's stream) and need a device.

TEST(Bf16ConvertTest, StagedUploadRoundTripsBf16ThroughDevice) {
  REDLINE_REQUIRE_GPU();

  // Every in-range pattern (36,862 elements) with deliberately small staging:
  // 16 KiB -> two 8 KiB slots -> 4,096-element chunks -> 9 chunks, so both
  // slots are reused several times and the event guards are exercised.
  std::vector<std::uint16_t> src_bits;
  for (std::uint32_t p = 0; p < kNumPatterns; ++p) {
    if (!Bf16OverflowsFp16(static_cast<std::uint16_t>(p))) {
      src_bits.push_back(static_cast<std::uint16_t>(p));
    }
  }
  const std::size_t count = src_bits.size();
  const std::size_t bytes = count * sizeof(std::uint16_t);

  std::vector<std::uint16_t> expected(count);
  ConvertBf16ToFp16(AsBytes(src_bits), count, expected.data(), "reference");

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void* dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, bytes), cudaSuccess);
  {
    StagedUploader uploader(std::size_t{16} << 10);
    ASSERT_EQ(uploader.chunk_elements(), 4096u);
    ASSERT_NO_THROW(
        uploader.Upload(dst, AsBytes(src_bits), count, Dtype::kBF16, "roundtrip", stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
  std::vector<std::uint16_t> got(count, 0);
  ASSERT_EQ(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(dst), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(got[i], expected[i])
        << "device round trip diverged at element " << i << " (BF16 " << Hex(src_bits[i]) << ")";
  }
}

TEST(Bf16ConvertTest, StagedUploadFp16PassthroughIsVerbatimAndRejectsOtherDtypes) {
  REDLINE_REQUIRE_GPU();

  // FP16 tensors are staged unmodified: arbitrary bit patterns (including
  // NaN/inf encodings) must arrive on the device verbatim.
  std::vector<std::uint16_t> src_bits(5000);
  for (std::size_t i = 0; i < src_bits.size(); ++i) {
    src_bits[i] = static_cast<std::uint16_t>((i * 2654435761u) >> 16); // arbitrary spread
  }
  src_bits[0] = 0x7C00; // +inf
  src_bits[1] = 0xFE00; // -NaN
  src_bits[2] = 0x7BFF; // FP16 max finite
  const std::size_t bytes = src_bits.size() * sizeof(std::uint16_t);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void* dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, bytes), cudaSuccess);

  StagedUploader uploader(std::size_t{4} << 10); // 1,024-element chunks -> 5 chunks
  ASSERT_NO_THROW(
      uploader.Upload(dst, AsBytes(src_bits), src_bits.size(), Dtype::kF16, "fp16", stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  std::vector<std::uint16_t> got(src_bits.size(), 0);
  ASSERT_EQ(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(std::memcmp(got.data(), src_bits.data(), bytes), 0);

  // Only BF16 and F16 sources are supported; anything else is a hard error
  // naming the tensor (the checkpoint is all-BF16 per docs/MODEL_SPEC.md).
  try {
    uploader.Upload(dst, AsBytes(src_bits), src_bits.size(), Dtype::kF32, "w_f32", stream);
    FAIL() << "expected std::runtime_error for unsupported dtype";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("w_f32"), std::string::npos) << e.what();
  }

  ASSERT_EQ(cudaFree(dst), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(Bf16ConvertTest, StagedUploadReportsOverflowBeyondFirstChunkAndStaysUsable) {
  REDLINE_REQUIRE_GPU();

  // Overflow deep inside the tensor (third chunk with 4,096-element chunks):
  // the reported index must be absolute across chunks, offset by index_base.
  std::vector<std::uint16_t> src_bits(10000, 0x3F80);
  src_bits[9000] = 0x4780; // 65536
  const std::size_t bytes = src_bits.size() * sizeof(std::uint16_t);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  void* dst = nullptr;
  ASSERT_EQ(cudaMalloc(&dst, bytes), cudaSuccess);

  StagedUploader uploader(std::size_t{16} << 10);
  try {
    uploader.Upload(dst, AsBytes(src_bits), src_bits.size(), Dtype::kBF16, "w_qkv", stream, 500);
    FAIL() << "expected Fp16OverflowError";
  } catch (const Fp16OverflowError& e) {
    EXPECT_EQ(e.tensor_name(), "w_qkv");
    EXPECT_EQ(e.element_index(), 9500u); // 9000 + index_base 500
    EXPECT_EQ(e.bf16_bits(), 0x4780);
  }

  // The uploader (and its staging slots) must remain usable after a failed
  // tensor: retry with clean data.
  src_bits[9000] = 0x3F80;
  ASSERT_NO_THROW(
      uploader.Upload(dst, AsBytes(src_bits), src_bits.size(), Dtype::kBF16, "w_qkv", stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  std::vector<std::uint16_t> got(src_bits.size(), 0);
  ASSERT_EQ(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], 0x3C00) << "element " << i;
  }

  ASSERT_EQ(cudaFree(dst), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
} // namespace redline
