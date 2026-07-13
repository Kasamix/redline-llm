#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include "core/types.hpp"

// BF16 -> FP16 weight conversion + pinned staging upload (docs/DESIGN.md
// section 4).
//
// The checkpoint stores every tensor in BF16 (docs/MODEL_SPEC.md section 8);
// the engine serves FP16. Conversion happens once, on the CPU, at load time:
//
//   1. expand BF16 to FP32 exactly (`u32 = u16 << 16` - BF16 is by
//      construction the top 16 bits of an IEEE-754 binary32), then
//   2. round FP32 to FP16 with round-to-nearest-even, subnormal results
//      produced exactly (no flush-to-zero), no clamping.
//
// This reproduces bit-for-bit what `torch.Tensor.to(float16)` does when HF
// loads the BF16 checkpoint as FP16, so converted weights are identical to
// the e2e reference's (validated exhaustively over all 65,536 BF16 bit
// patterns against a torch-generated golden table; see
// scripts/gen_bf16_golden.py and tests/test_bf16_convert.cpp).
//
// Overflow policy: any input with |x| > 65504 (the FP16 maximum finite
// value; this includes +-infinity) makes the checked conversion fail hard,
// naming the tensor and the element index. The expected overflow count for
// this checkpoint is zero, so a single hit means a corrupt file or the wrong
// checkpoint; clamping would silently diverge from the HF reference. NaN
// payloads are not gated by the magnitude predicate and convert to FP16 NaN.

namespace redline {

// ---------------------------------------------------------------- scalar core

// BF16 bits -> FP32 bits. Exact: BF16 is binary32 truncated to its top 16
// bits, so appending 16 zero bits reconstructs a binary32 with the same sign,
// exponent, and (zero-extended) mantissa.
constexpr std::uint32_t Bf16BitsToFp32Bits(std::uint16_t bf16_bits) noexcept {
  return static_cast<std::uint32_t>(bf16_bits) << 16;
}

// IEEE-754 binary32 -> binary16 with round-to-nearest-even.
//
//  - Subnormal FP16 results are computed exactly (no flush-to-zero): inputs
//    in [2^-25, 2^-14) round onto the FP16 subnormal grid k * 2^-24.
//  - Finite inputs whose rounded magnitude exceeds 65504 become +-infinity,
//    exactly as IEEE prescribes (and exactly what torch produces); the
//    checked tensor path below refuses such inputs before this matters.
//  - NaNs map to the sign-preserved canonical quiet NaN 0x7E00. Torch's own
//    NaN payload handling varies by code path (its scalar converter
//    canonicalizes, its vectorized F16C path keeps the top payload bits), so
//    the golden test compares NaN entries by "is NaN", not by payload; this
//    implementation picks the canonical form deterministically.
//
// Pure integer arithmetic: no dependence on the host FP environment, safe in
// constant expressions, deterministic across compilers.
constexpr std::uint16_t Fp32BitsToFp16BitsRne(std::uint32_t fp32_bits) noexcept {
  const auto sign = static_cast<std::uint16_t>((fp32_bits >> 16) & 0x8000u);
  const std::uint32_t abs = fp32_bits & 0x7FFFFFFFu;

  if (abs > 0x7F800000u) { // NaN (any payload) -> canonical quiet NaN
    return static_cast<std::uint16_t>(sign | 0x7E00u);
  }
  if (abs >= 0x47800000u) { // |x| >= 65536: infinity, or finite -> rounds to infinity
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }

  const std::uint32_t exp32 = abs >> 23; // biased binary32 exponent, now 0..142

  if (exp32 >= 113) {
    // Normal FP16 result (binary32 exponent 113 <=> 2^-14, the FP16 minimum
    // normal). Rebias 127 -> 15 by subtracting 112 in the exponent field,
    // then round the low 13 mantissa bits away with add-half-then-truncate;
    // the tie-breaking `lsb` term implements round-to-nearest-EVEN. A
    // mantissa carry propagates into the exponent field (IEEE-correct), and
    // a carry past exponent 30 yields exactly 0x7C00 = infinity, which is
    // the correct RNE overflow for values in [65520, 65536).
    const std::uint32_t rebased = abs - (112u << 23);
    const std::uint32_t lsb = (rebased >> 13) & 1u;
    const std::uint32_t rounded = rebased + 0x0FFFu + lsb;
    return static_cast<std::uint16_t>(sign | (rounded >> 13));
  }

  if (exp32 >= 102) {
    // Subnormal FP16 result: |x| in [2^-25, 2^-14). The result mantissa is
    // RNE(|x| * 2^24), an integer in [0, 1024]; 1024 (= 0x400) is the carry
    // into the minimum normal 2^-14 and is already its correct encoding.
    // Inputs here are binary32-normal (exp32 >= 102 > 0), so the implicit
    // leading bit is materialized before shifting.
    const std::uint32_t sig = 0x00800000u | (abs & 0x007FFFFFu); // 1.m as a 24-bit integer
    const std::uint32_t shift = 126u - exp32;                    // 14..24
    const std::uint32_t q = sig >> shift;
    const std::uint32_t rem = sig & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    const bool round_up = rem > halfway || (rem == halfway && (q & 1u) != 0u);
    return static_cast<std::uint16_t>(sign | (q + (round_up ? 1u : 0u)));
  }

  // |x| < 2^-25 (including binary32 subnormals and zero): rounds to +-0.
  // Exactly 2^-25 is handled above (tie between 0 and the minimum subnormal
  // 2^-24; RNE picks the even mantissa, 0).
  return sign;
}

// BF16 bits -> FP16 bits, unchecked. Composition of the two exact steps
// above; finite overflow yields +-infinity (the checked path throws instead).
constexpr std::uint16_t Bf16BitsToFp16BitsRne(std::uint16_t bf16_bits) noexcept {
  return Fp32BitsToFp16BitsRne(Bf16BitsToFp32Bits(bf16_bits));
}

// True iff the BF16 value violates the loader's overflow policy:
// |x| > 65504, i.e. finite magnitudes above the FP16 maximum, and +-infinity
// (|inf| > 65504 is true under IEEE comparison). NaN is excluded: the
// predicate is a magnitude test and NaN compares unordered.
//
// On the BF16 grid the smallest magnitude above 65504 is 65536 (bits 0x4780;
// the next value down, 0x477F, is 65280 and converts exactly), so the failing
// band is abs bits in [0x4780, 0x7F80] where 0x7F80 is infinity.
constexpr bool Bf16OverflowsFp16(std::uint16_t bf16_bits) noexcept {
  const std::uint16_t abs = bf16_bits & 0x7FFFu;
  return abs >= 0x4780u && abs <= 0x7F80u;
}

// FP16 bit-pattern classifiers (used by tests and error reporting).
constexpr bool IsNanFp16Bits(std::uint16_t fp16_bits) noexcept {
  return (fp16_bits & 0x7FFFu) > 0x7C00u;
}
constexpr bool IsInfFp16Bits(std::uint16_t fp16_bits) noexcept {
  return (fp16_bits & 0x7FFFu) == 0x7C00u;
}

// ------------------------------------------------------------ checked convert

// Thrown by the checked conversion when a weight element exceeds the FP16
// range (docs/DESIGN.md section 4: fail hard, never clamp).
class Fp16OverflowError : public std::runtime_error {
 public:
  Fp16OverflowError(std::string_view tensor_name, std::size_t element_index,
                    std::uint16_t bf16_bits);

  const std::string& tensor_name() const noexcept { return tensor_name_; }
  std::size_t element_index() const noexcept { return element_index_; }
  std::uint16_t bf16_bits() const noexcept { return bf16_bits_; }

 private:
  std::string tensor_name_;
  std::size_t element_index_;
  std::uint16_t bf16_bits_;
};

// Convert `count` BF16 elements at `src` (little-endian byte stream, as
// mmapped from a .safetensors data section; no alignment requirement) into
// `dst` FP16 bit patterns. Throws Fp16OverflowError on the FIRST element with
// |x| > 65504, reporting `tensor_name` and `index_base + i` - callers
// streaming a tensor in chunks pass the chunk's element offset as
// `index_base` so reported indices are tensor-absolute.
void ConvertBf16ToFp16(const std::byte* src, std::size_t count, std::uint16_t* dst,
                       std::string_view tensor_name, std::size_t index_base = 0);

// ----------------------------------------------------------- staging + upload

// Page-locked host buffer (cudaHostAlloc / cudaFreeHost). Pinned memory keeps
// every H2D chunk a true async DMA; a pageable source would silently degrade
// cudaMemcpyAsync to a staged synchronous copy. Requires a CUDA runtime;
// throws std::runtime_error on allocation failure.
class PinnedStagingBuffer {
 public:
  explicit PinnedStagingBuffer(std::size_t bytes);
  ~PinnedStagingBuffer();
  PinnedStagingBuffer(const PinnedStagingBuffer&) = delete;
  PinnedStagingBuffer& operator=(const PinnedStagingBuffer&) = delete;

  std::byte* data() const noexcept { return data_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::byte* data_ = nullptr;
  std::size_t bytes_ = 0;
};

// Streams host tensors to device memory through one bounded, reusable pinned
// staging buffer: convert (BF16, checked) or copy (FP16 passthrough) a chunk
// into a staging slot, then cudaMemcpyAsync it to the destination on the
// CALLER's stream. The buffer is split into two slots so the CPU converts
// chunk k+1 while the DMA engine drains chunk k; a cudaEvent per slot guards
// reuse (host-side wait before overwriting a slot with a copy in flight).
//
// Contract for the weight loader (consumed by src/loader/weights.hpp):
//   - All device work goes onto the stream passed to Upload(); the uploader
//     creates no stream of its own and never synchronizes the whole stream,
//     only its two slot events.
//   - Upload() returns once every chunk of the tensor has been ISSUED; the
//     last chunks may still be in flight. Staging memory outlives them (the
//     destructor waits on both slot events), but callers must sync the
//     stream before reading the destination or freeing it.
//   - Total staging is bounded by `staging_bytes` regardless of tensor size
//     (the largest checkpoint tensor, embed [151936, 1536], streams through
//     in staging_bytes/2-sized chunks).
class StagedUploader {
 public:
  static constexpr std::size_t kDefaultStagingBytes = std::size_t{32} << 20; // 2 x 16 MiB slots
  static constexpr std::size_t kMinStagingBytes = std::size_t{4} << 10;

  // Allocates the pinned buffer and the two slot events. `staging_bytes`
  // must be >= kMinStagingBytes; each slot holds staging_bytes/2 bytes.
  explicit StagedUploader(std::size_t staging_bytes = kDefaultStagingBytes);
  ~StagedUploader();
  StagedUploader(const StagedUploader&) = delete;
  StagedUploader& operator=(const StagedUploader&) = delete;

  // Convert-and-copy `count` elements of `src_dtype` (Dtype::kBF16, checked
  // conversion; or Dtype::kF16, passthrough) from host `src` to the device
  // pointer `dst_device` (room for `count` FP16 elements) on `stream`.
  // Any other dtype throws std::runtime_error naming the tensor.
  // Fp16OverflowError propagates from the BF16 path with element indices
  // offset by `index_base` (callers uploading a tensor region pass its
  // element offset). On error, previously issued chunks may still be in
  // flight; the uploader remains valid and its destructor waits for them.
  void Upload(void* dst_device, const std::byte* src, std::size_t count, Dtype src_dtype,
              std::string_view tensor_name, cudaStream_t stream, std::size_t index_base = 0);

  // Host-blocks until every chunk issued so far has been consumed by the
  // copy engine (slot events reached). NOT a stream sync: it guarantees the
  // staging buffer is reusable, which also means all issued copies have
  // completed since each event is recorded directly after its copy.
  void Synchronize();

  // Elements per staged chunk (slot capacity).
  std::size_t chunk_elements() const noexcept { return slot_bytes_ / sizeof(std::uint16_t); }

 private:
  void WaitSlot(int slot);

  PinnedStagingBuffer staging_; // declared first: outlives the event waits in ~StagedUploader
  std::size_t slot_bytes_ = 0;
  cudaEvent_t slot_done_[2] = {nullptr, nullptr};
  bool slot_pending_[2] = {false, false};
  int next_slot_ = 0;
};

} // namespace redline
