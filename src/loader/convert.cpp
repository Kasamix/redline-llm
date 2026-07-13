#include "loader/convert.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>

namespace redline {

namespace {

// Throwing CUDA-call check. Loader-path only (init time): failures are
// unrecoverable configuration/driver problems, never steady-state events.
void CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("convert: ") + what + " failed: " + cudaGetErrorName(err) +
                             ": " + cudaGetErrorString(err));
  }
}

std::string FormatOverflowMessage(std::string_view tensor_name, std::size_t element_index,
                                  std::uint16_t bf16_bits) {
  // The offending value is exact: BF16 widens to FP32 losslessly.
  const float value = std::bit_cast<float>(Bf16BitsToFp32Bits(bf16_bits));
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "': element %zu is %g (bf16 bits 0x%04X), outside the FP16 finite range "
                "(|x| <= 65504). Refusing to clamp (docs/DESIGN.md section 4): this checkpoint "
                "is expected to contain no such values, so the file is corrupt or is not the "
                "expected model.",
                element_index, static_cast<double>(value), bf16_bits);
  return "bf16->fp16 overflow in tensor '" + std::string(tensor_name) + buf;
}

} // namespace

Fp16OverflowError::Fp16OverflowError(std::string_view tensor_name, std::size_t element_index,
                                     std::uint16_t bf16_bits)
    : std::runtime_error(FormatOverflowMessage(tensor_name, element_index, bf16_bits)),
      tensor_name_(tensor_name), element_index_(element_index), bf16_bits_(bf16_bits) {}

void ConvertBf16ToFp16(const std::byte* src, std::size_t count, std::uint16_t* dst,
                       std::string_view tensor_name, std::size_t index_base) {
  // Byte-wise loads: safetensors data offsets carry no alignment guarantee,
  // and the file (like the host) is little-endian. memcpy of two bytes
  // compiles to a plain 16-bit load.
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t bf16 = 0;
    std::memcpy(&bf16, src + i * sizeof(std::uint16_t), sizeof(std::uint16_t));
    if (Bf16OverflowsFp16(bf16)) {
      throw Fp16OverflowError(tensor_name, index_base + i, bf16);
    }
    dst[i] = Bf16BitsToFp16BitsRne(bf16);
  }
}

// ------------------------------------------------------- PinnedStagingBuffer

PinnedStagingBuffer::PinnedStagingBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes == 0) {
    throw std::runtime_error("convert: pinned staging buffer size must be non-zero");
  }
  void* ptr = nullptr;
  CudaCheck(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc(staging)");
  data_ = static_cast<std::byte*>(ptr);
}

PinnedStagingBuffer::~PinnedStagingBuffer() {
  if (data_ != nullptr) {
    // Destructor: failure is not actionable here, and cudaFreeHost only
    // fails on invalid arguments or a torn-down context.
    (void)cudaFreeHost(data_);
  }
}

// ------------------------------------------------------------ StagedUploader

StagedUploader::StagedUploader(std::size_t staging_bytes)
    : staging_(staging_bytes),
      // Two equal slots, each a whole number of FP16 elements.
      slot_bytes_((staging_bytes / 2) & ~std::size_t{1}) {
  if (staging_bytes < kMinStagingBytes) {
    throw std::runtime_error("convert: staging buffer must be at least " +
                             std::to_string(kMinStagingBytes) + " bytes, got " +
                             std::to_string(staging_bytes));
  }
  for (cudaEvent_t& event : slot_done_) {
    CudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
              "cudaEventCreateWithFlags(slot event)");
  }
}

StagedUploader::~StagedUploader() {
  // Pending copies read from staging_; wait for them before the
  // PinnedStagingBuffer member (destroyed after this body) frees it.
  for (int slot = 0; slot < 2; ++slot) {
    if (slot_pending_[slot]) {
      (void)cudaEventSynchronize(slot_done_[slot]);
    }
    if (slot_done_[slot] != nullptr) {
      (void)cudaEventDestroy(slot_done_[slot]);
    }
  }
}

void StagedUploader::WaitSlot(int slot) {
  if (slot_pending_[slot]) {
    CudaCheck(cudaEventSynchronize(slot_done_[slot]), "cudaEventSynchronize(slot event)");
    slot_pending_[slot] = false;
  }
}

void StagedUploader::Upload(void* dst_device, const std::byte* src, std::size_t count,
                            Dtype src_dtype, std::string_view tensor_name, cudaStream_t stream,
                            std::size_t index_base) {
  if (src_dtype != Dtype::kBF16 && src_dtype != Dtype::kF16) {
    throw std::runtime_error("convert: tensor '" + std::string(tensor_name) +
                             "' has unsupported source dtype (expected BF16 or F16); "
                             "the checkpoint is expected to be all-BF16 "
                             "(docs/MODEL_SPEC.md section 8)");
  }
  const std::size_t chunk = chunk_elements();
  for (std::size_t offset = 0; offset < count; offset += chunk) {
    const std::size_t n = std::min(chunk, count - offset);
    const int slot = next_slot_;
    WaitSlot(slot); // the previous copy out of this slot must have drained
    std::byte* stage = staging_.data() + static_cast<std::size_t>(slot) * slot_bytes_;
    const std::byte* chunk_src = src + offset * sizeof(std::uint16_t);
    if (src_dtype == Dtype::kBF16) {
      ConvertBf16ToFp16(chunk_src, n, reinterpret_cast<std::uint16_t*>(stage), tensor_name,
                        index_base + offset);
    } else {
      std::memcpy(stage, chunk_src, n * sizeof(std::uint16_t)); // already FP16: passthrough
    }
    CudaCheck(cudaMemcpyAsync(static_cast<std::byte*>(dst_device) + offset * sizeof(std::uint16_t),
                              stage, n * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(H2D weight chunk)");
    CudaCheck(cudaEventRecord(slot_done_[slot], stream), "cudaEventRecord(slot event)");
    slot_pending_[slot] = true;
    next_slot_ = slot ^ 1;
  }
}

void StagedUploader::Synchronize() {
  WaitSlot(0);
  WaitSlot(1);
}

} // namespace redline
