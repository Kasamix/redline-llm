#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "core/config.hpp"

// Load-time weight preparation (fusions) + device weight store
// (docs/DESIGN.md sections 4-5).
//
// The checkpoint stores one tensor per HF module; the engine wants one fused
// FP16 device tensor per GEMM so the decode path runs a single QKV matmul and
// a single gate+up matmul per layer (docs/DESIGN.md section 6.1). Fusions are
// layout-only row concatenations, done once at load while streaming through
// the staged uploader (src/loader/convert.hpp); nothing is transposed and every fused tensor
// keeps the HF nn.Linear convention [out_features, in_features], row-major.
//
// Device layout (one tensor per row; FP16 = 2 bytes per element):
//
//   Device tensor    Shape [rows, cols]  Bytes        Built from HF tensors
//   ---------------  ------------------  -----------  --------------------------------------
//   embed            [151936, 1536]      466,747,392  model.embed_tokens.weight
//   per layer l = 0..27:
//     w_qkv          [2048, 1536]          6,291,456  q_proj.weight rows 0:1536,
//                                                     k_proj.weight rows 1536:1792,
//                                                     v_proj.weight rows 1792:2048
//     b_qkv          [2048]                    4,096  q/k/v_proj.bias at the same offsets
//     w_o            [1536, 1536]          4,718,592  o_proj.weight
//     w_gateup       [17920, 1536]        55,050,240  gate_proj.weight rows 0:8960,
//                                                     up_proj.weight rows 8960:17920
//     w_down         [1536, 8960]         27,525,120  down_proj.weight
//     input_norm     [1536]                    3,072  input_layernorm.weight
//     post_attn_norm [1536]                    3,072  post_attention_layernorm.weight
//     (per-layer subtotal)                93,595,648
//   final_norm       [1536]                    3,072  model.norm.weight
//   lm_head          alias of embed                0  lm_head.weight IF present in the file
//                                                     (absent for this checkpoint --
//                                                     tie_word_embeddings, MODEL_SPEC F2)
//
//   Total, tied lm_head: 466,747,392 + 28 * 93,595,648 + 3,072
//                      = 3,087,428,608 B (= 1,543,714,304 params * 2 B) ~= 2944 MiB.
//
// The store is ONE cudaMalloc slab carved at the byte offsets computed by
// ComputeWeightsLayout: every region size above is a multiple of 256 B, so
// regions pack back to back with zero padding, each region inherits
// cudaMalloc's >= 256 B base alignment, and the slab size equals the total
// exactly. The QKV row order Q(12x128) | K(2x128) | V(2x128) is what makes
// the executor's strided views into qkv_out (+0 / +1536 / +1792, row stride
// 2048) line up with the GEMM output (docs/DESIGN.md section 5); b_qkv uses
// the same offsets so the CUBLASLT_EPILOGUE_BIAS vector (length m = 2048 in
// the column-major dual) is per-output-feature on the fused axis.
//
// Contracts:
//   - Every device/host allocation this component ever makes happens inside
//     the DeviceWeights constructor (init time); none later. The pinned
//     staging memory belongs to the caller's StagedUploader.
//   - All uploads are issued on the CALLER's stream; the constructor drains
//     the uploader before returning, so on return the weights are fully
//     resident and no work from this loader is pending anywhere.
//   - A missing or mis-shaped checkpoint tensor is a hard WeightsError naming
//     the tensor (plus expected/actual shape), thrown BEFORE any device
//     allocation; a BF16 element outside FP16 range hard-fails upload with
//     Fp16OverflowError (loader/convert.hpp; never clamped).
//   - The name -> (destination, row range) mapping is exposed CPU-only
//     (BuildWeightMap / ComputeWeightsLayout / EntryByteOffset) and is the
//     single source of truth the upload loop executes, so the WeightsMapTest
//     unit suite covers the real load-path address math without a GPU.
//     Real-checkpoint load verification is covered by the e2e suites (tests/e2e/).

namespace redline {

class SafetensorsModel; // loader/safetensors.hpp
class StagedUploader;   // loader/convert.hpp

// Typed error for weight-preparation failures: a checkpoint tensor missing or
// mis-shaped, an unsupported dtype, a config whose geometry disagrees with
// the section 4 fusion constants, or a CUDA failure while building the store.
// Messages always name the offending tensor or field. Derives from
// std::runtime_error so generic catch sites keep working.
class WeightsError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Fusion geometry of the docs/DESIGN.md section 4 table, verbatim.
// BuildWeightMap/ComputeWeightsLayout re-derive every one
// of these from the validated ModelConfig and throw WeightsError on any
// disagreement -- the fused layout is defined for Qwen2.5-1.5B-Instruct
// exactly, and startup config validation already pins those values upstream.
namespace weight_layout {

inline constexpr std::int64_t kHiddenSize = 1536; // cols of every hidden-side GEMM weight
inline constexpr std::int64_t kNumLayers = 28;
inline constexpr std::int64_t kVocabSize = 151936;      // embed / lm_head rows
inline constexpr std::int64_t kIntermediateSize = 8960; // w_down cols

// w_qkv [2048, 1536] + b_qkv [2048]: q rows 0:1536, k rows 1536:1792,
// v rows 1792:2048 (bias elements at the same offsets).
inline constexpr std::int64_t kQRows = 1536; // num_q_heads (12) * head_dim (128)
inline constexpr std::int64_t kKvRows = 256; // num_kv_heads (2) * head_dim (128), K and V each
inline constexpr std::int64_t kQRowBegin = 0;
inline constexpr std::int64_t kKRowBegin = 1536;
inline constexpr std::int64_t kVRowBegin = 1792;
inline constexpr std::int64_t kQkvRows = 2048;

// w_gateup [17920, 1536]: gate rows 0:8960, up rows 8960:17920.
inline constexpr std::int64_t kGateRows = 8960; // == kIntermediateSize
inline constexpr std::int64_t kGateRowBegin = 0;
inline constexpr std::int64_t kUpRowBegin = 8960;
inline constexpr std::int64_t kGateUpRows = 17920;

// input_layernorm + post_attention_layernorm per layer, plus model.norm.
inline constexpr std::int64_t kNumNormVectors = 2 * kNumLayers + 1;

static_assert(kKRowBegin == kQRowBegin + kQRows, "K rows start where Q rows end");
static_assert(kVRowBegin == kKRowBegin + kKvRows, "V rows start where K rows end");
static_assert(kQkvRows == kVRowBegin + kKvRows, "QKV fusion is exactly tiled");
static_assert(kUpRowBegin == kGateRowBegin + kGateRows, "up rows start where gate rows end");
static_assert(kGateUpRows == kUpRowBegin + kGateRows, "gate+up fusion is exactly tiled");
static_assert(kNumNormVectors == 57, "docs/DESIGN.md section 4: 57 norm vectors");

// Total device bytes with lm_head aliased to embed (this checkpoint), pinned
// at compile time to the docs/DESIGN.md section 1 figure.
inline constexpr std::int64_t kTotalWeightBytesTied =
    2 * (kVocabSize * kHiddenSize                          // embed
         + kNumLayers * (kQkvRows * kHiddenSize + kQkvRows // w_qkv + b_qkv
                         + kHiddenSize * kHiddenSize       // w_o
                         + kGateUpRows * kHiddenSize       // w_gateup
                         + kHiddenSize * kIntermediateSize // w_down
                         + 2 * kHiddenSize)                // two layer norms
         + kHiddenSize);                                   // final norm
static_assert(kTotalWeightBytesTied == 3'087'428'608,
              "FP16 weight bytes must match docs/DESIGN.md section 1 (2944 MiB)");

} // namespace weight_layout

// The fused device tensor a checkpoint tensor lands in.
enum class DeviceTensorKind : std::uint8_t {
  kEmbed,        // [151936, 1536]
  kWqkv,         // [2048, 1536] per layer
  kBqkv,         // [2048] per layer (1-D)
  kWo,           // [1536, 1536] per layer
  kWgateup,      // [17920, 1536] per layer
  kWdown,        // [1536, 8960] per layer
  kInputNorm,    // [1536] per layer (1-D)
  kPostAttnNorm, // [1536] per layer (1-D)
  kFinalNorm,    // [1536] (1-D)
  kLmHead,       // [151936, 1536]; only when lm_head.weight exists in the file
};

// One checkpoint tensor -> fused-destination mapping row. The full table
// (BuildWeightMap) drives both the load loop and the CPU-only unit checks.
struct WeightMapEntry {
  std::string hf_name; // exact checkpoint tensor name
  DeviceTensorKind dest = DeviceTensorKind::kEmbed;
  std::int32_t layer = -1;                  // 0..num_layers-1; -1 for non-layer tensors
  std::int64_t dest_row_begin = 0;          // first destination row (2-D dests); for 1-D
                                            // dests (b_qkv, norms) directly an element offset
  std::vector<std::int64_t> expected_shape; // exact HF shape; mismatch is a hard error

  std::int64_t num_elements() const {
    std::int64_t n = 1;
    for (const std::int64_t dim : expected_shape) {
      n *= dim;
    }
    return n;
  }
  // Destination rows this entry occupies ([dest_row_begin, dest_row_begin + dest_rows())).
  std::int64_t dest_rows() const { return expected_shape.empty() ? 0 : expected_shape[0]; }
};

// Elements per destination row: 2-D fused tensors keep the HF row-major
// [out_features, in_features] layout so their row width is in_features; 1-D
// destinations use width 1 so dest_row_begin doubles as an element offset.
std::int64_t DestRowWidth(DeviceTensorKind kind, const ModelConfig& config);

// Builds the complete name -> (destination, row range) table for the fusion
// layout documented above: 338 entries for the tied checkpoint (1 embed +
// 28 layers x 12 tensors + model.norm -- exactly the MODEL_SPEC section 8
// inventory), plus one lm_head entry when `has_lm_head`. Pure CPU, no CUDA.
// Throws WeightsError if the config geometry disagrees with the
// weight_layout constants.
std::vector<WeightMapEntry> BuildWeightMap(const ModelConfig& config, bool has_lm_head);

// Byte offsets of every fused tensor inside the single device slab. Regions
// are placed in map order (embed, per-layer groups, final norm, then a
// separate lm_head only when present), each aligned to 256 B; every region's
// byte size is already a multiple of 256 for this model, so the packing has
// zero padding and total_bytes is exact.
struct WeightsLayout {
  struct LayerOffsets {
    std::int64_t w_qkv = 0;
    std::int64_t b_qkv = 0;
    std::int64_t w_o = 0;
    std::int64_t w_gateup = 0;
    std::int64_t w_down = 0;
    std::int64_t input_norm = 0;
    std::int64_t post_attn_norm = 0;
  };

  std::int64_t embed = 0;
  std::vector<LayerOffsets> layers; // num_layers entries
  std::int64_t final_norm = 0;
  std::int64_t lm_head = 0; // == embed when lm_head_aliases_embed
  bool lm_head_aliases_embed = true;
  std::int64_t total_bytes = 0;

  // Base byte offset of one fused tensor. `layer` is read only for the
  // per-layer kinds and must then be in [0, layers.size()).
  std::int64_t OffsetFor(DeviceTensorKind kind, std::int32_t layer) const;
};

// Pure CPU slab plan (no CUDA). Throws WeightsError on geometry mismatch.
WeightsLayout ComputeWeightsLayout(const ModelConfig& config, bool has_lm_head);

// Byte offset inside the slab where `entry`'s payload starts -- the exact
// address computation the upload loop uses (base offset of the destination
// plus dest_row_begin rows of DestRowWidth elements). Public so WeightsMapTest
// can verify, without a GPU, that all entries tile the slab exactly.
std::int64_t EntryByteOffset(const WeightsLayout& layout, const WeightMapEntry& entry,
                             const ModelConfig& config);

// The device weight store. Construction performs the entire load:
//
//   1. Detect lm_head.weight in the checkpoint (absent for this model);
//      compute the slab layout.
//   2. Validate EVERY mapped tensor against the checkpoint -- presence, exact
//      shape, dtype in {BF16, F16} -- before touching the device, so a wrong
//      or corrupt model directory fails fast with the tensor named.
//   3. cudaMalloc the single slab (the only device allocation this class
//      ever performs).
//   4. Stream each tensor through `uploader` onto `stream` (BF16 -> FP16
//      checked conversion per loader/convert.hpp; F16 passthrough), one
//      contiguous destination range per checkpoint tensor.
//   5. Drain the uploader. On return the weights are resident and no copies
//      are in flight; on ANY failure the slab is released (after draining
//      in-flight chunks that target it) and the exception propagates.
//
// Accessors hand out device pointers into the slab; they remain valid for
// the lifetime of this object. Typical engine init call sequence
// (docs/DESIGN.md section 2 ordering):
//
//   SafetensorsModel model = SafetensorsModel::OpenDir(model_dir);
//   ModelConfig config = ModelConfig::FromModelDir(model_dir);
//   StagedUploader uploader;                       // pinned staging, freed after load
//   DeviceWeights weights(model, config, uploader, stream);
class DeviceWeights {
 public:
  // Device byte budget of the store, for the engine's cudaMemGetInfo
  // preflight -- callable before any allocation. `has_separate_lm_head` is
  // SafetensorsModel::Contains("lm_head.weight") (false for this checkpoint:
  // 3,087,428,608 B; a separate head would add 466,747,392 B).
  static std::int64_t TotalDeviceBytes(const ModelConfig& config, bool has_separate_lm_head);

  DeviceWeights(const SafetensorsModel& model, const ModelConfig& config, StagedUploader& uploader,
                cudaStream_t stream);
  ~DeviceWeights();
  DeviceWeights(const DeviceWeights&) = delete;
  DeviceWeights& operator=(const DeviceWeights&) = delete;

  // [vocab_size, hidden] embedding table (embed_gather + tied lm_head GEMM).
  const half* embed() const { return At(layout_.embed); }
  // lm_head GEMM weight: the embed alias unless the file shipped lm_head.weight.
  const half* lm_head() const { return At(layout_.lm_head); }
  bool lm_head_is_tied() const { return layout_.lm_head_aliases_embed; }

  // Per-layer fused tensors; `layer` in [0, num_layers).
  const half* w_qkv(std::int32_t layer) const { return At(LayerAt(layer).w_qkv); }
  const half* b_qkv(std::int32_t layer) const { return At(LayerAt(layer).b_qkv); }
  const half* w_o(std::int32_t layer) const { return At(LayerAt(layer).w_o); }
  const half* w_gateup(std::int32_t layer) const { return At(LayerAt(layer).w_gateup); }
  const half* w_down(std::int32_t layer) const { return At(LayerAt(layer).w_down); }
  const half* input_norm(std::int32_t layer) const { return At(LayerAt(layer).input_norm); }
  const half* post_attn_norm(std::int32_t layer) const { return At(LayerAt(layer).post_attn_norm); }
  // Final model.norm vector [hidden].
  const half* final_norm() const { return At(layout_.final_norm); }

  std::int32_t num_layers() const { return num_layers_; }
  std::int64_t total_bytes() const { return layout_.total_bytes; }
  const WeightsLayout& layout() const { return layout_; }

 private:
  const half* At(std::int64_t byte_offset) const;
  const WeightsLayout::LayerOffsets& LayerAt(std::int32_t layer) const;

  WeightsLayout layout_;
  std::int32_t num_layers_ = 0;
  void* slab_ = nullptr; // single cudaMalloc; released in the destructor
};

} // namespace redline
