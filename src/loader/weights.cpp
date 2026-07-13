#include "loader/weights.hpp"

#include <cassert>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/types.hpp"
#include "loader/convert.hpp"
#include "loader/safetensors.hpp"

namespace redline {

namespace {

namespace wl = weight_layout;

std::string ShapeToString(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

// One geometry cross-check: the fused layout below is defined for
// Qwen2.5-1.5B-Instruct exactly (docs/DESIGN.md section 4 table). Startup
// config validation already hard-errors on any checkpoint drift, so a
// failure here means a ModelConfig that bypassed validation.
void CheckGeometry(std::int64_t actual, std::int64_t expected, const char* what) {
  if (actual != expected) {
    throw WeightsError("weights: " + std::string(what) + " = " + std::to_string(actual) +
                       " does not match the docs/DESIGN.md section 4 fusion constant " +
                       std::to_string(expected) +
                       "; the fused device layout is defined for Qwen2.5-1.5B-Instruct only");
  }
}

void RequireDesignGeometry(const ModelConfig& config) {
  CheckGeometry(config.hidden_size, wl::kHiddenSize, "hidden_size");
  CheckGeometry(config.num_layers, wl::kNumLayers, "num_hidden_layers");
  CheckGeometry(config.vocab_size, wl::kVocabSize, "vocab_size");
  CheckGeometry(config.intermediate_size, wl::kIntermediateSize, "intermediate_size");
  CheckGeometry(std::int64_t{config.num_q_heads} * config.head_dim, wl::kQRows,
                "num_attention_heads * head_dim (fused Q rows)");
  CheckGeometry(std::int64_t{config.num_kv_heads} * config.head_dim, wl::kKvRows,
                "num_key_value_heads * head_dim (fused K/V rows)");
  if (!config.qkv_bias) {
    throw WeightsError("weights: qkv_bias = false, but the fused layout carries b_qkv [2048] "
                       "(Qwen2 hardcodes Q/K/V projection bias -- docs/MODEL_SPEC.md section 2)");
  }
}

std::string LayerTensorName(std::int32_t layer, const char* suffix) {
  return "model.layers." + std::to_string(layer) + suffix;
}

// Presence + exact shape + supported dtype for one mapped tensor, before any
// device work (hard error naming the tensor). Byte-count
// consistency (shape x dtype size == data span) is already enforced by the
// safetensors parser at open time.
void ValidateEntry(const SafetensorsModel& model, const WeightMapEntry& entry) {
  if (!model.Contains(entry.hf_name)) {
    throw WeightsError("weights: required tensor '" + entry.hf_name +
                       "' is missing from the checkpoint -- wrong or corrupt model directory? "
                       "(expected shape " +
                       ShapeToString(entry.expected_shape) + ")");
  }
  const TensorView& view = model.Tensor(entry.hf_name);
  if (view.shape != entry.expected_shape) {
    throw WeightsError("weights: tensor '" + entry.hf_name + "' has shape " +
                       ShapeToString(view.shape) + ", expected " +
                       ShapeToString(entry.expected_shape) + " (docs/MODEL_SPEC.md section 8)");
  }
  if (view.dtype != Dtype::kBF16 && view.dtype != Dtype::kF16) {
    throw WeightsError("weights: tensor '" + entry.hf_name +
                       "' has an unsupported dtype; expected BF16 (this checkpoint) or F16 "
                       "(docs/MODEL_SPEC.md section 8)");
  }
}

void CudaCheck(cudaError_t err, const std::string& what) {
  if (err != cudaSuccess) {
    throw WeightsError("weights: " + what + " failed: " + cudaGetErrorName(err) + ": " +
                       cudaGetErrorString(err));
  }
}

} // namespace

std::int64_t DestRowWidth(DeviceTensorKind kind, const ModelConfig& config) {
  switch (kind) {
  case DeviceTensorKind::kEmbed:
  case DeviceTensorKind::kLmHead:
  case DeviceTensorKind::kWqkv:
  case DeviceTensorKind::kWo:
  case DeviceTensorKind::kWgateup:
    return config.hidden_size; // [out_features, hidden] row-major
  case DeviceTensorKind::kWdown:
    return config.intermediate_size; // [hidden, intermediate] row-major
  case DeviceTensorKind::kBqkv:
  case DeviceTensorKind::kInputNorm:
  case DeviceTensorKind::kPostAttnNorm:
  case DeviceTensorKind::kFinalNorm:
    return 1; // 1-D: dest_row_begin is directly an element offset
  }
  assert(false && "unhandled DeviceTensorKind");
  return 1;
}

std::vector<WeightMapEntry> BuildWeightMap(const ModelConfig& config, bool has_lm_head) {
  RequireDesignGeometry(config);

  const std::int64_t hidden = config.hidden_size;
  const std::int64_t inter = config.intermediate_size;
  const std::int64_t vocab = config.vocab_size;
  const std::int64_t q_rows = std::int64_t{config.num_q_heads} * config.head_dim;   // 1536
  const std::int64_t kv_rows = std::int64_t{config.num_kv_heads} * config.head_dim; // 256

  std::vector<WeightMapEntry> map;
  map.reserve(static_cast<std::size_t>(2 + 12 * config.num_layers + (has_lm_head ? 1 : 0)));

  map.push_back({"model.embed_tokens.weight", DeviceTensorKind::kEmbed, -1, 0, {vocab, hidden}});

  for (std::int32_t l = 0; l < config.num_layers; ++l) {
    map.push_back({LayerTensorName(l, ".input_layernorm.weight"),
                   DeviceTensorKind::kInputNorm,
                   l,
                   0,
                   {hidden}});
    // QKV fusion (docs/DESIGN.md section 4): q rows 0:1536, k rows 1536:1792,
    // v rows 1792:2048; bias elements at the same offsets so the fused GEMM's
    // BIAS epilogue vector lines up with the fused output-feature axis.
    map.push_back({LayerTensorName(l, ".self_attn.q_proj.weight"),
                   DeviceTensorKind::kWqkv,
                   l,
                   wl::kQRowBegin,
                   {q_rows, hidden}});
    map.push_back({LayerTensorName(l, ".self_attn.q_proj.bias"),
                   DeviceTensorKind::kBqkv,
                   l,
                   wl::kQRowBegin,
                   {q_rows}});
    map.push_back({LayerTensorName(l, ".self_attn.k_proj.weight"),
                   DeviceTensorKind::kWqkv,
                   l,
                   wl::kKRowBegin,
                   {kv_rows, hidden}});
    map.push_back({LayerTensorName(l, ".self_attn.k_proj.bias"),
                   DeviceTensorKind::kBqkv,
                   l,
                   wl::kKRowBegin,
                   {kv_rows}});
    map.push_back({LayerTensorName(l, ".self_attn.v_proj.weight"),
                   DeviceTensorKind::kWqkv,
                   l,
                   wl::kVRowBegin,
                   {kv_rows, hidden}});
    map.push_back({LayerTensorName(l, ".self_attn.v_proj.bias"),
                   DeviceTensorKind::kBqkv,
                   l,
                   wl::kVRowBegin,
                   {kv_rows}});
    map.push_back({LayerTensorName(l, ".self_attn.o_proj.weight"),
                   DeviceTensorKind::kWo,
                   l,
                   0,
                   {hidden, hidden}});
    map.push_back({LayerTensorName(l, ".post_attention_layernorm.weight"),
                   DeviceTensorKind::kPostAttnNorm,
                   l,
                   0,
                   {hidden}});
    // gate+up fusion: gate rows 0:8960, up rows 8960:17920.
    map.push_back({LayerTensorName(l, ".mlp.gate_proj.weight"),
                   DeviceTensorKind::kWgateup,
                   l,
                   wl::kGateRowBegin,
                   {inter, hidden}});
    map.push_back({LayerTensorName(l, ".mlp.up_proj.weight"),
                   DeviceTensorKind::kWgateup,
                   l,
                   wl::kUpRowBegin,
                   {inter, hidden}});
    map.push_back({LayerTensorName(l, ".mlp.down_proj.weight"),
                   DeviceTensorKind::kWdown,
                   l,
                   0,
                   {hidden, inter}});
  }

  map.push_back({"model.norm.weight", DeviceTensorKind::kFinalNorm, -1, 0, {hidden}});

  if (has_lm_head) {
    // docs/DESIGN.md section 4: alias embed UNLESS lm_head.weight exists in
    // the file -- file presence wins. Absent for this checkpoint (F2).
    map.push_back({"lm_head.weight", DeviceTensorKind::kLmHead, -1, 0, {vocab, hidden}});
  }
  return map;
}

std::int64_t WeightsLayout::OffsetFor(DeviceTensorKind kind, std::int32_t layer) const {
  switch (kind) {
  case DeviceTensorKind::kEmbed:
    return embed;
  case DeviceTensorKind::kLmHead:
    return lm_head;
  case DeviceTensorKind::kFinalNorm:
    return final_norm;
  default:
    break;
  }
  assert(layer >= 0 && static_cast<std::size_t>(layer) < layers.size());
  const LayerOffsets& at = layers[static_cast<std::size_t>(layer)];
  switch (kind) {
  case DeviceTensorKind::kWqkv:
    return at.w_qkv;
  case DeviceTensorKind::kBqkv:
    return at.b_qkv;
  case DeviceTensorKind::kWo:
    return at.w_o;
  case DeviceTensorKind::kWgateup:
    return at.w_gateup;
  case DeviceTensorKind::kWdown:
    return at.w_down;
  case DeviceTensorKind::kInputNorm:
    return at.input_norm;
  case DeviceTensorKind::kPostAttnNorm:
    return at.post_attn_norm;
  default:
    assert(false && "unhandled DeviceTensorKind");
    return 0;
  }
}

WeightsLayout ComputeWeightsLayout(const ModelConfig& config, bool has_lm_head) {
  RequireDesignGeometry(config);

  const std::int64_t hidden = config.hidden_size;
  const std::int64_t inter = config.intermediate_size;
  const std::int64_t vocab = config.vocab_size;
  constexpr std::int64_t kElem = 2; // FP16
  // Every region start stays >= 256-B aligned on top of cudaMalloc's base
  // alignment (vectorized kernel loads + cuBLASLt alignment hints). All
  // region sizes of this model are multiples of 256 B, so the alignment
  // padding is zero and total_bytes is the exact sum.
  constexpr std::int64_t kAlign = 256;

  WeightsLayout layout;
  std::int64_t offset = 0;
  const auto place = [&offset](std::int64_t bytes) {
    const std::int64_t at = offset;
    offset = (at + bytes + kAlign - 1) / kAlign * kAlign;
    return at;
  };

  layout.embed = place(vocab * hidden * kElem);
  layout.layers.resize(static_cast<std::size_t>(config.num_layers));
  for (WeightsLayout::LayerOffsets& l : layout.layers) {
    l.w_qkv = place(wl::kQkvRows * hidden * kElem);
    l.b_qkv = place(wl::kQkvRows * kElem);
    l.w_o = place(hidden * hidden * kElem);
    l.w_gateup = place(wl::kGateUpRows * hidden * kElem);
    l.w_down = place(hidden * inter * kElem);
    l.input_norm = place(hidden * kElem);
    l.post_attn_norm = place(hidden * kElem);
  }
  layout.final_norm = place(hidden * kElem);
  if (has_lm_head) {
    layout.lm_head = place(vocab * hidden * kElem);
    layout.lm_head_aliases_embed = false;
  } else {
    layout.lm_head = layout.embed; // tied embeddings (docs/MODEL_SPEC.md F2)
    layout.lm_head_aliases_embed = true;
  }
  layout.total_bytes = offset;
  return layout;
}

std::int64_t EntryByteOffset(const WeightsLayout& layout, const WeightMapEntry& entry,
                             const ModelConfig& config) {
  return layout.OffsetFor(entry.dest, entry.layer) +
         entry.dest_row_begin * DestRowWidth(entry.dest, config) * std::int64_t{2};
}

// ---------------------------------------------------------------- DeviceWeights

std::int64_t DeviceWeights::TotalDeviceBytes(const ModelConfig& config, bool has_separate_lm_head) {
  return ComputeWeightsLayout(config, has_separate_lm_head).total_bytes;
}

DeviceWeights::DeviceWeights(const SafetensorsModel& model, const ModelConfig& config,
                             StagedUploader& uploader, cudaStream_t stream)
    : num_layers_(config.num_layers) {
  const bool has_lm_head = model.Contains("lm_head.weight");
  if (!has_lm_head && !config.tie_word_embeddings) {
    // Unreachable for a validated config (config validation pins tie_word_embeddings ==
    // true), kept so an unvalidated config cannot alias silently.
    throw WeightsError("weights: tie_word_embeddings is false but the checkpoint has no "
                       "'lm_head.weight' tensor to load");
  }
  layout_ = ComputeWeightsLayout(config, has_lm_head);
  const std::vector<WeightMapEntry> map = BuildWeightMap(config, has_lm_head);

  // Validate the whole inventory before the first device byte moves: a wrong
  // checkpoint fails in milliseconds, not after a partial 3 GB upload.
  for (const WeightMapEntry& entry : map) {
    ValidateEntry(model, entry);
  }

  CudaCheck(cudaMalloc(&slab_, static_cast<std::size_t>(layout_.total_bytes)),
            "cudaMalloc(" + std::to_string(layout_.total_bytes) + " B weight slab)");

  try {
    for (const WeightMapEntry& entry : map) {
      const TensorView& view = model.Tensor(entry.hf_name);
      std::byte* dst = static_cast<std::byte*>(slab_) + EntryByteOffset(layout_, entry, config);
      // Each fusion is a contiguous row concatenation, so every checkpoint
      // tensor maps to ONE contiguous destination range: a single streamed
      // upload per tensor, element indices tensor-absolute (index_base 0).
      uploader.Upload(dst, view.data, static_cast<std::size_t>(entry.num_elements()), view.dtype,
                      entry.hf_name, stream, /*index_base=*/0);
    }
    // Slot events recorded after each chunk copy have all been reached once
    // this returns (loader/convert.hpp), so the weights are resident and the
    // caller may free the uploader immediately.
    uploader.Synchronize();
  } catch (...) {
    // A failed upload (e.g. Fp16OverflowError) can leave chunks in flight
    // that target the slab; drain them before freeing it.
    try {
      uploader.Synchronize();
    } catch (...) {
      // Draining is best-effort on the error path; the original error wins.
    }
    (void)cudaFree(slab_);
    slab_ = nullptr;
    throw;
  }
}

DeviceWeights::~DeviceWeights() {
  if (slab_ != nullptr) {
    // Destructor: failure is not actionable; cudaFree only fails on invalid
    // arguments or a torn-down context.
    (void)cudaFree(slab_);
  }
}

const half* DeviceWeights::At(std::int64_t byte_offset) const {
  assert(slab_ != nullptr);
  return reinterpret_cast<const half*>(static_cast<const std::byte*>(slab_) + byte_offset);
}

const WeightsLayout::LayerOffsets& DeviceWeights::LayerAt(std::int32_t layer) const {
  assert(layer >= 0 && layer < num_layers_);
  return layout_.layers[static_cast<std::size_t>(layer)];
}

} // namespace redline
