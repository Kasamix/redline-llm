// Weight-preparation mapping-table unit tests (docs/DESIGN.md section 4).
// Everything here is CPU-only: BuildWeightMap,
// ComputeWeightsLayout, DestRowWidth, EntryByteOffset, and TotalDeviceBytes
// perform no CUDA calls, so the fusion offsets and the load-path address math
// are verified without a GPU or a checkpoint. Real-checkpoint load
// verification happens in the e2e suites (tests/e2e/).

#include "loader/weights.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/config.hpp"

namespace redline {
namespace {

namespace wl = weight_layout;

// The validated ModelConfig for the pinned checkpoint (docs/MODEL_SPEC.md
// sections 1-2). Constructed directly: FromJsonText validation already pins
// every one of these values against the same spec.
ModelConfig SpecConfig() {
  ModelConfig config;
  config.hidden_size = 1536;
  config.num_layers = 28;
  config.num_q_heads = 12;
  config.num_kv_heads = 2;
  config.head_dim = 128;
  config.intermediate_size = 8960;
  config.vocab_size = 151936;
  config.rms_norm_eps = 1e-6f;
  config.rope_theta = 1e6f;
  config.max_position_embeddings = 32768;
  config.tie_word_embeddings = true;
  config.qkv_bias = true;
  config.eos_token_ids = {151645, 151643};
  return config;
}

const WeightMapEntry* FindEntry(const std::vector<WeightMapEntry>& map, const std::string& name) {
  for (const WeightMapEntry& entry : map) {
    if (entry.hf_name == name) {
      return &entry;
    }
  }
  return nullptr;
}

// Locates `name` in the map and checks every field of its mapping row.
void ExpectEntry(const std::vector<WeightMapEntry>& map, const std::string& name,
                 DeviceTensorKind dest, std::int32_t layer, std::int64_t dest_row_begin,
                 const std::vector<std::int64_t>& expected_shape) {
  const WeightMapEntry* entry = FindEntry(map, name);
  ASSERT_NE(entry, nullptr) << "map is missing tensor '" << name << "'";
  EXPECT_EQ(entry->dest, dest) << name;
  EXPECT_EQ(entry->layer, layer) << name;
  EXPECT_EQ(entry->dest_row_begin, dest_row_begin) << name;
  EXPECT_EQ(entry->expected_shape, expected_shape) << name;
}

// Runs `fn`, requires it to throw WeightsError, and returns the message.
template <typename Fn> std::string WeightsErrorMessage(Fn&& fn) {
  try {
    std::forward<Fn>(fn)();
  } catch (const WeightsError& error) {
    return error.what();
  }
  ADD_FAILURE() << "expected WeightsError";
  return {};
}

TEST(WeightsMapTest, InventoryMatchesCheckpointTensorList) {
  const ModelConfig config = SpecConfig();
  const auto map = BuildWeightMap(config, /*has_lm_head=*/false);

  // 1 embed + 28 layers x 12 tensors + model.norm = 338 -- exactly the
  // MODEL_SPEC section 8 inventory (which has no lm_head.weight).
  ASSERT_EQ(map.size(), 338u);

  // Every name unique.
  std::vector<std::string> names;
  names.reserve(map.size());
  for (const WeightMapEntry& entry : map) {
    names.push_back(entry.hf_name);
  }
  std::sort(names.begin(), names.end());
  EXPECT_EQ(std::adjacent_find(names.begin(), names.end()), names.end())
      << "duplicate tensor name in the weight map";

  // Exactly 12 tensors per layer, layers 0..27 all present.
  for (std::int32_t l = 0; l < 28; ++l) {
    const std::string prefix = "model.layers." + std::to_string(l) + ".";
    const auto in_layer = std::count_if(map.begin(), map.end(), [&](const WeightMapEntry& e) {
      return e.hf_name.rfind(prefix, 0) == 0;
    });
    EXPECT_EQ(in_layer, 12) << "layer " << l;
    // Per-layer entries carry their layer index; non-layer entries carry -1.
    for (const WeightMapEntry& entry : map) {
      if (entry.hf_name.rfind(prefix, 0) == 0) {
        EXPECT_EQ(entry.layer, l) << entry.hf_name;
      }
    }
  }
  const WeightMapEntry* embed = FindEntry(map, "model.embed_tokens.weight");
  const WeightMapEntry* final_norm = FindEntry(map, "model.norm.weight");
  ASSERT_NE(embed, nullptr);
  ASSERT_NE(final_norm, nullptr);
  EXPECT_EQ(embed->layer, -1);
  EXPECT_EQ(final_norm->layer, -1);
  EXPECT_EQ(FindEntry(map, "lm_head.weight"), nullptr)
      << "tied checkpoint must not map an lm_head tensor";
}

TEST(WeightsMapTest, MappingRowsMatchDesignTableVerbatim) {
  const ModelConfig config = SpecConfig();
  const auto map = BuildWeightMap(config, /*has_lm_head=*/false);

  ExpectEntry(map, "model.embed_tokens.weight", DeviceTensorKind::kEmbed, -1, 0, {151936, 1536});
  ExpectEntry(map, "model.norm.weight", DeviceTensorKind::kFinalNorm, -1, 0, {1536});

  // Every layer, all 12 tensors: docs/DESIGN.md section 4 fusion offsets --
  // q rows 0:1536, k rows 1536:1792, v rows 1792:2048 (bias elements at the
  // same offsets); gate rows 0:8960, up rows 8960:17920. Shapes are the
  // MODEL_SPEC section 8 per-layer pattern.
  for (std::int32_t l = 0; l < 28; ++l) {
    const std::string p = "model.layers." + std::to_string(l);
    ExpectEntry(map, p + ".input_layernorm.weight", DeviceTensorKind::kInputNorm, l, 0, {1536});
    ExpectEntry(map, p + ".self_attn.q_proj.weight", DeviceTensorKind::kWqkv, l, 0, {1536, 1536});
    ExpectEntry(map, p + ".self_attn.q_proj.bias", DeviceTensorKind::kBqkv, l, 0, {1536});
    ExpectEntry(map, p + ".self_attn.k_proj.weight", DeviceTensorKind::kWqkv, l, 1536, {256, 1536});
    ExpectEntry(map, p + ".self_attn.k_proj.bias", DeviceTensorKind::kBqkv, l, 1536, {256});
    ExpectEntry(map, p + ".self_attn.v_proj.weight", DeviceTensorKind::kWqkv, l, 1792, {256, 1536});
    ExpectEntry(map, p + ".self_attn.v_proj.bias", DeviceTensorKind::kBqkv, l, 1792, {256});
    ExpectEntry(map, p + ".self_attn.o_proj.weight", DeviceTensorKind::kWo, l, 0, {1536, 1536});
    ExpectEntry(map, p + ".post_attention_layernorm.weight", DeviceTensorKind::kPostAttnNorm, l, 0,
                {1536});
    ExpectEntry(map, p + ".mlp.gate_proj.weight", DeviceTensorKind::kWgateup, l, 0, {8960, 1536});
    ExpectEntry(map, p + ".mlp.up_proj.weight", DeviceTensorKind::kWgateup, l, 8960, {8960, 1536});
    ExpectEntry(map, p + ".mlp.down_proj.weight", DeviceTensorKind::kWdown, l, 0, {1536, 8960});
  }

  // The offsets above are the section 4 constants themselves.
  EXPECT_EQ(wl::kQRowBegin, 0);
  EXPECT_EQ(wl::kKRowBegin, 1536);
  EXPECT_EQ(wl::kVRowBegin, 1792);
  EXPECT_EQ(wl::kQkvRows, 2048);
  EXPECT_EQ(wl::kGateRowBegin, 0);
  EXPECT_EQ(wl::kUpRowBegin, 8960);
  EXPECT_EQ(wl::kGateUpRows, 17920);
  EXPECT_EQ(wl::kNumNormVectors, 57);
}

TEST(WeightsMapTest, DestRowWidthsMatchFusedShapes) {
  const ModelConfig config = SpecConfig();
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kEmbed, config), 1536);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kLmHead, config), 1536);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kWqkv, config), 1536);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kWo, config), 1536);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kWgateup, config), 1536);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kWdown, config), 8960); // [1536, 8960] row-major
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kBqkv, config), 1);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kInputNorm, config), 1);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kPostAttnNorm, config), 1);
  EXPECT_EQ(DestRowWidth(DeviceTensorKind::kFinalNorm, config), 1);
}

TEST(WeightsMapTest, FusedDestinationsAreExactlyTiledByTheirSources) {
  const ModelConfig config = SpecConfig();
  const auto map = BuildWeightMap(config, /*has_lm_head=*/false);

  // For every fused per-layer destination, the source row ranges must cover
  // [0, rows) with no gap and no overlap.
  const std::vector<std::pair<DeviceTensorKind, std::int64_t>> fused = {
      {DeviceTensorKind::kWqkv, wl::kQkvRows},
      {DeviceTensorKind::kBqkv, wl::kQkvRows},
      {DeviceTensorKind::kWgateup, wl::kGateUpRows},
  };
  for (const auto& [kind, total_rows] : fused) {
    for (std::int32_t l = 0; l < 28; ++l) {
      std::vector<std::pair<std::int64_t, std::int64_t>> spans; // [begin, end)
      for (const WeightMapEntry& entry : map) {
        if (entry.dest == kind && entry.layer == l) {
          spans.emplace_back(entry.dest_row_begin, entry.dest_row_begin + entry.dest_rows());
        }
      }
      std::sort(spans.begin(), spans.end());
      ASSERT_FALSE(spans.empty());
      EXPECT_EQ(spans.front().first, 0);
      for (std::size_t i = 1; i < spans.size(); ++i) {
        EXPECT_EQ(spans[i].first, spans[i - 1].second)
            << "gap or overlap in fused destination, layer " << l;
      }
      EXPECT_EQ(spans.back().second, total_rows);
    }
  }
}

TEST(WeightsMapTest, EntryByteSpansTileTheSlabExactly) {
  const ModelConfig config = SpecConfig();
  for (const bool has_lm_head : {false, true}) {
    const auto map = BuildWeightMap(config, has_lm_head);
    const WeightsLayout layout = ComputeWeightsLayout(config, has_lm_head);

    // Compute each entry's device byte span exactly as the upload loop does;
    // together they must tile [0, total_bytes) -- no overlap (no tensor
    // clobbers another), no gap (no uninitialized weight bytes), and no span
    // out of bounds.
    std::vector<std::pair<std::int64_t, std::int64_t>> spans;
    spans.reserve(map.size());
    for (const WeightMapEntry& entry : map) {
      const std::int64_t begin = EntryByteOffset(layout, entry, config);
      spans.emplace_back(begin, begin + entry.num_elements() * 2);
    }
    std::sort(spans.begin(), spans.end());
    ASSERT_EQ(spans.front().first, 0);
    for (std::size_t i = 1; i < spans.size(); ++i) {
      ASSERT_EQ(spans[i].first, spans[i - 1].second)
          << "gap or overlap between uploaded regions at span " << i;
    }
    EXPECT_EQ(spans.back().second, layout.total_bytes);
  }
}

TEST(WeightsMapTest, TotalBytesMatchTheDesignBudget) {
  const ModelConfig config = SpecConfig();

  // Sum of all mapped tensor payloads (FP16) for the tied checkpoint.
  const auto map = BuildWeightMap(config, /*has_lm_head=*/false);
  std::int64_t payload_bytes = 0;
  for (const WeightMapEntry& entry : map) {
    payload_bytes += entry.num_elements() * 2;
  }
  EXPECT_EQ(payload_bytes, std::int64_t{3'087'428'608}); // 1,543,714,304 params * 2 B

  // The slab is packed with zero padding, so the allocation equals the
  // payload exactly (docs/DESIGN.md section 1: 2944 MiB of weights).
  EXPECT_EQ(DeviceWeights::TotalDeviceBytes(config, /*has_separate_lm_head=*/false), payload_bytes);
  EXPECT_EQ(wl::kTotalWeightBytesTied, payload_bytes);

  // A separate lm_head (not this checkpoint) adds one [151936, 1536] tensor.
  EXPECT_EQ(DeviceWeights::TotalDeviceBytes(config, /*has_separate_lm_head=*/true),
            payload_bytes + std::int64_t{151936} * 1536 * 2);
}

TEST(WeightsMapTest, LayoutOffsetsAreAlignedAndLmHeadAliasesEmbed) {
  const ModelConfig config = SpecConfig();
  const WeightsLayout tied = ComputeWeightsLayout(config, /*has_lm_head=*/false);

  ASSERT_EQ(tied.layers.size(), 28u);
  const auto expect_aligned = [](std::int64_t offset) { EXPECT_EQ(offset % 256, 0) << offset; };
  expect_aligned(tied.embed);
  for (const WeightsLayout::LayerOffsets& l : tied.layers) {
    expect_aligned(l.w_qkv);
    expect_aligned(l.b_qkv);
    expect_aligned(l.w_o);
    expect_aligned(l.w_gateup);
    expect_aligned(l.w_down);
    expect_aligned(l.input_norm);
    expect_aligned(l.post_attn_norm);
  }
  expect_aligned(tied.final_norm);

  // Tied checkpoint: lm_head is the embed region, not a copy.
  EXPECT_TRUE(tied.lm_head_aliases_embed);
  EXPECT_EQ(tied.lm_head, tied.embed);
  EXPECT_EQ(tied.OffsetFor(DeviceTensorKind::kLmHead, -1),
            tied.OffsetFor(DeviceTensorKind::kEmbed, -1));

  // Untied file: a distinct region and a larger slab.
  const WeightsLayout untied = ComputeWeightsLayout(config, /*has_lm_head=*/true);
  EXPECT_FALSE(untied.lm_head_aliases_embed);
  EXPECT_NE(untied.lm_head, untied.embed);
  expect_aligned(untied.lm_head);
  EXPECT_EQ(untied.total_bytes, tied.total_bytes + std::int64_t{151936} * 1536 * 2);

  // OffsetFor agrees with the named fields for a middle layer.
  EXPECT_EQ(tied.OffsetFor(DeviceTensorKind::kWqkv, 7), tied.layers[7].w_qkv);
  EXPECT_EQ(tied.OffsetFor(DeviceTensorKind::kBqkv, 7), tied.layers[7].b_qkv);
  EXPECT_EQ(tied.OffsetFor(DeviceTensorKind::kWdown, 27), tied.layers[27].w_down);
  EXPECT_EQ(tied.OffsetFor(DeviceTensorKind::kFinalNorm, -1), tied.final_norm);
}

TEST(WeightsMapTest, LmHeadEntryPresentOnlyWhenTheFileHasOne) {
  const ModelConfig config = SpecConfig();
  const auto map = BuildWeightMap(config, /*has_lm_head=*/true);
  ASSERT_EQ(map.size(), 339u);
  ExpectEntry(map, "lm_head.weight", DeviceTensorKind::kLmHead, -1, 0, {151936, 1536});
}

TEST(WeightsMapTest, GeometryMismatchIsAHardErrorNamingTheField) {
  // The fused layout is defined for the pinned checkpoint only; a config that
  // dodged config validation must not silently produce a wrong layout.
  {
    ModelConfig config = SpecConfig();
    config.hidden_size = 2048;
    EXPECT_NE(WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("hidden_size"),
              std::string::npos);
    EXPECT_THROW(ComputeWeightsLayout(config, false), WeightsError);
    EXPECT_THROW(DeviceWeights::TotalDeviceBytes(config, false), WeightsError);
  }
  {
    ModelConfig config = SpecConfig();
    config.num_layers = 24;
    EXPECT_NE(WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("num_hidden_layers"),
              std::string::npos);
  }
  {
    ModelConfig config = SpecConfig();
    config.intermediate_size = 11008;
    EXPECT_NE(WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("intermediate_size"),
              std::string::npos);
  }
  {
    ModelConfig config = SpecConfig();
    config.vocab_size = 32000;
    EXPECT_NE(WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("vocab_size"),
              std::string::npos);
  }
  {
    // 4 KV heads would change the fused K/V row count (4 * 128 = 512 != 256).
    ModelConfig config = SpecConfig();
    config.num_kv_heads = 4;
    EXPECT_NE(
        WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("num_key_value_heads"),
        std::string::npos);
  }
  {
    ModelConfig config = SpecConfig();
    config.qkv_bias = false;
    EXPECT_NE(WeightsErrorMessage([&] { BuildWeightMap(config, false); }).find("qkv_bias"),
              std::string::npos);
  }
}

// Sanity anchor for the strided-view rule (docs/DESIGN.md section 5): the
// fused QKV row offsets ARE the executor's qkv_out view offsets (+0 / +1536 /
// +1792 with row stride 2048), because GEMM output feature f of row r comes
// from weight row f. This is what makes K/V readable in place after the
// fused GEMM without a split kernel.
TEST(WeightsMapTest, QkvFusionOffsetsMatchTheStridedViewContract) {
  EXPECT_EQ(wl::kQRowBegin, 0);    // q = qkv_out + 0
  EXPECT_EQ(wl::kKRowBegin, 1536); // k = qkv_out + 1536
  EXPECT_EQ(wl::kVRowBegin, 1792); // v = qkv_out + 1792
  EXPECT_EQ(wl::kQkvRows, 2048);   // row stride 2048
}

} // namespace
} // namespace redline
