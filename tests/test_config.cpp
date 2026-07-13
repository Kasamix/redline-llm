// ModelConfig loader/validation unit tests (docs/DESIGN.md sections 0-1,
// docs/MODEL_SPEC.md). The parse core takes JSON text, so every case below
// runs from in-memory fixture strings - no checkpoint download. Fixture
// values mirror the pinned Qwen2.5-1.5B-Instruct files as documented in
// docs/MODEL_SPEC.md (sources S1/S2).

#include "core/config.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/types.hpp"

namespace redline {
namespace {

using nlohmann::json;

// Mirrors <model_dir>/config.json for the pinned checkpoint (MODEL_SPEC S1).
constexpr char kConfigJson[] = R"json({
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
})json";

// Mirrors <model_dir>/generation_config.json (MODEL_SPEC S2). The sampling
// knobs are irrelevant to the engine (greedy only); the EOS *set* is the
// payload (MODEL_SPEC F4).
constexpr char kGenerationConfigJson[] = R"json({
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [151645, 151643],
  "pad_token_id": 151643,
  "repetition_penalty": 1.1,
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.8,
  "transformers_version": "4.43.1"
})json";

// Re-serializes the fixture with one key replaced by `value_literal` (a JSON
// literal, e.g. "2048" or "\"llama\"").
std::string WithOverride(const std::string& base_json, const std::string& key,
                         const std::string& value_literal) {
  json j = json::parse(base_json);
  j[key] = json::parse(value_literal);
  return j.dump();
}

// Re-serializes the fixture with one key removed.
std::string WithoutKey(const std::string& base_json, const std::string& key) {
  json j = json::parse(base_json);
  j.erase(key);
  return j.dump();
}

// Runs `fn`, requires it to throw redline::ConfigError, and returns the
// message for substring assertions.
template <typename Fn> std::string ConfigErrorMessage(Fn&& fn) {
  try {
    (void)fn();
  } catch (const ConfigError& e) {
    return e.what();
  } catch (const std::exception& e) {
    ADD_FAILURE() << "expected redline::ConfigError, caught a different exception: " << e.what();
    return {};
  }
  ADD_FAILURE() << "expected redline::ConfigError, but nothing was thrown";
  return {};
}

void ExpectContains(const std::string& haystack, const std::string& needle) {
  EXPECT_NE(haystack.find(needle), std::string::npos)
      << "expected error message to contain \"" << needle << "\"\n  actual message: \"" << haystack
      << "\"";
}

// Temp directory for the FromModelDir file-plumbing tests; removed on scope
// exit. All parse-core tests stay file-free.
class ScopedTempDir {
 public:
  explicit ScopedTempDir(const std::string& name)
      : path_(std::filesystem::path(testing::TempDir()) / name) {
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec); // best-effort cleanup
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void WriteFile(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open()) << "failed to open " << path;
  out << text;
  ASSERT_TRUE(out.good()) << "failed to write " << path;
}

// ---------------------------------------------------------------- happy path

TEST(ConfigTest, HappyPathPopulatesEveryField) {
  const ModelConfig mc = ModelConfig::FromJsonText(kConfigJson, kGenerationConfigJson);

  EXPECT_EQ(mc.hidden_size, 1536);
  EXPECT_EQ(mc.num_layers, 28);
  EXPECT_EQ(mc.num_q_heads, 12);
  EXPECT_EQ(mc.num_kv_heads, 2);
  EXPECT_EQ(mc.head_dim, 128); // derived: 1536 / 12 (MODEL_SPEC F7)
  EXPECT_EQ(mc.intermediate_size, 8960);
  EXPECT_EQ(mc.vocab_size, 151936);
  EXPECT_FLOAT_EQ(mc.rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(mc.rope_theta, 1e6f);
  EXPECT_EQ(mc.max_position_embeddings, 32768);
  EXPECT_TRUE(mc.tie_word_embeddings);
  EXPECT_TRUE(mc.qkv_bias);

  // 2 (K,V) * 28 layers * 2 kv heads * 128 head_dim * 2 B = 28,672 B/token.
  EXPECT_EQ(mc.KvBytesPerToken(), 28672);
}

TEST(ConfigTest, EosSetExtractionPreservesFileOrder) {
  const ModelConfig mc = ModelConfig::FromJsonText(kConfigJson, kGenerationConfigJson);

  ASSERT_EQ(mc.eos_token_ids.size(), 2u);
  EXPECT_EQ(mc.eos_token_ids[0], 151645); // <|im_end|>, first in the file
  EXPECT_EQ(mc.eos_token_ids[1], 151643); // <|endoftext|>

  EXPECT_TRUE(mc.IsEosToken(151645));
  EXPECT_TRUE(mc.IsEosToken(151643));
  EXPECT_FALSE(mc.IsEosToken(151644)); // <|im_start|> is not a stop token
  EXPECT_FALSE(mc.IsEosToken(0));
}

TEST(ConfigTest, EosSetExtractionDeduplicates) {
  const std::string gen =
      WithOverride(kGenerationConfigJson, "eos_token_id", "[151645, 151643, 151645]");
  const ModelConfig mc = ModelConfig::FromJsonText(kConfigJson, gen);
  ASSERT_EQ(mc.eos_token_ids.size(), 2u);
  EXPECT_EQ(mc.eos_token_ids[0], 151645);
  EXPECT_EQ(mc.eos_token_ids[1], 151643);
}

TEST(ConfigTest, ExplicitHeadDimKeyMatchingDerivedIsAccepted) {
  // config.json normally has no head_dim key (MODEL_SPEC F7); if present it
  // must agree with hidden_size / num_attention_heads.
  const std::string cfg = WithOverride(kConfigJson, "head_dim", "128");
  const ModelConfig mc = ModelConfig::FromJsonText(cfg, kGenerationConfigJson);
  EXPECT_EQ(mc.head_dim, 128);
}

TEST(ConfigTest, AbsentArchitecturesKeyIsAccepted) {
  // model_type is the required identity; the architectures array is
  // cross-checked only when present.
  const std::string cfg = WithoutKey(kConfigJson, "architectures");
  const ModelConfig mc = ModelConfig::FromJsonText(cfg, kGenerationConfigJson);
  EXPECT_EQ(mc.hidden_size, 1536);
}

TEST(ConfigTest, AbsentOptionalTokenIdKeysAreAccepted) {
  const std::string cfg = WithoutKey(WithoutKey(kConfigJson, "bos_token_id"), "eos_token_id");
  const ModelConfig mc = ModelConfig::FromJsonText(cfg, kGenerationConfigJson);
  ASSERT_EQ(mc.eos_token_ids.size(), 2u); // stop set still comes from generation_config.json
}

// ------------------------------------------------- per-field mismatch errors

struct MismatchCase {
  const char* key;           // config.json key to override
  const char* wrong_literal; // JSON literal for the wrong value
  const char* actual_marker; // substring proving the message carries the actual value
};

TEST(ConfigTest, EveryMismatchedFieldFailsNamingFieldExpectedAndActual) {
  const MismatchCase kCases[] = {
      {"hidden_size", "2048", "got 2048"},
      {"num_hidden_layers", "24", "got 24"},
      {"num_attention_heads", "16", "got 16"},
      {"num_key_value_heads", "4", "got 4"},
      {"intermediate_size", "11008", "got 11008"},
      {"vocab_size", "32000", "got 32000"},
      {"max_position_embeddings", "4096", "got 4096"},
      {"rope_theta", "10000.0", "got 10000"},
      {"rms_norm_eps", "1e-05", "got 1e-05"},
      {"tie_word_embeddings", "false", "got false"},
      {"torch_dtype", "\"float16\"", "float16"},
      {"model_type", "\"llama\"", "llama"},
      {"hidden_act", "\"gelu\"", "gelu"},
  };
  for (const MismatchCase& c : kCases) {
    SCOPED_TRACE(c.key);
    const std::string cfg = WithOverride(kConfigJson, c.key, c.wrong_literal);
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, c.key);           // names the field
    ExpectContains(msg, "expected");      // carries the expected value
    ExpectContains(msg, c.actual_marker); // carries the actual value
  }
}

TEST(ConfigTest, ExplicitHeadDimKeyContradictingDerivedFails) {
  const std::string cfg = WithOverride(kConfigJson, "head_dim", "64");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
  ExpectContains(msg, "head_dim");
  ExpectContains(msg, "got 64");
}

TEST(ConfigTest, WrongArchitecturesArrayFails) {
  const std::string cfg = WithOverride(kConfigJson, "architectures", "[\"LlamaForCausalLM\"]");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
  ExpectContains(msg, "architectures");
  ExpectContains(msg, "Qwen2ForCausalLM");
}

TEST(ConfigTest, SlidingWindowEnabledIsRejected) {
  // use_sliding_window: false makes the window keys inert (MODEL_SPEC §1);
  // true would require windowed attention the engine does not implement.
  const std::string cfg = WithOverride(kConfigJson, "use_sliding_window", "true");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
  ExpectContains(msg, "use_sliding_window");
}

TEST(ConfigTest, PresentButWrongOptionalTokenIdsFail) {
  {
    const std::string cfg = WithOverride(kConfigJson, "bos_token_id", "42");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "bos_token_id");
    ExpectContains(msg, "got 42");
  }
  {
    const std::string cfg = WithOverride(kConfigJson, "eos_token_id", "42");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "eos_token_id");
    ExpectContains(msg, "got 42");
  }
}

// ------------------------------------------------------ EOS-set validation

TEST(ConfigTest, EosScalarFormYieldsTruncatedSetAndFails) {
  // A scalar eos_token_id parses (HF supports the spelling) but produces
  // {151645}, which is not the full stop set - hard error, never a silent
  // single-token stop condition (MODEL_SPEC F4).
  const std::string gen = WithOverride(kGenerationConfigJson, "eos_token_id", "151645");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
  ExpectContains(msg, "eos_token_id");
  ExpectContains(msg, "{151645, 151643}"); // expected set
  ExpectContains(msg, "{151645}");         // actual set
}

TEST(ConfigTest, EosSubsetFails) {
  const std::string gen = WithOverride(kGenerationConfigJson, "eos_token_id", "[151643]");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
  ExpectContains(msg, "eos_token_id");
  ExpectContains(msg, "{151643}");
}

TEST(ConfigTest, EosSupersetFails) {
  const std::string gen =
      WithOverride(kGenerationConfigJson, "eos_token_id", "[151645, 151643, 42]");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
  ExpectContains(msg, "eos_token_id");
  ExpectContains(msg, "42");
}

TEST(ConfigTest, MissingEosKeyInGenerationConfigFails) {
  const std::string gen = WithoutKey(kGenerationConfigJson, "eos_token_id");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
  ExpectContains(msg, "generation_config.json");
  ExpectContains(msg, "missing required key 'eos_token_id'");
}

TEST(ConfigTest, EmptyEosArrayFails) {
  const std::string gen = WithOverride(kGenerationConfigJson, "eos_token_id", "[]");
  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
  ExpectContains(msg, "eos_token_id");
  ExpectContains(msg, "must not be empty");
}

TEST(ConfigTest, NonIntegerEosEntriesFail) {
  {
    const std::string gen =
        WithOverride(kGenerationConfigJson, "eos_token_id", "[\"151645\", 151643]");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
    ExpectContains(msg, "eos_token_id");
    ExpectContains(msg, "expected an integer");
  }
  {
    // 2^32 does not fit the int32 TokenId type.
    const std::string gen =
        WithOverride(kGenerationConfigJson, "eos_token_id", "[151645, 4294967296]");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
    ExpectContains(msg, "eos_token_id");
    ExpectContains(msg, "not a valid non-negative int32");
  }
  {
    const std::string gen = WithOverride(kGenerationConfigJson, "eos_token_id", "[-1, 151645]");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, gen); });
    ExpectContains(msg, "not a valid non-negative int32");
  }
}

// ------------------------------------------- missing keys / types / parsing

TEST(ConfigTest, EveryMissingRequiredConfigKeyFailsNamingTheKey) {
  const char* kRequiredKeys[] = {
      "model_type",          "hidden_size",
      "num_hidden_layers",   "num_attention_heads",
      "num_key_value_heads", "intermediate_size",
      "vocab_size",          "rms_norm_eps",
      "rope_theta",          "max_position_embeddings",
      "tie_word_embeddings", "torch_dtype",
      "hidden_act",
  };
  for (const char* key : kRequiredKeys) {
    SCOPED_TRACE(key);
    const std::string cfg = WithoutKey(kConfigJson, key);
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "config.json");
    ExpectContains(msg, std::string("missing required key '") + key + "'");
  }
}

TEST(ConfigTest, WrongJsonTypesFailNamingTheKey) {
  {
    const std::string cfg = WithOverride(kConfigJson, "hidden_size", "\"1536\"");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "hidden_size");
    ExpectContains(msg, "expected an integer");
  }
  {
    const std::string cfg = WithOverride(kConfigJson, "tie_word_embeddings", "1");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "tie_word_embeddings");
    ExpectContains(msg, "expected a boolean");
  }
  {
    const std::string cfg = WithOverride(kConfigJson, "rope_theta", "\"1e6\"");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "rope_theta");
    ExpectContains(msg, "expected a number");
  }
  {
    const std::string cfg = WithOverride(kConfigJson, "torch_dtype", "16");
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(cfg, kGenerationConfigJson); });
    ExpectContains(msg, "torch_dtype");
    ExpectContains(msg, "expected a string");
  }
}

TEST(ConfigTest, MalformedJsonFailsForEitherFile) {
  {
    const std::string msg = ConfigErrorMessage(
        [&] { return ModelConfig::FromJsonText("{ this is not JSON", kGenerationConfigJson); });
    ExpectContains(msg, "config.json");
    ExpectContains(msg, "malformed JSON");
  }
  {
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText(kConfigJson, ""); });
    ExpectContains(msg, "generation_config.json");
    ExpectContains(msg, "malformed JSON");
  }
  {
    const std::string msg =
        ConfigErrorMessage([&] { return ModelConfig::FromJsonText("[]", kGenerationConfigJson); });
    ExpectContains(msg, "config.json");
    ExpectContains(msg, "must be an object");
  }
}

// ------------------------------------------------------- file-level plumbing

TEST(ConfigTest, FromModelDirReadsBothFiles) {
  ScopedTempDir dir("redline_config_test_happy");
  WriteFile(dir.path() / "config.json", kConfigJson);
  WriteFile(dir.path() / "generation_config.json", kGenerationConfigJson);

  const ModelConfig mc = ModelConfig::FromModelDir(dir.path().string());
  EXPECT_EQ(mc.hidden_size, 1536);
  EXPECT_EQ(mc.num_layers, 28);
  EXPECT_EQ(mc.head_dim, 128);
  ASSERT_EQ(mc.eos_token_ids.size(), 2u);
  EXPECT_EQ(mc.eos_token_ids[0], 151645);
  EXPECT_EQ(mc.eos_token_ids[1], 151643);
}

TEST(ConfigTest, FromModelDirMissingGenerationConfigIsFatal) {
  // The stop set only exists in generation_config.json; falling back to
  // config.json's scalar would silently drop 151643 (MODEL_SPEC F4).
  ScopedTempDir dir("redline_config_test_nogen");
  WriteFile(dir.path() / "config.json", kConfigJson);

  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromModelDir(dir.path().string()); });
  ExpectContains(msg, "generation_config.json");
}

TEST(ConfigTest, FromModelDirMissingConfigIsFatal) {
  ScopedTempDir dir("redline_config_test_nocfg");

  const std::string msg =
      ConfigErrorMessage([&] { return ModelConfig::FromModelDir(dir.path().string()); });
  ExpectContains(msg, "config.json");
}

} // namespace
} // namespace redline
