// Model config loader + validation (docs/DESIGN.md sections 0-1).
//
// Policy: this engine serves exactly one pinned checkpoint
// (Qwen/Qwen2.5-1.5B-Instruct - docs/MODEL_SPEC.md). Every value parsed from
// the checkpoint's JSON is cross-checked against the MODEL_SPEC expected
// constants; any disagreement is a hard ConfigError naming the field plus the
// expected and actual values. There is deliberately no fallback: a mismatch
// means the wrong (or corrupt) checkpoint, and continuing would only surface
// later as silent numerical divergence from the HF reference.

#include "core/config.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/types.hpp"

namespace redline {
namespace {

using nlohmann::json;

// Expected constants for the pinned checkpoint (docs/MODEL_SPEC.md sections
// 1, 2, 6). Used for validation only: the engine consumes the values parsed
// from the checkpoint's own JSON, never these copies.
namespace expected {
constexpr std::int64_t kHiddenSize = 1536;
constexpr std::int64_t kNumLayers = 28;
constexpr std::int64_t kNumQHeads = 12;
constexpr std::int64_t kNumKvHeads = 2;
constexpr std::int64_t kHeadDim = 128; // derived: hidden_size / num_q_heads (MODEL_SPEC F7)
constexpr std::int64_t kIntermediateSize = 8960;
constexpr std::int64_t kVocabSize = 151936;
constexpr double kRopeTheta = 1e6; // not the common 1e4 default
constexpr double kRmsNormEps = 1e-6;
constexpr std::int64_t kMaxPositionEmbeddings = 32768;
constexpr bool kTieWordEmbeddings = true;
constexpr char kModelType[] = "qwen2";
constexpr char kArchitecture[] = "Qwen2ForCausalLM";
constexpr char kTorchDtype[] = "bfloat16"; // converted to FP16 at load (MODEL_SPEC F1)
constexpr char kHiddenAct[] = "silu";
constexpr std::int64_t kBosTokenId = 151643;
constexpr std::int64_t kConfigEosTokenId = 151645; // config.json scalar; the stop SET is below
// Full generation stop set from generation_config.json (MODEL_SPEC F4):
// {151645 <|im_end|>, 151643 <|endoftext|>}. Compared as a set.
constexpr TokenId kEosSet[] = {151645, 151643};
} // namespace expected

[[noreturn]] void Fail(const char* file, const std::string& message) {
  throw ConfigError(std::string(file) + ": " + message);
}

std::string FormatTokenSet(const std::vector<TokenId>& ids) {
  std::ostringstream oss;
  oss << "{";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << ids[i];
  }
  oss << "}";
  return oss.str();
}

json ParseJsonText(const char* file, const std::string& text) {
  try {
    json parsed = json::parse(text);
    if (!parsed.is_object()) {
      Fail(file, "top-level JSON value must be an object");
    }
    return parsed;
  } catch (const json::parse_error& e) {
    Fail(file, std::string("malformed JSON: ") + e.what());
  }
}

const json& Require(const json& root, const char* file, const char* key) {
  const auto it = root.find(key);
  if (it == root.end()) {
    Fail(file, std::string("missing required key '") + key + "'");
  }
  return *it;
}

std::int64_t RequireInt(const json& root, const char* file, const char* key) {
  const json& value = Require(root, file, key);
  if (!value.is_number_integer()) {
    Fail(file, std::string("key '") + key + "': expected an integer, got " + value.dump());
  }
  return value.get<std::int64_t>();
}

double RequireNumber(const json& root, const char* file, const char* key) {
  const json& value = Require(root, file, key);
  if (!value.is_number()) {
    Fail(file, std::string("key '") + key + "': expected a number, got " + value.dump());
  }
  return value.get<double>();
}

bool RequireBool(const json& root, const char* file, const char* key) {
  const json& value = Require(root, file, key);
  if (!value.is_boolean()) {
    Fail(file, std::string("key '") + key + "': expected a boolean, got " + value.dump());
  }
  return value.get<bool>();
}

std::string RequireString(const json& root, const char* file, const char* key) {
  const json& value = Require(root, file, key);
  if (!value.is_string()) {
    Fail(file, std::string("key '") + key + "': expected a string, got " + value.dump());
  }
  return value.get<std::string>();
}

void CheckInt(const char* file, const char* field, std::int64_t expected_value,
              std::int64_t actual) {
  if (actual != expected_value) {
    std::ostringstream oss;
    oss << field << ": expected " << expected_value << ", got " << actual;
    Fail(file, oss.str());
  }
}

// Exact comparison, intentionally: both sides are the nearest double to the
// same decimal literal (e.g. "1e-06"), so a correct checkpoint compares
// bit-equal and anything else is a real difference worth failing on.
void CheckNumber(const char* file, const char* field, double expected_value, double actual) {
  if (actual != expected_value) {
    std::ostringstream oss;
    oss << std::setprecision(12) << field << ": expected " << expected_value << ", got " << actual;
    Fail(file, oss.str());
  }
}

void CheckBool(const char* file, const char* field, bool expected_value, bool actual) {
  if (actual != expected_value) {
    std::ostringstream oss;
    oss << std::boolalpha << field << ": expected " << expected_value << ", got " << actual;
    Fail(file, oss.str());
  }
}

void CheckString(const char* file, const char* field, const std::string& expected_value,
                 const std::string& actual) {
  if (actual != expected_value) {
    Fail(file,
         std::string(field) + ": expected \"" + expected_value + "\", got \"" + actual + "\"");
  }
}

// If `key` is present it must be an integer equal to `expected_value`;
// absence is fine (the key is informational for this checkpoint).
void CheckOptionalInt(const json& root, const char* file, const char* key,
                      std::int64_t expected_value) {
  const auto it = root.find(key);
  if (it == root.end()) {
    return;
  }
  if (!it->is_number_integer()) {
    Fail(file, std::string("key '") + key + "': expected an integer, got " + it->dump());
  }
  CheckInt(file, key, expected_value, it->get<std::int64_t>());
}

// generation_config.json's eos_token_id may be a single integer or an array
// of integers (HF supports both spellings). Returns the ids in file order
// with duplicates removed.
std::vector<TokenId> ExtractEosSet(const json& gen, const char* file) {
  const json& value = Require(gen, file, "eos_token_id");
  std::vector<TokenId> out;
  const auto append = [&](const json& element) {
    if (!element.is_number_integer()) {
      Fail(file, "key 'eos_token_id': expected an integer or an array of integers, got " +
                     element.dump());
    }
    const std::int64_t id64 = element.get<std::int64_t>();
    if (id64 < 0 || id64 > std::numeric_limits<TokenId>::max()) {
      Fail(file, "key 'eos_token_id': value " + std::to_string(id64) +
                     " is not a valid non-negative int32 token id");
    }
    const TokenId id = static_cast<TokenId>(id64);
    if (std::find(out.begin(), out.end(), id) == out.end()) {
      out.push_back(id);
    }
  };
  if (value.is_array()) {
    if (value.empty()) {
      Fail(file, "key 'eos_token_id': array must not be empty");
    }
    for (const json& element : value) {
      append(element);
    }
  } else {
    append(value);
  }
  return out;
}

// The extracted stop set must equal the MODEL_SPEC expected set exactly
// (order-insensitive). A truncated set (e.g. only 151645) would let
// generation run through <|endoftext|>; an enlarged one would stop early.
void CheckEosSet(const char* file, const std::vector<TokenId>& actual) {
  std::vector<TokenId> actual_sorted = actual;
  std::sort(actual_sorted.begin(), actual_sorted.end());
  std::vector<TokenId> expected_sorted(std::begin(expected::kEosSet), std::end(expected::kEosSet));
  std::sort(expected_sorted.begin(), expected_sorted.end());
  if (actual_sorted != expected_sorted) {
    const std::vector<TokenId> expected_ids(std::begin(expected::kEosSet),
                                            std::end(expected::kEosSet));
    Fail(file, "eos_token_id: expected the stop set " + FormatTokenSet(expected_ids) + ", got " +
                   FormatTokenSet(actual));
  }
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw ConfigError("missing or unreadable file: " + path.string() +
                      " (the model directory must contain config.json and "
                      "generation_config.json)");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  if (in.bad()) {
    throw ConfigError("I/O error while reading " + path.string());
  }
  return buffer.str();
}

} // namespace

ModelConfig ModelConfig::FromJsonText(const std::string& config_json,
                                      const std::string& generation_config_json) {
  constexpr const char* kCfg = "config.json";
  constexpr const char* kGen = "generation_config.json";

  const json cfg = ParseJsonText(kCfg, config_json);
  const json gen = ParseJsonText(kGen, generation_config_json);

  // Architecture identity first: every expectation below assumes the pinned
  // Qwen2 checkpoint, so a foreign config should fail on the clearest field.
  CheckString(kCfg, "model_type", expected::kModelType, RequireString(cfg, kCfg, "model_type"));
  if (const auto it = cfg.find("architectures"); it != cfg.end()) {
    const bool ok =
        it->is_array() && std::any_of(it->begin(), it->end(), [](const json& entry) {
          return entry.is_string() && entry.get<std::string>() == expected::kArchitecture;
        });
    if (!ok) {
      Fail(kCfg, std::string("architectures: expected to contain \"") + expected::kArchitecture +
                     "\", got " + it->dump());
    }
  }

  const std::int64_t hidden_size = RequireInt(cfg, kCfg, "hidden_size");
  CheckInt(kCfg, "hidden_size", expected::kHiddenSize, hidden_size);

  const std::int64_t num_layers = RequireInt(cfg, kCfg, "num_hidden_layers");
  CheckInt(kCfg, "num_hidden_layers", expected::kNumLayers, num_layers);

  const std::int64_t num_q_heads = RequireInt(cfg, kCfg, "num_attention_heads");
  CheckInt(kCfg, "num_attention_heads", expected::kNumQHeads, num_q_heads);

  const std::int64_t num_kv_heads = RequireInt(cfg, kCfg, "num_key_value_heads");
  CheckInt(kCfg, "num_key_value_heads", expected::kNumKvHeads, num_kv_heads);

  const std::int64_t intermediate_size = RequireInt(cfg, kCfg, "intermediate_size");
  CheckInt(kCfg, "intermediate_size", expected::kIntermediateSize, intermediate_size);

  const std::int64_t vocab_size = RequireInt(cfg, kCfg, "vocab_size");
  CheckInt(kCfg, "vocab_size", expected::kVocabSize, vocab_size);

  const std::int64_t max_position_embeddings = RequireInt(cfg, kCfg, "max_position_embeddings");
  CheckInt(kCfg, "max_position_embeddings", expected::kMaxPositionEmbeddings,
           max_position_embeddings);

  const double rope_theta = RequireNumber(cfg, kCfg, "rope_theta");
  CheckNumber(kCfg, "rope_theta", expected::kRopeTheta, rope_theta);

  const double rms_norm_eps = RequireNumber(cfg, kCfg, "rms_norm_eps");
  CheckNumber(kCfg, "rms_norm_eps", expected::kRmsNormEps, rms_norm_eps);

  const bool tie_word_embeddings = RequireBool(cfg, kCfg, "tie_word_embeddings");
  CheckBool(kCfg, "tie_word_embeddings", expected::kTieWordEmbeddings, tie_word_embeddings);

  // The checkpoint stores BF16; the loader converts to FP16 once at load
  // (MODEL_SPEC F1). Any other dtype means the wrong checkpoint export.
  CheckString(kCfg, "torch_dtype", expected::kTorchDtype, RequireString(cfg, kCfg, "torch_dtype"));

  // The MLP kernel computes silu(gate) * up; a different activation would
  // silently compute the wrong function, so it is validated here.
  CheckString(kCfg, "hidden_act", expected::kHiddenAct, RequireString(cfg, kCfg, "hidden_act"));

  // head_dim is derived - config.json has no head_dim key for this
  // checkpoint (MODEL_SPEC F7). If a config does carry the key, it must
  // agree with the derived value.
  const std::int64_t head_dim = hidden_size / num_q_heads; // divisor validated == 12 above
  CheckInt(kCfg, "head_dim (derived hidden_size / num_attention_heads)", expected::kHeadDim,
           head_dim);
  if (const auto it = cfg.find("head_dim"); it != cfg.end()) {
    if (!it->is_number_integer()) {
      Fail(kCfg, "key 'head_dim': expected an integer, got " + it->dump());
    }
    CheckInt(kCfg, "head_dim", head_dim, it->get<std::int64_t>());
  }

  // The engine implements full causal attention only; a checkpoint that
  // enables sliding-window attention would be served incorrectly.
  // `use_sliding_window: false` (this checkpoint) makes `sliding_window` and
  // `max_window_layers` inert, so those keys are ignored (MODEL_SPEC §1).
  if (const auto it = cfg.find("use_sliding_window"); it != cfg.end()) {
    if (!it->is_boolean()) {
      Fail(kCfg, "key 'use_sliding_window': expected a boolean, got " + it->dump());
    }
    if (it->get<bool>()) {
      Fail(kCfg, "use_sliding_window: expected false (this engine implements full causal "
                 "attention only), got true");
    }
  }

  // Informational token ids in config.json - cross-checked when present.
  // The authoritative stop set comes from generation_config.json below.
  CheckOptionalInt(cfg, kCfg, "bos_token_id", expected::kBosTokenId);
  CheckOptionalInt(cfg, kCfg, "eos_token_id", expected::kConfigEosTokenId);

  ModelConfig mc;
  mc.hidden_size = static_cast<std::int32_t>(hidden_size);
  mc.num_layers = static_cast<std::int32_t>(num_layers);
  mc.num_q_heads = static_cast<std::int32_t>(num_q_heads);
  mc.num_kv_heads = static_cast<std::int32_t>(num_kv_heads);
  mc.head_dim = static_cast<std::int32_t>(head_dim);
  mc.intermediate_size = static_cast<std::int32_t>(intermediate_size);
  mc.vocab_size = static_cast<std::int32_t>(vocab_size);
  mc.rms_norm_eps = static_cast<float>(rms_norm_eps);
  mc.rope_theta = static_cast<float>(rope_theta);
  mc.max_position_embeddings = static_cast<std::int32_t>(max_position_embeddings);
  mc.tie_word_embeddings = tie_word_embeddings;
  // Bias on Q/K/V projections only - hardcoded in the HF Qwen2 implementation
  // rather than keyed in config.json (MODEL_SPEC §2, F7).
  mc.qkv_bias = true;

  mc.eos_token_ids = ExtractEosSet(gen, kGen);
  CheckEosSet(kGen, mc.eos_token_ids);

  return mc;
}

ModelConfig ModelConfig::FromModelDir(const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  const std::string config_json = ReadTextFile(dir / "config.json");
  // Required, not optional: generation_config.json is the only source of the
  // full EOS stop set (config.json's scalar lacks 151643 - MODEL_SPEC F4).
  // Falling back to the scalar would silently ship a wrong stop condition.
  const std::string generation_config_json = ReadTextFile(dir / "generation_config.json");
  return FromJsonText(config_json, generation_config_json);
}

} // namespace redline
