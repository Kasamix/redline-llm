#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"

// Model constants, populated at runtime from the checkpoint's own
// `<model_dir>/config.json` (plus the EOS set from generation_config.json) -
// no model constants are hardcoded in the engine.
//
// Ground truth for every expected value lives in docs/MODEL_SPEC.md, fetched
// from the pinned checkpoint revision. The loader cross-checks every parsed
// value against those expected constants and fails hard on any mismatch,
// naming the field plus the expected and actual values - never a silent
// fallback. Field mapping, with the values expected for
// Qwen2.5-1.5B-Instruct:
//
//   source key                 ModelConfig field         expected value
//   ------------------------   -----------------------   -----------------------------
//   hidden_size                hidden_size               1536
//   num_hidden_layers          num_layers                28
//   num_attention_heads        num_q_heads               12
//   num_key_value_heads        num_kv_heads              2 (grouped-query attention)
//   intermediate_size          intermediate_size         8960
//   vocab_size                 vocab_size                151936
//   rms_norm_eps               rms_norm_eps              1e-6
//   rope_theta                 rope_theta                1000000.0
//   max_position_embeddings    max_position_embeddings   32768 (engine context ceiling)
//   tie_word_embeddings        tie_word_embeddings       true (lm_head reuses embed_tokens)
//   eos_token_id (generation)  eos_token_ids             {151645, 151643} - a set, F4
//   torch_dtype                (checkpoint dtype)        "bfloat16" -> converted to FP16 at load
//
// Derived (absent from config.json, MODEL_SPEC F7):
//   head_dim = hidden_size / num_q_heads = 128
//
// Qwen2 attention applies a bias on the Q/K/V projections and no bias on any
// other linear layer (MODEL_SPEC §2); modeled by `qkv_bias`.

namespace redline {

// Thrown for every model-configuration failure: unreadable or malformed
// files, missing/mistyped keys, an unsupported architecture, and any value
// that disagrees with the pinned checkpoint constants in docs/MODEL_SPEC.md.
// The message always names the source file and field; value mismatches carry
// both the expected and the actual value. Derives from std::runtime_error so
// generic engine-level handlers still work.
class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ModelConfig {
  std::int32_t hidden_size = 0;
  std::int32_t num_layers = 0;
  std::int32_t num_q_heads = 0;
  std::int32_t num_kv_heads = 0;
  std::int32_t head_dim = 0; // derived: hidden_size / num_q_heads
  std::int32_t intermediate_size = 0;
  std::int32_t vocab_size = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  std::int32_t max_position_embeddings = 0;
  bool tie_word_embeddings = false;
  bool qkv_bias = true; // Qwen2: bias on Q/K/V projections only

  // Generation stop set from generation_config.json (docs/MODEL_SPEC.md F4:
  // stop on membership, not equality - {151645 <|im_end|>, 151643
  // <|endoftext|>} for this checkpoint). File order preserved, duplicates
  // removed.
  std::vector<TokenId> eos_token_ids;

  // Stop-condition membership test (MODEL_SPEC F4). The set has two entries
  // for this checkpoint, so a linear scan is exact and branch-cheap.
  bool IsEosToken(TokenId token) const {
    return std::find(eos_token_ids.begin(), eos_token_ids.end(), token) != eos_token_ids.end();
  }

  // FP16 KV-cache bytes per token across all layers:
  //   2 (K and V) * num_layers * num_kv_heads * head_dim * sizeof(fp16)
  // = 28,672 bytes for Qwen2.5-1.5B. Used to size the paged pool.
  std::int64_t KvBytesPerToken() const {
    return std::int64_t{2} * num_layers * num_kv_heads * head_dim * 2;
  }

  // Parse core - no file I/O. Takes the raw JSON text of config.json and
  // generation_config.json, validates every value against the expected
  // constants of docs/MODEL_SPEC.md (hard ConfigError on mismatch, naming
  // field/expected/actual), and returns the populated config. Unit tests
  // feed fixture strings, so no checkpoint download is needed.
  static ModelConfig FromJsonText(const std::string& config_json,
                                  const std::string& generation_config_json);

  // Read `<model_dir>/config.json` and `<model_dir>/generation_config.json`
  // and delegate to FromJsonText. generation_config.json is required, not
  // optional: it is the only source of the full EOS stop set (config.json's
  // scalar eos_token_id lacks 151643 - MODEL_SPEC F4), so a missing file is
  // a hard ConfigError rather than a silent single-token fallback.
  static ModelConfig FromModelDir(const std::string& model_dir);
};

} // namespace redline
