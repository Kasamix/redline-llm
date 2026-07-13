# MODEL_SPEC - Qwen/Qwen2.5-1.5B-Instruct

Single ground-truth reference for the model Redline serves. All loader, kernel, scheduler,
and test code is written against the values in this file. Every value below was fetched from
the Hugging Face Hub on **2026-07-09** at the pinned revision; nothing is quoted from memory.

| | |
|---|---|
| Model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Pinned revision (`main` @ fetch time) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Repo last modified | 2024-09-25T12:32:50Z |
| Architecture class | `Qwen2ForCausalLM` (`model_type: "qwen2"`) |
| Saved with | `transformers` 4.43.1 (per `config.json`) |

Sources (fetched 2026-07-09):

- **S1** - [`config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json)
- **S2** - [`generation_config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/generation_config.json)
- **S3** - [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer_config.json)
- **S4** - [HF model API](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct) (file list, revision sha, param count)
- **S5** - `model.safetensors` JSON header, read via HTTP range request (bytes `[8, 8+38528)`) at the pinned revision
- **S6** - [`modeling_qwen2.py` @ transformers v4.43.1](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py) (the version the checkpoint was saved with); cross-checked against [v5.13.0](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py) - the math cited below is identical in both.

Use `scripts/fetch_model.py` to download the checkpoint into `models/` (gitignored) at the pinned revision.

---

## 1. Core architecture constants (S1)

| Constant | Value | Notes |
|---|---:|---|
| `hidden_size` | **1536** | |
| `num_hidden_layers` | **28** | |
| `num_attention_heads` | **12** | query heads |
| `num_key_value_heads` | **2** | GQA: 6 query heads share each KV head |
| `head_dim` | **128** | **derived** = 1536 / 12; `config.json` has no `head_dim` key |
| `intermediate_size` | **8960** | MLP inner dim |
| `vocab_size` | **151936** | embedding rows; see §6 for used-ID range |
| `rope_theta` | **1000000.0** | 1e6, not the LLaMA-default 1e4 |
| `rms_norm_eps` | **1e-06** | |
| `max_position_embeddings` | **32768** | hard context ceiling for the engine |
| `tie_word_embeddings` | **true** | no `lm_head.weight` tensor exists - see §5 |
| `torch_dtype` | **"bfloat16"** | checkpoint dtype; engine serves FP16 - see §10 flag F1 |
| `hidden_act` | **"silu"** | SwiGLU MLP: `down(silu(gate(x)) * up(x))` |
| `attention_dropout` | 0.0 | |
| `initializer_range` | 0.02 | training-time only, irrelevant to inference |
| `use_cache` | true | |
| `bos_token_id` | 151643 | see §6 |
| `eos_token_id` | 151645 | see §6 - generation uses a *set*, S2 |

### Sliding window (S1) - present in config, **disabled**

| Key | Value |
|---|---:|
| `use_sliding_window` | **false** |
| `sliding_window` | 32768 |
| `max_window_layers` | 21 |

`use_sliding_window: false` means **full (non-windowed) causal attention in all 28 layers**.
`sliding_window` and `max_window_layers` are inert for this checkpoint; the engine ignores them.

## 2. Attention block

- **GQA layout:** 12 query heads, 2 KV heads, `head_dim` 128. Q projection outputs
  12×128 = 1536; K and V projections each output 2×128 = **256** (shapes confirmed in §8).
- **QKV bias: YES.** Qwen2 uses bias on `q_proj`/`k_proj`/`v_proj` and **no bias** on
  `o_proj`. `config.json` has **no** `attention_bias` key - the bias choice is hardcoded in the
  HF implementation (S6, v4.43.1 lines 216-219: `bias=True` for q/k/v, `bias=False` for o;
  identical at v5.13.0 lines 200-203) and is confirmed by the checkpoint itself: the only bias
  tensors in `model.safetensors` are `self_attn.{q,k,v}_proj.bias` (S5, §8).
- MLP `gate_proj`/`up_proj`/`down_proj` and the output projection have **no bias** (S5, S6).
- Attention scale: `1/sqrt(head_dim)` = 1/√128 (standard; S6).
- Causal mask only (no sliding window, §1).

## 3. RoPE - GPT-NeoX half-rotation style (S6)

HF `transformers` applies rotary embeddings to Qwen2 with the **NeoX "half-rotation" (non-interleaved)**
convention, *not* the GPT-J interleaved-pairs convention:

```python
# transformers v4.43.1, modeling_qwen2.py, lines 120-124 (identical at v5.13.0, line 116)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# q_embed = (q * cos) + (rotate_half(q) * sin)   (apply_rotary_pos_emb)
```

Kernel contract (must match HF bit-for-bit at the math level):

- Element `i` pairs with element `i + 64` within each 128-wide head (halves), **not** `(2i, 2i+1)`.
- Frequencies: `inv_freq[j] = rope_theta^(-2j/128)` for `j = 0..63`
  (v4.43.1 line 90: `1.0 / (base ** (arange(0, dim, 2).float() / dim))`), angle at position `m` is
  `m * inv_freq[j]`. cos/sin tables are built as `cat(freqs, freqs)`, so `cos[i] == cos[i+64]`.
- Rotary covers the **full** `head_dim` (128) - no partial-rotary factor.
- Applied to **Q and K only**, per head, **after** the projection bias add; positions are absolute
  token indices (0-based).
- HF computes the cos/sin cache in **float32**, then casts to the activation dtype at use
  (v4.43.1 lines 90-116). Engine kernels compute angles/tables in FP32 to match.

## 4. Normalization and MLP (S6)

- **RMSNorm** (`eps = 1e-6`), pre-norm: `input_layernorm` before attention,
  `post_attention_layernorm` before MLP, plus a final `model.norm` before the LM head.
- HF reference semantics (v4.43.1 lines 74-79): compute in FP32 -
  `x32 * rsqrt(mean(x32²) + eps)` - then **cast back to the activation dtype *before*
  multiplying by the weight**. The fused RMSNorm kernel must reproduce this order (FP32
  accumulate, FP16 round, then FP16 weight multiply) or token-match tests will drift.
- **MLP:** `down_proj( silu(gate_proj(x)) * up_proj(x) )`, `silu(x) = x·σ(x)`, no biases.
- Residual adds around both sub-blocks (standard pre-norm transformer).

## 5. Tied embeddings - there is no `lm_head` tensor

`tie_word_embeddings: true` (S1) and the safetensors header (S5) confirm:
**`lm_head.weight` does not exist in the checkpoint.** The loader must alias the output
projection to `model.embed_tokens.weight` (`[151936, 1536]`). Logits GEMM:
`logits = hidden @ embed_tokens.weightᵀ` → `[*, 151936]`.

## 6. Token IDs, EOS/BOS/PAD (S1, S2, S3)

| Item | Value | Source |
|---|---|---|
| `bos_token_id` (config) | 151643 (`<|endoftext|>`) | S1 |
| **BOS actually used?** | **No** - `add_bos_token: false`, `bos_token: null`; the chat template never prepends one | S3 |
| `eos_token_id` (config) | 151645 (`<|im_end|>`) | S1 |
| **`eos_token_id` (generation)** | **[151645, 151643]** - a *set*; stop on either | S2 |
| `pad_token_id` (generation) | 151643 | S2 |
| `pad_token` (tokenizer) | `<|endoftext|>` (151643) | S3 |
| `unk_token` | null | S3 |
| `model_max_length` (tokenizer) | 131072 - **exceeds the model's 32768**; engine cap is 32768 (flag F5) | S3 |
| Tokenizer class | `Qwen2Tokenizer` (byte-level BPE; `vocab.json` + `merges.txt` / `tokenizer.json`) | S3 |

Special tokens: 22 added tokens occupy IDs **151643-151664** (S3), including
`<|endoftext|>` = 151643, `<|im_start|>` = 151644, `<|im_end|>` = 151645, plus tool-call and
vision markers. Regular BPE tokens occupy 0-151642. Highest producible token ID is **151664**;
embedding rows **151665-151935 (271 rows) are alignment padding** and are never produced by the
tokenizer. The logits GEMM and argmax still run over the full `vocab_size` = 151936, exactly as
HF does, so greedy comparisons remain bit-comparable.

### Generation defaults (S2) - and a token-match trap

`generation_config.json`: `do_sample: true`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`,
**`repetition_penalty: 1.1`**.

HF `generate()` applies `repetition_penalty` **even when `do_sample=False`**. Any greedy
token-match harness comparing Redline against HF must explicitly pass
`do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None`
(or strip the generation config), otherwise mismatches are guaranteed by construction.

### Chat template (S3)

`tokenizer_config.json` contains a **2,507-char Jinja chat template** (ChatML):
`<|im_start|>{role}\n{content}<|im_end|>\n` turns, default system prompt
("You are Qwen, created by Alibaba Cloud…"), tool-call support, and
`add_generation_prompt` appending `<|im_start|>assistant\n`. Per the engine contract,
templating/tokenization happen client-side; the engine consumes token IDs. The template is
recorded here so test fixtures can produce correct prompts.

## 7. Checkpoint files (S4, pinned revision)

| File | Role |
|---|---|
| `model.safetensors` | **single** weight file - there is **no** `model.safetensors.index.json` |
| `config.json` | architecture constants |
| `generation_config.json` | sampling defaults, EOS set |
| `tokenizer.json` | fast-tokenizer bundle |
| `vocab.json`, `merges.txt` | BPE vocab/merges (slow tokenizer) |
| `tokenizer_config.json` | special tokens, chat template |
| `LICENSE`, `README.md`, `.gitattributes` | not needed by the engine |

HF API reports `safetensors.parameters = {BF16: 1543714304}` - all weights BF16, matching §8.

## 8. `model.safetensors` tensor inventory (S5)

Header: 38,528-byte JSON, `__metadata__ = {"format": "pt"}`, **338 tensors, all `BF16`**,
data region 3,087,428,608 bytes. Layer indices run 0-27 and every layer has identical
shapes/dtypes (verified programmatically from the header). PyTorch `nn.Linear` convention:
`weight` shape is `[out_features, in_features]`, `y = x·Wᵀ + b`.

Per-layer pattern (`{i}` = 0…27), 12 tensors per layer:

| Tensor | Shape | dtype |
|---|---|---|
| `model.embed_tokens.weight` | `[151936, 1536]` | BF16 |
| `model.layers.{i}.input_layernorm.weight` | `[1536]` | BF16 |
| `model.layers.{i}.self_attn.q_proj.weight` | `[1536, 1536]` | BF16 |
| `model.layers.{i}.self_attn.q_proj.bias` | `[1536]` | BF16 |
| `model.layers.{i}.self_attn.k_proj.weight` | `[256, 1536]` | BF16 |
| `model.layers.{i}.self_attn.k_proj.bias` | `[256]` | BF16 |
| `model.layers.{i}.self_attn.v_proj.weight` | `[256, 1536]` | BF16 |
| `model.layers.{i}.self_attn.v_proj.bias` | `[256]` | BF16 |
| `model.layers.{i}.self_attn.o_proj.weight` | `[1536, 1536]` | BF16 |
| `model.layers.{i}.post_attention_layernorm.weight` | `[1536]` | BF16 |
| `model.layers.{i}.mlp.gate_proj.weight` | `[8960, 1536]` | BF16 |
| `model.layers.{i}.mlp.up_proj.weight` | `[8960, 1536]` | BF16 |
| `model.layers.{i}.mlp.down_proj.weight` | `[1536, 8960]` | BF16 |
| `model.norm.weight` | `[1536]` | BF16 |

There is **no** `lm_head.weight` (§5) and no other tensor outside the pattern above.

<details>
<summary>Complete 338-tensor listing (generated from the safetensors header at revision 989aa79)</summary>

| Tensor | dtype | Shape |
|---|---|---|
| `model.embed_tokens.weight` | BF16 | [151936, 1536] |
| `model.layers.0.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.0.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.0.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.0.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.0.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.0.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.0.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.0.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.0.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.0.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.0.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.0.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.1.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.1.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.1.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.1.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.1.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.1.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.1.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.1.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.1.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.1.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.1.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.1.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.2.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.2.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.2.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.2.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.2.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.2.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.2.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.2.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.2.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.2.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.2.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.2.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.3.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.3.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.3.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.3.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.3.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.3.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.3.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.3.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.3.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.3.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.3.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.3.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.4.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.4.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.4.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.4.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.4.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.4.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.4.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.4.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.4.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.4.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.4.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.4.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.5.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.5.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.5.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.5.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.5.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.5.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.5.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.5.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.5.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.5.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.5.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.5.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.6.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.6.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.6.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.6.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.6.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.6.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.6.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.6.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.6.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.6.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.6.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.6.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.7.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.7.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.7.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.7.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.7.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.7.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.7.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.7.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.7.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.7.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.7.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.7.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.8.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.8.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.8.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.8.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.8.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.8.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.8.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.8.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.8.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.8.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.8.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.8.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.9.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.9.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.9.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.9.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.9.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.9.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.9.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.9.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.9.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.9.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.9.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.9.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.10.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.10.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.10.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.10.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.10.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.10.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.10.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.10.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.10.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.10.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.10.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.10.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.11.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.11.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.11.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.11.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.11.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.11.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.11.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.11.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.11.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.11.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.11.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.11.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.12.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.12.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.12.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.12.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.12.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.12.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.12.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.12.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.12.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.12.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.12.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.12.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.13.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.13.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.13.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.13.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.13.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.13.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.13.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.13.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.13.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.13.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.13.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.13.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.14.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.14.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.14.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.14.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.14.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.14.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.14.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.14.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.14.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.14.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.14.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.14.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.15.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.15.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.15.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.15.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.15.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.15.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.15.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.15.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.15.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.15.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.15.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.15.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.16.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.16.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.16.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.16.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.16.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.16.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.16.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.16.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.16.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.16.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.16.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.16.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.17.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.17.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.17.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.17.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.17.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.17.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.17.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.17.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.17.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.17.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.17.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.17.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.18.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.18.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.18.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.18.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.18.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.18.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.18.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.18.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.18.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.18.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.18.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.18.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.19.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.19.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.19.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.19.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.19.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.19.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.19.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.19.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.19.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.19.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.19.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.19.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.20.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.20.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.20.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.20.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.20.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.20.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.20.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.20.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.20.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.20.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.20.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.20.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.21.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.21.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.21.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.21.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.21.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.21.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.21.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.21.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.21.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.21.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.21.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.21.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.22.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.22.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.22.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.22.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.22.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.22.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.22.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.22.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.22.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.22.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.22.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.22.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.23.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.23.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.23.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.23.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.23.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.23.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.23.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.23.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.23.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.23.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.23.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.23.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.24.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.24.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.24.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.24.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.24.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.24.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.24.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.24.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.24.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.24.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.24.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.24.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.25.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.25.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.25.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.25.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.25.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.25.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.25.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.25.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.25.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.25.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.25.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.25.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.26.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.26.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.26.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.26.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.26.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.26.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.26.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.26.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.26.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.26.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.26.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.26.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.layers.27.input_layernorm.weight` | BF16 | [1536] |
| `model.layers.27.mlp.down_proj.weight` | BF16 | [1536, 8960] |
| `model.layers.27.mlp.gate_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.27.mlp.up_proj.weight` | BF16 | [8960, 1536] |
| `model.layers.27.post_attention_layernorm.weight` | BF16 | [1536] |
| `model.layers.27.self_attn.k_proj.bias` | BF16 | [256] |
| `model.layers.27.self_attn.k_proj.weight` | BF16 | [256, 1536] |
| `model.layers.27.self_attn.o_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.27.self_attn.q_proj.bias` | BF16 | [1536] |
| `model.layers.27.self_attn.q_proj.weight` | BF16 | [1536, 1536] |
| `model.layers.27.self_attn.v_proj.bias` | BF16 | [256] |
| `model.layers.27.self_attn.v_proj.weight` | BF16 | [256, 1536] |
| `model.norm.weight` | BF16 | [1536] |

</details>

## 9. Derived quantities

| Quantity | Value | Derivation |
|---|---:|---|
| Parameters (stored) | **1,543,714,304** | Σ shape products over all 338 tensors; matches S4 |
| - embedding | 233,373,696 | 151936 × 1536 (counted once; tied) |
| - non-embedding | 1,310,340,608 | total − embedding |
| - per layer | 46,797,824 | Σ over `model.layers.0.*` |
| FP16 weight bytes | **3,087,428,608 B = 2.875 GiB** | params × 2 B (BF16 file is the same size) |
| KV bytes/token (FP16) | **28,672 B = 28 KiB** | 2(K,V) × 28 layers × 2 kv_heads × 128 head_dim × 2 B |
| KV bytes / 16-token block (all layers) | 458,752 B = 448 KiB | 16 × 28,672 |
| KV bytes / 16-token block (one layer, K+V) | 16,384 B = 16 KiB | 16 × 2 × 2 × 128 × 2 |
| KV pool 1.5 GiB (dev GPU config) | 56,173 tokens ≈ 3,510 blocks | pool / 28,672 |
| KV pool 16 GiB (bench GPU config) | 599,186 tokens ≈ 37,449 blocks | pool / 28,672 |

Sanity checks against engine configs: batch 8 × 4096-token contexts = 32,768 tokens = 0.875 GiB KV
(fits the 1.5 GiB dev pool); batch 64 × 1280 tokens (1K prefill + 256 decode) = 81,920 tokens =
2.24 GiB KV (trivially fits a 24 GB card next to 2.875 GiB of weights).

## 10. Flags: checkpoint facts vs. current engine assumptions

Discrepancies between this checkpoint and the assumptions in `README.md` (planned
architecture) that the implementation must resolve:

- **F1 - checkpoint is BF16, engine serves FP16.** `torch_dtype: "bfloat16"` and every tensor
  in the file is BF16. A pure "mmap and use" load is impossible for an FP16 engine: the loader
  must mmap the file, then **convert BF16→FP16 once at load** into the FP16 weight buffers.
  BF16→FP16 is exact in mantissa (8 → 10 bits) - the only edge cases are overflow
  (|w| > 65504) and values landing in FP16's subnormal range (|w| < ~6.1e-5 normal floor).
  The conversion must be **bit-identical to `torch.Tensor.to(torch.float16)`** - the operation
  HF performs when loading this BF16 checkpoint as FP16, and therefore what the greedy-parity
  reference actually runs: **round-to-nearest-even, subnormals preserved (no flush-to-zero),
  and a hard failure on any overflow** (torch produces `inf`; the loader must abort instead -
  the expected overflow count for this checkpoint is zero, so failing is free). Any clamp,
  flush, or truncation rounding yields weights that differ from the reference. The converter
  is unit-tested exhaustively over all 65,536 BF16 bit patterns against a torch-generated
  golden table. Dev GPU (sm_75) has no BF16 compute, so serving BF16 natively is not an
  option there.
- **F2 - no `lm_head.weight`.** `tie_word_embeddings: true`; loaders that expect a distinct
  output-projection tensor will fail. Alias `model.embed_tokens.weight` (§5).
- **F3 - QKV projections have biases.** The GEMM/epilogue path (and any fused QKV kernel) must
  apply per-channel bias for Q (1536), K (256), V (256) in every layer. No other linear layer
  has bias. The planned-kernel list in `README.md` does not mention bias handling.
- **F4 - EOS is a set.** Stop detection must test membership in **{151645, 151643}** (S2), not
  equality with the single `config.json` value 151645.
- **F5 - context ceiling is 32768,** even though the tokenizer advertises
  `model_max_length: 131072`. Engine max sequence length must be clamped to
  `max_position_embeddings` = 32768.
- **F6 - greedy parity harness must neutralize `generation_config.json`** (`do_sample: true`,
  `repetition_penalty: 1.1`, etc.) or HF output will not be greedy-comparable (§6).
- **F7 - `config.json` omits `head_dim` and `attention_bias`.** Both are implicit
  (128 derived; bias hardcoded in HF Qwen2 - S6). The loader should compute/assume them rather
  than require the keys.
- **F8 - dev-GPU memory budget is tight but feasible:** 2.875 GiB weights + ≤1.5 GiB KV pool
  + activations/workspaces on a 6 GB card leaves ≲1.5 GiB headroom; keep the KV pool size and
  max batch runtime-configurable (they already are in the plan) and account for
  WSL2/driver/display overhead when sizing the dev pool.
- **Confirmations (no change needed):** GQA is real (12 Q / 2 KV heads - the paged-attention
  kernel's GQA assumption holds); `head_dim` 128 suits the planned kernel tiling; sliding window
  is disabled so full causal attention is correct everywhere; single safetensors file means the
  mmap loader needs no index-file handling; RoPE is NeoX-style half-rotation with θ = 1e6 (§3).

---

*Every value in this file was fetched on 2026-07-09 from revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. If the pinned revision ever changes, re-verify this
entire file and re-run `scripts/fetch_model.py --revision <new-sha>`.*
