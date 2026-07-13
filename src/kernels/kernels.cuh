#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Host-side launch wrappers for every custom device kernel. Implementations
// live in the sibling .cu files; these are plain host functions so non-CUDA
// translation units can call them without kernel syntax.
//
// Conventions
//   - All pointers are device pointers.
//   - All launches are asynchronous on `stream`; no internal synchronization.
//   - Tensors are row-major with the innermost dimension last.
//   - Every 2-D activation argument carries an explicit row stride (in
//     elements). This is the strided-view rule of docs/DESIGN.md section 5:
//     the fused QKV GEMM writes rows laid out as Q(12x128)|K(2x128)|V(2x128),
//     and downstream kernels read Q/K/V *in place* as views into that buffer
//     - q = qkv_out + 0, k = qkv_out + 1536, v = qkv_out + 1792, all with row
//     stride 2048 - so no split/repack kernel exists on the decode path.
//     Dense-packed tensors simply pass stride == row width.
//   - Numerics: FP16 storage, FP32 accumulation inside every reduction.
//   - Paged KV pool: one allocation at init, layer-outer
//     (docs/DESIGN.md section 5). Kernels receive one layer's slice:
//       [num_blocks, 2 (0 = K, 1 = V), num_kv_heads, kKvBlockSize = 16, head_dim]
//     Block-table indirection follows the PagedAttention design (Kwon et
//     al., 2023, arXiv:2309.06180); all kernel implementations are original.
//   - Device block-table rows are padded with the dummy block id (block 0,
//     reserved at init) so speculative address math stays in bounds;
//     kInvalidBlockId is a host-only debug sentinel and never reaches these
//     kernels (docs/DESIGN.md section 9 sentinel policy).

namespace redline::kernels {

// Token-embedding gather - the first launch of every decode step and prefill
// chunk (docs/DESIGN.md section 6.2 #1):
//   out[t, :] = embed[input_ids[t], :]
// One 256-thread block copies each row, half2-vectorized (hidden_size 1536 =
// 3 KiB per row); no shared memory, no reduction.
//   out:       [num_tokens, hidden_size], rows spaced out_row_stride elems
//   embed:     [vocab_size, hidden_size] dense device weight table (weights
//              carry no stride parameter under the strided-view rule)
//   input_ids: [num_tokens] token ids (int32), each in [0, vocab_size) -
//              range-validated at request admission, never here. Padded
//              batch slots carry token id 0 and harmlessly gather embedding
//              row 0; there is no seq_lens early-exit in this kernel.
void LaunchEmbedGather(half* out, std::int64_t out_row_stride, const half* embed,
                       const std::int32_t* input_ids, std::int32_t num_tokens,
                       std::int32_t hidden_size, cudaStream_t stream);

// out[t, :] = rmsnorm(input[t, :]) * weight, with FP32 accumulation.
//   input, out: [num_tokens, hidden_size]; weight: [hidden_size].
void LaunchRmsNorm(half* out, const half* input, const half* weight, std::int32_t num_tokens,
                   std::int32_t hidden_size, float eps, cudaStream_t stream);

// Fused residual-add + RMSNorm (the per-block norm pattern of the model),
// reproducing HF's operation order exactly (docs/DESIGN.md section 6.2):
//   h = fp16(fp32(residual[t]) + fp32(input[t]))   // == native half add;
//   residual[t] = h                                 // written back in place
//   mean_sq accumulated over fp32(h)                // the *rounded* residual,
//                                                   // never the raw FP32 sum
//   out[t] = fp16(fp32(h) * rsqrt(mean_sq + eps)) * weight
// Both FP16 roundings (residual before the mean-square; norm before the
// weight multiply) are load-bearing for HF token parity.
void LaunchRmsNormResidual(half* out, half* residual, const half* input, const half* weight,
                           std::int32_t num_tokens, std::int32_t hidden_size, float eps,
                           cudaStream_t stream);

// Rotary position embeddings applied in place to Q and K.
// NeoX-style half-rotation: element i pairs with i + head_dim/2, matching the
// HF Qwen2 rotate_half path (convention and inv_freq formula verified in
// docs/MODEL_SPEC.md section 3; angles computed in FP32 like HF).
//   q: [num_tokens, num_q_heads * head_dim]  rows spaced q_row_stride elems
//   k: [num_tokens, num_kv_heads * head_dim] rows spaced k_row_stride elems
//   positions: [num_tokens] absolute positions (int32)
// Canonical decode call: q = qkv_out, k = qkv_out + 1536, both strides 2048.
void LaunchRopeInplace(half* q, half* k, std::int64_t q_row_stride, std::int64_t k_row_stride,
                       const std::int32_t* positions, std::int32_t num_tokens,
                       std::int32_t num_q_heads, std::int32_t num_kv_heads, std::int32_t head_dim,
                       float rope_theta, cudaStream_t stream);

// Scatter this step's K/V vectors into one layer's paged pool. The pool slot
// is derived in-kernel - p = positions[t], block = block_table_row[p >> 4],
// slot = p & 15 - so no host-computed slot-index array exists and the CUDA
// graph replay uploads stay at the four arrays of docs/DESIGN.md section 5.
//   k, v: [num_tokens, num_kv_heads * head_dim], rows spaced k/v_row_stride
//         (decode: views into qkv_out at +1536 / +1792, stride 2048; K
//         already carries RoPE)
//   block_tables: int32 rows of physical block ids;
//         block_table_row_stride selects the mapping -
//           decode:  row t belongs to token t (stride = max_blocks_per_seq)
//           prefill: all chunk tokens share one sequence's row (stride = 0)
//   positions: [num_tokens] absolute positions (int32)
void LaunchKvScatter(half* kv_pool_layer, const half* k, const half* v, std::int64_t k_row_stride,
                     std::int64_t v_row_stride, const std::int32_t* block_tables,
                     std::int64_t block_table_row_stride, const std::int32_t* positions,
                     std::int32_t num_tokens, std::int32_t num_kv_heads, std::int32_t head_dim,
                     cudaStream_t stream);

// Gather one sequence's cached K/V from the paged pool into dense staging
// buffers - the prefill-only inverse of LaunchKvScatter (docs/DESIGN.md
// section 6.2 #5; never inside a CUDA graph capture). Positions [0, ctx_len)
// are copied per KV head; the slot derives in-kernel from the destination
// row itself - block = block_table_row[t >> 4], slot = t & 15 - so no
// positions array and no host-computed slot array exist. ctx_len need not
// be a multiple of the block size (the final block may be partial). Copies
// are pure bit moves: scatter-then-gather of the same positions is
// bit-exact.
//   khat, vhat: [num_kv_heads, ctx_len, head_dim] staging views - head
//        plane h starts at h * *_head_stride and token rows are spaced
//        *_row_stride elements. Canonical prefill call: row stride
//        head_dim (the dense tiles the section 6.3 GEMM recipes read with
//        lda = head_dim) and head stride max_seq_len * head_dim (the
//        allocated plane spacing of the khat/vhat scratch, section 5).
//   block_table_row: the ONE sequence's row of physical block ids; must
//        cover ceil(ctx_len / 16) entries (every gathered position was
//        previously scattered, so its block is allocated and mapped)
void LaunchKvGather(half* khat, half* vhat, std::int64_t khat_head_stride,
                    std::int64_t khat_row_stride, std::int64_t vhat_head_stride,
                    std::int64_t vhat_row_stride, const half* kv_pool_layer,
                    const std::int32_t* block_table_row, std::int32_t ctx_len,
                    std::int32_t num_kv_heads, std::int32_t head_dim, cudaStream_t stream);

// Paged grouped-query decode attention: one query token per sequence attends
// to that sequence's full cached history through its block table. Primary v1
// mapping is KV-head-centric - grid (num_seqs, num_kv_heads), each CTA
// serving the num_q_heads/num_kv_heads query heads of its group so K/V tiles
// stream from DRAM once, not once per Q head (docs/DESIGN.md section 6.2).
// Online softmax with FP32 running max/denominator and FP32 accumulation.
// Masking is select-style (pos < seq_len ? dot : -inf), so garbage or NaN in
// pool slots beyond seq_len can never contaminate results.
// Qwen2.5-1.5B shape: 12 Q heads sharing 2 KV heads (6:1), head_dim 128.
//   out: [num_seqs, num_q_heads * head_dim], rows spaced out_row_stride
//   q:   [num_seqs, num_q_heads * head_dim], rows spaced q_row_stride
//        (decode: q = qkv_out, stride 2048; out = attn_out, stride 1536)
//   block_tables: [num_seqs, max_blocks_per_seq], padded with the dummy block
//   seq_lens: [num_seqs] tokens in cache per sequence (including this
//        step's); 0 marks a padded slot and the CTA exits immediately
//   scale: 1 / sqrt(head_dim)
void LaunchPagedAttentionDecode(half* out, std::int64_t out_row_stride, const half* q,
                                std::int64_t q_row_stride, const half* kv_pool_layer,
                                const std::int32_t* block_tables, const std::int32_t* seq_lens,
                                std::int32_t num_seqs, std::int32_t max_blocks_per_seq,
                                std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                std::int32_t head_dim, float scale, cudaStream_t stream);

// Test oracle: same contract as LaunchPagedAttentionDecode, naive per-Q-head
// mapping (grid (num_seqs, num_q_heads); each CTA re-streams its group's
// K/V, ~6x the DRAM traffic). Exists ONLY so unit tests can cross-check the
// primary kernel (docs/DESIGN.md section 12a); never called on the hot path.
void LaunchPagedAttentionDecodeOracle(half* out, std::int64_t out_row_stride, const half* q,
                                      std::int64_t q_row_stride, const half* kv_pool_layer,
                                      const std::int32_t* block_tables,
                                      const std::int32_t* seq_lens, std::int32_t num_seqs,
                                      std::int32_t max_blocks_per_seq, std::int32_t num_q_heads,
                                      std::int32_t num_kv_heads, std::int32_t head_dim, float scale,
                                      cudaStream_t stream);

// Prefill causal softmax - the custom middle stage of the chunked-prefill
// attention composite (docs/DESIGN.md sections 6.2 #7 and 6.3):
//   kv_gather -> scores GEMMs -> prefill softmax -> PV GEMMs.
// The scores and PV GEMMs run as strided-batched cuBLASLt calls owned by the
// executor's prefill composite, which calls this kernel between them.
//   scores: FP32 [num_q_heads, chunk_len, kv_len], densely packed (head
//           plane stride chunk_len * kv_len, query-row stride kv_len) - the
//           layout the scores GEMMs write (ldd = kv_len, batch stride
//           chunk_len * kv_len), already scaled by 1/sqrt(head_dim)
//   probs:  FP16, identical shape and packing - the layout the PV GEMMs
//           read. Dense by construction of those GEMM descriptors, so
//           neither tensor carries a stride parameter.
// Causal mask: the gathered khat/vhat hold key positions 0 .. kv_len-1 of
// the one sequence being prefilled, and query row i has global position
// q = chunk_start + i, so key j is live iff j <= q. Masking is select-style
// (masked entries are never read; their probs are written as exact zeros so
// the PV GEMMs' full-kv_len contraction drops them). kv_len > chunk_len is
// the normal case for every chunk after the first - chunk n attends to all
// previously scattered positions plus itself. FP32 max/sum per row, one
// FP16 rounding of the normalized value; one 256-thread block per
// (query row, head). The composite always calls with
// chunk_start + chunk_len == kv_len; the launcher debug-asserts the weaker
// chunk_start + chunk_len <= kv_len, under which every query attends at
// least to its own position - a fully masked row is impossible by
// construction.
void LaunchPrefillSoftmax(half* probs, const float* scores, std::int32_t num_q_heads,
                          std::int32_t chunk_len, std::int32_t kv_len, std::int32_t chunk_start,
                          cudaStream_t stream);

// SwiGLU activation over the fused gate+up GEMM output:
//   out[t, i] = silu(gate_up[t, i]) * gate_up[t, intermediate_size + i]
//   gate_up: [num_tokens, 2 * intermediate_size]; out: [num_tokens, intermediate_size]
void LaunchSiluMul(half* out, const half* gate_up, std::int32_t num_tokens,
                   std::int32_t intermediate_size, cudaStream_t stream);

// Greedy sampling: per-sequence argmax over the vocab dimension. FP32
// compares; ties resolve to the lowest index (matches HF greedy decoding).
//   logits: [num_seqs, vocab_size]; out_tokens: [num_seqs]
void LaunchGreedyArgmax(std::int32_t* out_tokens, const half* logits, std::int32_t num_seqs,
                        std::int32_t vocab_size, cudaStream_t stream);

} // namespace redline::kernels
