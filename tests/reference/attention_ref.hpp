#pragma once

// FP64 CPU references for the attention kernels (docs/DESIGN.md section 12a):
//   - paged GQA decode attention (LaunchPagedAttentionDecode / ...Oracle),
//     computed as a textbook two-pass softmax per (sequence, query head)
//     over the CPU pool mirror - structurally unlike both GPU kernels
//     (which use online softmax / warp reductions);
//   - prefill causal softmax (LaunchPrefillSoftmax).
// Header-only, no dependency on the kernels under test.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "fp16_ref.hpp"
#include "pool_ref.hpp"

namespace redline::reference {

// Decode attention over FP16-quantized inputs, all math in FP64, one FP16
// rounding of the final normalized accumulation (the kernels' epilogue
// rounding). Rows with seq_lens[r] <= 0 (padded slots) are left as +0 bits;
// the kernels leave the real output rows untouched instead, so tests compare
// only live rows against this reference. Only positions < seq_len are read -
// the reference is NaN-clean over poisoned pools by the same select-style
// argument the kernel is tested for.
//   q_packed: [num_seqs, num_q_heads * head_dim] FP16 bits (dense)
//   pool:     one layer's slice, section 5 layout (pool_ref.hpp)
//   returns:  [num_seqs, num_q_heads * head_dim] FP16 bits (dense)
inline std::vector<std::uint16_t> PagedAttentionDecodeReference(
    const std::vector<std::uint16_t>& q_packed, const std::vector<std::uint16_t>& pool,
    const std::vector<std::int32_t>& block_tables, const std::vector<std::int32_t>& seq_lens,
    std::int32_t num_seqs, std::int32_t max_blocks_per_seq, std::int32_t num_q_heads,
    std::int32_t num_kv_heads, std::int32_t head_dim, double scale) {
  const std::int32_t group = num_q_heads / num_kv_heads;
  std::vector<std::uint16_t> out(static_cast<std::size_t>(num_seqs) * num_q_heads * head_dim, 0);
  std::vector<double> scores;
  for (std::int32_t r = 0; r < num_seqs; ++r) {
    const std::int32_t len = seq_lens[r];
    if (len <= 0)
      continue;
    const std::int32_t* table_row = &block_tables[static_cast<std::size_t>(r) * max_blocks_per_seq];
    for (std::int32_t h = 0; h < num_q_heads; ++h) {
      const std::int32_t kv_head = h / group;
      const std::size_t q_base =
          (static_cast<std::size_t>(r) * num_q_heads + h) * static_cast<std::size_t>(head_dim);
      scores.assign(static_cast<std::size_t>(len), 0.0);
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::int32_t t = 0; t < len; ++t) {
        const std::int32_t block = table_row[t / kKvBlockSize];
        const std::int32_t slot = t % kKvBlockSize;
        double dot = 0.0;
        for (std::int32_t d = 0; d < head_dim; ++d) {
          dot += HalfBitsToDouble(q_packed[q_base + d]) *
                 HalfBitsToDouble(pool[static_cast<std::size_t>(
                     PoolElemIndex(block, 0, kv_head, slot, d, num_kv_heads, head_dim))]);
        }
        scores[t] = dot * scale;
        max_score = std::max(max_score, scores[t]);
      }
      double denom = 0.0;
      for (std::int32_t t = 0; t < len; ++t) {
        scores[t] = std::exp(scores[t] - max_score); // reuse as weights
        denom += scores[t];
      }
      for (std::int32_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (std::int32_t t = 0; t < len; ++t) {
          const std::int32_t block = table_row[t / kKvBlockSize];
          const std::int32_t slot = t % kKvBlockSize;
          acc += scores[t] * HalfBitsToDouble(pool[static_cast<std::size_t>(PoolElemIndex(
                                 block, 1, kv_head, slot, d, num_kv_heads, head_dim))]);
        }
        out[q_base + d] = QuantizeDoubleToHalfBits(acc / denom);
      }
    }
  }
  return out;
}

// Prefill causal softmax (section 6.2 #7): scores FP32
// [num_q_heads, chunk_len, kv_len] dense; query row i has global position
// q = chunk_start + i and attends keys j <= q. Live entries: FP64 max/sum,
// one FP16 rounding. Masked entries: exact +0 bits, never read - matching
// the kernel's select-style contract (their score values, NaN included,
// must not influence anything).
inline std::vector<std::uint16_t>
PrefillSoftmaxReference(const std::vector<float>& scores, std::int32_t num_q_heads,
                        std::int32_t chunk_len, std::int32_t kv_len, std::int32_t chunk_start) {
  std::vector<std::uint16_t> probs(static_cast<std::size_t>(num_q_heads) * chunk_len * kv_len, 0);
  for (std::int32_t h = 0; h < num_q_heads; ++h) {
    for (std::int32_t i = 0; i < chunk_len; ++i) {
      const std::size_t base =
          (static_cast<std::size_t>(h) * chunk_len + i) * static_cast<std::size_t>(kv_len);
      const std::int32_t live = std::min(chunk_start + i + 1, kv_len);
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::int32_t j = 0; j < live; ++j) {
        max_score = std::max(max_score, static_cast<double>(scores[base + j]));
      }
      double denom = 0.0;
      for (std::int32_t j = 0; j < live; ++j) {
        denom += std::exp(static_cast<double>(scores[base + j]) - max_score);
      }
      for (std::int32_t j = 0; j < live; ++j) {
        probs[base + j] = QuantizeDoubleToHalfBits(
            std::exp(static_cast<double>(scores[base + j]) - max_score) / denom);
      }
      // j >= live stays exact +0x0000.
    }
  }
  return probs;
}

} // namespace redline::reference
