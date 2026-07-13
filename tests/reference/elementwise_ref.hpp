#pragma once

// FP64 CPU references for the elementwise/scan kernels of docs/DESIGN.md
// section 12a: silu_mul (section 6.2 #8) and greedy argmax (section 6.2 #9).
// Header-only, no dependency on the kernels under test.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "fp16_ref.hpp"

namespace redline::reference {

// out[t, c] = fp16(silu(gate) * up), silu(x) = x / (1 + exp(-x)), FP64 math
// on FP16-quantized inputs, one final rounding. The kernel computes silu and
// the product in FP32 (precise expf) - differences sit far below the
// section 12a elementwise tolerances.
//   gate_up: [rows, 2 * intermediate] FP16 bits (dense); gate at column c,
//            up at column intermediate + c.
//   returns: [rows, intermediate] FP16 bits (dense).
inline std::vector<std::uint16_t> SiluMulReference(const std::vector<std::uint16_t>& gate_up,
                                                   std::int32_t rows, std::int32_t intermediate) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * intermediate);
  for (std::int32_t r = 0; r < rows; ++r) {
    const std::size_t in_base = static_cast<std::size_t>(r) * 2 * intermediate;
    const std::size_t out_base = static_cast<std::size_t>(r) * intermediate;
    for (std::int32_t c = 0; c < intermediate; ++c) {
      const double g = HalfBitsToDouble(gate_up[in_base + c]);
      const double u = HalfBitsToDouble(gate_up[in_base + intermediate + c]);
      const double silu = g / (1.0 + std::exp(-g));
      out[out_base + c] = QuantizeDoubleToHalfBits(silu * u);
    }
  }
  return out;
}

// Greedy argmax with the kernel's documented semantics (section 6.2 #9 and
// the kernel header): FP32-order compares (FP16 order is preserved exactly
// by any widening, so FP64 here is equivalent), ties resolve to the LOWEST
// index (torch.argmax first occurrence), a NaN candidate is never adopted,
// an all-NaN row deterministically yields token 0 (documented deviation from
// torch, which propagates NaN as the maximum), and an all -inf row yields
// token 0 through the ordinary index tiebreak.
inline std::vector<std::int32_t> ArgmaxReference(const std::vector<std::uint16_t>& logits,
                                                 std::int32_t rows, std::int32_t vocab) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(rows));
  for (std::int32_t r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * vocab;
    double best_v = -std::numeric_limits<double>::infinity();
    std::int64_t best_i = vocab; // sentinel: ordered AFTER every real column
    for (std::int32_t i = 0; i < vocab; ++i) {
      const double v = HalfBitsToDouble(logits[base + i]);
      // Both arms are IEEE compares: false against NaN, so NaN never wins.
      if (v > best_v || (v == best_v && i < best_i)) {
        best_v = v;
        best_i = i;
      }
    }
    out[r] = (best_i >= vocab) ? 0 : static_cast<std::int32_t>(best_i);
  }
  return out;
}

} // namespace redline::reference
