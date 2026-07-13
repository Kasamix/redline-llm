#pragma once

// FP64 CPU reference for LaunchRopeInplace (docs/DESIGN.md section 6.2 #3,
// docs/MODEL_SPEC.md section 3). Header-only, no dependency on the kernel
// under test.
//
// Convention pinned here: GPT-NeoX half-rotation - element d of each head
// pairs with element d + head_dim/2 (HF Qwen2 rotate_half), NOT the GPT-J
// interleaved (2d, 2d+1) pairing. The committed HF-captured fixture
// (tests/data/rope_fixture.json) pins the same decision to captured
// evidence; this reference must agree with that fixture within the
// section 12a elementwise tolerances, which the test suite checks CPU-only.
//
// Rounding chain: HF casts its FP32 cos/sin cache to the activation dtype at
// use and applies q*cos + rotate_half(q)*sin as FP16 elementwise torch ops
// (FP32 opmath, one correctly rounded FP16 result per op). The reference
// reproduces the three FP16 rounding points - cos/sin, each product, the
// final sum - but computes angles and trig in FP64 where the kernel uses
// FP32 (positions * inv_freq, sincosf): those last-ulp FP32 differences can
// flip an FP16 rounding only when the trig value sits within ~1e-7 relative
// of a boundary, and the resulting one-ulp output differences sit far below
// the section 12a tolerances (max-abs 1e-2 / mean-abs 1e-3).
//
// Sign note: HF's first-half term is q1*cos + (-q2)*sin; fp16((-x2) * s) ==
// -fp16(x2 * s) exactly (round-to-nearest is sign-symmetric), so the
// reference folds the sign into the final subtraction.

#include <cmath>
#include <cstdint>
#include <vector>

#include "fp16_ref.hpp"

namespace redline::reference {

// heads: packed [num_tokens, num_heads * head_dim] FP16 bits, rotated in
// place - the caller extracts strided GPU views into packed CPU arrays
// first. Every head in the buffer is rotated (callers pass Q and K
// separately or as one packed q||k buffer with num_heads = q + kv heads).
inline void RopeReferenceInplace(std::vector<std::uint16_t>& heads,
                                 const std::vector<std::int32_t>& positions,
                                 std::int32_t num_tokens, std::int32_t num_heads,
                                 std::int32_t head_dim, double theta) {
  const std::int32_t half = head_dim / 2;
  for (std::int32_t t = 0; t < num_tokens; ++t) {
    const double pos = static_cast<double>(positions[t]);
    for (std::int32_t h = 0; h < num_heads; ++h) {
      const std::size_t base =
          (static_cast<std::size_t>(t) * num_heads + h) * static_cast<std::size_t>(head_dim);
      for (std::int32_t d = 0; d < half; ++d) {
        // inv_freq[d] = theta^(-2d/head_dim) (MODEL_SPEC section 3).
        const double inv_freq =
            std::pow(theta, -2.0 * static_cast<double>(d) / static_cast<double>(head_dim));
        const double angle = pos * inv_freq;
        // FP16 rounding point 1: cos/sin cast to the activation dtype.
        const double c = RoundDoubleToHalfValue(std::cos(angle));
        const double s = RoundDoubleToHalfValue(std::sin(angle));
        // THE CONVENTION SWITCH POINT (reference side): pair (d, d + half).
        const double x1 = HalfBitsToDouble(heads[base + d]);
        const double x2 = HalfBitsToDouble(heads[base + d + half]);
        // FP16 rounding point 2: each product rounds.
        const double x1c = RoundDoubleToHalfValue(x1 * c);
        const double x2s = RoundDoubleToHalfValue(x2 * s);
        const double x2c = RoundDoubleToHalfValue(x2 * c);
        const double x1s = RoundDoubleToHalfValue(x1 * s);
        // FP16 rounding point 3: the sum rounds once.
        heads[base + d] = QuantizeDoubleToHalfBits(x1c - x2s);
        heads[base + d + half] = QuantizeDoubleToHalfBits(x2c + x1s);
      }
    }
  }
}

} // namespace redline::reference
