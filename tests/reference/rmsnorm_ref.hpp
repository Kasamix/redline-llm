#pragma once

// FP64 CPU references for LaunchRmsNorm / LaunchRmsNormResidual
// (docs/DESIGN.md section 6.2 #2, section 12a), including the half-ULP
// rounding-boundary fixture and the two deliberately wrong "mutant"
// references that keep that fixture honest. Header-only, no dependency on
// the kernels under test.
//
// Operation order being pinned (HF Qwen2 semantics):
//   fused:  h   = fp16(r + x)            ROUNDING POINT 1 - the mean-square
//           ms  = mean(fp64(h)^2)          runs over the ROUNDED residual
//           inv = 1/sqrt(ms + eps)
//           n   = fp16(fp64(h) * inv)    ROUNDING POINT 2 - norm rounds
//           out = fp16(fp64(n) * fp64(w))  BEFORE the weight multiply
//   plain:  same tail with h replaced by the raw input (no add, no
//           write-back).
//
// The final n * w product of the correct order is exact in FP32 on both
// sides (11-bit x 11-bit significands), so its single rounding here is
// bit-identical to the kernel's; the only noise-sensitive rounding is
// fp16(h64 * inv), where the kernel's FP32-accumulated inv_rms differs from
// this reference's FP64 one by ~1e-6 relative - the boundary fixture
// therefore excludes columns whose pre-rounding normalized value sits within
// kBoundaryExclusionRelFloor (relative) of an FP16 rounding boundary, and
// bit-exactness is asserted only on the rest.

#include <cmath>
#include <cstdint>
#include <vector>

#include "fp16_ref.hpp"

namespace redline::reference {

// --------------------------------------------------------------------- plain

inline std::vector<std::uint16_t> RmsNormReference(const std::vector<std::uint16_t>& input,
                                                   const std::vector<std::uint16_t>& weight,
                                                   int rows, int hidden, float eps) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * hidden);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * hidden;
    double sum_sq = 0.0;
    for (int c = 0; c < hidden; ++c) {
      const double x = HalfBitsToDouble(input[base + c]);
      sum_sq += x * x;
    }
    const double inv =
        1.0 / std::sqrt(sum_sq / static_cast<double>(hidden) + static_cast<double>(eps));
    for (int c = 0; c < hidden; ++c) {
      const double x = HalfBitsToDouble(input[base + c]);
      const double normed = HalfBitsToDouble(QuantizeDoubleToHalfBits(x * inv));
      out[base + c] = QuantizeDoubleToHalfBits(normed * HalfBitsToDouble(weight[c]));
    }
  }
  return out;
}

// --------------------------------------------------------------------- fused

struct FusedRmsNormReference {
  std::vector<std::uint16_t> out_bits;      // [rows * hidden]
  std::vector<std::uint16_t> residual_bits; // written-back h = fp16(r + x), [rows * hidden]
  std::vector<double> inv_rms;              // per row, pre-rounding (diagnostics)
  std::vector<char> excluded;               // per element; only filled when
                                            // rel_exclusion_floor > 0
};

// Correct-order fused reference. When rel_exclusion_floor > 0, marks every
// element whose pre-rounding normalized value h64 * inv_rms lies within that
// RELATIVE distance of an FP16 rounding boundary (see header comment).
inline FusedRmsNormReference
ComputeFusedRmsNormReference(const std::vector<std::uint16_t>& residual_in,
                             const std::vector<std::uint16_t>& input,
                             const std::vector<std::uint16_t>& weight, int rows, int hidden,
                             float eps, double rel_exclusion_floor = 0.0) {
  const std::size_t total = static_cast<std::size_t>(rows) * hidden;
  FusedRmsNormReference ref;
  ref.out_bits.resize(total);
  ref.residual_bits.resize(total);
  ref.inv_rms.resize(static_cast<std::size_t>(rows));
  ref.excluded.assign(total, 0);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * hidden;
    double sum_sq = 0.0;
    for (int c = 0; c < hidden; ++c) {
      // ROUNDING POINT 1: the residual sum is rounded to FP16 before the
      // mean-square consumes it. HalfAddRn == fp16(fp32(r) + fp32(x)) ==
      // native half add (innocuous double rounding; fp16_ref.hpp).
      const std::uint16_t h = HalfAddRn(residual_in[base + c], input[base + c]);
      ref.residual_bits[base + c] = h;
      const double h64 = HalfBitsToDouble(h);
      sum_sq += h64 * h64;
    }
    const double inv =
        1.0 / std::sqrt(sum_sq / static_cast<double>(hidden) + static_cast<double>(eps));
    ref.inv_rms[r] = inv;
    for (int c = 0; c < hidden; ++c) {
      const double h64 = HalfBitsToDouble(ref.residual_bits[base + c]);
      const double v = h64 * inv; // pre-rounding normalized value
      // ROUNDING POINT 2: normalize rounds to FP16 before the weight multiply.
      const double normed = HalfBitsToDouble(QuantizeDoubleToHalfBits(v));
      ref.out_bits[base + c] = QuantizeDoubleToHalfBits(normed * HalfBitsToDouble(weight[c]));
      if (rel_exclusion_floor > 0.0 && RelDistanceToHalfBoundary(v) < rel_exclusion_floor) {
        ref.excluded[base + c] = 1;
      }
    }
  }
  return ref;
}

// ------------------------------------------------------------------- mutants
//
// Two deliberately wrong operation orders. The boundary fixture asserts that
// each of them disagrees with the correct reference on at least
// kBoundaryMinMutantFlips non-excluded columns per row - if a future edit
// makes the fixture blind to either error, the assert (not a silent green)
// fires. Verified at authoring time by a bit-exact numpy simulation of this
// exact construction.

// Mutant 1: normalizes the UNROUNDED sum - both the mean-square and the
// numerator consume the raw r64 + x64 (this is the "systematic drift across
// all 57 norm sites" failure of docs/DESIGN.md section 6.2). The
// norm-before-weight rounding is kept.
inline std::vector<std::uint16_t> FusedRmsNormMutantUnroundedSum(
    const std::vector<std::uint16_t>& residual_in, const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& weight, int rows, int hidden, float eps) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * hidden);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * hidden;
    double sum_sq = 0.0;
    for (int c = 0; c < hidden; ++c) {
      const double s = HalfBitsToDouble(residual_in[base + c]) +
                       HalfBitsToDouble(input[base + c]); // unrounded - the defect
      sum_sq += s * s;
    }
    const double inv =
        1.0 / std::sqrt(sum_sq / static_cast<double>(hidden) + static_cast<double>(eps));
    for (int c = 0; c < hidden; ++c) {
      const double s = HalfBitsToDouble(residual_in[base + c]) + HalfBitsToDouble(input[base + c]);
      const double normed = HalfBitsToDouble(QuantizeDoubleToHalfBits(s * inv));
      out[base + c] = QuantizeDoubleToHalfBits(normed * HalfBitsToDouble(weight[c]));
    }
  }
  return out;
}

// Mutant 2: keeps the rounded residual and its mean-square but SINGLE-rounds
// the tail - out = fp16(h64 * inv * w64) - skipping the norm-before-weight
// rounding point.
inline std::vector<std::uint16_t> FusedRmsNormMutantSingleRounding(
    const std::vector<std::uint16_t>& residual_in, const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& weight, int rows, int hidden, float eps) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * hidden);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * hidden;
    double sum_sq = 0.0;
    std::vector<double> h64(static_cast<std::size_t>(hidden));
    for (int c = 0; c < hidden; ++c) {
      h64[c] = HalfBitsToDouble(HalfAddRn(residual_in[base + c], input[base + c]));
      sum_sq += h64[c] * h64[c];
    }
    const double inv =
        1.0 / std::sqrt(sum_sq / static_cast<double>(hidden) + static_cast<double>(eps));
    for (int c = 0; c < hidden; ++c) {
      out[base + c] =
          QuantizeDoubleToHalfBits(h64[c] * inv * HalfBitsToDouble(weight[c])); // one round
    }
  }
  return out;
}

// ----------------------------------------------------- boundary fixture (12a #2)
//
// Construction per the reviewed derivation (residual variant): rows are
// NON-constant (RMSNorm of a constant row is scale-invariant, which makes
// whole-row boundary fixtures blind to both mutants) and each row's boundary
// columns share ONE tie direction (mixing pair kinds cancels the
// unrounded-vs-rounded mean-square shift to below noise).
//
//   pair A: residual 0x3C00 (1.0)          + input 0x1000 (2^-11)
//           -> exact midpoint of [0x3C00, 0x3C01]; RN-even rounds DOWN.
//   pair B: residual 0x3C01 (1.0009765625) + input 0x1000
//           -> exact midpoint of [0x3C01, 0x3C02]; RN-even rounds UP.
//
// Both FP64 sums are exact, isolating the FP16 rounding decision. Filler
// columns: residual ~ U(0.25, 4) quantized to FP16, input +0.0 - the filler
// sums are exact, so no rounding decision exists outside the boundary
// columns, while the varied magnitudes break the scale invariance that blinds
// constant rows.

inline constexpr int kBoundaryFixtureRows = 8; // 2 pair kinds x 4 filler streams
inline constexpr int kBoundaryFixtureHidden = 1536;
inline constexpr int kBoundaryColumnsPerRow = 256;
inline constexpr std::uint16_t kPairAResidualBits = 0x3C00;         // 1.0
inline constexpr std::uint16_t kPairBResidualBits = 0x3C01;         // 1.0 + 2^-10
inline constexpr std::uint16_t kBoundaryInputBits = 0x1000;         // 2^-11
inline constexpr std::uint16_t kPairAExpectedResidualBits = 0x3C00; // tie -> even (down)
inline constexpr std::uint16_t kPairBExpectedResidualBits = 0x3C02; // tie -> even (up)
inline constexpr double kBoundaryExclusionRelFloor = 1e-5; // >= 10x above FP32-vs-FP64 noise
inline constexpr int kBoundaryMinMutantFlips = 16;         // per row, non-excluded columns

struct RmsNormBoundaryFixture {
  std::vector<std::uint16_t> residual; // [rows * hidden]
  std::vector<std::uint16_t> input;    // [rows * hidden]
  std::vector<std::uint16_t> weight;   // [hidden]
  std::vector<char> is_boundary;       // [rows * hidden]
  std::vector<char> is_pair_b;         // [rows]; false = pair A
};

// Deterministic pure function of `seed`; mirrored bit-for-bit by the numpy
// simulation that pinned the committed seed (splitmix64 streams + IEEE
// float64 arithmetic + RN-even FP16 quantization on both sides).
inline RmsNormBoundaryFixture BuildRmsNormBoundaryFixture(std::uint64_t seed) {
  constexpr int rows = kBoundaryFixtureRows;
  constexpr int hidden = kBoundaryFixtureHidden;
  RmsNormBoundaryFixture fx;
  fx.residual.assign(static_cast<std::size_t>(rows) * hidden, 0);
  fx.input.assign(static_cast<std::size_t>(rows) * hidden, 0);
  fx.weight.resize(hidden);
  fx.is_boundary.assign(static_cast<std::size_t>(rows) * hidden, 0);
  fx.is_pair_b.resize(rows);
  for (int r = 0; r < rows; ++r) {
    const bool pair_b = (r % 2) == 1;
    fx.is_pair_b[r] = pair_b ? 1 : 0;
    const std::size_t base = static_cast<std::size_t>(r) * hidden;
    // Boundary columns interleave through the row: c = 6*i + (r mod 6),
    // i < 256 (max index 6*255 + 5 = 1535).
    for (int i = 0; i < kBoundaryColumnsPerRow; ++i) {
      fx.is_boundary[base + 6 * i + (r % 6)] = 1;
    }
    SplitMix64 stream(seed + static_cast<std::uint64_t>(r));
    for (int c = 0; c < hidden; ++c) {
      if (fx.is_boundary[base + c]) {
        fx.residual[base + c] = pair_b ? kPairBResidualBits : kPairAResidualBits;
        fx.input[base + c] = kBoundaryInputBits;
      } else {
        // Draw order matters for the cross-language mirror: one draw per
        // filler column, ascending c.
        fx.residual[base + c] = stream.NextHalfBitsUniform(0.25, 4.0);
        fx.input[base + c] = 0x0000; // exact filler sums: h == residual, no rounding decision
      }
    }
  }
  SplitMix64 wstream(seed + 1000003ull);
  for (int c = 0; c < hidden; ++c) {
    fx.weight[c] = wstream.NextHalfBitsUniform(0.5, 2.0);
  }
  return fx;
}

} // namespace redline::reference
