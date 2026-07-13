#pragma once

// IEEE-754 binary16 bit-level helpers for the kernel unit-test references
// (docs/DESIGN.md section 12a). Header-only, pure host C++ - deliberately
// independent of the kernels under test AND of CUDA's conversion intrinsics,
// so the references remain a second opinion rather than a mirror. The test
// binary cross-checks these helpers against the CUDA host conversions once
// (all 65,536 bit patterns round-trip; crafted rounding-boundary doubles),
// which pins the two implementations to each other without either depending
// on the other.
//
// Conventions:
//   - FP16 values travel as std::uint16_t bit patterns ("bits").
//   - bits -> double is exact (binary16 embeds losslessly in binary64).
//   - double -> bits rounds to nearest, ties to even, subnormals exact
//     (no flush-to-zero), overflow (|x| >= 65520) to the signed infinity,
//     NaN to the canonical quiet NaN 0x7E00.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace redline::reference {

inline constexpr std::uint16_t kHalfSignMask = 0x8000u;
inline constexpr std::uint16_t kHalfExpMask = 0x7C00u;
inline constexpr std::uint16_t kHalfManMask = 0x03FFu;
inline constexpr std::uint16_t kHalfPosInfBits = 0x7C00u;
inline constexpr std::uint16_t kHalfCanonicalNanBits = 0x7E00u;
inline constexpr std::uint16_t kHalfOneBits = 0x3C00u;
inline constexpr std::uint16_t kHalfMaxFiniteBits = 0x7BFFu; // 65504
// Smallest |x| that rounds to infinity under RN-even: the midpoint between
// 65504 (mantissa odd) and the would-be 65536 (mantissa even) ties *away*
// from the finite lattice, so 65520 itself already overflows.
inline constexpr double kHalfOverflowThreshold = 65520.0;

inline bool IsNanHalfBits(std::uint16_t bits) {
  return (bits & kHalfExpMask) == kHalfExpMask && (bits & kHalfManMask) != 0;
}

inline bool IsInfHalfBits(std::uint16_t bits) {
  return (bits & kHalfExpMask) == kHalfExpMask && (bits & kHalfManMask) == 0;
}

// Exact widening: every binary16 value is representable in binary64.
inline double HalfBitsToDouble(std::uint16_t bits) {
  const double sign = (bits & kHalfSignMask) ? -1.0 : 1.0;
  const int exp_field = (bits >> 10) & 0x1F;
  const int man = bits & kHalfManMask;
  if (exp_field == 0x1F) {
    if (man != 0)
      return std::numeric_limits<double>::quiet_NaN();
    return sign * std::numeric_limits<double>::infinity();
  }
  if (exp_field == 0) {
    return sign * std::ldexp(static_cast<double>(man), -24); // subnormal: man * 2^-24
  }
  // normal: (1024 + man) * 2^(exp_field - 15 - 10)
  return sign * std::ldexp(static_cast<double>(1024 + man), exp_field - 25);
}

// Round a binary64 value to binary16, nearest-even, in ONE rounding step.
// (Rounding double -> float -> half instead would double-round; FP32's 24-bit
// significand meets the p2 >= 2*p1 + 2 innocuous-double-rounding bound for
// FP16 sums/products but not for arbitrary doubles, so this path stays
// single-step on principle.)
inline std::uint16_t QuantizeDoubleToHalfBits(double x) {
  if (std::isnan(x))
    return kHalfCanonicalNanBits;
  const std::uint16_t sign = std::signbit(x) ? kHalfSignMask : 0u;
  const double a = std::fabs(x);
  if (a >= kHalfOverflowThreshold)
    return static_cast<std::uint16_t>(sign | kHalfPosInfBits);
  if (a == 0.0)
    return sign;

  // Determine the target lattice: results are integer multiples of
  // 2^ulp_exp; subnormal band (|x| < 2^-14) has ulp 2^-24, the normal band
  // [2^e, 2^(e+1)) has ulp 2^(e-10).
  int ulp_exp;
  if (a < std::ldexp(1.0, -14)) {
    ulp_exp = -24;
  } else {
    ulp_exp = std::ilogb(a) - 10;
  }

  // q = a / 2^ulp_exp exactly (power-of-two scaling; q < 2^41, exact in
  // binary64). Round q to an integer, ties to even, with exact arithmetic:
  // floor is exact and q - floor(q) is exact by Sterbenz.
  const double q = std::ldexp(a, -ulp_exp);
  const double f = std::floor(q);
  const double frac = q - f;
  std::int64_t r = static_cast<std::int64_t>(f);
  if (frac > 0.5) {
    r += 1;
  } else if (frac == 0.5) {
    r += (r & 1); // tie: round to the even integer
  }

  if (ulp_exp == -24) {
    // Subnormal band. r in [0, 1024]; r == 1024 lands exactly on the minimum
    // normal 2^-14, whose bit pattern 0x0400 the arithmetic below produces
    // naturally (exponent field 1, mantissa 0).
    return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(r));
  }
  // Normal band: r in [1024, 2048]; r == 2048 carries into the next binade.
  int e = ulp_exp + 10; // exponent of the leading bit
  if (r == 2048) {
    r = 1024;
    e += 1;
  }
  if (e > 15)
    return static_cast<std::uint16_t>(sign | kHalfPosInfBits); // unreachable: a < 65520
  return static_cast<std::uint16_t>(sign |
                                    static_cast<std::uint16_t>(((e + 15) << 10) | (r - 1024)));
}

// Convenience: value of x after one FP16 round trip.
inline double RoundDoubleToHalfValue(double x) {
  return HalfBitsToDouble(QuantizeDoubleToHalfBits(x));
}

// fp16(a + b) for FP16 operands, single rounding of the exact sum. This is
// bit-identical to the kernel-side fp16(fp32(a) + fp32(b)): the FP32 sum of
// two FP16 values double-rounds innocuously (binary32's p = 24 meets the
// p2 >= 2*p1 + 2 bound for p1 = 11), and both equal the native half add.
inline std::uint16_t HalfAddRn(std::uint16_t a, std::uint16_t b) {
  return QuantizeDoubleToHalfBits(HalfBitsToDouble(a) + HalfBitsToDouble(b));
}

// Relative distance from finite nonzero v to the NEAREST FP16 rounding
// boundary (the midpoint between two adjacent representable magnitudes).
// Used by the fused-rmsnorm boundary fixture to exclude columns whose
// pre-rounding normalized value sits so close to a boundary that benign
// FP32-vs-FP64 noise (reduction order, rsqrtf ulps) could legitimately flip
// the rounded bit (docs/DESIGN.md section 12a mandatory case 2). Returns 0
// for degenerate inputs (zero, non-finite, FP16 overflow) so callers exclude
// them.
inline double RelDistanceToHalfBoundary(double v) {
  const double a = std::fabs(v);
  if (!std::isfinite(a) || a == 0.0)
    return 0.0;
  const std::uint16_t q = QuantizeDoubleToHalfBits(a); // sign bit clear
  if ((q & kHalfExpMask) == kHalfExpMask)
    return 0.0; // overflowed to inf
  const double qv = HalfBitsToDouble(q);
  // Midpoint below: between q and its next-smaller magnitude (below the
  // smallest subnormal the neighbor is zero, handled by q == 0 -> qv/... ).
  const double m_lo = (q == 0) ? -std::ldexp(1.0, -25) // midpoint of [-min_subnormal, 0] band edge
                               : 0.5 * (qv + HalfBitsToDouble(static_cast<std::uint16_t>(q - 1)));
  // Midpoint above: between q and its next-larger magnitude; above the max
  // finite value the boundary is the overflow threshold itself.
  const double m_hi = (q == kHalfMaxFiniteBits)
                          ? kHalfOverflowThreshold
                          : 0.5 * (qv + HalfBitsToDouble(static_cast<std::uint16_t>(q + 1)));
  const double dist = std::min(std::fabs(a - m_lo), std::fabs(a - m_hi));
  return dist / a;
}

// ---------------------------------------------------------------------------
// Deterministic cross-language RNG: the stateful splitmix64 stream, using the
// same mixer constants as the project's normative workload generator
// (bench/workload.py). Hand-rolled so the numpy simulation that pinned the
// rmsnorm boundary-fixture seed reproduces the exact same streams - stdlib
// and numpy RNGs can never produce identical sequences across languages.
// ---------------------------------------------------------------------------
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t Next() {
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  // Uniform in [0, 1): top 53 bits scaled by 2^-53 (exact in binary64).
  double NextUniform() { return static_cast<double>(Next() >> 11) * 0x1.0p-53; }

  // Uniform in [lo, hi), quantized to FP16.
  std::uint16_t NextHalfBitsUniform(double lo, double hi) {
    return QuantizeDoubleToHalfBits(lo + NextUniform() * (hi - lo));
  }

 private:
  std::uint64_t state_;
};

// ---------------------------------------------------------------------------
// Tolerance comparison per docs/DESIGN.md section 12a: max-abs and mean-abs
// error between an FP16 buffer (bits) and a double reference, computed over
// value space. NaNs compare by class (both NaN = match, else counted as a
// max-abs violation via +inf error).
// ---------------------------------------------------------------------------
struct ToleranceReport {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  std::size_t worst_index = 0;
  double worst_actual = 0.0;
  double worst_expected = 0.0;
  std::size_t count = 0;
};

inline ToleranceReport CompareHalfAgainstDouble(const std::uint16_t* actual_bits,
                                                const double* expected, std::size_t n) {
  ToleranceReport rep;
  rep.count = n;
  double sum_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = HalfBitsToDouble(actual_bits[i]);
    const double e = expected[i];
    double err;
    if (std::isnan(a) || std::isnan(e)) {
      err = (std::isnan(a) && std::isnan(e)) ? 0.0 : std::numeric_limits<double>::infinity();
    } else {
      err = std::fabs(a - e);
    }
    sum_abs += err;
    if (i == 0 || err > rep.max_abs) {
      rep.max_abs = err;
      rep.worst_index = i;
      rep.worst_actual = a;
      rep.worst_expected = e;
    }
  }
  rep.mean_abs = (n == 0) ? 0.0 : sum_abs / static_cast<double>(n);
  return rep;
}

} // namespace redline::reference
