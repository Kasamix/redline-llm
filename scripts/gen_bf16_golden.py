#!/usr/bin/env python3
"""Generate tests/data/bf16_fp16_golden.bin - the BF16->FP16 conversion golden table.

File format (exactly 131,072 bytes, committed):
    65,536 little-endian uint16 values. Entry i is the FP16 bit pattern that torch
    produces for the BF16 bit pattern i:

        torch.tensor(<bits i>, dtype=torch.bfloat16).to(torch.float16)

    i.e. widen BF16 to FP32 exactly (BF16 is the top 16 bits of binary32), then round
    to FP16 with IEEE-754 round-to-nearest-even: subnormal results exact (no
    flush-to-zero), no clamping, finite values beyond +-65504 rounding to +-inf.

The engine's loader (src/loader/convert.hpp) must match this table bit-exactly for
every pattern; tests/test_bf16_convert.cpp iterates all 65,536 entries. NaN entries
are compared by "is NaN" rather than by payload: torch's NaN payload handling depends
on the code path it dispatches to (scalar converter canonicalizes to 0x7E00, the
vectorized F16C path keeps the top payload bits), while the loader deterministically
emits the sign-preserved canonical quiet NaN. Patterns whose value has |x| > 65504
appear in the table as +-inf (torch's IEEE behavior); the loader instead fails hard
on them (docs/DESIGN.md section 4) - the unit test asserts that divergence
explicitly.

Regeneration is deterministic; the output is platform-independent (written explicitly
little-endian).

Usage:
    python scripts/gen_bf16_golden.py                # -> <repo>/tests/data/bf16_fp16_golden.bin
    python scripts/gen_bf16_golden.py --out FILE     # custom path

Requires: torch, numpy (CPU only).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "data" / "bf16_fp16_golden.bin"
NUM_PATTERNS = 1 << 16


def build_table() -> "numpy.ndarray":  # noqa: F821 - imported in main()
    import numpy as np
    import torch

    # All 65,536 BF16 bit patterns, bit-reinterpreted (never value-converted).
    bits = np.arange(NUM_PATTERNS, dtype=np.uint16)
    bf16 = torch.tensor(bits.view(np.int16), dtype=torch.int16).view(torch.bfloat16)

    # The operation under test: torch's BF16 -> FP16 conversion (CPU).
    fp16 = bf16.to(torch.float16)

    out = fp16.view(torch.int16).numpy().view(np.uint16).copy()

    # Independent cross-check: widen to FP32 via u32 = u16 << 16 and let numpy's
    # IEEE RNE float32->float16 cast do the rounding. numpy and torch must agree
    # on every non-NaN entry. NaN entries are excluded from the bit-equality
    # check (payload policies differ across libraries and torch code paths);
    # both sides must still produce a NaN.
    fp32 = (bits.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(over="ignore"):  # finite -> inf casts are expected here
        np_fp16 = fp32.astype(np.float16).view(np.uint16)
    nan_in = np.isnan(fp32)
    if not np.array_equal(out[~nan_in], np_fp16[~nan_in]):
        bad = np.nonzero(out[~nan_in] != np_fp16[~nan_in])[0]
        raise AssertionError(f"torch and numpy disagree on non-NaN entries, e.g. index {bad[:5]}")
    if not (np.isnan(fp32[nan_in].astype(np.float16)).all()
            and np.isnan(fp16.numpy()[nan_in]).all()):
        raise AssertionError("a NaN input did not convert to a NaN")

    # Landmark values pin the semantics the loader depends on.
    landmarks = {
        0x0000: 0x0000,  # +0
        0x8000: 0x8000,  # -0
        0x3F80: 0x3C00,  # 1.0
        0xBF80: 0xBC00,  # -1.0
        0x477F: 0x7BF8,  # 65280, largest BF16 magnitude below FP16 max 65504: exact
        0x4780: 0x7C00,  # 65536: beyond FP16 max, IEEE RNE -> +inf (loader hard-fails instead)
        0x7F80: 0x7C00,  # +inf -> +inf (loader hard-fails instead)
        0xFF80: 0xFC00,  # -inf -> -inf (loader hard-fails instead)
        0x3300: 0x0000,  # 2^-25: tie between 0 and the min subnormal 2^-24, RNE -> even (0)
        0x3340: 0x0001,  # 1.5 * 2^-25 = 0.75 * 2^-24 -> min subnormal (no flush-to-zero)
        0x3880: 0x0400,  # 2^-14: FP16 minimum normal
        0x387F: 0x03FC,  # largest BF16 below 2^-14 -> subnormal 1020 * 2^-24, exact
        0x0001: 0x0000,  # BF16 min subnormal (~9.2e-41): far below 2^-25, rounds to +0
    }
    for pattern, expected in landmarks.items():
        got = int(out[pattern])
        if got != expected:
            raise AssertionError(
                f"landmark mismatch at BF16 0x{pattern:04X}: got 0x{got:04X}, "
                f"expected 0x{expected:04X}")
    if not (int(out[0x7FC0]) & 0x7FFF) > 0x7C00:  # quiet NaN stays NaN
        raise AssertionError("0x7FC0 (NaN) did not convert to an FP16 NaN")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"output file (default: {DEFAULT_OUT.relative_to(REPO_ROOT).as_posix()})",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    table = build_table()
    data = table.astype("<u2").tobytes()  # explicit little-endian
    assert len(data) == NUM_PATTERNS * 2, len(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    n_inf = int(np.sum((table & 0x7FFF) == 0x7C00))
    n_nan = int(np.sum((table & 0x7FFF) > 0x7C00))
    print(f"wrote    : {out_path} ({len(data):,} bytes)")
    print(f"sha256   : {hashlib.sha256(data).hexdigest()}")
    print(f"entries  : {NUM_PATTERNS:,} (inf: {n_inf:,}, nan: {n_nan:,})")
    print(f"versions : torch {torch.__version__}, numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
