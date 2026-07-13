#!/usr/bin/env python3
"""Generate tests/data/rope_fixture.json - the HF-captured RoPE fixture.

Purpose (docs/DESIGN.md section 12a; docs/MODEL_SPEC.md section 3): pin the
engine's rotary-embedding convention - GPT-NeoX half-rotation, element d
paired with d + head_dim/2, NOT the GPT-J interleaved (2d, 2d+1) pairing -
to captured evidence from Hugging Face's own Qwen2 implementation rather
than to a re-derivation that could share a misunderstanding with the kernel.
The kernel unit suite compares LaunchRopeInplace against this file within
the section 12a elementwise tolerances, and additionally cross-checks the
committed data against the suite's independent FP64 reference on CPU-only
builds.

What is captured
    q_out, k_out = transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb(
        q_in, k_in, *Qwen2RotaryEmbedding(config)(q_in, position_ids))
    on fixed FP16 inputs (torch CPU), rope_theta = 1e6, head_dim = 128,
    12 Q heads / 2 KV heads, 8 tokens at positions
    [0, 1, 2, 15, 16, 17, 2047, 4095].

Self-checks before anything is written (the script hard-fails on any):
    1. An independent emulation of the documented math (MODEL_SPEC section 3:
       float32 inv_freq/angles/cos/sin, cast to FP16 at use, FP16 elementwise
       apply) must match the transformers capture BIT-EXACTLY. A mismatch
       means the installed transformers changed its rotary math - investigate
       before regenerating, never commit ambiguous data.
    2. Anti-vacuity: a GPT-J interleaved-pairing emulation must DISAGREE with
       the capture on a substantial fraction of rotated elements - if the two
       conventions coincided on these inputs, the fixture could not pin the
       decision.
    3. The position-0 token must pass through bit-identically (identity
       rotation), which any convention satisfies - a capture-pipeline sanity
       check.

File format (committed): a single JSON object.
    meta:  model constants, positions, generator provenance (torch /
           transformers versions), capture method, RNG seed.
    q_in, q_out: [num_tokens][num_q_heads * head_dim] FP16 bit patterns
           (uint16), row-major, token-major with heads concatenated - the
           packed row layout the kernel's launcher sees.
    k_in, k_out: same with num_kv_heads.
Bit patterns (not decimal floats) make the round trip exact by construction.

Usage:
    python scripts/gen_rope_fixture.py                 # -> tests/data/rope_fixture.json
    python scripts/gen_rope_fixture.py --out FILE

Requires: torch, numpy, transformers (CPU only; no model download).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "data" / "rope_fixture.json"

ROPE_THETA = 1e6
HEAD_DIM = 128
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
POSITIONS = [0, 1, 2, 15, 16, 17, 2047, 4095]
SEED = 20260709


def build_inputs():
    """Fixed FP16 inputs: N(0, 1) draws quantized to FP16, PCG64-seeded."""
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(SEED))
    t = len(POSITIONS)
    q = rng.standard_normal((t, NUM_Q_HEADS, HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((t, NUM_KV_HEADS, HEAD_DIM)).astype(np.float16)
    return q, k


def hf_capture(q_np, k_np):
    """The operation under test: transformers' own Qwen2 rotary application."""
    import torch
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (Qwen2RotaryEmbedding,
                                                          apply_rotary_pos_emb)

    cfg = Qwen2Config(
        hidden_size=NUM_Q_HEADS * HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=32768,
    )
    # [T, H, D] -> [1, H, T, D] (the layout apply_rotary_pos_emb documents).
    q = torch.from_numpy(q_np.copy()).permute(1, 0, 2).unsqueeze(0).contiguous()
    k = torch.from_numpy(k_np.copy()).permute(1, 0, 2).unsqueeze(0).contiguous()
    position_ids = torch.tensor([POSITIONS], dtype=torch.long)
    try:
        rot = Qwen2RotaryEmbedding(config=cfg)
        cos, sin = rot(q, position_ids)  # cast to q.dtype (FP16) inside
        method = "Qwen2RotaryEmbedding(config).forward(x, position_ids) + apply_rotary_pos_emb"
        q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
    except TypeError:
        # Legacy (< 4.44) module API: forward(x, seq_len) returns full tables
        # indexed by position_ids inside apply_rotary_pos_emb.
        rot = Qwen2RotaryEmbedding(HEAD_DIM, max_position_embeddings=32768, base=ROPE_THETA)
        cos, sin = rot(q, seq_len=max(POSITIONS) + 1)
        method = "legacy Qwen2RotaryEmbedding(dim).forward(x, seq_len) + apply_rotary_pos_emb"
        q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    assert q_emb.dtype == torch.float16 and k_emb.dtype == torch.float16, (
        q_emb.dtype, k_emb.dtype)
    to_np = lambda x: x.squeeze(0).permute(1, 0, 2).contiguous().numpy()  # noqa: E731
    return to_np(q_emb), to_np(k_emb), method


def emulate_documented_math(q_np, k_np, interleaved=False):
    """Independent emulation of MODEL_SPEC section 3 (or the WRONG GPT-J
    interleaved pairing when interleaved=True, used only as the anti-vacuity
    probe): float32 inv_freq / angles / cos / sin, cast to FP16, applied as
    FP16 elementwise torch ops."""
    import torch

    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64)
                                     .float() / HEAD_DIM))          # float32 [64]
    pos = torch.tensor(POSITIONS, dtype=torch.float32)              # float32 [T]
    freqs = pos[:, None] * inv_freq[None, :]                        # float32 [T, 64]

    def apply(x_np):
        x = torch.from_numpy(x_np.copy())                           # [T, H, D] fp16
        if not interleaved:
            # NeoX half-rotation: emb = cat(freqs, freqs) -> cos[d] == cos[d+64];
            # pair (d, d + D/2).
            emb = torch.cat((freqs, freqs), dim=-1)                 # [T, 128] float32
            cos = emb.cos().to(torch.float16)[:, None, :]           # fp16 [T, 1, 128]
            sin = emb.sin().to(torch.float16)[:, None, :]
            x1 = x[..., : HEAD_DIM // 2]
            x2 = x[..., HEAD_DIM // 2:]
            rotated = torch.cat((-x2, x1), dim=-1)
            return (x * cos) + (rotated * sin)                      # fp16 ops
        # GPT-J interleaved: pair (2j, 2j+1), both sharing angle j.
        cos = freqs.cos().to(torch.float16)[:, None, :]             # fp16 [T, 1, 64]
        sin = freqs.sin().to(torch.float16)[:, None, :]
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = (xe * cos) - (xo * sin)
        out[..., 1::2] = (xo * cos) + (xe * sin)
        return out

    return apply(q_np).numpy(), apply(k_np).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"output file (default: {DEFAULT_OUT.relative_to(REPO_ROOT).as_posix()})")
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers

    q_in, k_in = build_inputs()
    q_out, k_out, method = hf_capture(q_in, k_in)

    # Self-check 1: documented-math emulation must match the capture bit-exactly.
    q_emu, k_emu = emulate_documented_math(q_in, k_in)
    for name, cap, emu in (("q", q_out, q_emu), ("k", k_out, k_emu)):
        cap_bits = cap.view(np.uint16)
        emu_bits = emu.view(np.uint16)
        if not np.array_equal(cap_bits, emu_bits):
            n = int((cap_bits != emu_bits).sum())
            raise AssertionError(
                f"transformers capture and MODEL_SPEC-documented emulation disagree on "
                f"{n} {name} elements - the installed transformers "
                f"({transformers.__version__}) changed its rotary math; investigate before "
                f"regenerating this fixture.")

    # Self-check 2 (anti-vacuity): the interleaved convention must disagree.
    q_int, _ = emulate_documented_math(q_in, k_in, interleaved=True)
    live = np.asarray(POSITIONS) > 0  # position 0 is identity under BOTH conventions
    frac = float((q_out[live].view(np.uint16) != q_int[live].view(np.uint16)).mean())
    if frac < 0.10:
        raise AssertionError(
            f"interleaved emulation agrees with the capture on {(1 - frac) * 100:.1f}% of "
            f"rotated elements - this fixture would not pin the pairing convention.")

    # Self-check 3: position 0 is the identity rotation, bit-exact.
    for name, before, after in (("q", q_in, q_out), ("k", k_in, k_out)):
        if not np.array_equal(before[0].view(np.uint16), after[0].view(np.uint16)):
            raise AssertionError(f"position-0 {name} row was not passed through bit-exactly")

    flat = lambda a: a.reshape(a.shape[0], -1).view(np.uint16).tolist()  # noqa: E731
    doc = {
        "meta": {
            "description": "HF-captured Qwen2 rotary fixture; see scripts/gen_rope_fixture.py",
            "convention": "neox-half-rotation (pair d with d + head_dim/2)",
            "rope_theta": ROPE_THETA,
            "head_dim": HEAD_DIM,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "num_tokens": len(POSITIONS),
            "positions": POSITIONS,
            "seed": SEED,
            "encoding": "IEEE binary16 bit patterns as uint16; rows are "
                        "[num_tokens][heads * head_dim], token-major, heads concatenated",
            "capture_method": method,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "interleaved_disagreement_fraction": round(frac, 4),
        },
        "q_in": flat(q_in),
        "k_in": flat(k_in),
        "q_out": flat(q_out),
        "k_out": flat(k_out),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")

    print(f"wrote    : {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"capture  : {method}")
    print(f"versions : torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"anti-vacuity: interleaved pairing disagrees on {frac * 100:.1f}% "
          f"of rotated q elements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
