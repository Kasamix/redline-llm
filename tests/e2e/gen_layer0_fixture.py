#!/usr/bin/env python3
"""Generate the HF layer-0 activation fixture for suite (f) (docs/DESIGN.md §12f).

For ONE fixed plain-text prompt this script runs a single HF ``transformers``
FP16 forward pass on the Qwen2.5-1.5B-Instruct checkpoint and captures, via
forward hooks on layer 0:

  ``normed``    output of ``model.layers[0].input_layernorm``        [T, 1536]
  ``qkv``       ``q_proj|k_proj|v_proj`` outputs concatenated on the
                feature axis - PRE-rotary (HF applies RoPE later,
                inside attention), matching the engine's pre-RoPE
                ``qkv_out`` capture                                  [T, 2048]
  ``attn_out``  output of ``model.layers[0].self_attn`` (post
                ``o_proj``, before the residual add)                 [T, 1536]
  ``mlp_out``   output of ``model.layers[0].mlp`` (before the
                residual add)                                        [T, 1536]

``tests/e2e/test_layer0.py`` later compares these stage-by-stage against the
engine's ``debug_dump_dir`` capture of the same prompt's first prefill chunk.
This localizes fused-weight packing mistakes - K rows swapped with V inside
``w_qkv``, a QKV bias applied on the wrong axis, a ``w_gateup`` concat
off-by-one - that otherwise surface only as fluent-looking garbage 28 layers
downstream (§12f). Like ``gen_reference.py``, it is designed to run as a
SEPARATE OS process before the engine so HF and the engine never share one
6 GiB card.

Fixture ``.npz`` schema (schema_version = 1)
--------------------------------------------
  prompt_token_ids  int32   [T]        tokenized WITHOUT chat template /
                                       special tokens (§12c convention)
  normed            float16 [T, 1536]
  qkv               float16 [T, 2048]  columns [q 0:1536 | k 1536:1792 |
                                       v 1792:2048], pre-rotary
  attn_out          float16 [T, 1536]
  mlp_out           float16 [T, 1536]
  meta_json         str (0-d)          JSON: schema_version, model_dir,
                                       model_id, model_revision,
                                       model_revision_verified (best-effort,
                                       see --revision), gpu_name,
                                       torch/transformers versions,
                                       attn_implementation, dtype,
                                       prompt_text, num_prompt_tokens,
                                       hidden_size, qkv_width, num_q_heads,
                                       num_kv_heads, head_dim,
                                       qkv_rope_applied (false),
                                       hf_capture_points

Usage
-----
    python tests/e2e/gen_layer0_fixture.py --model $MODEL --out ~/redline_ref/layer0.npz
    # optional: --attn-implementation sdpa --device cuda --prompt-text "..."

Requires ``torch`` and ``transformers`` (the e2e venv). Runs on GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

SCHEMA_VERSION = 1

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Pinned revision documented in docs/MODEL_SPEC.md / scripts/fetch_model.py;
# recorded in the fixture meta as PROVENANCE (a claim, not a checksum of the
# local dir). When the checkout carries huggingface_hub download metadata
# (the fetch_model.py layout), the claim is cross-checked best-effort and
# meta["model_revision_verified"] says whether that check passed; nothing
# downstream gates on either field. Override with --revision if the pin
# changes.
DEFAULT_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# The one fixed prompt of §12f. Deliberately self-contained (NOT imported from
# prompts.py: suite (c) may re-tune its prompt set without silently changing
# this fixture) and plain factual text so tokenization is stable across
# tokenizer releases. About 70 tokens under the Qwen2.5 tokenizer - small
# enough that the dev preset's 1024-token prefill chunk covers it in one
# chunk, large enough to exercise multi-block paging (>= 5 KV blocks).
FIXED_PROMPT = (
    "The Rhine rises in the Swiss Alps and flows north through Basel, "
    "Strasbourg, and Cologne before reaching the North Sea near Rotterdam. "
    "Along the way it collects the Neckar, the Main, and the Moselle, and "
    "for centuries its valley has carried grain, timber, and wine between "
    "the interior of Europe and the Atlantic ports. The river remains one "
    "of the busiest waterways in the world."
)

# The engine-side dump must fit one prefill chunk (dev preset 1024); the test
# asserts the same bound against its actual engine configuration.
MAX_PROMPT_TOKENS = 1024
MIN_PROMPT_TOKENS = 8


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument("--out", required=True, help="Output fixture .npz path.")
    parser.add_argument(
        "--prompt-text",
        default=FIXED_PROMPT,
        help="Override the fixed prompt (debug only; the committed test expects the default).",
    )
    parser.add_argument(
        "--attn-implementation",
        # Default sdpa, NOT eager: HF eager attention materializes the raw
        # (unscaled) q@k^T score matrix in FP16 and applies 1/sqrt(head_dim)
        # only afterwards. Qwen2.5-1.5B's layer-0 attention-sink logits reach
        # |q.k| ~ 2.6e5 on this prompt - past FP16's 65504 max - so the eager
        # capture overflows to inf and NaN-poisons attn_out/mlp_out (measured
        # on the dev GPU: 3,520 overflowing score entries, half of attn_out NaN).
        # SDPA keeps scores in FP32 inside the kernel (as does the engine's
        # FP32 score buffer, docs/DESIGN.md section 6.3) and its layer-0
        # attn_out sits within 2.1e-3 max-abs of an FP64 recomputation.
        default="sdpa",
        help="HF attn backend to pin and record (eager|sdpa|flash_attention_2; default sdpa - "
        "eager overflows FP16 on this model's layer-0 attention-sink logits and NaN-poisons "
        "the capture).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="Model revision recorded in the fixture meta as provenance (default: MODEL_SPEC "
        "pin). Cross-checked best-effort against the checkout's huggingface_hub download "
        "metadata when present; a mismatch warns (and sets model_revision_verified false) "
        "but never fails the run.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda).")
    return parser.parse_args(argv)


def _local_revision_evidence(model_dir: str) -> set[str]:
    """Best-effort revision evidence for a local checkout: the distinct commit
    hashes huggingface_hub recorded when it materialized files into
    ``model_dir`` (the ``scripts/fetch_model.py`` ``snapshot_download(
    local_dir=...)`` layout keeps one ``.cache/huggingface/download/**/
    *.metadata`` file per downloaded file: line 1 a float timestamp, line 2
    the commit sha, line 3 the etag). Parsed positionally with a format
    sanity check so any layout drift degrades to "no evidence" - an empty
    set means UNVERIFIABLE (hand-copied checkouts, other tools), never a
    mismatch."""
    import glob  # noqa: PLC0415
    import re  # noqa: PLC0415

    hashes: set[str] = set()
    pattern = os.path.join(model_dir, ".cache", "huggingface", "download", "**", "*.metadata")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read(4096).splitlines()
        except OSError:
            continue
        if len(lines) < 2 or not re.fullmatch(r"[0-9a-f]{40}", lines[1].strip()):
            continue
        try:
            float(lines[0].strip())  # writer format check: line 1 is a timestamp
        except ValueError:
            continue
        hashes.add(lines[1].strip())
    return hashes


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("error: CUDA device requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    print(f"model  : {args.model}")
    print(f"attn   : {args.attn_implementation}")
    print(f"device : {device}")

    # Provenance cross-check: --revision is recorded, not enforced, but when
    # the checkout carries download metadata the fixture meta must not
    # silently claim the wrong commit for a drifted local model dir.
    revision_evidence = _local_revision_evidence(args.model)
    revision_verified = bool(revision_evidence) and revision_evidence == {args.revision}
    if revision_evidence and not revision_verified:
        print(
            f"warning: checkout metadata under {args.model!r} records revision(s) "
            f"{sorted(revision_evidence)} but the fixture meta will record --revision "
            f"{args.revision!r} (model_revision_verified: false) - re-fetch with "
            "scripts/fetch_model.py or pass the matching --revision",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,  # `torch_dtype` is deprecated in transformers 5.x
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_q_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden_size // num_q_heads)
    qkv_width = (num_q_heads + 2 * num_kv_heads) * head_dim

    # Plain-text continuation tokenization: NO chat template, NO special
    # tokens, so the engine consumes byte-identical IDs (§12c convention).
    enc = tokenizer(args.prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    num_tokens = int(input_ids.shape[1])
    if not (MIN_PROMPT_TOKENS <= num_tokens <= MAX_PROMPT_TOKENS):
        print(
            f"error: prompt tokenizes to {num_tokens} tokens, outside "
            f"[{MIN_PROMPT_TOKENS}, {MAX_PROMPT_TOKENS}] - the engine-side dump covers the "
            "FIRST prefill chunk only, so the prompt must fit one dev-preset chunk",
            file=sys.stderr,
        )
        return 2

    # ---- forward hooks on layer 0 ------------------------------------------
    layer0 = model.model.layers[0]
    captures: dict[str, Any] = {}

    def save(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[name] = tensor.detach().to("cpu")

        return hook

    handles = [
        layer0.input_layernorm.register_forward_hook(save("normed")),
        layer0.self_attn.q_proj.register_forward_hook(save("q")),
        layer0.self_attn.k_proj.register_forward_hook(save("k")),
        layer0.self_attn.v_proj.register_forward_hook(save("v")),
        layer0.self_attn.register_forward_hook(save("attn_out")),
        layer0.mlp.register_forward_hook(save("mlp_out")),
    ]
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = {"normed", "q", "k", "v", "attn_out", "mlp_out"} - set(captures)
    if missing:
        print(f"error: forward hooks captured nothing for {sorted(missing)} - "
              "HF module layout changed?", file=sys.stderr)
        return 2

    # Pre-rotary q|k|v concat, mirroring the engine's fused qkv_out layout.
    captures["qkv"] = torch.cat([captures.pop("q"), captures.pop("k"), captures.pop("v")], dim=-1)

    expected_shapes = {
        "normed": (num_tokens, hidden_size),
        "qkv": (num_tokens, qkv_width),
        "attn_out": (num_tokens, hidden_size),
        "mlp_out": (num_tokens, hidden_size),
    }
    arrays: dict[str, Any] = {}
    for name, want in expected_shapes.items():
        tensor = captures[name]
        if tensor.dim() == 3:  # [batch=1, T, features]
            tensor = tensor.squeeze(0)
        if tensor.dtype != torch.float16:
            print(
                f"error: captured {name!r} has dtype {tensor.dtype}, expected torch.float16 - "
                "the fixture must be the FP16 values HF actually computed",
                file=sys.stderr,
            )
            return 2
        if tuple(tensor.shape) != want:
            print(
                f"error: captured {name!r} has shape {tuple(tensor.shape)}, expected {want}",
                file=sys.stderr,
            )
            return 2
        bad = int((~torch.isfinite(tensor)).sum())
        if bad:
            # A NaN/inf-poisoned reference must fail HERE with the real cause,
            # not downstream in test_layer0.py as a misleading stage "diff".
            # Known trigger: --attn-implementation eager (unscaled FP16 q@k^T
            # overflow on this model's attention-sink logits, see --help).
            print(
                f"error: captured {name!r} contains {bad} non-finite values under "
                f"attn_implementation={args.attn_implementation!r} - the fixture would be "
                "unusable as a parity reference; use the sdpa default (FP32-accumulated "
                "attention, no FP16 score materialization)",
                file=sys.stderr,
            )
            return 2
        arrays[name] = tensor.numpy()

    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_dir": os.path.abspath(args.model),
        "model_id": MODEL_ID,
        "model_revision": args.revision,
        # True only when huggingface_hub download metadata in the checkout
        # unanimously names exactly model_revision; false means the claim
        # could not be substantiated (or contradicted evidence - warned
        # above). Informational: nothing gates on it.
        "model_revision_verified": revision_verified,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attn_implementation": args.attn_implementation,
        "dtype": "float16",
        "prompt_text": args.prompt_text,
        "num_prompt_tokens": num_tokens,
        "hidden_size": hidden_size,
        "qkv_width": qkv_width,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        # qkv is captured at the q/k/v_proj outputs, BEFORE HF applies rotary
        # inside attention - must match the engine dump's qkv_rope_applied.
        "qkv_rope_applied": False,
        "hf_capture_points": {
            "normed": "model.layers[0].input_layernorm (output)",
            "qkv": "model.layers[0].self_attn.{q,k,v}_proj (outputs, concatenated)",
            "attn_out": "model.layers[0].self_attn (output[0], post o_proj)",
            "mlp_out": "model.layers[0].mlp (output)",
        },
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        prompt_token_ids=np.asarray(input_ids[0].tolist(), dtype=np.int32),
        normed=arrays["normed"],
        qkv=arrays["qkv"],
        attn_out=arrays["attn_out"],
        mlp_out=arrays["mlp_out"],
        meta_json=np.asarray(json.dumps(meta)),
    )

    print(f"\nwrote {out_path}")
    print(f"  prompt tokens : {num_tokens}")
    for name in ("normed", "qkv", "attn_out", "mlp_out"):
        arr = arrays[name]
        absa = abs(arr.astype("float64"))
        print(f"  {name:9s} shape={arr.shape!s:14s} abs-max={absa.max():.4f} "
              f"abs-mean={absa.mean():.5f}")
    return 0


if __name__ == "__main__":
    # torch/transformers import inside main() only: pytest collection of the
    # sibling test module must never drag torch into the engine process.
    sys.exit(main())
