#!/usr/bin/env python3
"""Generate the HF greedy-parity reference for suite (c) (docs/DESIGN.md §12c).

This script runs HF ``transformers`` on the Qwen2.5-1.5B-Instruct checkpoint in
FP16 and caches, for each of the 20 fixed prompts in ``prompts.py``, the greedy
continuation plus per-step top-k logits and top1-top2 margins. ``test_hf_parity``
later replays this reference against the Redline engine.

It is designed to run **as a separate OS process before** the engine: HF and the
engine each need most of a 6 GiB card, so they never share one interpreter (§12c).

Greedy neutralization (§12c, MODEL_SPEC F6)
-------------------------------------------
The shipped ``generation_config.json`` sets ``do_sample=true`` and, critically,
``repetition_penalty=1.1`` - which HF's ``generate()`` applies *even when*
``do_sample=False``, so a reference built with ``do_sample=False`` alone is not
truly greedy and could never match the engine. This script sidesteps the trap
entirely by decoding with an explicit argmax over the model's **raw** logits (no
generation config, no logits processors, no repetition penalty, no EOS-logit
suppression), which is exactly the neutralized greedy the engine is compared
against. The effective neutralized settings are recorded in the JSON and the
test refuses to run against a reference whose neutralization record is absent or
wrong.

Reference JSON schema (schema_version = 1)
------------------------------------------
{
  "schema_version": 1,
  "meta": {
    "model_dir": str,
    "model_id": str,
    "model_revision": str | null,     # best-effort / --revision
    "gpu_name": str,
    "torch_version": str,
    "transformers_version": str,
    "attn_implementation": str,       # pinned; e.g. "sdpa"
    "dtype": "float16",
    "max_new_tokens": int,            # requested cap (128)
    "topk": int,                      # entries stored per step (>= 5)
    "eos_token_ids": [int, ...],      # {151645, 151643} (MODEL_SPEC F4)
    "add_special_tokens": false,      # tokenized WITHOUT chat template
    "num_positions": int              # sum of per-prompt generated lengths
  },
  "neutralization": {
    "do_sample": false,
    "repetition_penalty": 1.0,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "method": "manual-argmax-on-raw-logits",
    "shipped_generation_config": { ... risky shipped values, for the record ... }
  },
  "prompts": [
    {
      "index": int,
      "category": "short"|"medium"|"long",
      "prompt_text": str,
      "prompt_token_ids": [int, ...],
      "num_prompt_tokens": int,
      "generated_token_ids": [int, ...],     # HF greedy continuation, len L<=128
      "topk_ids":    [[int]*topk, ... x L],  # per step, argmax first
      "topk_logits": [[float]*topk, ... x L],# per step, FP32, descending
      "margins":     [float, ... x L],       # top1 - top2 (FP32) per step
      "stop_reason": "length" | "eos",
      "eos_hit": bool
    },
    ...
  ]
}

Note ``generated_token_ids[n] == topk_ids[n][0]`` and
``margins[n] == topk_logits[n][0] - topk_logits[n][1]`` by construction.

Usage
-----
    python tests/e2e/gen_reference.py --model $MODEL --out ~/redline_ref/hf_ref.json
    # optional: --max-new-tokens 128 --topk 10 --attn-implementation sdpa

Requires ``torch`` and ``transformers`` (the e2e venv). Runs on GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Sibling import (prompts.py lives next to this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CATEGORY_TOKEN_BANDS, all_prompts  # noqa: E402

SCHEMA_VERSION = 1

# EOS is a *set* for this checkpoint (MODEL_SPEC F4): stop on either member.
EOS_TOKEN_IDS = [151645, 151643]

# Pinned revision documented in docs/MODEL_SPEC.md / scripts/fetch_model.py.
# Recorded as a cache key; override with --revision if the pin ever changes.
DEFAULT_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument("--out", required=True, help="Output reference JSON path.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Greedy tokens per prompt (default 128)."
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="Top-k logit entries stored per step (>= 5)."
    )
    parser.add_argument(
        "--attn-implementation",
        # Default sdpa, NOT eager: HF eager attention materializes the raw
        # (unscaled) q@k^T score matrix in FP16 and applies 1/sqrt(head_dim)
        # only afterwards. Qwen2.5-1.5B's layer-0 attention-sink logits reach
        # |q.k| ~ 2.6e5 - past FP16's 65504 max - so eager full-model FP16
        # forwards overflow to inf and NaN-poison the logits (measured
        # on the layer-0 capture; same mechanism here). SDPA keeps
        # scores in FP32 inside the kernel (as does the engine's FP32 score
        # buffer, docs/DESIGN.md section 6.3).
        default="sdpa",
        help="HF attn backend to pin and record (eager|sdpa|flash_attention_2; default sdpa - "
        "eager overflows FP16 on this model's attention-sink logits and NaN-poisons "
        "the reference).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="Model revision to record in the cache key (default: MODEL_SPEC pin).",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda).")
    return parser.parse_args(argv)


def _shipped_generation_config(model_dir: str) -> dict[str, Any]:
    """Read the risky shipped sampling defaults, for the neutralization record."""
    path = os.path.join(model_dir, "generation_config.json")
    keep = ("do_sample", "repetition_penalty", "temperature", "top_p", "top_k")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: cfg[k] for k in keep if k in cfg}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.topk < 5:
        print("error: --topk must be >= 5 (first-token check needs top-5)", file=sys.stderr)
        return 2

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("error: CUDA device requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    print(f"model    : {args.model}")
    print(f"attn     : {args.attn_implementation}")
    print(f"device   : {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,  # `torch_dtype` is deprecated in transformers 5.x
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    eos_set = set(EOS_TOKEN_IDS)

    prompt_records: list[dict[str, Any]] = []
    total_positions = 0

    for prompt in all_prompts():
        # Plain-text continuation: NO chat template, NO special tokens, so the
        # engine and HF consume byte-identical IDs (§12c).
        enc = tokenizer(prompt.text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        num_prompt_tokens = int(input_ids.shape[1])

        lo, hi = CATEGORY_TOKEN_BANDS[prompt.category]
        if not (lo <= num_prompt_tokens <= hi):
            print(
                f"warning: prompt {prompt.index} ({prompt.category}) has "
                f"{num_prompt_tokens} tokens, outside nominal band [{lo}, {hi}]",
                file=sys.stderr,
            )

        generated: list[int] = []
        topk_ids: list[list[int]] = []
        topk_logits: list[list[float]] = []
        margins: list[float] = []
        stop_reason = "length"
        eos_hit = False

        cur_ids = input_ids
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                attention_mask = torch.ones_like(cur_ids)
                out = model(input_ids=cur_ids, attention_mask=attention_mask, use_cache=False)
                # Raw next-token logits in FP32 (matches the engine's fp32 argmax).
                logits = out.logits[:, -1, :].float()
                bad = int((~torch.isfinite(logits)).sum())
                if bad:
                    # A NaN/inf-poisoned reference must fail HERE with the real
                    # cause, not downstream in test_hf_parity.py as misleading
                    # disagreements. Known trigger: --attn-implementation eager
                    # (unscaled FP16 q@k^T overflow on this model's
                    # attention-sink logits, see --help).
                    print(
                        f"error: prompt {prompt.index} step {len(generated)}: logits contain "
                        f"{bad} non-finite values under attn_implementation="
                        f"{args.attn_implementation!r} -- the reference would be unusable; "
                        "use the sdpa default (FP32-accumulated attention, no FP16 score "
                        "materialization)",
                        file=sys.stderr,
                    )
                    return 2
                topv, topi = torch.topk(logits, k=args.topk, dim=-1)
                ids_row = [int(x) for x in topi[0].tolist()]
                logits_row = [float(x) for x in topv[0].tolist()]
                argmax = ids_row[0]
                margin = logits_row[0] - logits_row[1]

                generated.append(argmax)
                topk_ids.append(ids_row)
                topk_logits.append(logits_row)
                margins.append(margin)

                if argmax in eos_set:
                    stop_reason = "eos"
                    eos_hit = True
                    break

                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[argmax]], device=device, dtype=cur_ids.dtype)],
                    dim=1,
                )

        if len(generated) < args.max_new_tokens:
            print(
                f"warning: prompt {prompt.index} generated only {len(generated)} tokens "
                f"(stop_reason={stop_reason}); the nominal 128-position count will be short",
                file=sys.stderr,
            )

        total_positions += len(generated)
        prompt_records.append(
            {
                "index": prompt.index,
                "category": prompt.category,
                "prompt_text": prompt.text,
                "prompt_token_ids": [int(x) for x in input_ids[0].tolist()],
                "num_prompt_tokens": num_prompt_tokens,
                "generated_token_ids": generated,
                "topk_ids": topk_ids,
                "topk_logits": topk_logits,
                "margins": margins,
                "stop_reason": stop_reason,
                "eos_hit": eos_hit,
            }
        )
        print(f"  prompt {prompt.index:2d} [{prompt.category:6s}] "
              f"prompt_tokens={num_prompt_tokens:4d} generated={len(generated):3d}")

    reference = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "model_dir": os.path.abspath(args.model),
            "model_id": MODEL_ID,
            "model_revision": args.revision,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "attn_implementation": args.attn_implementation,
            "dtype": "float16",
            "max_new_tokens": args.max_new_tokens,
            "topk": args.topk,
            "eos_token_ids": EOS_TOKEN_IDS,
            "add_special_tokens": False,
            "num_positions": total_positions,
        },
        # NEUTRALIZED AND ASSERTED (§12c). Manual argmax on raw logits is
        # do_sample=False with repetition_penalty=1.0 and no warpers, by
        # construction; recorded so the test can refuse a non-neutralized ref.
        "neutralization": {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "method": "manual-argmax-on-raw-logits",
            "shipped_generation_config": _shipped_generation_config(args.model),
        },
        "prompts": prompt_records,
    }

    # Self-assert the neutralization record before writing, so a corrupted run
    # can never emit a reference the test would (correctly) reject.
    n = reference["neutralization"]
    assert n["do_sample"] is False
    assert n["repetition_penalty"] == 1.0
    assert n["temperature"] is None and n["top_p"] is None and n["top_k"] is None

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reference, f)

    print(f"\nwrote {out_path}")
    print(f"  prompts        : {len(prompt_records)}")
    print(f"  total positions: {total_positions} (nominal 2560 = 20 x 128)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
