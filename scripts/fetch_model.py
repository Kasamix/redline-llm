#!/usr/bin/env python3
"""Download the Qwen2.5-1.5B-Instruct checkpoint into models/ (gitignored).

The default revision is pinned to the commit documented in docs/MODEL_SPEC.md so that
every checkout builds and tests against byte-identical weights and configs.

Usage:
    python scripts/fetch_model.py                     # pinned revision -> models/Qwen2.5-1.5B-Instruct
    python scripts/fetch_model.py --revision <sha>    # override the pin (re-verify MODEL_SPEC.md!)
    python scripts/fetch_model.py --dest /path/dir    # custom target directory

Requires:
    pip install "huggingface_hub>=0.23"

The model repo is public; no HF token is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# `main` as of 2026-07-09 (repo last modified 2024-09-25). See docs/MODEL_SPEC.md.
PINNED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything the engine and its test harness consume. LICENSE is kept for attribution.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",  # absent for this model (single shard) but harmless
    "LICENSE",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-id", default=MODEL_ID, help=f"HF repo id (default: {MODEL_ID})")
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help="Commit sha / tag / branch to download (default: the pin in docs/MODEL_SPEC.md)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Target directory (default: <repo>/models/<model-name>)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub is not installed. Run: pip install 'huggingface_hub>=0.23'",
              file=sys.stderr)
        return 1

    dest = Path(args.dest) if args.dest else REPO_ROOT / "models" / args.model_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"model    : {args.model_id}")
    print(f"revision : {args.revision}")
    print(f"dest     : {dest}")
    if args.revision != PINNED_REVISION:
        print("warning  : revision differs from the pin in docs/MODEL_SPEC.md - "
              "re-verify the spec against the new revision.", file=sys.stderr)

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(dest),
        allow_patterns=ALLOW_PATTERNS,
    )

    print("\ndownloaded files:")
    total = 0
    for f in sorted(Path(path).rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            size = f.stat().st_size
            total += size
            print(f"  {size:>13,} B  {f.relative_to(path)}")
    print(f"  {total:>13,} B  total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
