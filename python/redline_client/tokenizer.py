"""Wrapper around the HF ``tokenizers`` library (no ``transformers`` needed).

Loads ``tokenizer.json`` from the model directory - Qwen2.5 ships one - so
encode/decode stays exactly consistent with the checkpoint the engine serves.
The engine itself never sees text (token-id-only contract); this module is
the client-side half of that split.
"""

from __future__ import annotations

import json
import os


def load_tokenizer(model_dir: str) -> "Tokenizer":
    """Load the checkpoint's tokenizer: alias for ``Tokenizer(model_dir)``."""
    return Tokenizer(model_dir)


def _read_eos_token_ids(model_dir: str, hf_tokenizer) -> list[int]:
    """EOS ids from checkpoint metadata, never hardcoded.

    Primary source is ``generation_config.json`` (``eos_token_id``: int or
    list - Qwen2.5-Instruct ships ``[151645, 151643]``, i.e. ``<|im_end|>``
    first). Fallback is ``tokenizer_config.json``'s ``eos_token`` (a string
    or an ``added_token`` dict) mapped through the loaded tokenizer. The
    engine reads the same generation_config set server-side for its stop
    check; these ids exist client-side for display/manual-stop logic only.
    """
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.isfile(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            eos = json.load(f).get("eos_token_id")
        if isinstance(eos, int):
            return [eos]
        if isinstance(eos, list) and eos and all(isinstance(t, int) for t in eos):
            return list(eos)

    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.isfile(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            eos_token = json.load(f).get("eos_token")
        if isinstance(eos_token, dict):  # added_token serialization
            eos_token = eos_token.get("content")
        if isinstance(eos_token, str):
            token_id = hf_tokenizer.token_to_id(eos_token)
            if token_id is not None:
                return [int(token_id)]

    raise ValueError(
        f"could not determine the EOS token id under {model_dir!r}: neither "
        "generation_config.json (eos_token_id) nor tokenizer_config.json "
        "(eos_token) yielded one"
    )


class Tokenizer:
    """Thin encode/decode wrapper plus special-token IDs.

    ``tokenizers`` is imported lazily inside ``__init__`` so this package
    stays importable in environments that only need type information.
    """

    def __init__(self, model_dir: str) -> None:
        try:
            from tokenizers import Tokenizer as _HFTokenizer
        except ImportError as error:
            raise ImportError(
                "redline_client.Tokenizer requires the 'tokenizers' package "
                "(pip install tokenizers)"
            ) from error

        path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no tokenizer.json under {model_dir!r} - expected an HF "
                "checkpoint directory (Qwen2.5 ships one)"
            )
        self._tokenizer = _HFTokenizer.from_file(path)
        self._eos_token_ids = _read_eos_token_ids(model_dir, self._tokenizer)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Text -> token IDs. Chat templating is the caller's concern."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Token IDs -> text."""
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    @property
    def eos_token_id(self) -> int:
        """Primary EOS id - the first entry of the checkpoint's EOS set
        (``<|im_end|>`` = 151645 for Qwen2.5-Instruct)."""
        return self._eos_token_ids[0]

    @property
    def eos_token_ids(self) -> list[int]:
        """Every EOS id the engine stops on, in checkpoint order."""
        return list(self._eos_token_ids)
