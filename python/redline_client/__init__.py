"""Client-side utilities for the redline engine.

The engine consumes and produces token IDs only; this package provides the
client half of that contract - tokenizer glue (:class:`Tokenizer`) and a
minimal ``step()`` pump loop (:func:`generate`) - for examples, end-to-end
tests, and quick smoke runs. The bench harness ships its own asyncio driver
(``bench/adapters/redline_a.py``); this helper is deliberately synchronous.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from redline_client.tokenizer import Tokenizer, load_tokenizer

__all__ = ["Completion", "Tokenizer", "generate", "load_tokenizer"]
__version__ = "0.1.0"

# Consecutive empty step() results tolerated while requests are in flight.
# Legitimate empty steps exist (a non-final prefill chunk emits no token) but
# are bounded by ceil(max_seq_len / prefill_chunk) per request; a run of 4096
# means the engine is wedged and the helper fails instead of spinning forever.
_MAX_IDLE_STEPS = 4096


@dataclass
class Completion:
    """One request's result, as collected by :func:`generate`.

    ``token_ids`` holds the generated ids (an aborted request keeps whatever
    was produced before the abort); ``finish_reason`` is the engine's string:
    ``"eos"``, ``"length"``, or ``"aborted"``.
    """

    request_id: int
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""


def generate(
    engine: Any,
    prompts: Iterable[Sequence[int]],
    max_new_tokens: int | Sequence[int],
    *,
    ignore_eos: bool = False,
    on_token: Callable[[int, int, bool, str], None] | None = None,
) -> list[Completion]:
    """Run token-id prompts to completion; results in submission order.

    The minimal pump loop of docs/DESIGN.md section 10: submit while
    ``engine.has_capacity()`` grants backpressure, then ``engine.step()``
    until every submitted request finishes. ``generate`` must be the
    engine's only driver while it runs (the engine is single-driver by
    contract); an emission for an id it did not submit, or for one that
    already finished, raises.

    ``max_new_tokens`` is one budget for all prompts (int) or one per prompt
    (sequence of equal length). ``on_token(request_id, token_id, finished,
    finish_reason)``, when given, is called for every produced token as it
    arrives - abort notices carry no token and are not forwarded.
    """
    prompt_lists = [list(p) for p in prompts]
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(prompt_lists)
    else:
        budgets = [int(n) for n in max_new_tokens]
        if len(budgets) != len(prompt_lists):
            raise ValueError(
                f"max_new_tokens has {len(budgets)} entries for "
                f"{len(prompt_lists)} prompts; pass an int or one budget per prompt"
            )

    completions: list[Completion] = []
    by_id: dict[int, Completion] = {}
    unfinished = 0
    next_index = 0
    idle_steps = 0

    while next_index < len(prompt_lists) or unfinished:
        while next_index < len(prompt_lists) and engine.has_capacity():
            request_id = engine.add_request(
                prompt_lists[next_index], budgets[next_index], ignore_eos
            )
            completion = Completion(request_id=request_id)
            completions.append(completion)
            by_id[request_id] = completion
            unfinished += 1
            next_index += 1

        results = engine.step()
        if not results:
            idle_steps += 1
            if idle_steps > _MAX_IDLE_STEPS:
                raise RuntimeError(
                    f"engine made no progress across {_MAX_IDLE_STEPS} consecutive "
                    f"step() calls with {unfinished} request(s) in flight"
                )
            continue
        idle_steps = 0

        for request_id, token_id, finished, finish_reason in results:
            completion = by_id.get(request_id)
            if completion is None:
                raise RuntimeError(
                    f"engine emitted a result for unknown request id {request_id}; "
                    "generate() must be the engine's only driver while it runs"
                )
            if completion.finish_reason:
                # The engine erases a request once it finishes, so any further
                # emission for its id violates the single-driver contract -
                # and counting it again would corrupt `unfinished` and end the
                # pump with incomplete results.
                raise RuntimeError(
                    f"engine emitted a result for request id {request_id} after it "
                    f"finished ({completion.finish_reason!r}); generate() must be "
                    "the engine's only driver while it runs"
                )
            if finish_reason != "aborted":
                completion.token_ids.append(token_id)
                if on_token is not None:
                    on_token(request_id, token_id, finished, finish_reason)
            if finished:
                completion.finish_reason = finish_reason
                unfinished -= 1

    return completions
