"""End-to-end HF greedy-parity suite (c) - docs/DESIGN.md §12c.

Consumes the reference JSON produced by ``gen_reference.py`` (a *separate* OS
process) and drives the Redline engine through the documented section-10 API,
so torch/transformers are never imported here and HF never shares the engine's
interpreter or GPU.

Three checks, in order of authority (§12c):

1. Teacher-forced agreement (PRIMARY, gating). Feed HF's generated token at
   every step via ``forced_tokens`` so the engine is always conditioned on the
   identical reference prefix (no divergence compounding); compare the engine's
   argmax at every position. Pass = >= 99% position-wise agreement over all
   positions AND every disagreement sits on a documented FP16 coin-flip (HF
   top1-top2 margin < 0.1). A single larger-margin disagreement fails the suite
   regardless of the aggregate rate.
2. Free-running greedy prefix match (SECONDARY, reported, NOT gating). The
   engine self-decodes; per-prompt matched-prefix length and the first
   divergence (position, both token IDs, HF margin) are reported into the
   terminal summary.
3. First-token check (gating). The engine's first-step argmax must lie in HF's
   top-5 set, and where it differs from HF's argmax the HF margin must be < 0.1.
   (The section-10 API exposes only the argmax token, not logits, so "top-5
   sets match" is enforced as "engine top-1 in HF top-5".)

Plus one harness-integrity tripwire (gating, not part of the sec-12c trio):

4. Forced-token echo tripwire. Re-runs one prompt teacher-forced with a single
   garbage token substituted at the reference's highest-margin position and
   asserts the engine's emission there differs from the garbage. If the glue
   ever echoed the appended ``forced_tokens`` entry as ``StepResult.token``
   instead of the engine's own argmax, checks (1) and (3) would pass vacuously
   at 100%; this closes that hole end-to-end through the pybind layer.

Auditable receipt: every run writes a JSON receipt on module teardown (even
after failures) to ``$REDLINE_PARITY_OUT`` or ``./hf_parity_report.json`` -
the same teardown-report convention as suites (d)/(e). It records the
agreement counts and rate, per-check outcomes, thresholds, reference and
engine identity. A run that backs a published parity claim commits its
receipt under ``bench/results/parity/`` - the flagship correctness number
must never live only in a terminal summary or a commit message.

Run:
    python tests/e2e/gen_reference.py --model $MODEL --out ref.json   # separate process
    PYTHONPATH=$BUILD pytest tests/e2e/test_hf_parity.py --model $MODEL --ref ref.json -v
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import warnings
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_options import get_option  # noqa: E402
from prompts import all_prompts  # noqa: E402

# Gating thresholds (§12c).
AGREEMENT_FLOOR = 0.99
COINFLIP_MARGIN = 0.1
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Module-level receipt (written on teardown, even after failures)
# --------------------------------------------------------------------------

_REPORT: dict[str, Any] = {
    "suite": "c (HF greedy parity)",
    "design": "docs/DESIGN.md section 12c",
    "thresholds": {
        "agreement_floor": AGREEMENT_FLOOR,
        "coinflip_margin": COINFLIP_MARGIN,
    },
}


@pytest.fixture(scope="module", autouse=True)
def parity_receipt(request):
    """The suite's committable receipt (see the module docstring): structured
    outcomes land in ``_REPORT`` as each check computes them (before its
    asserts, so a failing run still leaves its numbers on disk), and the
    teardown writes the JSON to ``$REDLINE_PARITY_OUT`` or
    ``./hf_parity_report.json``."""
    _REPORT["meta"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model_dir": get_option(request.config, "--model", "REDLINE_TEST_MODEL"),
        "reference_path": get_option(request.config, "--ref", "REDLINE_HF_REF"),
        "engine_kwargs_option": get_option(
            request.config, "--engine-kwargs", "REDLINE_ENGINE_KWARGS"
        ),
    }
    yield _REPORT
    out = os.environ.get("REDLINE_PARITY_OUT") or os.path.join(
        os.getcwd(), "hf_parity_report.json"
    )
    try:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_REPORT, fh, indent=2, default=str)
        os.replace(tmp, out)
        print(f"\n[hf-parity] receipt JSON written to {out}")
    except OSError as err:  # never mask test results with a reporting error
        warnings.warn(f"could not write HF-parity receipt JSON to {out}: {err}")


# --------------------------------------------------------------------------
# Reference loading + validation (suite-c JSON schema)
# --------------------------------------------------------------------------
def _validate_reference(ref: dict[str, Any]) -> None:
    """Refuse to run against a missing/wrong neutralization record (§12c).

    Raises (does not skip) when a reference is present but not neutralized, so
    the suite errors loudly rather than silently passing on a non-greedy ref.
    """
    assert ref.get("schema_version") == SCHEMA_VERSION, (
        f"reference schema_version {ref.get('schema_version')!r} != {SCHEMA_VERSION}; "
        "regenerate with gen_reference.py"
    )

    neutralization = ref.get("neutralization")
    assert isinstance(neutralization, dict), (
        "reference has no 'neutralization' record -- it was not built by "
        "gen_reference.py's neutralized greedy path; refusing to run (sec 12c)"
    )
    problems = []
    if neutralization.get("do_sample") is not False:
        problems.append(f"do_sample={neutralization.get('do_sample')!r} (want False)")
    if neutralization.get("repetition_penalty") != 1.0:
        problems.append(
            f"repetition_penalty={neutralization.get('repetition_penalty')!r} (want 1.0)"
        )
    for key in ("temperature", "top_p", "top_k"):
        if neutralization.get(key) is not None:
            problems.append(f"{key}={neutralization.get(key)!r} (want null)")
    assert not problems, (
        "reference generation config is NOT neutralized (sec 12c / MODEL_SPEC F6): "
        + "; ".join(problems)
        + " -- HF applies repetition_penalty even under do_sample=False, so this "
        "reference is not greedy-comparable. Regenerate with gen_reference.py."
    )

    prompts = ref.get("prompts")
    assert isinstance(prompts, list) and prompts, "reference has no prompts"


def _crosscheck_prompts(ref: dict[str, Any]) -> None:
    """Ensure the reference was built from the current prompts.py (catch drift)."""
    expected = all_prompts()
    got = ref["prompts"]
    assert len(got) == len(expected), (
        f"reference has {len(got)} prompts, prompts.py has {len(expected)} -- "
        "stale reference; regenerate with gen_reference.py"
    )
    for exp, rec in zip(expected, got):
        assert rec["index"] == exp.index and rec["category"] == exp.category, (
            f"reference prompt {rec.get('index')} metadata drifted from prompts.py"
        )
        assert rec["prompt_text"] == exp.text, (
            f"reference prompt {exp.index} text differs from prompts.py -- "
            "stale reference; regenerate with gen_reference.py"
        )
        # Internal consistency of the reference itself.
        gen = rec["generated_token_ids"]
        topk_ids = rec["topk_ids"]
        margins = rec["margins"]
        assert len(gen) == len(topk_ids) == len(margins), (
            f"reference prompt {exp.index}: ragged per-step arrays"
        )
        assert gen and all(gen[n] == topk_ids[n][0] for n in range(len(gen))), (
            f"reference prompt {exp.index}: generated token != top-1 at some step"
        )


@pytest.fixture(scope="session")
def reference(pytestconfig: pytest.Config) -> dict[str, Any]:
    path = pytestconfig.getoption("--ref")
    if not path:
        pytest.skip("--ref not provided (run gen_reference.py first)")
    with open(path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    _validate_reference(ref)
    _crosscheck_prompts(ref)
    _REPORT["reference"] = {
        "path": path,
        "schema_version": ref.get("schema_version"),
        "meta": ref.get("meta"),
        "neutralization": ref.get("neutralization"),
        "prompts": len(ref["prompts"]),
        "positions": sum(len(r["generated_token_ids"]) for r in ref["prompts"]),
    }
    return ref


@pytest.fixture(scope="module")
def engine(model_dir: str, engine_kwargs: dict[str, Any]):
    """One engine shared by every test in this module, released at module end.

    Module scope (not session) with an explicit teardown: a whole-directory
    e2e run constructs further engines in later modules (suite d), and two
    engines never fit one device - weights + KV pool are duplicated per
    instance (e.g. 2 x ~12.4 GiB under the bench preset on a 24 GiB card).
    Same drop-then-collect discipline as test_invariance._engine.
    """
    redline = pytest.importorskip(
        "redline", reason="redline extension not built (set PYTHONPATH=$BUILD)"
    )
    eng = redline.Engine(model_dir, **engine_kwargs)
    _REPORT["engine"] = {
        "module_file": getattr(redline, "__file__", None),
        "module_version": getattr(redline, "__version__", None),
        "engine_kwargs": dict(engine_kwargs),
    }
    yield eng
    del eng
    gc.collect()


# --------------------------------------------------------------------------
# Engine driving
# --------------------------------------------------------------------------
def _drive(
    engine: Any,
    prompt_ids: list[int],
    max_new: int,
    *,
    ignore_eos: bool,
    forced: list[int] | None = None,
) -> list[int]:
    """Submit one request, pump ``step()`` to completion, return per-position
    engine argmax tokens (StepResult.token, which carries the engine's own
    sampled token even under teacher forcing)."""
    kwargs: dict[str, Any] = {"ignore_eos": ignore_eos}
    if forced is not None:
        kwargs["forced_tokens"] = list(forced)
    rid = engine.add_request(list(prompt_ids), int(max_new), **kwargs)

    out: list[int] = []
    guard = len(prompt_ids) + int(max_new) + 1024
    steps = 0
    done = False
    while not done:
        for req_id, token, finished, finish_reason in engine.step():
            if req_id != rid:
                continue
            if finish_reason == "aborted":
                raise AssertionError(
                    f"request {rid} was aborted mid-run (unexpected under this suite)"
                )
            out.append(token)
            if finished:
                done = True
        steps += 1
        if steps > guard:
            raise AssertionError(
                f"engine did not finish request {rid} within {guard} steps "
                f"(collected {len(out)}/{max_new} tokens)"
            )
    return out


@pytest.fixture(scope="module")
def engine_outputs(engine: Any, reference: dict[str, Any]) -> dict[str, list[list[int]]]:
    """Run teacher-forced and free-running decode once for every prompt.

    Module-scoped to match ``engine`` (a session-scoped fixture may not depend
    on a module-scoped one); this file is one module, so the outputs are still
    computed exactly once per run.
    """
    teacher: list[list[int]] = []
    free: list[list[int]] = []
    for rec in reference["prompts"]:
        prompt_ids = rec["prompt_token_ids"]
        ref_tokens = rec["generated_token_ids"]
        length = len(ref_tokens)

        tf = _drive(engine, prompt_ids, length, ignore_eos=True, forced=ref_tokens)
        assert len(tf) == length, (
            f"prompt {rec['index']}: teacher-forced run emitted {len(tf)} tokens, "
            f"expected {length}"
        )
        teacher.append(tf)

        fr = _drive(engine, prompt_ids, length, ignore_eos=True, forced=None)
        assert len(fr) == length, (
            f"prompt {rec['index']}: free-running run emitted {len(fr)} tokens, "
            f"expected {length}"
        )
        free.append(fr)

    return {"teacher": teacher, "free": free}


# --------------------------------------------------------------------------
# (1) Teacher-forced agreement - PRIMARY, gating
# --------------------------------------------------------------------------
def test_teacher_forced_agreement(
    reference: dict[str, Any],
    engine_outputs: dict[str, list[list[int]]],
    report_sink: list,
) -> None:
    prompts = reference["prompts"]
    total = 0
    agree = 0
    high_margin: list[str] = []  # disagreements with margin >= COINFLIP_MARGIN (fatal)
    coinflip = 0  # disagreements with margin < COINFLIP_MARGIN (allowed)

    for i, rec in enumerate(prompts):
        ref_tokens = rec["generated_token_ids"]
        margins = rec["margins"]
        eng_tokens = engine_outputs["teacher"][i]
        for n, (eng_tok, hf_tok, margin) in enumerate(zip(eng_tokens, ref_tokens, margins)):
            total += 1
            if eng_tok == hf_tok:
                agree += 1
                continue
            if margin >= COINFLIP_MARGIN:
                high_margin.append(
                    f"prompt {rec['index']} pos {n}: engine={eng_tok} hf={hf_tok} "
                    f"margin={margin:.4f} (>= {COINFLIP_MARGIN})"
                )
            else:
                coinflip += 1

    assert total > 0, "no positions compared -- empty reference"
    rate = agree / total

    # Receipt entry before the asserts: a failing run still leaves its numbers.
    _REPORT["teacher_forced_agreement"] = {
        "gating": True,
        "positions": total,
        "agreements": agree,
        "rate": rate,
        "coinflip_disagreements_below_margin": coinflip,
        "fatal_disagreements_at_or_above_margin": len(high_margin),
        "fatal_detail": high_margin[:20],
        "pass": not high_margin and rate >= AGREEMENT_FLOOR,
    }
    report_sink.append(
        f"teacher-forced agreement: {agree}/{total} = {rate:.4%} "
        f"(coin-flip disagreements < {COINFLIP_MARGIN}: {coinflip}; "
        f"fatal >= {COINFLIP_MARGIN}: {len(high_margin)})"
    )

    # A large-margin disagreement fails regardless of the aggregate rate.
    assert not high_margin, (
        f"{len(high_margin)} disagreement(s) exceed the FP16 coin-flip margin "
        f"({COINFLIP_MARGIN}) -- these are real numeric errors, not reduction-order "
        f"noise:\n  " + "\n  ".join(high_margin[:20])
    )
    assert rate >= AGREEMENT_FLOOR, (
        f"teacher-forced agreement {rate:.4%} < {AGREEMENT_FLOOR:.0%} over {total} positions "
        f"({total - agree} disagreements, all below the {COINFLIP_MARGIN} margin)"
    )


# --------------------------------------------------------------------------
# (3) First-token check - gating
# --------------------------------------------------------------------------
def test_first_token_check(
    reference: dict[str, Any],
    engine_outputs: dict[str, list[list[int]]],
    report_sink: list,
) -> None:
    prompts = reference["prompts"]
    failures: list[str] = []
    argmax_matches = 0

    for i, rec in enumerate(prompts):
        engine_first = engine_outputs["teacher"][i][0]  # position 0 == free-run first token
        hf_top5 = rec["topk_ids"][0][:5]
        hf_top1 = hf_top5[0]
        margin0 = rec["margins"][0]

        if engine_first == hf_top1:
            argmax_matches += 1
            continue
        if engine_first not in hf_top5:
            failures.append(
                f"prompt {rec['index']}: engine first token {engine_first} not in HF top-5 "
                f"{hf_top5} (margin={margin0:.4f})"
            )
        elif margin0 >= COINFLIP_MARGIN:
            failures.append(
                f"prompt {rec['index']}: engine first token {engine_first} != HF argmax "
                f"{hf_top1} with margin {margin0:.4f} (>= {COINFLIP_MARGIN})"
            )

    _REPORT["first_token_check"] = {
        "gating": True,
        "prompts": len(prompts),
        "exact_argmax_matches": argmax_matches,
        "failures": failures,
        "pass": not failures,
    }
    report_sink.append(
        f"first-token check: {argmax_matches}/{len(prompts)} exact argmax matches; "
        f"{len(failures)} failure(s)"
    )
    assert not failures, "first-token check failed:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------
# (2) Free-running greedy prefix match - SECONDARY, reported, NOT gating
# --------------------------------------------------------------------------
def test_free_running_prefix_match(
    reference: dict[str, Any],
    engine_outputs: dict[str, list[list[int]]],
    report_sink: list,
) -> None:
    prompts = reference["prompts"]
    lines = ["free-running greedy prefix match (informational, non-gating):"]
    matched_lengths: list[int] = []
    per_prompt: list[dict[str, Any]] = []

    for i, rec in enumerate(prompts):
        ref_tokens = rec["generated_token_ids"]
        margins = rec["margins"]
        eng_tokens = engine_outputs["free"][i]

        divergence = None
        for n, (eng_tok, hf_tok) in enumerate(zip(eng_tokens, ref_tokens)):
            if eng_tok != hf_tok:
                divergence = (n, eng_tok, hf_tok, margins[n])
                break

        matched = divergence[0] if divergence else min(len(eng_tokens), len(ref_tokens))
        matched_lengths.append(matched)
        per_prompt.append(
            {
                "index": rec["index"],
                "category": rec["category"],
                "matched_prefix": matched,
                "length": len(ref_tokens),
                "first_divergence": None
                if divergence is None
                else {
                    "pos": divergence[0],
                    "engine_token": divergence[1],
                    "hf_token": divergence[2],
                    "hf_margin": divergence[3],
                },
            }
        )

        if divergence is None:
            lines.append(
                f"  prompt {rec['index']:2d} [{rec['category']:6s}] "
                f"matched FULL {matched}/{len(ref_tokens)}"
            )
        else:
            pos, eng_tok, hf_tok, margin = divergence
            lines.append(
                f"  prompt {rec['index']:2d} [{rec['category']:6s}] "
                f"matched {matched}/{len(ref_tokens)}, first divergence at pos {pos}: "
                f"engine={eng_tok} hf={hf_tok} hf_margin={margin:.4f}"
            )

    if matched_lengths:
        avg = sum(matched_lengths) / len(matched_lengths)
        lines.append(
            f"  summary: mean matched prefix {avg:.1f} tokens; "
            f"min {min(matched_lengths)}, max {max(matched_lengths)}"
        )
    _REPORT["free_running_prefix_match"] = {
        "gating": False,
        "per_prompt": per_prompt,
        "mean_matched_prefix": (
            sum(matched_lengths) / len(matched_lengths) if matched_lengths else None
        ),
        "min_matched_prefix": min(matched_lengths) if matched_lengths else None,
        "max_matched_prefix": max(matched_lengths) if matched_lengths else None,
    }
    report = "\n".join(lines)
    report_sink.append(report)
    print(report)
    # Intentionally no assertion: divergence compounding under self-feeding is
    # expected on FP16 coin-flips; the teacher-forced metric is the gate.


# --------------------------------------------------------------------------
# (4) Forced-token echo tripwire - harness integrity, gating
# --------------------------------------------------------------------------
def test_forced_token_echo_tripwire(
    engine: Any,
    reference: dict[str, Any],
    report_sink: list,
) -> None:
    """Prove ``StepResult.token`` is the engine's argmax, not a forced-token echo.

    Failure mode guarded against: if the section-10 glue reported the appended
    ``forced_tokens[n]`` instead of the engine's own argmax, teacher-forced
    agreement and the first-token check would pass vacuously at 100% (upstream,
    tests/test_scheduler.cpp unit-asserts the same contract, but only up to the
    C++ scheduler -- this exercises it end-to-end through pybind).

    Mechanics: at step ``m`` the emission is conditioned only on the appended
    prefix ``0..m-1``, and the forced token at ``m`` is appended *after* that
    argmax is taken (ApplySampledToken). So corrupting the forced token
    at the reference's single highest-margin position ``m`` must NOT change the
    emission at ``m``: a correct engine reports its own argmax, which cannot
    plausibly be a token outside HF's top-k at a high-certainty position, while
    echoing glue reports the garbage token verbatim.
    """
    # Site the tripwire at the highest-margin position in the whole reference
    # (where the engine's genuine argmax is most certain).
    best: tuple[float, int, int] | None = None  # (margin, prompt_list_idx, pos)
    for i, rec in enumerate(reference["prompts"]):
        for n, margin in enumerate(rec["margins"]):
            if best is None or margin > best[0]:
                best = (float(margin), i, n)
    assert best is not None, "no positions in reference -- cannot site tripwire"
    margin, i, m = best
    rec = reference["prompts"][i]
    assert margin >= COINFLIP_MARGIN, (
        f"degenerate reference: max HF margin {margin:.4f} < {COINFLIP_MARGIN} across all "
        "positions, no high-certainty site for the echo tripwire -- regenerate the reference"
    )

    ref_tokens = rec["generated_token_ids"]
    # Garbage = smallest token id that is neither in HF's top-k at m nor an EOS
    # id: a valid vocab id (Qwen2 ids are dense from 0) that the engine's real
    # argmax at a margin>=0.1 position cannot legitimately equal.
    excluded = set(rec["topk_ids"][m]) | set(reference["meta"]["eos_token_ids"])
    garbage = next(t for t in range(len(excluded) + 1) if t not in excluded)

    corrupted = list(ref_tokens)
    corrupted[m] = garbage
    out = _drive(
        engine, rec["prompt_token_ids"], len(ref_tokens), ignore_eos=True, forced=corrupted
    )
    assert len(out) == len(ref_tokens), (
        f"tripwire run emitted {len(out)} tokens, expected {len(ref_tokens)}"
    )

    _REPORT["forced_token_echo_tripwire"] = {
        "gating": True,
        "prompt_index": rec["index"],
        "position": m,
        "hf_margin": margin,
        "garbage_token": garbage,
        "hf_argmax_token": ref_tokens[m],
        "engine_emitted": out[m],
        "pass": out[m] != garbage,
    }
    report_sink.append(
        f"forced-token echo tripwire: prompt {rec['index']} pos {m} forced garbage token "
        f"{garbage} over hf argmax {ref_tokens[m]} (hf_margin={margin:.4f}); "
        f"engine emitted {out[m]}"
    )
    assert out[m] != garbage, (
        f"engine emission at prompt {rec['index']} pos {m} EQUALS the forced garbage token "
        f"{garbage} (hf argmax {ref_tokens[m]}, hf_margin {margin:.4f} >= {COINFLIP_MARGIN}) "
        "-- StepResult.token is echoing forced_tokens instead of reporting the engine's own "
        "argmax, so the teacher-forced and first-token gates are vacuous. Check the "
        "pybind glue / ApplySampledToken emission path."
    )
