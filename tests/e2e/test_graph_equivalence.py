"""Suite (e) - graph equivalence (docs/DESIGN.md section 12e).

With identical request sets, ``enable_cuda_graphs`` on vs off must emit
identical tokens. Exact token equality across the two execution paths is only
sound under the same-shape rule of section 12d: both sides must run identical
GEMM shapes, or cuBLASLt may pick a different algorithm/split-K for a
different ``m`` and legitimately flip near-tie tokens. The graph path pads
every decode step to its bucket by construction - a replay executes at the
captured bucket shape with dummy rows in the padded slots (section 9) - so
every GATING comparison here runs the eager side with
``pad_eager_to_bucket=1``: pad every eager decode batch to the smallest
configured bucket >= the live batch, which is the SAME bucket the graph
replay uses. Identical shape => same cached cuBLASLt algo => each row's
result is independent of the other rows' contents => any token mismatch is
definitively a graph capture/replay bug (stale persistent-buffer contents
between replays, a missed contents-only mirror update, padded-slot handling),
never numerics. Gating assertions are exact token equality - no tolerance.
Paged-KV concepts follow Kwon et al. 2023 (docs/DESIGN.md section 17).

Protocol sides (per comparison; forced kwargs per side)
-------------------------------------------------------
  graph side - ``enable_cuda_graphs=True``: every decode step must REPLAY a
               captured bucket graph (audited via ``stats()``; a capture
               failure silently degrading to eager per section 9 would make
               this suite vacuously compare eager to eager, so zero replays
               or any eager decode step on this side FAILS);
  eager side - ``enable_cuda_graphs=False, pad_eager_to_bucket=1``: the
               permanently maintained eager path, padded to the same bucket
               ladder the graphs use.

Tests
-----
1. ``test_fixed_prompts_are_stable`` - pins every prompt, max_new schedule,
   and the mixed-finish schedule model by digest; CPU-only.
2. ``test_bucket_graph_vs_padded_eager_identical[b]`` - GATING, one per dev
   bucket b in {1,2,4,8}: b concurrent requests with equal max_new, so every
   decode step of both sides executes at exactly bucket b's shape.
3. ``test_mixed_finish_shrinking_batch_identical`` - GATING: 8 requests with
   staggered max_new so the live batch shrinks 8 -> 7 -> 5 -> 4 -> 3 -> 2 -> 1
   mid-run, crossing every dev bucket and exercising padded live sizes on
   both sides (7, 5 pad to bucket 8; 3 pads to bucket 4).
4. ``test_graph_vs_unpadded_eager_informational`` - INFORMATIONAL, never
   gates on token mismatch (section 12e): graphs on vs TRUE unpadded eager on
   the mixed-finish workload - the one workload where the unpadded side runs
   genuinely different GEMM shapes (m = 7/5/3 at the shrunken live sizes
   instead of the padded 8/8/4). Cross-shape near-tie flips are expected FP16
   behavior; per-request matched prefixes are reported, never asserted.

Engine-side contracts this suite depends on
-------------------------------------------
* ``enable_cuda_graphs`` Engine ctor kwarg (docs/DESIGN.md sections 9/10):
  True captures one decode graph per configured bucket at init and replays
  the smallest bucket covering the live batch; False bypasses capture.
* ``pad_eager_to_bucket`` Engine ctor debug kwarg (sections 10/12d/12e):
  integer, 0 = off; 1 = pad every eager decode batch to the smallest
  configured bucket >= the live batch (exactly the graph-replay padding).
* ``stats()`` counters (section 10): ``graph_replays`` / ``eager_decodes``
  split decode steps by execution path; ``bucket_histogram`` ({bucket:
  decode steps executed at exactly that bucket's shape}) is attributed from
  the EXECUTED row count - the replayed bucket, or what pad_eager_to_bucket
  padded to - which is what makes the same-bucket rule auditable here.
* cuBLASLt algo report via ``Engine.algo_report`` (sections 6.1/12d): configurations probed by BOTH sides must have selected
  identical algorithms; one-sided configs are recorded, never failed on.

Running
-------
    PYTHONPATH=$BUILD pytest $REPO/tests/e2e/test_graph_equivalence.py --model $MODEL -v

``--model`` / ``--engine-kwargs`` are registered by tests/e2e/conftest.py;
env fallbacks ``REDLINE_TEST_MODEL`` / ``REDLINE_ENGINE_KWARGS``
work without them. Under the bench-preset overlay
(``--engine-kwargs 'kv_pool_gb=8,max_batch=64,...'``) the workloads still cap
at 8 live sequences, so exactly the {1,2,4,8} prefix of the bigger bucket
ladder is exercised; max_batch < 8 fails loudly (the dev ladder premise).
Per-side protocol kwargs (graphs flag, padding, ``reserve_full`` - watermark
aborts would break token-equality gating) are always forced over the overlay.

Output JSON (auditable record): written on module teardown to
``$REDLINE_GRAPH_EQUIV_OUT`` or ``./graph_equivalence_report.json`` -
per-side engine stats, bucket/engagement audits, expected schedule, full
token streams, first divergences, and the Lt algo reports per engine phase.

Memory note: engines are constructed strictly sequentially (context manager
drops the previous one and runs gc.collect() first) - weights + KV pool are
~4 GiB per engine and the dev GPU has 6 GiB.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import pytest

# Shared option parsing (one coercion table for the whole e2e tree; the
# registering conftest sits in this directory, but insert it on sys.path
# explicitly so this module also imports standalone).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_options import get_option, parse_engine_kwargs  # noqa: E402

redline = pytest.importorskip(
    "redline",
    reason="built `redline` pybind module not importable - build it and run with "
    "PYTHONPATH=$BUILD (docs/DESIGN.md section 10)",
)

# --------------------------------------------------------------------------
# Fixed workload constants
# --------------------------------------------------------------------------

# Dev-preset decode bucket ladder (docs/DESIGN.md section 11). Every workload
# below keeps the live batch <= max(BUCKETS), so under a bigger ladder (the
# bench preset) the smallest-covering-bucket rule still selects exactly
# these four buckets and the schedule model below stays valid.
BUCKETS = (1, 2, 4, 8)

# Per-bucket gating workloads: bucket test b submits the first b requests of
# this fixed 8-prompt family (lengths include non-multiples of 16 so block
# tables straddle the 16-token block boundary), all with MAX_NEW tokens, so
# all finish on the same decode step and every decode step runs at batch b.
BUCKET_FAMILY_LENS = (192, 80, 144, 33, 128, 65, 96, 47)
MAX_NEW = 64

# Mixed-finish gating workload: 8 requests whose max_new values stagger the
# finishes so the live batch shrinks across buckets mid-run. A request with
# max_new=m takes its first token from its final prefill chunk and is then
# live for decode steps 1..m-1, so this multiset {7,13,13,19,25,31,37,43}
# (assigned non-monotonically: finish order != submission order) yields
#   decode steps  1..6   7..12  13..18  19..24  25..30  31..36  37..42
#   live batch      8      7       5       4       3       2       1
#   bucket          8      8       8       4       4       2       1
# - every dev bucket, with padded live sizes 7, 5 (bucket 8) and 3 (bucket
# 4) exercised on BOTH sides. Pinned as MIXED_EXPECTED_HISTOGRAM and
# cross-checked against the schedule model in test_fixed_prompts_are_stable.
MIXED_PROMPT_LENS = (128, 96, 160, 64, 144, 80, 112, 48)
MIXED_MAX_NEW = (43, 13, 7, 25, 13, 37, 19, 31)
MIXED_EXPECTED_HISTOGRAM = {8: 18, 4: 12, 2: 6, 1: 6}

# Deterministic prompt streams: the normative splitmix64 token-id generator
# of bench/FAIRNESS.md (identical constants to bench/workload.py and
# src/bench_main.cpp - duplicated here because tests/e2e does not import
# bench.*). Distinct (seed, stream) tags keep this suite's prompts disjoint
# from suite (d)'s.
SEED = 27
_MASK64 = (1 << 64) - 1
TOKEN_ID_LOW = 1000  # ordinary BPE ids only, far below specials (>= 151643)
TOKEN_ID_RANGE = 99_000
_STREAM_BUCKET = 9300  # + request index within the bucket family
_STREAM_MIXED = 9500  # + submission index of the mixed-finish wave

# Pinned digest of the complete fixed workload (prompts + lengths + max_new
# schedules + bucket ladder); see test_fixed_prompts_are_stable.
_WORKLOAD_DIGEST16 = "5993f08d9b975679"

# Engine construction: dev preset defaults (docs/DESIGN.md sections 10-11);
# --engine-kwargs / REDLINE_ENGINE_KWARGS overlays them (bench preset); the
# per-side FORCED_* kwargs always win (protocol requirements,
# see module docstring).
DEV_PRESET_KWARGS = {
    "kv_pool_gb": 1.0,
    "max_batch": 8,
    "max_seq_len": 2048,
    "prefill_chunk": 1024,
    "admission_policy": "reserve_full",
}
FORCED_GRAPH_KWARGS = {
    "enable_cuda_graphs": True,
    "pad_eager_to_bucket": 0,  # graph replay pads by construction; keep eager padding out
    "admission_policy": "reserve_full",
}
FORCED_EAGER_PADDED_KWARGS = {
    "enable_cuda_graphs": False,
    "pad_eager_to_bucket": 1,  # smallest bucket covering the live batch = the graph's bucket
    "admission_policy": "reserve_full",
}
FORCED_EAGER_UNPADDED_KWARGS = {
    "enable_cuda_graphs": False,
    "pad_eager_to_bucket": 0,
    "admission_policy": "reserve_full",
}

# --------------------------------------------------------------------------
# Module-level output report (written on teardown, even after failures)
# --------------------------------------------------------------------------

_REPORT = {
    "suite": "e (graph equivalence)",
    "design": "docs/DESIGN.md section 12e",
}
_ALGO_CACHE = {}  # sha16 of report -> stored label (dedupe across phases)

# Graph-side run of the mixed-finish workload, cached by merged-kwargs key so
# the informational test can reuse the gating test's run instead of paying a
# third engine construction (sound because replays are deterministic: same
# request set + same engine kwargs => byte-identical schedule and tokens).
# Only the INFORMATIONAL test reads this; gating tests always run fresh.
_MIXED_GRAPH_CACHE = {}


@pytest.fixture(scope="module", autouse=True)
def graph_equivalence_report(request):
    _REPORT["meta"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model_dir": get_option(request.config, "--model", "REDLINE_TEST_MODEL"),
        "engine_kwargs_option": get_option(
            request.config, "--engine-kwargs", "REDLINE_ENGINE_KWARGS"
        ),
        "forced_kwargs": {
            "graph": dict(FORCED_GRAPH_KWARGS),
            "eager_padded": dict(FORCED_EAGER_PADDED_KWARGS),
            "eager_unpadded": dict(FORCED_EAGER_UNPADDED_KWARGS),
        },
        "module_file": getattr(redline, "__file__", None),
        "module_version": getattr(redline, "__version__", None),
        "workload": {
            "algorithm": "splitmix64 (normative, bench/FAIRNESS.md)",
            "seed": SEED,
            "workload_digest16": _WORKLOAD_DIGEST16,
        },
    }
    yield _REPORT
    out = os.environ.get("REDLINE_GRAPH_EQUIV_OUT") or os.path.join(
        os.getcwd(), "graph_equivalence_report.json"
    )
    try:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_REPORT, fh, indent=2, default=str)
        os.replace(tmp, out)
        print(f"\n[graph-equivalence] report JSON written to {out}")
    except OSError as err:  # never mask test results with a reporting error
        warnings.warn(f"could not write graph-equivalence report JSON to {out}: {err}")


# --------------------------------------------------------------------------
# Options / fixtures - option access and --engine-kwargs parsing come from
# tests/e2e/engine_options.py (single implementation for the e2e tree); the
# `model_dir` session fixture comes from tests/e2e/conftest.py, which honors
# the same REDLINE_TEST_MODEL fallback this module used to duplicate.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_engine_kwargs(request):
    merged = dict(DEV_PRESET_KWARGS)
    text = get_option(request.config, "--engine-kwargs", "REDLINE_ENGINE_KWARGS")
    if text:
        try:
            merged.update(parse_engine_kwargs(text))
        except ValueError as error:
            pytest.fail(str(error))
    if int(merged.get("max_batch", 0)) < max(BUCKETS):
        pytest.fail(
            f"suite (e) exercises the dev bucket ladder {list(BUCKETS)}, so it requires "
            f"max_batch >= {max(BUCKETS)}; got max_batch={merged.get('max_batch')!r} from "
            "the engine-kwargs overlay"
        )
    return merged


# --------------------------------------------------------------------------
# Deterministic prompts (normative splitmix64, bench/FAIRNESS.md)
# --------------------------------------------------------------------------


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _token_id(seed, stream, position):
    x = ((seed << 40) ^ (stream << 20) ^ position) & _MASK64
    return TOKEN_ID_LOW + _splitmix64(x) % TOKEN_ID_RANGE


def _prompt(stream, length):
    return [_token_id(SEED, stream, p) for p in range(length)]


def _bucket_submissions(batch):
    """The per-bucket workload: the first `batch` requests of the fixed
    family, all sharing MAX_NEW + ignore_eos, so every sequence prefills
    before any decode step (dedicated prefill steps, docs/DESIGN.md section
    8) and all finish on the same decode step - the live batch is exactly
    `batch` for every decode step."""
    return [
        {
            "key": f"req{i}",
            "prompt": _prompt(_STREAM_BUCKET + i, BUCKET_FAMILY_LENS[i]),
            "max_new": MAX_NEW,
        }
        for i in range(batch)
    ]


def _mixed_submissions():
    """The mixed-finish workload (see the MIXED_* constants above)."""
    return [
        {
            "key": f"mixed{i}",
            "prompt": _prompt(_STREAM_MIXED + i, MIXED_PROMPT_LENS[i]),
            "max_new": MIXED_MAX_NEW[i],
        }
        for i in range(len(MIXED_MAX_NEW))
    ]


# --------------------------------------------------------------------------
# Schedule model (docs/DESIGN.md section 8, closed wave)
# --------------------------------------------------------------------------


def _expected_schedule(max_news):
    """Independent model of the section-8 step loop for a closed wave of
    requests submitted before the first step: every prompt prefills in
    dedicated prefill steps first (each emitting the request's first token
    from its final chunk), then decode steps run all live sequences; a
    request with max_new=m is live for decode steps 1..m-1. Returns the
    expected live-batch trajectory, the smallest-covering-bucket per step
    (over BUCKETS - valid because live <= max(BUCKETS) always, see the
    BUCKETS comment), the bucket histogram both sides must report, and the
    decode step/token counters."""
    live_by_step = []
    histogram = {}
    step = 1
    while True:
        live = sum(1 for m in max_news if m - 1 >= step)
        if live == 0:
            break
        bucket = min(b for b in BUCKETS if b >= live)
        histogram[bucket] = histogram.get(bucket, 0) + 1
        live_by_step.append(live)
        step += 1
    return {
        "live": live_by_step,
        "histogram": histogram,
        "decode_steps": len(live_by_step),
        "decode_tokens": sum(live_by_step),
    }


# --------------------------------------------------------------------------
# Engine driving helpers
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _engine(model_dir, kwargs):
    try:
        eng = redline.Engine(model_dir, **kwargs)
    except TypeError as err:
        pytest.fail(
            f"redline.Engine rejected kwargs {sorted(kwargs)}: {err}. Suite (e) needs the "
            "section-10 ctor (enable_cuda_graphs) plus the debug kwarg pad_eager_to_bucket "
            "(int; 0 = off, 1 = pad every eager decode batch to the smallest configured "
            "bucket >= the live batch) - see the engine-side contracts in this module's "
            "docstring (docs/DESIGN.md sections 10/12e)"
        )
    try:
        yield eng
    finally:
        # Engine dtor frees all device memory; the 6 GiB dev card cannot hold
        # two engines (weights + pool ~4 GiB each), so drop before the next one.
        del eng
        gc.collect()


def _drive(engine, submissions):
    """Submit all requests up front (closed wave), then pump step() to
    completion. Returns results mapping submission key -> {"tokens": [...],
    "finish_reason": str}."""
    id_to_key = {}
    for sub in submissions:
        rid = engine.add_request(list(sub["prompt"]), sub["max_new"], True)  # ignore_eos
        id_to_key[rid] = sub["key"]
    results = {sub["key"]: {"tokens": [], "finish_reason": None} for sub in submissions}
    finished = set()

    max_steps = 512 + 8 * (len(submissions) * 8 + sum(s["max_new"] for s in submissions))
    steps = 0
    empty_streak = 0
    while len(finished) < len(submissions):
        if steps >= max_steps:
            pytest.fail(
                f"engine did not drain {len(submissions)} requests within {max_steps} steps "
                f"({len(finished)} finished) - scheduler stall"
            )
        emissions = engine.step()
        steps += 1
        if not emissions:
            empty_streak += 1
            if empty_streak > 4096:
                pytest.fail("engine stalled: 4096 consecutive empty step() results")
            continue
        empty_streak = 0
        for rid, token, is_finished, reason in emissions:
            key = id_to_key.get(rid)
            if key is None:
                pytest.fail(f"engine emitted unknown request id {rid}")
            if key in finished:
                pytest.fail(f"engine emitted a token for already-finished request {key!r}")
            results[key]["tokens"].append(int(token))
            if is_finished:
                results[key]["finish_reason"] = reason
                finished.add(key)
    return results


def _check_run_mechanics(label, results, submissions, stats, stats_init):
    """Hard mechanical checks shared by every phase: forced lengths
    (ignore_eos => exactly max_new tokens, finish_reason 'length'), zero
    aborts (reserve_full is forced, so an abort is an engine bug, not load),
    and full block release after drain (free-list conservation)."""
    for sub in submissions:
        got = results[sub["key"]]
        if len(got["tokens"]) != sub["max_new"] or got["finish_reason"] != "length":
            pytest.fail(
                f"[{label}] request {sub['key']!r} produced {len(got['tokens'])} tokens with "
                f"finish_reason={got['finish_reason']!r}; expected exactly {sub['max_new']} "
                "tokens and 'length' (ignore_eos=True forces full length, docs/DESIGN.md "
                "section 8)"
            )
    aborts = stats.get("aborts") if isinstance(stats, dict) else None
    if aborts not in (None, 0):
        pytest.fail(
            f"[{label}] stats()['aborts'] == {aborts}; suite (e) forces reserve_full and its "
            "workloads trivially fit the pool, so aborts indicate an engine bug"
        )
    free_init = stats_init.get("free_blocks") if isinstance(stats_init, dict) else None
    free_after = stats.get("free_blocks") if isinstance(stats, dict) else None
    reserved_after = stats.get("reserved_blocks") if isinstance(stats, dict) else None
    if free_init is not None and free_after is not None and free_init != free_after:
        pytest.fail(
            f"[{label}] drained wave did not fully release its KV blocks: free_blocks "
            f"{free_init} -> {free_after} (docs/DESIGN.md section 8 finish handling)"
        )
    if reserved_after not in (None, 0):
        pytest.fail(
            f"[{label}] stats()['reserved_blocks'] == {reserved_after} after drain; "
            "expected 0 (all reservations released with their requests)"
        )


def _run_phase(model_dir, kwargs, submissions, entry, side):
    """One engine lifetime: build, drive the wave, capture stats + the algo
    report. Records everything into the report entry."""
    with _engine(model_dir, kwargs) as eng:
        stats_init = eng.stats()
        results = _drive(eng, submissions)
        stats = eng.stats()
        _check_run_mechanics(side, results, submissions, stats, stats_init)
        algo = _algo_report_of(eng)
    entry[f"{side}_engine_kwargs"] = dict(kwargs)
    entry[f"{side}_stats"] = stats
    entry[f"{side}_lt_algo_report"] = _store_algo_report(algo)
    return results, stats, algo


# --------------------------------------------------------------------------
# Audits: graph engagement, same-bucket rule, Lt algo consistency
# --------------------------------------------------------------------------


def _audit_graph_side_counters(entry, stats, expect):
    """The graphs-on side must have REPLAYED every decode step. Section 9's
    degrade rule turns a capture failure into silent eager execution with a
    one-time warning - acceptable in production, but fatal here: the suite
    would vacuously compare eager to eager. Every live batch in these
    workloads has a covering bucket, so eager decode steps on this side can
    only mean degraded capture or dispatch drift."""
    replays = stats.get("graph_replays")
    eager = stats.get("eager_decodes")
    entry["graph_engagement"] = {
        "graph_replays": replays,
        "eager_decodes": eager,
        "expected_decode_steps": expect["decode_steps"],
    }
    if not replays:
        pytest.fail(
            f"[graph] stats()['graph_replays'] == {replays!r}: the graphs-on engine never "
            "replayed a decode graph - capture failed and degraded to eager (section 9 "
            "one-time warning; check the engine init log line 'cuda_graphs=...') or the "
            "flag was dropped. Suite (e) cannot gate graph equivalence without replays "
            "(docs/DESIGN.md section 12e)."
        )
    if eager:
        pytest.fail(
            f"[graph] {eager} decode step(s) executed eagerly on the graphs-on side "
            f"(graph_replays={replays}); every live batch of this workload has a covering "
            "bucket, so eager decode steps mean a partially degraded capture ladder or "
            "bucket-dispatch drift (docs/DESIGN.md sections 9/12e)"
        )
    if replays != expect["decode_steps"]:
        pytest.fail(
            f"[graph] graph_replays == {replays}, but the section-8 schedule model expects "
            f"{expect['decode_steps']} decode steps for this wave (live trajectory "
            f"{expect['live']}) - the executed schedule diverged from the closed-wave model"
        )
    if stats.get("decode_tokens") != expect["decode_tokens"]:
        pytest.fail(
            f"[graph] decode_tokens == {stats.get('decode_tokens')}, expected "
            f"{expect['decode_tokens']} (sum of live rows over the modeled decode steps)"
        )


def _audit_eager_side_counters(entry, stats, expect):
    """The graphs-off side must have run every decode step on the eager path
    (a nonzero replay counter with enable_cuda_graphs=False is a kwargs
    mix-up or an engine bug), padded onto the bucket ladder."""
    replays = stats.get("graph_replays")
    eager = stats.get("eager_decodes")
    entry["eager_engagement"] = {
        "graph_replays": replays,
        "eager_decodes": eager,
        "expected_decode_steps": expect["decode_steps"],
    }
    if replays not in (None, 0):
        pytest.fail(
            f"[eager_padded] stats()['graph_replays'] == {replays} with "
            "enable_cuda_graphs=False - the eager side of the comparison replayed graphs; "
            "the two sides are not distinct execution paths (kwargs mix-up?)"
        )
    if eager != expect["decode_steps"]:
        pytest.fail(
            f"[eager_padded] eager_decodes == {eager}, but the section-8 schedule model "
            f"expects {expect['decode_steps']} decode steps for this wave (live trajectory "
            f"{expect['live']}) - the executed schedule diverged from the closed-wave model"
        )
    if stats.get("decode_tokens") != expect["decode_tokens"]:
        pytest.fail(
            f"[eager_padded] decode_tokens == {stats.get('decode_tokens')}, expected "
            f"{expect['decode_tokens']} (sum of live rows over the modeled decode steps)"
        )


def _normalize_bucket_histogram(value):
    """stats()['bucket_histogram'] as {bucket: count}; accepts a dict or a
    list of (bucket, count) pairs (the EngineStats layout). A bare count list
    is unparseable without the engine's bucket list; the audit then warns
    instead of guessing (the raw stats dump is recorded in the report JSON
    either way). None when unparseable."""
    try:
        if isinstance(value, dict):
            return {int(k): int(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            items = list(value)
            if not items:
                return {}
            if all(isinstance(e, (list, tuple)) and len(e) == 2 for e in items):
                return {int(b): int(c) for b, c in items}
    except (TypeError, ValueError):
        return None
    return None


def _audit_same_bucket(entry, graph_stats, eager_stats, expect):
    """The section-12e gating rule: the eager side must have executed every
    decode step at the SAME bucket the graph replay used. Both sides'
    bucket histograms must be identical AND equal the schedule model's
    histogram (which pins WHICH buckets the wave was designed to hit). The
    sides' step/prefill counters must also agree - a divergent schedule
    would void the whole comparison. Unparseable histograms only warn
    (attribution is a preferred contract; the engagement audits above
    already gate the path split)."""
    gh = _normalize_bucket_histogram(
        graph_stats.get("bucket_histogram") if isinstance(graph_stats, dict) else None
    )
    eh = _normalize_bucket_histogram(
        eager_stats.get("bucket_histogram") if isinstance(eager_stats, dict) else None
    )
    if gh is None or eh is None:
        entry["same_bucket_audit"] = "unattributed"
        warnings.warn(
            "bucket_histogram is unparseable on at least one side - cannot audit that the "
            "eager side padded to the same bucket the graph replayed (preferred contract: "
            "{bucket: decode steps executed at exactly that bucket's shape}, docs/DESIGN.md "
            "section 10)"
        )
        return
    graph_nz = {b: c for b, c in gh.items() if c}
    eager_nz = {b: c for b, c in eh.items() if c}
    if graph_nz != eager_nz:
        entry["same_bucket_audit"] = {"graph": graph_nz, "eager_padded": eager_nz}
        pytest.fail(
            f"[same-bucket rule] bucket histograms differ: graph side {graph_nz} vs "
            f"padded-eager side {eager_nz}; section 12e requires the eager side to run "
            "every decode step padded to the SAME bucket the graph used - the comparison "
            "is cross-shape and exact token equality is not a sound gate here"
        )
    if graph_nz != expect["histogram"]:
        entry["same_bucket_audit"] = {
            "both_sides": graph_nz,
            "expected": expect["histogram"],
        }
        pytest.fail(
            f"[same-bucket rule] both sides report bucket histogram {graph_nz}, but the "
            f"schedule model expects {expect['histogram']} (live trajectory "
            f"{expect['live']}) - the wave did not hit the buckets this test is required "
            "to exercise (docs/DESIGN.md section 12e)"
        )
    for counter in ("steps", "prefill_tokens", "decode_tokens"):
        if graph_stats.get(counter) != eager_stats.get(counter):
            pytest.fail(
                f"[same-bucket rule] stats()['{counter}'] differs between sides: graph "
                f"{graph_stats.get(counter)} vs padded-eager {eager_stats.get(counter)} - "
                "the two sides did not execute the same schedule"
            )
    entry["same_bucket_audit"] = {"histogram": graph_nz, "matches_model": True}


def _algo_report_of(engine):
    """cuBLASLt algo-selection report, probed exactly like
    bench/adapters/redline_a.py; parsed from JSON text when possible."""
    for attr in ("algo_report", "lt_algo_report", "algos"):
        probe = getattr(engine, attr, None)
        if callable(probe):
            try:
                report = probe()
            except Exception:
                return None
            if isinstance(report, str):
                try:
                    return json.loads(report)
                except ValueError:
                    return {"raw": report}
            return report
    return None


def _store_algo_report(report):
    """Dedupe identical reports across phases; full copies live under
    _REPORT['lt_algo_reports']."""
    if report is None:
        return None
    blob = json.dumps(report, sort_keys=True, default=str)
    key = hashlib.sha256(blob.encode()).hexdigest()[:16]
    if key in _ALGO_CACHE:
        return {"same_as": _ALGO_CACHE[key]}
    label = f"algo_report_{len(_ALGO_CACHE)}"
    _ALGO_CACHE[key] = label
    _REPORT.setdefault("lt_algo_reports", {})[label] = report
    return {"stored_as": label}


def _algo_identity(report):
    """{config-identity: selection outcome} from a parsed report; tolerant of
    the exact layout (looks for the list of per-configuration entries
    carrying a 'tag', per src/gemm/cublaslt_gemm.hpp AlgoReportJson). The key
    is the FULL configuration identity - the tag plus every descriptor field
    the entry carries - not the bare tag: runtime-probed GEMM plans
    legitimately share one call-site tag across many shapes. The
    outcome value is the availability verdict plus the selected algo."""
    if report is None:
        return None
    entries = None
    if isinstance(report, list):
        entries = report
    elif isinstance(report, dict):
        for value in report.values():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(e, dict) for e in value)
                and any("tag" in e for e in value)
            ):
                entries = value
                break
    if not entries or not all(isinstance(e, dict) for e in entries):
        return None
    identity = {}
    for e in entries:
        tag = e.get("tag") or e.get("name")
        if tag is None:
            continue
        config = {k: v for k, v in e.items() if k not in ("algo", "available")}
        outcome = {k: e[k] for k in ("available", "algo") if k in e}
        identity[json.dumps(config, sort_keys=True, default=str)] = json.dumps(
            outcome or e, sort_keys=True, default=str
        )
    return identity or None


def _tag_histogram(keys):
    """{tag: count} summary of config-identity keys (the full configs already
    live in the stored reports under _REPORT['lt_algo_reports'])."""
    hist = {}
    for key in keys:
        try:
            tag = str(json.loads(key).get("tag"))
        except (ValueError, AttributeError):
            tag = key
        hist[tag] = hist.get(tag, 0) + 1
    return hist


def _check_algo_consistency(entry, graph_report, eager_report):
    """The premise that makes exact token equality a sound gate is PER SHAPE:
    every GEMM configuration executed by BOTH sides must have selected the
    same algorithm (same algo => same split-K/reduction order). Any config
    probed by both sides with different outcomes breaks that premise and
    fails here. One-sided configs are recorded, never failed on - the two
    engines may legitimately probe at different times (e.g. decode-bucket
    plans during graph warmup vs on first eager decode; docs/DESIGN.md
    section 16)."""
    ids_graph = _algo_identity(graph_report)
    ids_eager = _algo_identity(eager_report)
    if ids_graph is None or ids_eager is None:
        entry["algo_consistency"] = "unavailable"
        warnings.warn(
            "engine exposes no parseable cuBLASLt algo report (probed algo_report / "
            "lt_algo_report / algos) - Lt algo IDs cannot be recorded; the shape-equality "
            "premise is not independently auditable in this run (docs/DESIGN.md section 12d)"
        )
        return
    common = set(ids_graph) & set(ids_eager)
    mismatched = sorted(key for key in common if ids_graph[key] != ids_eager[key])
    if mismatched:
        entry["algo_consistency"] = {
            "mismatched_configs": [
                {"config": key, "graph": ids_graph[key], "eager": ids_eager[key]}
                for key in mismatched
            ]
        }
        pytest.fail(
            f"graph and eager engines selected different cuBLASLt algorithms for "
            f"{len(mismatched)} configuration(s) probed by BOTH sides - the "
            "identical-shape => same-algo premise of suite (e) is broken "
            f"(docs/DESIGN.md sections 12d/12e). First mismatch: {mismatched[0]} -> "
            f"graph {ids_graph[mismatched[0]]} vs eager {ids_eager[mismatched[0]]}"
        )
    if not common:
        entry["algo_consistency"] = "no_common_configs"
        warnings.warn(
            "graph and eager algo reports share no GEMM configuration - the shape-equality "
            "premise is not independently auditable in this run (report layout drift?; "
            "docs/DESIGN.md section 12d)"
        )
        return
    entry["algo_consistency"] = {
        "identical_over_common_configs": True,
        "common_configs": len(common),
        "graph_only_config_tags": _tag_histogram(set(ids_graph) - common),
        "eager_only_config_tags": _tag_histogram(set(ids_eager) - common),
    }


# --------------------------------------------------------------------------
# Gating comparison
# --------------------------------------------------------------------------


def _assert_tokens_identical(entry, name, submissions, graph_results, eager_results):
    """Exact token equality per request - no tolerance. Failure prints the
    first divergent (request, position) and both tokens."""
    tokens = {}
    entry["tokens"] = tokens
    for sub in submissions:
        key = sub["key"]
        g = graph_results[key]["tokens"]
        e = eager_results[key]["tokens"]
        tokens[key] = {"graph": list(g), "eager_padded": list(e)}
        common = min(len(g), len(e))
        for pos in range(common):
            if g[pos] != e[pos]:
                entry["match"] = False
                entry["first_divergence"] = {
                    "request": key,
                    "pos": pos,
                    "graph_token": g[pos],
                    "eager_token": e[pos],
                }
                lo, hi = max(0, pos - 2), min(common, pos + 3)
                pytest.fail(
                    f"[{name}] token mismatch for request {key!r} at first divergent "
                    f"position {pos}: graph token {g[pos]} != padded-eager token {e[pos]} "
                    f"(context graph[{lo}:{hi}]={g[lo:hi]}, eager[{lo}:{hi}]={e[lo:hi]}). "
                    "Same-shape rule: both sides executed identical GEMM shapes at every "
                    "decode step, so this is a graph capture/replay bug (stale persistent-"
                    "buffer contents between replays, missed contents-only mirror update, "
                    "padded-slot handling), not numerics (docs/DESIGN.md sections 9/12e)."
                )
        if len(g) != len(e):
            entry["match"] = False
            entry["first_divergence"] = {
                "request": key,
                "pos": common,
                "graph_len": len(g),
                "eager_len": len(e),
            }
            pytest.fail(
                f"[{name}] request {key!r}: identical common prefix but different lengths "
                f"(graph {len(g)} tokens vs padded-eager {len(e)})"
            )
    entry["match"] = True


def _run_equivalence(name, model_dir, base_kwargs, submissions, expect, cache_graph=False):
    """One gating comparison: graphs-on run vs padded-eager run of the same
    request set; protocol audits first (a shape/engagement violation is the
    right first error), then exact token equality."""
    entry = {
        "name": name,
        "gating": True,
        "requests": [
            {"key": s["key"], "prompt_len": len(s["prompt"]), "max_new": s["max_new"]}
            for s in submissions
        ],
        "expected_schedule": {
            "live_trajectory": expect["live"],
            "histogram": {str(b): c for b, c in sorted(expect["histogram"].items())},
            "decode_steps": expect["decode_steps"],
            "decode_tokens": expect["decode_tokens"],
        },
        "match": None,
    }
    _REPORT.setdefault("comparisons", []).append(entry)

    graph_kwargs = {**base_kwargs, **FORCED_GRAPH_KWARGS}
    graph_results, graph_stats, graph_algo = _run_phase(
        model_dir, graph_kwargs, submissions, entry, "graph"
    )
    if cache_graph:
        _MIXED_GRAPH_CACHE[json.dumps(graph_kwargs, sort_keys=True, default=str)] = (
            graph_results,
            graph_stats,
            graph_algo,
        )

    eager_kwargs = {**base_kwargs, **FORCED_EAGER_PADDED_KWARGS}
    eager_results, eager_stats, eager_algo = _run_phase(
        model_dir, eager_kwargs, submissions, entry, "eager_padded"
    )

    _audit_graph_side_counters(entry, graph_stats, expect)
    _audit_eager_side_counters(entry, eager_stats, expect)
    _audit_same_bucket(entry, graph_stats, eager_stats, expect)
    _check_algo_consistency(entry, graph_algo, eager_algo)

    _assert_tokens_identical(entry, name, submissions, graph_results, eager_results)
    return entry


def _common_prefix(a, b):
    i = 0
    limit = min(len(a), len(b))
    while i < limit and a[i] == b[i]:
        i += 1
    return i


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_fixed_prompts_are_stable():
    """Guards the 'identical request sets' premise of section 12e: the whole
    workload (prompts, lengths, max_new schedules, bucket ladder) is pinned
    by digest, so a comparison can never drift because a constant was edited.
    Also self-checks the mixed-finish schedule model: it must cross every dev
    bucket, shrink monotonically, and contain padded live sizes. CPU-only;
    needs neither GPU nor checkpoint."""
    bucket_family = [_prompt(_STREAM_BUCKET + i, n) for i, n in enumerate(BUCKET_FAMILY_LENS)]
    mixed = [_prompt(_STREAM_MIXED + i, n) for i, n in enumerate(MIXED_PROMPT_LENS)]
    assert bucket_family[0][:4] == [17601, 7333, 70777, 52483]
    fixed = {
        "buckets": list(BUCKETS),
        "bucket_family_lens": list(BUCKET_FAMILY_LENS),
        "bucket_max_new": MAX_NEW,
        "mixed_lens": list(MIXED_PROMPT_LENS),
        "mixed_max_new": list(MIXED_MAX_NEW),
        "prompts": {"bucket_family": bucket_family, "mixed": mixed},
    }
    blob = json.dumps(fixed, sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:16] == _WORKLOAD_DIGEST16
    every_token = [t for group in fixed["prompts"].values() for row in group for t in row]
    assert all(TOKEN_ID_LOW <= t < TOKEN_ID_LOW + TOKEN_ID_RANGE for t in every_token)
    assert max(every_token) < 151_643  # below the special-token range (MODEL_SPEC section 6)

    # Per-bucket workloads: b equal-max_new requests decode at exactly bucket b.
    for b in BUCKETS:
        expect = _expected_schedule([MAX_NEW] * b)
        assert expect["histogram"] == {b: MAX_NEW - 1}
        assert expect["live"] == [b] * (MAX_NEW - 1)

    # Mixed-finish schedule model: crosses every dev bucket while shrinking,
    # with padded live sizes (no covering bucket equals them) on the way.
    assert min(MIXED_MAX_NEW) >= 2  # nobody may finish straight from prefill
    expect = _expected_schedule(MIXED_MAX_NEW)
    assert expect["histogram"] == MIXED_EXPECTED_HISTOGRAM
    assert set(expect["histogram"]) == set(BUCKETS)
    assert expect["live"] == sorted(expect["live"], reverse=True)  # shrinks mid-run
    assert set(expect["live"]) == {8, 7, 5, 4, 3, 2, 1}
    assert {7, 5, 3} == set(expect["live"]) - set(BUCKETS)  # padded live sizes
    assert expect["decode_tokens"] == sum(m - 1 for m in MIXED_MAX_NEW)


@pytest.mark.parametrize("bucket", BUCKETS)
def test_bucket_graph_vs_padded_eager_identical(model_dir, base_engine_kwargs, bucket):
    """GATING (section 12e, same-shape rule): `bucket` concurrent requests
    with equal max_new - every decode step of the graphs-on run replays the
    bucket-`bucket` graph and every decode step of the graphs-off run is
    padded eager at the same bucket; tokens must be byte-identical."""
    submissions = _bucket_submissions(bucket)
    expect = _expected_schedule([s["max_new"] for s in submissions])
    _run_equivalence(
        f"bucket{bucket}", model_dir, base_engine_kwargs, submissions, expect
    )


def test_mixed_finish_shrinking_batch_identical(model_dir, base_engine_kwargs):
    """GATING: the mixed-finish schedule - staggered max_new shrinks the live
    batch 8 -> 7 -> 5 -> 4 -> 3 -> 2 -> 1 mid-run, so one run replays (and
    the eager side pads to) buckets 8, 4, 2 and 1 in sequence, including
    padded live sizes 7, 5 (bucket 8) and 3 (bucket 4). Tokens of all eight
    requests must be byte-identical between the sides."""
    submissions = _mixed_submissions()
    expect = _expected_schedule(MIXED_MAX_NEW)
    _run_equivalence(
        "mixed_finish_shrinking_batch",
        model_dir,
        base_engine_kwargs,
        submissions,
        expect,
        cache_graph=True,
    )


def test_graph_vs_unpadded_eager_informational(model_dir, base_engine_kwargs, request):
    """INFORMATIONAL, never gates on token mismatch (section 12e): graphs on
    vs TRUE unpadded eager on the mixed-finish workload. The unpadded side
    runs the shrunken live batches at their real sizes (m = 7/5/3 GEMM rows
    where the graph side executed 8/8/4), so this is a cross-shape
    comparison: different cuBLASLt algo/split-K reduction orders may
    legitimately flip near-tie tokens. Reported per request; mechanical
    checks (forced lengths, zero aborts, block release) still apply."""
    submissions = _mixed_submissions()
    info = {
        "name": "graph_vs_unpadded_eager",
        "gating": False,
        "requests": [
            {"key": s["key"], "prompt_len": len(s["prompt"]), "max_new": s["max_new"]}
            for s in submissions
        ],
    }
    _REPORT.setdefault("informational", []).append(info)

    graph_kwargs = {**base_engine_kwargs, **FORCED_GRAPH_KWARGS}
    cached = _MIXED_GRAPH_CACHE.get(json.dumps(graph_kwargs, sort_keys=True, default=str))
    if cached is not None:
        # Reuse the gating test's graphs-on run (deterministic replays: same
        # request set + same kwargs => byte-identical tokens).
        graph_results, graph_stats, graph_algo = cached
        info["graph_run"] = "reused from test_mixed_finish_shrinking_batch_identical"
        info["graph_stats"] = graph_stats
        info["graph_lt_algo_report"] = _store_algo_report(graph_algo)
    else:
        graph_results, graph_stats, _ = _run_phase(
            model_dir, graph_kwargs, submissions, info, "graph"
        )

    unpadded_kwargs = {**base_engine_kwargs, **FORCED_EAGER_UNPADDED_KWARGS}
    eager_results, eager_stats, _ = _run_phase(
        model_dir, unpadded_kwargs, submissions, info, "eager_unpadded"
    )

    per_request = []
    identical = True
    matched = total = 0
    for sub in submissions:
        g = graph_results[sub["key"]]["tokens"]
        e = eager_results[sub["key"]]["tokens"]
        prefix = _common_prefix(g, e)
        rec = {
            "key": sub["key"],
            "max_new": sub["max_new"],
            "matched_prefix": prefix,
            "identical": prefix == len(g) == len(e),
        }
        if not rec["identical"]:
            identical = False
            rec["first_divergence"] = {
                "pos": prefix,
                "graph_token": g[prefix] if prefix < len(g) else None,
                "eager_token": e[prefix] if prefix < len(e) else None,
            }
        per_request.append(rec)
        matched += prefix
        total += len(g)
    info["per_request"] = per_request
    info["identical"] = identical

    if identical:
        summary = (
            f"[graph-equivalence/informational] graphs-on vs unpadded eager "
            f"(mixed-finish, cross-shape at live 7/5/3): identical tokens for all "
            f"{len(submissions)} requests ({total} tokens)"
        )
    else:
        diverged = [r for r in per_request if not r["identical"]]
        first = diverged[0]
        summary = (
            f"[graph-equivalence/informational, non-gating] graphs-on vs unpadded eager "
            f"diverged on {len(diverged)}/{len(submissions)} requests (matched "
            f"{matched}/{total} tokens); first: request {first['key']!r} at position "
            f"{first['first_divergence']['pos']} (graph token "
            f"{first['first_divergence']['graph_token']} vs eager "
            f"{first['first_divergence']['eager_token']}). Cross-shape near-tie flips are "
            "expected FP16 behavior (docs/DESIGN.md section 12e); full details in the "
            "graph-equivalence report JSON."
        )
    # Expected outcome either way - report, never warn: a warning would turn
    # into a failure under filterwarnings=error and this test must never gate
    # on token divergence.
    print("\n" + summary)
    # Surface through the shared terminal-summary sink when the shared conftest
    # is present (tests/e2e/conftest.py `report_sink`); harmless without it.
    try:
        request.getfixturevalue("report_sink").append(summary)
    except Exception:
        pass
