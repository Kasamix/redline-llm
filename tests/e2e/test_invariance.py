"""Suite (d) - paged/batching invariance (docs/DESIGN.md section 12d).

Byte-identical token equality across configurations is only sound when both
sides run identical GEMM shapes: a different ``m`` lets cuBLASLt pick a
different algorithm/split-K, changing FP32 reduction order and legitimately
flipping near-tie tokens. Every GATING comparison here therefore exploits the
engine's own bucket padding (same-shape protocol):

  solo run   - the tracked prompt greedy-decoded alone, with
               ``pad_eager_to_bucket=8`` so every decode step executes at
               bucket-8 shape (7 padded dummy rows);
  batch run  - the same prompt as one of 8 concurrent sequences (7 fixed
               filler prompts), same padding flag, so every decode step is
               also bucket-8 shape.

Identical shape => same cached cuBLASLt algo => each row's result is
independent of the other rows' contents => any token mismatch is definitively
a block-table / kv_scatter / paged-attention bug, never numerics. Gating
assertions are exact token equality - no tolerance. Paged-KV concepts follow
Kwon et al. 2023 (see docs/DESIGN.md section 17).

Tests
-----
1. ``test_fixed_prompts_are_stable``          - pins the fixed synthetic prompts (digest).
2. ``test_solo_padded_vs_batch8_identical``   - main gating comparison, 128-token prompt.
3. ``test_block_boundary_prompt_lengths_identical`` - gating, prompt lengths {15, 16, 17}
   (block size is 16 tokens; these straddle the first block boundary).
4. ``test_fragmented_pool_identical``         - gating; the allocator is pre-churned through
   the public API (staggered short requests) so the tracked sequence's physical blocks are
   non-contiguous, which the test asserts via the block-table debug hook.
5. ``test_batch1_unpadded_vs_batch8_informational`` - INFORMATIONAL, never gates on token
   mismatch: true batch-1 (unpadded) vs batch-8, reported with HF top1-top2 logit margins
   where a ``--ref`` reference (tests/e2e/gen_reference.py) is available.

Engine-side contracts this suite depends on
-------------------------------------------
* ``pad_eager_to_bucket`` Engine ctor debug kwarg (docs/DESIGN.md sections
  10/12d/12e): integer-valued, default 0/False = off.
  Value ``b`` pads every eager decode batch to the smallest configured bucket
  ``>= max(live_batch, b)`` with dummy rows (seq_len=0, dummy-block table
  rows). ``True == 1`` reproduces the suite-(e) behavior of padding to the
  same bucket a graph would use; this suite passes ``8``.
* Fragmentation probe: ``Engine.debug_block_table(req_id) -> list[int]``
  (allocated physical block ids of a live request, allocation order) or
  ``stats()["debug_block_tables"]`` (dict req_id -> list[int]). Host-side
  scheduler state, zero hot-path cost. Required by the fragmented-pool test.
* ``stats()["bucket_histogram"]`` should attribute padded eager decode steps
  to the bucket they were padded to (audited; warns if unattributed, fails if
  it shows a non-8 bucket was used).
* cuBLASLt algo report (docs/DESIGN.md sections 6.1/12d): probed via
  ``Engine.algo_report`` / ``lt_algo_report`` / ``algos`` (same probe order as
  bench/adapters/redline_a.py); recorded into the output JSON so the
  shape-equality premise stays auditable.

Running
-------
    PYTHONPATH=$BUILD pytest $REPO/tests/e2e/test_invariance.py --model $MODEL -v

``--model`` / ``--ref`` are registered by tests/e2e/conftest.py; env
fallbacks ``REDLINE_TEST_MODEL`` / ``REDLINE_HF_REF`` / ``REDLINE_ENGINE_KWARGS``
work without them. An ``--engine-kwargs 'k=v,k=v'`` option (the bench
preset) is honored when the conftest registers it; ``enable_cuda_graphs=False``
and ``admission_policy="reserve_full"`` are always forced - graphs-vs-eager
equality is suite (e), and watermark aborts would break token-equality gating.

Output JSON (auditable record): written on module teardown to
``$REDLINE_INVARIANCE_OUT`` or ``./invariance_report.json`` - comparisons,
first divergences, engine stats, block ids, bucket audits, and the Lt algo
reports per engine phase.

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

# Same-shape bucket of the gating protocol (docs/DESIGN.md section 12d).
BUCKET = 8

# Tracked prompt: 128 tokens (8 full KV blocks), 64 greedy tokens out.
TRACKED_LEN = 128
MAX_NEW = 64
# 7 fixed filler prompts for the concurrent-of-8 run (lengths in tokens).
FILLER_LENS = (96, 160, 64, 144, 80, 112, 48)
# Submission slot of the tracked prompt among the 8 ("one of 8", not first).
TRACKED_SUBMIT_INDEX = 3
# Block size is 16 tokens: {15,16,17} straddle the first block boundary.
BOUNDARY_LENS = (15, 16, 17)

# Fragmented-pool churn: 8 requests x 6 blocks each (84-token prompts,
# 84+max_new <= 96 = 6 blocks, so no decode-time allocation), with staggered
# max_new values chosen so the finish (block release) order interleaves the
# requests' block ranges: finish order is [0, 2, 4, 6, 1, 3, 5, 7]. The
# allocator free list is a LIFO stack (docs/DESIGN.md section 7), so any
# subsequent allocation window of >= 7 blocks crosses two released ranges
# that are 12 block ids apart - the tracked sequence's 8 prompt blocks are
# non-contiguous by construction, whatever the intra-request release order.
CHURN_PROMPT_LEN = 84
CHURN_MAX_NEW = (2, 9, 3, 10, 4, 11, 5, 12)

# Deterministic prompt streams: the normative splitmix64 token-id generator of
# bench/FAIRNESS.md (identical constants to bench/workload.py and
# src/bench_main.cpp - duplicated here because tests/e2e does not import
# bench.*). Distinct stream tags keep every prompt in this suite unique.
SEED = 25
_MASK64 = (1 << 64) - 1
TOKEN_ID_LOW = 1000  # ordinary BPE ids only, far below specials (>= 151643)
TOKEN_ID_RANGE = 99_000
_STREAM_CHURN = 9000
_STREAM_FILLER = 9100
_STREAM_TRACKED = 9200
QWEN25_VOCAB_SIZE = 151_936

# Pinned digest of every fixed prompt in this suite (test_fixed_prompts_are_stable).
_PROMPT_DIGEST16 = "6c21af2633f00270"

# Engine construction: dev preset defaults (docs/DESIGN.md sections 10-11);
# --engine-kwargs / REDLINE_ENGINE_KWARGS overlays them (bench preset);
# FORCED_KWARGS always win (protocol requirements, see docstring).
DEV_PRESET_KWARGS = {
    "kv_pool_gb": 1.0,
    "max_batch": 8,
    "max_seq_len": 2048,
    "prefill_chunk": 1024,
    "admission_policy": "reserve_full",
}
FORCED_KWARGS = {
    "enable_cuda_graphs": False,
    "admission_policy": "reserve_full",
}

# --------------------------------------------------------------------------
# Module-level output report (written on teardown, even after failures)
# --------------------------------------------------------------------------

_REPORT = {
    "suite": "d (paged/batching invariance)",
    "design": "docs/DESIGN.md section 12d",
}
_ALGO_CACHE = {}  # sha16 of report -> stored label (dedupe across phases)


@pytest.fixture(scope="module", autouse=True)
def invariance_report(request):
    _REPORT["meta"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model_dir": get_option(request.config, "--model", "REDLINE_TEST_MODEL"),
        "engine_kwargs_option": get_option(request.config, "--engine-kwargs", "REDLINE_ENGINE_KWARGS"),
        "forced_kwargs": dict(FORCED_KWARGS),
        "module_file": getattr(redline, "__file__", None),
        "module_version": getattr(redline, "__version__", None),
        "workload": {
            "algorithm": "splitmix64 (normative, bench/FAIRNESS.md)",
            "seed": SEED,
            "prompt_digest16": _PROMPT_DIGEST16,
        },
    }
    yield _REPORT
    out = os.environ.get("REDLINE_INVARIANCE_OUT") or os.path.join(
        os.getcwd(), "invariance_report.json"
    )
    try:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_REPORT, fh, indent=2, default=str)
        os.replace(tmp, out)
        print(f"\n[invariance] report JSON written to {out}")
    except OSError as err:  # never mask test results with a reporting error
        warnings.warn(f"could not write invariance report JSON to {out}: {err}")


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
    merged.update(FORCED_KWARGS)
    if int(merged.get("max_batch", 0)) < BUCKET:
        pytest.fail(
            f"suite (d) pads to bucket {BUCKET}, so it requires max_batch >= {BUCKET}; "
            f"got max_batch={merged.get('max_batch')!r} from the engine-kwargs overlay"
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


def _batch_submissions(tracked_prompt):
    """The 8 concurrent requests: 7 fixed fillers with the tracked prompt at
    submission slot TRACKED_SUBMIT_INDEX. All share max_new + ignore_eos so
    every sequence prefills before any decode step (dedicated prefill steps,
    docs/DESIGN.md section 8) and all finish on the same decode step."""
    subs = []
    filler = 0
    for i in range(BUCKET):
        if i == TRACKED_SUBMIT_INDEX:
            subs.append({"key": "tracked", "prompt": tracked_prompt, "max_new": MAX_NEW})
        else:
            subs.append(
                {
                    "key": f"filler{filler}",
                    "prompt": _prompt(_STREAM_FILLER + filler, FILLER_LENS[filler]),
                    "max_new": MAX_NEW,
                }
            )
            filler += 1
    return subs


# --------------------------------------------------------------------------
# Engine driving helpers
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _engine(model_dir, kwargs):
    try:
        eng = redline.Engine(model_dir, **kwargs)
    except TypeError as err:
        pytest.fail(
            f"redline.Engine rejected kwargs {sorted(kwargs)}: {err}. Suite (d) needs the "
            "section-10 ctor plus the debug kwarg pad_eager_to_bucket (int; 0/False = off, "
            "b = pad every eager decode batch to the smallest configured bucket >= "
            "max(live batch, b)) -- see the engine-side contracts in this module's "
            "docstring (docs/DESIGN.md sections 10/12d)"
        )
    try:
        yield eng
    finally:
        # Engine dtor frees all device memory; the 6 GiB dev card cannot hold
        # two engines (weights + pool ~4 GiB each), so drop before the next one.
        del eng
        gc.collect()


def _drive(engine, submissions, probe_key=None):
    """Submit all requests up front (closed wave), then pump step() to
    completion. Returns (results, probed_block_ids) where results maps
    submission key -> {"tokens": [...], "finish_reason": str}. When
    ``probe_key`` is given, the request's physical block ids are captured at
    its FIRST emission (prompt blocks are live and allocated at that point)."""
    id_to_key = {}
    for sub in submissions:
        rid = engine.add_request(list(sub["prompt"]), sub["max_new"], True)  # ignore_eos
        id_to_key[rid] = sub["key"]
    results = {sub["key"]: {"tokens": [], "finish_reason": None} for sub in submissions}
    finished = set()
    probed_blocks = None

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
            if probe_key is not None and key == probe_key and probed_blocks is None:
                probed_blocks = _block_ids(engine, rid)
            if is_finished:
                results[key]["finish_reason"] = reason
                finished.add(key)
    return results, probed_blocks


def _block_ids(engine, rid):
    """Physical block ids of a live request via the debug hook (see module
    docstring); None when the engine exposes neither surface."""
    probe = getattr(engine, "debug_block_table", None)
    if callable(probe):
        try:
            ids = probe(rid)
        except Exception:
            ids = None
        if ids:
            return [int(b) for b in ids]
    try:
        stats = engine.stats()
    except Exception:
        return None
    tables = stats.get("debug_block_tables") if isinstance(stats, dict) else None
    if isinstance(tables, dict):
        ids = tables.get(rid, tables.get(str(rid)))
        if ids:
            return [int(b) for b in ids]
    return None


def _check_run_mechanics(label, results, submissions, stats):
    """Hard mechanical checks shared by every run: forced lengths (ignore_eos
    => exactly max_new tokens, finish_reason 'length') and zero aborts
    (reserve_full is forced, so an abort is an engine bug, not load)."""
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
            f"[{label}] stats()['aborts'] == {aborts}; suite (d) forces reserve_full and its "
            "workloads trivially fit the pool, so aborts indicate an engine bug"
        )


def _churn_pool(engine, entry, side):
    """Fragment the free list through the public API only: run the staggered
    churn wave to completion and verify full release (free-list conservation)."""
    stats_before = engine.stats()
    subs = [
        {
            "key": f"churn{i}",
            "prompt": _prompt(_STREAM_CHURN + i, CHURN_PROMPT_LEN),
            "max_new": max_new,
        }
        for i, max_new in enumerate(CHURN_MAX_NEW)
    ]
    results, _ = _drive(engine, subs)
    stats_after = engine.stats()
    _check_run_mechanics(f"{side}:churn", results, subs, stats_after)
    free_before = stats_before.get("free_blocks") if isinstance(stats_before, dict) else None
    free_after = stats_after.get("free_blocks") if isinstance(stats_after, dict) else None
    if free_before is not None and free_after is not None and free_before != free_after:
        pytest.fail(
            f"[{side}] churn wave did not fully release its blocks: free_blocks "
            f"{free_before} -> {free_after}"
        )
    entry[f"{side}_churn"] = {
        "requests": len(subs),
        "free_blocks_before": free_before,
        "free_blocks_after": free_after,
    }


def _run_phase(model_dir, kwargs, submissions, entry, side, probe_key=None, churn=False):
    """One engine lifetime: (optional churn wave), the measured wave, stats and
    algo-report capture. Records everything into the report entry."""
    with _engine(model_dir, kwargs) as eng:
        if churn:
            _churn_pool(eng, entry, side)
        results, probed_blocks = _drive(eng, submissions, probe_key=probe_key)
        stats = eng.stats()
        _check_run_mechanics(side, results, submissions, stats)
        algo = _algo_report_of(eng)
    entry[f"{side}_engine_kwargs"] = dict(kwargs)
    entry[f"{side}_stats"] = stats
    if probe_key is not None:
        entry[f"{side}_block_ids"] = probed_blocks
    entry[f"{side}_lt_algo_report"] = _store_algo_report(algo)
    return results, probed_blocks, stats, algo


# --------------------------------------------------------------------------
# Audits: bucket histogram (same-shape), Lt algo report (auditable premise)
# --------------------------------------------------------------------------


def _normalize_bucket_histogram(value):
    """stats()['bucket_histogram'] as {bucket: count}; accepts a dict or a
    list of (bucket, count) pairs (the EngineStats layout, which stores pairs
    in bucket order). A bare count list is treated as unparseable: without the
    engine's configured bucket list its positions are ambiguous - they could
    be bucket ids or bucket-order indices ([0, 0, 0, 63] under dev-preset
    buckets {1, 2, 4, 8} most plausibly means 63 steps at bucket 8, but an
    index read would report bucket 3) - and guessing wrong would fabricate a
    same-shape violation. The audit then warns instead (the raw stats dump is
    recorded in the report JSON either way). None when unparseable."""
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


def _audit_padded_bucket(entry, side, stats, min_decode_steps):
    """Every decode step of a padded run must have executed at bucket 8 - this
    is what makes exact token equality a sound gate. A nonzero count at any
    other bucket fails; an empty/unattributed histogram only warns (recorded),
    since attribution of padded-eager steps is a preferred, not gating,
    contract (see the engine-side contracts in the module docstring)."""
    hist = _normalize_bucket_histogram(
        stats.get("bucket_histogram") if isinstance(stats, dict) else None
    )
    nonzero = {b: c for b, c in (hist or {}).items() if c}
    if not nonzero:
        entry[f"{side}_bucket_audit"] = "unattributed"
        warnings.warn(
            f"[{side}] bucket_histogram is empty/unparseable -- cannot audit that padded "
            "eager decode steps ran at bucket 8. Preferred contract: every decode step "
            "(graph replay or padded eager) increments its bucket's histogram entry, "
            "exposed as a dict or (bucket, count) pairs (engine-side contracts in the "
            "module docstring)."
        )
        return
    unexpected = sorted(set(nonzero) - {BUCKET})
    if unexpected:
        entry[f"{side}_bucket_audit"] = {"histogram": nonzero, "violation": unexpected}
        pytest.fail(
            f"[{side}] same-shape protocol violated: decode steps recorded at buckets "
            f"{unexpected} (histogram {nonzero}); with pad_eager_to_bucket={BUCKET} every "
            f"decode step must run at bucket {BUCKET} (docs/DESIGN.md section 12d)"
        )
    if nonzero.get(BUCKET, 0) < min_decode_steps:
        warnings.warn(
            f"[{side}] bucket-{BUCKET} step count {nonzero.get(BUCKET, 0)} is below the "
            f"expected minimum {min_decode_steps} - check histogram counting semantics"
        )
    entry[f"{side}_bucket_audit"] = {"bucket": BUCKET, "steps": nonzero.get(BUCKET, 0)}


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
    the exact layout (looks for the list of per-configuration entries carrying
    a 'tag', per src/gemm/cublaslt_gemm.hpp AlgoReportJson). The key is the
    FULL configuration identity - the tag plus every descriptor field the
    entry carries (m/n/k, ops, leading dims, epilogue, batch, strides) - not
    the bare tag: runtime-probed GEMM plans legitimately share one call-site
    tag ('linear', 'linear_bias') across many shapes, and a tag-keyed dict
    would collapse them into whichever shape happened to be probed last. The
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


def _check_algo_consistency(entry, solo_report, batch_report):
    """The premise that makes exact token equality a sound gate is PER SHAPE:
    every GEMM configuration executed by BOTH phases must have selected the
    same algorithm (same algo => same split-K/reduction order => each row's
    result independent of the other rows' contents). Any config probed by
    both phases with different outcomes breaks that premise and fails here.
    The engine probes plans lazily on first encounter (docs/DESIGN.md section
    16), so the two phases' REPORTED config sets differ by construction -
    only the batch phase executes the 7 filler prompts' prefill shapes. Those
    one-sided configs never touch the tracked sequence's rows; they are
    recorded for the audit trail (tag histogram here, full configs in the
    stored reports), never failed on."""
    ids_solo = _algo_identity(solo_report)
    ids_batch = _algo_identity(batch_report)
    if ids_solo is None or ids_batch is None:
        entry["algo_consistency"] = "unavailable"
        warnings.warn(
            "engine exposes no parseable cuBLASLt algo report (probed algo_report / "
            "lt_algo_report / algos) -- Lt algo IDs cannot be recorded; the shape-equality "
            "premise is not independently auditable in this run (docs/DESIGN.md section 12d)"
        )
        return
    common = set(ids_solo) & set(ids_batch)
    mismatched = sorted(key for key in common if ids_solo[key] != ids_batch[key])
    if mismatched:
        entry["algo_consistency"] = {
            "mismatched_configs": [
                {"config": key, "solo": ids_solo[key], "batch": ids_batch[key]}
                for key in mismatched
            ]
        }
        pytest.fail(
            f"solo and batch engines selected different cuBLASLt algorithms for "
            f"{len(mismatched)} configuration(s) probed by BOTH phases - the "
            "identical-shape => same-algo premise of suite (d) is broken "
            f"(docs/DESIGN.md section 12d). First mismatch: {mismatched[0]} -> "
            f"solo {ids_solo[mismatched[0]]} vs batch {ids_batch[mismatched[0]]}"
        )
    if not common:
        entry["algo_consistency"] = "no_common_configs"
        warnings.warn(
            "solo and batch algo reports share no GEMM configuration - the shape-equality "
            "premise is not independently auditable in this run (report layout drift?; "
            "docs/DESIGN.md section 12d)"
        )
        return
    entry["algo_consistency"] = {
        "identical_over_common_configs": True,
        "common_configs": len(common),
        "solo_only_config_tags": _tag_histogram(set(ids_solo) - common),
        "batch_only_config_tags": _tag_histogram(set(ids_batch) - common),
    }


# --------------------------------------------------------------------------
# Gating comparison
# --------------------------------------------------------------------------


def _assert_tokens_identical(entry, label, solo_tokens, batch_tokens):
    """Exact token equality - no tolerance. Failure prints the first divergent
    position and both tokens."""
    entry["solo_tokens"] = list(solo_tokens)
    entry["batch_tokens"] = list(batch_tokens)
    common = min(len(solo_tokens), len(batch_tokens))
    for pos in range(common):
        if solo_tokens[pos] != batch_tokens[pos]:
            entry["match"] = False
            entry["first_divergence"] = {
                "pos": pos,
                "solo_token": solo_tokens[pos],
                "batch_token": batch_tokens[pos],
            }
            lo, hi = max(0, pos - 2), min(common, pos + 3)
            pytest.fail(
                f"[{label}] token mismatch at first divergent position {pos}: "
                f"solo(padded-to-{BUCKET}) token {solo_tokens[pos]} != batch-of-{BUCKET} "
                f"token {batch_tokens[pos]} (context solo[{lo}:{hi}]={solo_tokens[lo:hi]}, "
                f"batch[{lo}:{hi}]={batch_tokens[lo:hi]}). Same-shape protocol: both runs "
                "executed identical GEMM shapes, so this is a block-table/kv_scatter/"
                "paged-attention bug, not numerics (docs/DESIGN.md section 12d)."
            )
    if len(solo_tokens) != len(batch_tokens):
        entry["match"] = False
        entry["first_divergence"] = {
            "pos": common,
            "solo_len": len(solo_tokens),
            "batch_len": len(batch_tokens),
        }
        pytest.fail(
            f"[{label}] identical common prefix but different lengths: solo produced "
            f"{len(solo_tokens)} tokens, batch produced {len(batch_tokens)}"
        )
    entry["match"] = True


def _is_contiguous_range(ids):
    ordered = sorted(ids)
    return ordered == list(range(ordered[0], ordered[0] + len(ordered)))


def _assert_fragmented(entry, label, solo_blocks, batch_blocks):
    for side, ids in (("solo", solo_blocks), ("batch", batch_blocks)):
        if not ids:
            pytest.fail(
                f"[{label}:{side}] token equality PASSED but fragmentation is unverifiable: "
                "the engine exposes no block-table debug hook. The fragmented-pool variant "
                "must assert the tracked sequence's physical block ids are non-contiguous "
                "(docs/DESIGN.md section 12d). Expose Engine.debug_block_table(req_id) -> "
                "list[int] or stats()['debug_block_tables'] (dict req_id -> list[int]) -- "
                "host-side scheduler state, zero hot-path cost (engine-side contracts in "
                "the module docstring)."
            )
        if len(ids) < 3:
            pytest.fail(
                f"[{label}:{side}] block-table probe returned {ids} - expected the tracked "
                "prompt's blocks (>= 3) at first-token time"
            )
        if _is_contiguous_range(ids):
            pytest.fail(
                f"[{label}:{side}] expected NON-contiguous physical blocks after allocator "
                f"churn, got {ids}: the fragmented-pool variant did not exercise scattered "
                "block placement (churn design: see the CHURN_MAX_NEW comment in this module)"
            )
        entry[f"{side}_blocks_noncontiguous"] = True


def _run_comparison(name, model_dir, base_kwargs, tracked_len, churn):
    tracked_prompt = _prompt(_STREAM_TRACKED + tracked_len, tracked_len)
    entry = {
        "name": name,
        "gating": True,
        "tracked_len": tracked_len,
        "max_new": MAX_NEW,
        "tracked_submit_index": TRACKED_SUBMIT_INDEX,
        "fragmented_pool": churn,
        "match": None,
    }
    _REPORT.setdefault("comparisons", []).append(entry)

    padded_kwargs = dict(base_kwargs)
    padded_kwargs["pad_eager_to_bucket"] = BUCKET

    solo_subs = [{"key": "tracked", "prompt": tracked_prompt, "max_new": MAX_NEW}]
    solo_results, solo_blocks, solo_stats, solo_algo = _run_phase(
        model_dir, padded_kwargs, solo_subs, entry, "solo", probe_key="tracked", churn=churn
    )

    batch_subs = _batch_submissions(tracked_prompt)
    batch_results, batch_blocks, batch_stats, batch_algo = _run_phase(
        model_dir, padded_kwargs, batch_subs, entry, "batch", probe_key="tracked", churn=churn
    )

    # Protocol audits first: a shape violation is the right first error.
    _audit_padded_bucket(entry, "solo", solo_stats, MAX_NEW - 1)
    _audit_padded_bucket(entry, "batch", batch_stats, MAX_NEW - 1)
    _check_algo_consistency(entry, solo_algo, batch_algo)

    _assert_tokens_identical(
        entry, name, solo_results["tracked"]["tokens"], batch_results["tracked"]["tokens"]
    )
    if churn:
        _assert_fragmented(entry, name, solo_blocks, batch_blocks)
    return entry


# --------------------------------------------------------------------------
# HF reference margin lookup for the informational comparison
# --------------------------------------------------------------------------


def _first_typed_list(record, keys, kind):
    for key in keys:
        value = record.get(key)
        if isinstance(value, list) and value:
            if kind is int and all(
                isinstance(v, int) and not isinstance(v, bool) for v in value
            ):
                return [int(v) for v in value]
            if kind is float and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            ):
                return [float(v) for v in value]
    return None


def _iter_ref_records(data):
    if isinstance(data, list):
        for i, e in enumerate(data):
            if isinstance(e, dict):
                yield e, f"[{i}]"
        return
    if not isinstance(data, dict):
        return
    for key in ("prompts", "records", "results", "reference", "data"):
        value = data.get(key)
        if isinstance(value, list):
            for i, e in enumerate(value):
                if isinstance(e, dict):
                    yield e, f"{key}[{i}]"
        elif isinstance(value, dict):
            for sub_key, e in value.items():
                if isinstance(e, dict):
                    yield e, f"{key}[{sub_key}]"


def _ref_record(config, base_kwargs):
    """Best-effort extraction of one usable prompt record (prompt token ids +
    generated tokens + per-step top1-top2 margins) from the gen_reference.py reference
    JSON. Schema-tolerant and entirely optional: returns None on any doubt -
    the informational test then reports margins as unavailable."""
    path = get_option(config, "--ref", "REDLINE_HF_REF")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    max_prompt = min(512, int(base_kwargs.get("max_seq_len", 2048)) - MAX_NEW)
    for record, label in _iter_ref_records(data):
        prompt_ids = _first_typed_list(
            record,
            ("prompt_token_ids", "prompt_ids", "input_ids", "prompt_tokens", "context_token_ids"),
            int,
        )
        gen_tokens = _first_typed_list(
            record,
            (
                "generated_tokens",
                "generated_token_ids",
                "output_token_ids",
                "output_tokens",
                "tokens",
                "gen_tokens",
            ),
            int,
        )
        margins = _first_typed_list(
            record, ("top1_top2_margins", "margins", "top12_margins", "step_margins"), float
        )
        if not prompt_ids or not gen_tokens:
            continue
        if not 8 <= len(prompt_ids) <= max_prompt:
            continue
        if not all(0 <= t < QWEN25_VOCAB_SIZE for t in prompt_ids):
            continue
        return {"prompt_ids": prompt_ids, "tokens": gen_tokens, "margins": margins, "label": label}
    return None


def _common_prefix(a, b):
    i = 0
    limit = min(len(a), len(b))
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _informational_divergences(batch1_tokens, batch8_tokens, ref):
    divergences = []
    for pos in range(min(len(batch1_tokens), len(batch8_tokens))):
        if batch1_tokens[pos] == batch8_tokens[pos]:
            continue
        d = {
            "pos": pos,
            "batch1_token": batch1_tokens[pos],
            "batch8_token": batch8_tokens[pos],
            "hf_margin": None,
        }
        if ref is None:
            d["hf_margin_note"] = "unavailable (no --ref / REDLINE_HF_REF reference)"
        else:
            on_hf_path = (
                batch1_tokens[:pos] == batch8_tokens[:pos] == ref["tokens"][:pos]
            )
            if on_hf_path and ref.get("margins") and pos < len(ref["margins"]):
                d["hf_margin"] = float(ref["margins"][pos])
            elif on_hf_path:
                d["hf_margin_note"] = "reference carries no margin at this position"
            else:
                d["hf_margin_note"] = "off the HF greedy path at this position"
        divergences.append(d)
        if len(divergences) >= 16:
            break
    return divergences


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_fixed_prompts_are_stable():
    """Guards the 'fixed prompts' premise of section 12d: every prompt this
    suite submits is pinned by digest, so a comparison can never drift because
    a constant was edited. CPU-only; needs neither GPU nor checkpoint."""
    tracked = _prompt(_STREAM_TRACKED + TRACKED_LEN, TRACKED_LEN)
    assert tracked[:4] == [18471, 27314, 28159, 83207]
    fixed = {
        "tracked": tracked,
        "fillers": [_prompt(_STREAM_FILLER + i, n) for i, n in enumerate(FILLER_LENS)],
        "churn": [_prompt(_STREAM_CHURN + i, CHURN_PROMPT_LEN) for i in range(len(CHURN_MAX_NEW))],
        "b15": _prompt(_STREAM_TRACKED + 15, 15),
        "b16": _prompt(_STREAM_TRACKED + 16, 16),
        "b17": _prompt(_STREAM_TRACKED + 17, 17),
    }
    blob = json.dumps(fixed, sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:16] == _PROMPT_DIGEST16
    every_token = [t for group in fixed.values() for row in (group if isinstance(group[0], list) else [group]) for t in row]
    assert all(TOKEN_ID_LOW <= t < TOKEN_ID_LOW + TOKEN_ID_RANGE for t in every_token)
    assert max(every_token) < 151_643  # below the special-token range (MODEL_SPEC section 6)


def test_solo_padded_vs_batch8_identical(model_dir, base_engine_kwargs):
    """GATING (section 12d same-shape protocol): the tracked prompt decoded
    alone with pad_eager_to_bucket=8 must be byte-identical in tokens to the
    same prompt decoded as one of 8 concurrent sequences."""
    _run_comparison("clean_pool_len128", model_dir, base_engine_kwargs, TRACKED_LEN, churn=False)


@pytest.mark.parametrize("tracked_len", BOUNDARY_LENS)
def test_block_boundary_prompt_lengths_identical(model_dir, base_engine_kwargs, tracked_len):
    """GATING: boundary prompt lengths {15, 16, 17} around the 16-token block
    size - the first prompt block is partially filled / exactly filled /
    spills one token into the second block."""
    _run_comparison(
        f"boundary_len{tracked_len}", model_dir, base_engine_kwargs, tracked_len, churn=False
    )


def test_fragmented_pool_identical(model_dir, base_engine_kwargs):
    """GATING: same comparison on a fragmented pool. Both engines first run
    the staggered churn wave through the public API so the free list is a
    permutation of interleaved block ranges; the tracked sequence's physical
    blocks are then non-contiguous on BOTH sides (asserted via the debug hook;
    the two runs' placements also differ by construction, but only per-side
    non-contiguity is asserted) - yet tokens must be byte-identical."""
    _run_comparison(
        "fragmented_pool_len128", model_dir, base_engine_kwargs, TRACKED_LEN, churn=True
    )


def test_batch1_unpadded_vs_batch8_informational(model_dir, base_engine_kwargs, request):
    """INFORMATIONAL, never gates on token mismatch (section 12d): true
    batch-1 (unpadded, bucket-1 GEMM shapes) vs batch-of-8. Cross-shape runs
    may legitimately flip near-tie tokens through different cuBLASLt
    algo/split-K reduction orders; divergences are reported with HF top1-top2
    margins when a reference is supplied (--ref / REDLINE_HF_REF)."""
    ref = _ref_record(request.config, base_engine_kwargs)
    if ref is not None:
        tracked_prompt = ref["prompt_ids"]
        prompt_source = f"hf_reference:{ref['label']}"
    else:
        tracked_prompt = _prompt(_STREAM_TRACKED + TRACKED_LEN, TRACKED_LEN)
        prompt_source = "synthetic"
    info = {
        "name": "batch1_unpadded_vs_batch8",
        "gating": False,
        "prompt_source": prompt_source,
        "prompt_len": len(tracked_prompt),
        "max_new": MAX_NEW,
    }
    _REPORT.setdefault("informational", []).append(info)

    padded_kwargs = dict(base_engine_kwargs)
    padded_kwargs["pad_eager_to_bucket"] = BUCKET
    batch_results, _, _, _ = _run_phase(
        model_dir, padded_kwargs, _batch_submissions(tracked_prompt), info, "batch8"
    )

    # True batch-1: no padding kwarg at all (engine default off). Strip any
    # user-supplied pad_eager_to_bucket from the --engine-kwargs overlay -
    # a padded "batch-1" would silently void the unpadded premise of this
    # comparison (the strip is recorded for the report JSON).
    batch1_kwargs = dict(base_engine_kwargs)
    if batch1_kwargs.pop("pad_eager_to_bucket", None) not in (None, 0, False):
        info["batch1_stripped_pad_eager_to_bucket"] = True
    solo_subs = [{"key": "tracked", "prompt": tracked_prompt, "max_new": MAX_NEW}]
    solo_results, _, _, _ = _run_phase(
        model_dir, batch1_kwargs, solo_subs, info, "batch1"
    )

    batch1 = solo_results["tracked"]["tokens"]
    batch8 = batch_results["tracked"]["tokens"]
    prefix = _common_prefix(batch1, batch8)
    info["matched_prefix"] = prefix
    info["identical"] = prefix == len(batch1) == len(batch8)
    info["divergences"] = _informational_divergences(batch1, batch8, ref)
    if ref is not None:
        info["hf_prefix_batch1"] = _common_prefix(batch1, ref["tokens"])
        info["hf_prefix_batch8"] = _common_prefix(batch8, ref["tokens"])

    if info["identical"]:
        summary = (
            f"[invariance/informational] batch-1 unpadded vs batch-8: identical tokens "
            f"({len(batch1)} tokens, prompt={prompt_source})"
        )
        print("\n" + summary)
    else:
        first = info["divergences"][0] if info["divergences"] else {"pos": prefix}
        summary = (
            "[invariance/informational, non-gating] batch-1 unpadded vs batch-8 diverged "
            f"at position {first.get('pos')}: token {first.get('batch1_token')} vs "
            f"{first.get('batch8_token')} (hf_margin={first.get('hf_margin')}, "
            f"note={first.get('hf_margin_note', '')!r}); matched prefix {prefix}/{MAX_NEW}; "
            f"prompt={prompt_source}. Cross-shape near-tie flips are expected FP16 behavior "
            "(docs/DESIGN.md section 12d); full details in the invariance report JSON."
        )
        # Expected outcome of a cross-shape comparison - report, never warn:
        # a warning would turn into a failure under filterwarnings=error and
        # this test must never gate on token divergence.
        print("\n" + summary)
    # Surface through the shared terminal-summary sink when the shared conftest
    # is present (tests/e2e/conftest.py `report_sink`); harmless without it.
    try:
        request.getfixturevalue("report_sink").append(summary)
    except Exception:
        pass
