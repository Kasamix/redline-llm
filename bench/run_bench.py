"""Benchmark runner - the measured path for ALL published cross-engine numbers.

Every engine (redline included) runs through this harness with the same
asyncio client structure, so client overhead is shared rather than
engine-specific (bench/FAIRNESS.md "Measured path"). The redline_bench C++
CLI is a profiling tool only.

Usage::

    python -m bench.run_bench --engine redline --case primary --trials 3 \\
        --model-dir /path/to/Qwen2.5-1.5B-Instruct \\
        --out bench/results/raw/redline_primary.json

Metrics, computed engine-agnostically from per-token timestamps (normative
definitions in bench/FAIRNESS.md "Metrics"):
  - output token throughput (headline) - sum_i n_i / (max_i t_i,last -
    min_i t_sub_i); the full wall window, INCLUDING all prefill work, and
    labeled as such (no shared-window metric may claim "prefill excluded")
  - per-request decode rate (secondary; genuinely prefill-free) -
    (n_i - 1) / (t_i,last - t_i,0) per request, reported as a distribution
  - TTFT - t_i,0 - t_sub_i, p50/p99 across requests
  - ITL p50 / p99 - inter-token latencies pooled across requests

``--offline-crosscheck`` (vLLM) reruns the same workload through the engine's
offline batch entrypoint (``LLM.generate``) in place of the streaming client,
with the same warmup count. Only aggregate wall time exists on that path (no
per-token timestamps), so the output JSON carries ``mode:
"offline_crosscheck"`` and an ``offline`` block instead of ``trials``.
Comparing its throughput against the streaming run's bounds the streaming
client's overhead (bench/FAIRNESS.md "Measured path").

``--equivalence-gate`` runs the cross-engine output-equivalence gate
(bench/FAIRNESS.md "Model and inputs") instead of a measured case::

    python -m bench.run_bench --equivalence-gate --engines redline,vllm,llamacpp \\
        --model-dir /path/to/Qwen2.5-1.5B-Instruct \\
        --out bench/results/equivalence/transcript.json

The 5 short greedy prompts run through every engine sequentially (one engine
on the GPU at a time; ``--llamacpp-server-cmd`` lets the gate own the
llama-server lifecycle around the llamacpp turn). Token streams must match,
or every first divergence must be reviewed as a documented FP16 coin-flip
(HF top1-top2 logit margin < 0.1, the section-12c protocol; the review runs
automatically when divergences exist). The transcript JSON is committed next
to the results; a failing gate (exit 3) blocks every measured trial.

Protocol: one unmeasured warmup run, then 3 measured trials per engine/case;
the median trial (by headline throughput) is reported with min-max across
trials; dispersion rule (spread > 3% of median on the headline metric -> one
additional measured trial, flag if the spread persists); all raw trials kept.
Persistence is evaluated over ALL measured trials including the re-run
(every raw trial is committed): max-min never shrinks when a trial is
appended, so a fired rule publishes flagged in all but degenerate cases --
deliberately conservative (over-flagging, never under-flagging).
The harness asserts every request produced exactly its forced output length
(EOS ignored on every engine) and refuses the trial otherwise (exit code 2).
Environment (GPU via `nvidia-smi -q`, CPU via `lscpu`, RAM, kernel, compiler
and library versions, `pip freeze`, repo commit, exact command line) is
captured into the output JSON.

Results JSON schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "engine": "redline" | "vllm" | "vllm024" | "llamacpp",
                                    # the two vLLM baseline arms are distinct
                                    # engines end to end (same adapter code
                                    # path, own venv/freeze file per arm -
                                    # bench/FAIRNESS.md "Engine configuration")
      "case": "primary" | "arrival" | "longout" | "batch1",
      "seed": int,
      "workload": {"name", "num_requests", "input_len", "output_len",
                   "arrival_stagger_s"},
      "engine_config": <adapter.version_info(): engine pins + full config>,
      "environment": <capture_environment()>,
      "warmup": [<metrics dict per warmup run>],          # unmeasured
      "trials": [                                         # every measured trial,
        {"index", "started_utc", "metrics": {...},        # incl. dispersion re-run
         "requests": [{"request_id", "enqueue_time_s", "token_times_s",
                       "token_ids", "finish_reason"}, ...]},   # timestamps rebased
      ],                                                  # to the trial's first
      "median_trial_index": int,                          # submit, seconds
      "metrics": {<name>: {"value": <median trial>, "min", "max", "label"}},
      "dispersion": {"metric", "threshold_fraction", "values",
                     "reran": bool, "flag": bool},
      "forced_length_ok": true,                           # false never written:
      "engine_stats": <redline Engine.stats() dump> | null,   # violations abort
      "lt_algo_report": <cuBLASLt algo dump when the engine exposes one> | null
    }
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import json
import platform
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from bench import adapters as adapters_mod
from bench.adapters import EngineAdapter, RequestTrace
from bench.workload import CASES, Workload, equivalence_workload

RESULTS_SCHEMA_VERSION = 1

HEADLINE_METRIC = "throughput_tok_s"
DISPERSION_FRACTION = 0.03  # FAIRNESS.md "Trials, dispersion, reporting"

# Cross-engine output-equivalence gate: a divergence is a documented FP16
# coin-flip only when HF's top1-top2 logit margin at that position is below
# this threshold (docs/DESIGN.md section 12c, the same protocol as the
# HF-parity suite; bench/FAIRNESS.md "Model and inputs").
EQUIVALENCE_MARGIN = 0.1

METRIC_LABELS = {
    "throughput_tok_s": (
        "output token throughput over the full wall window "
        "(includes all prefill work), tok/s"
    ),
    "decode_rate_p50_tok_s": (
        "per-request decode rate (n_i - 1) / (t_i,last - t_i,0), "
        "p50 across requests, tok/s"
    ),
    "ttft_p50_s": "time to first token, p50 across requests, s",
    "ttft_p99_s": "time to first token, p99 across requests, s",
    "itl_p50_s": "inter-token latency pooled across requests, p50, s",
    "itl_p99_s": "inter-token latency pooled across requests, p99, s",
}


class ForcedLengthError(RuntimeError):
    """A request did not produce exactly its forced output length; the trial
    is void (bench/FAIRNESS.md "Cases")."""


# --------------------------------------------------------------------------
# Metric functions (pure; CPU-only doctest/pytest coverage)
# --------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile over ``values`` (numpy's default rule):
    rank = p/100 * (n-1) on the sorted data.

    >>> percentile([1.0, 2.0, 3.0, 4.0], 50)
    2.5
    >>> round(percentile([1.0, 2.0, 3.0, 4.0], 99), 6)
    3.97
    >>> percentile([10.0], 99)
    10.0
    """
    if not values:
        raise ValueError("percentile of empty list")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (p / 100.0) * (len(data) - 1)
    low = int(rank)
    high = min(low + 1, len(data) - 1)
    fraction = rank - low
    return data[low] + fraction * (data[high] - data[low])


def inter_token_latencies(trace: RequestTrace) -> list[float]:
    """Successive token deltas within one request, first token excluded.

    >>> t = RequestTrace(0, 0.0, token_times=[1.0, 1.5, 2.5], token_ids=[7, 8, 9])
    >>> inter_token_latencies(t)
    [0.5, 1.0]
    """
    times = trace.token_times
    return [times[i] - times[i - 1] for i in range(1, len(times))]


def compute_metrics(traces: list[RequestTrace]) -> dict:
    """Engine-agnostic metrics from raw per-token timestamps (definitions
    normative in bench/FAIRNESS.md "Metrics").

    Requests with a single token contribute no ITL and no decode rate (both
    need at least two tokens); every case in this harness forces >= 32. A
    request whose tokens all share one timestamp (e.g. a single-chunk
    delivery) contributes ITLs of zero but no decode rate: its decode span
    is zero, so the rate is undefined rather than infinite.

    >>> a = RequestTrace(0, 0.0, token_times=[1.0, 1.1, 1.2, 1.3],
    ...                  token_ids=[1, 2, 3, 4])
    >>> b = RequestTrace(1, 0.5, token_times=[2.0, 2.2, 2.4, 2.6],
    ...                  token_ids=[5, 6, 7, 8])
    >>> m = compute_metrics([a, b])
    >>> m["total_new_tokens"], round(m["wall_window_s"], 6)
    (8, 2.6)
    >>> round(m["throughput_tok_s"], 4)       # 8 tokens / 2.6 s, incl. prefill
    3.0769
    >>> round(m["ttft_p50_s"], 6), round(m["ttft_p99_s"], 6)
    (1.25, 1.495)
    >>> round(m["itl_p50_s"], 6), round(m["itl_p99_s"], 6)
    (0.15, 0.2)
    >>> round(m["decode_rate_p50_tok_s"], 3)  # p50 of 3/0.3 and 3/0.6
    7.5
    >>> z = RequestTrace(2, 0.0, token_times=[3.0, 3.0], token_ids=[9, 9])
    >>> "decode_rate_p50_tok_s" in compute_metrics([z])  # zero span: no rate
    False
    """
    if not traces:
        raise ValueError("compute_metrics needs at least one trace")
    for trace in traces:
        if not trace.token_times:
            raise ValueError(f"request {trace.request_id} produced no tokens")

    total_tokens = sum(t.num_tokens for t in traces)
    window_start = min(t.enqueue_time for t in traces)
    window_end = max(t.token_times[-1] for t in traces)
    wall_window = window_end - window_start
    if wall_window <= 0:
        raise ValueError("non-positive wall window; timestamps are inconsistent")

    ttfts = [t.token_times[0] - t.enqueue_time for t in traces]
    pooled_itls = [delta for t in traces for delta in inter_token_latencies(t)]
    decode_rates = [
        (t.num_tokens - 1) / (t.token_times[-1] - t.token_times[0])
        for t in traces
        if t.num_tokens >= 2 and t.token_times[-1] > t.token_times[0]
    ]

    metrics = {
        "total_new_tokens": total_tokens,
        "wall_window_s": wall_window,
        "throughput_tok_s": total_tokens / wall_window,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p99_s": percentile(ttfts, 99),
        "num_requests": len(traces),
    }
    if pooled_itls:
        metrics["itl_p50_s"] = percentile(pooled_itls, 50)
        metrics["itl_p99_s"] = percentile(pooled_itls, 99)
    if decode_rates:
        metrics["decode_rate_p50_tok_s"] = percentile(decode_rates, 50)
        metrics["decode_rate_min_tok_s"] = min(decode_rates)
        metrics["decode_rate_max_tok_s"] = max(decode_rates)
    return metrics


def assert_forced_lengths(traces: list[RequestTrace], workload: Workload) -> None:
    """Every request must have produced exactly its forced output length
    (EOS ignored on every engine); a violation voids the trial.

    >>> from bench.workload import synthetic_workload
    >>> w = synthetic_workload("t", 1, 2, 2)
    >>> ok = RequestTrace(0, 0.0, token_times=[1.0, 1.1], token_ids=[3, 4])
    >>> assert_forced_lengths([ok], w)
    >>> short = RequestTrace(0, 0.0, token_times=[1.0], token_ids=[3])
    >>> assert_forced_lengths([short], w)
    Traceback (most recent call last):
        ...
    bench.run_bench.ForcedLengthError: forced-length violation on 1 request(s): request 0 produced 1 tokens, expected 2 (finish_reason='')
    """
    expected = {r.request_id: r.max_new_tokens for r in workload.requests}
    seen = {t.request_id for t in traces}
    violations = []
    for trace in traces:
        want = expected.get(trace.request_id)
        if want is None:
            violations.append(f"request {trace.request_id} not in workload")
        elif trace.num_tokens != want:
            violations.append(
                f"request {trace.request_id} produced {trace.num_tokens} tokens, "
                f"expected {want} (finish_reason={trace.finish_reason!r})"
            )
    for request_id in sorted(set(expected) - seen):
        violations.append(f"request {request_id} returned no trace")
    if violations:
        raise ForcedLengthError(
            f"forced-length violation on {len(violations)} request(s): "
            + "; ".join(violations)
        )


def select_median_trial(values: list[float]) -> int:
    """Index of the median trial by a headline metric (lower median when the
    count is even, so one real trial is always reported).

    >>> select_median_trial([5.0, 1.0, 3.0])
    2
    >>> select_median_trial([5.0, 1.0, 3.0, 4.0])
    2
    >>> select_median_trial([2.0])
    0
    """
    if not values:
        raise ValueError("no trials")
    order = sorted(range(len(values)), key=lambda i: values[i])
    return order[(len(values) - 1) // 2]


def spread_exceeds(values: list[float], fraction: float = DISPERSION_FRACTION) -> bool:
    """Dispersion rule predicate: (max - min) > ``fraction`` of the median.

    >>> spread_exceeds([100.0, 101.0, 102.0])
    False
    >>> spread_exceeds([100.0, 90.0, 101.0])
    True
    >>> spread_exceeds([100.0])
    False
    """
    if len(values) < 2:
        return False
    return (max(values) - min(values)) > fraction * percentile(values, 50)


def summarize_metrics(trials: list[dict], median_index: int) -> dict:
    """Reported summary: the median trial's value per metric, with min-max
    across all measured trials, each labeled per bench/FAIRNESS.md.

    >>> trials = [{"metrics": {"throughput_tok_s": 10.0}},
    ...           {"metrics": {"throughput_tok_s": 12.0}},
    ...           {"metrics": {"throughput_tok_s": 11.0}}]
    >>> s = summarize_metrics(trials, 2)
    >>> s["throughput_tok_s"]["value"], s["throughput_tok_s"]["min"], s["throughput_tok_s"]["max"]
    (11.0, 10.0, 12.0)
    """
    summary: dict = {}
    for name in METRIC_LABELS:
        values = [t["metrics"][name] for t in trials if name in t["metrics"]]
        if not values or name not in trials[median_index]["metrics"]:
            continue
        summary[name] = {
            "value": trials[median_index]["metrics"][name],
            "min": min(values),
            "max": max(values),
            "label": METRIC_LABELS[name],
        }
    return summary


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------


def _capture_cmd(cmd: list[str], timeout_s: float = 60.0) -> dict:
    """Run one capture command; failures are recorded, never fatal."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        ok = proc.returncode == 0
        return {
            "cmd": cmd,
            "ok": ok,
            "output": proc.stdout if ok else None,
            "error": None if ok else (proc.stderr or f"exit {proc.returncode}"),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"cmd": cmd, "ok": False, "output": None, "error": repr(error)}


def _mem_total_kb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def capture_environment() -> dict:
    """Everything bench/FAIRNESS.md "Hardware and environment" requires in the
    results JSON. Individual captures degrade to recorded errors so the
    harness also runs on machines without the tools (CPU-only tests)."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    return {
        "timestamp_utc": _utc_now(),
        "command_line": list(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "mem_total_kb": _mem_total_kb(),
        "nvidia_smi_q": _capture_cmd(["nvidia-smi", "-q"]),
        "lscpu": _capture_cmd(["lscpu"]),
        "nvcc_version": _capture_cmd(["nvcc", "--version"]),
        "gcc_version": _capture_cmd(["gcc", "--version"]),
        "pip_freeze": _capture_cmd([sys.executable, "-m", "pip", "freeze"]),
        "git_commit": _capture_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"]),
    }


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Trial driver
# --------------------------------------------------------------------------


def _trace_to_json(trace: RequestTrace, epoch: float) -> dict:
    """Raw per-token record, timestamps rebased to the trial's first submit
    (differences are what every metric uses, so rebasing is lossless)."""
    return {
        "request_id": trace.request_id,
        "enqueue_time_s": round(trace.enqueue_time - epoch, 7),
        "token_times_s": [round(t - epoch, 7) for t in trace.token_times],
        "token_ids": list(trace.token_ids),
        "finish_reason": trace.finish_reason,
    }


def run_trial(adapter: EngineAdapter, workload: Workload) -> dict:
    """One measured trial: drive the adapter over the workload (arrival times
    honored by the shared asyncio client), assert forced lengths, compute
    metrics, and keep the raw per-token data."""
    started = _utc_now()
    traces = adapter.run(workload)
    assert_forced_lengths(traces, workload)
    metrics = compute_metrics(traces)
    epoch = min(t.enqueue_time for t in traces)
    return {
        "started_utc": started,
        "metrics": metrics,
        "requests": [_trace_to_json(t, epoch) for t in traces],
    }


def _workload_summary(workload: Workload) -> dict:
    requests = workload.requests
    stagger = requests[1].arrival_time_s - requests[0].arrival_time_s if len(requests) > 1 else 0.0
    return {
        "name": workload.name,
        "num_requests": len(requests),
        "input_len": len(requests[0].prompt_token_ids),
        "output_len": requests[0].max_new_tokens,
        "arrival_stagger_s": stagger,
    }


# --------------------------------------------------------------------------
# Client-overhead cross-check (bench/FAIRNESS.md "Measured path")
# --------------------------------------------------------------------------


def run_offline_crosscheck(
    adapter: EngineAdapter,
    args: argparse.Namespace,
    workload: Workload,
    config: dict,
    environment: dict,
) -> int:
    """Client-overhead cross-check: the same workload through the engine's
    offline batch entrypoint (vLLM ``LLM.generate``) instead of the streaming
    client, with the same warmup count. The offline path exposes no per-token
    timestamps, so the JSON records aggregate wall time and throughput under
    an ``offline`` block with ``mode: "offline_crosscheck"`` - report.py
    validation refuses this shape for the cross-engine table, which is
    exactly right: it is a bound on client overhead, not a published result.
    Offline generation is a closed batch: staggered-arrival cases cannot be
    represented on it and are refused. Forced output lengths are asserted the
    same way as streaming trials (violation -> exit 2, nothing written)."""
    run_offline = getattr(adapter, "run_offline", None)
    if not callable(run_offline):
        raise SystemExit(
            f"--offline-crosscheck: engine '{args.engine}' has no offline "
            "path (vLLM only; the llama.cpp cross-check is llama-batched-bench, "
            "bench/FAIRNESS.md 'Measured path')"
        )
    if any(r.arrival_time_s > 0 for r in workload.requests):
        raise SystemExit(
            f"--offline-crosscheck: case '{args.case}' staggers arrivals; the "
            "offline batch path submits everything at t=0 and would not "
            "measure the same workload"
        )

    print(f"[bench] offline cross-check ({args.warmup} unmeasured warmup run(s))")
    offline = run_offline(args.model_dir, workload, config, warmup=args.warmup)
    if not offline.get("forced_length_ok", False):
        print(
            "[bench] TRIAL VOID: offline run violated forced lengths: "
            f"{offline.get('per_request_lengths')}",
            file=sys.stderr,
        )
        return 2

    result = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "engine": args.engine,
        "case": args.case,
        "seed": args.seed,
        "mode": "offline_crosscheck",
        "workload": _workload_summary(workload),
        "engine_config": adapter.version_info(),
        "environment": environment,
        "offline": offline,
        "forced_length_ok": True,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[bench] offline wall_s: {offline['wall_s']:.6g}")
    print(
        f"[bench] offline throughput_tok_s: {offline['throughput_tok_s']:.6g} "
        "(closed batch, includes all prefill work)"
    )
    print(f"[bench] wrote {out_path}")
    return 0


# --------------------------------------------------------------------------
# Cross-engine output-equivalence gate (bench/FAIRNESS.md "Model and inputs")
# --------------------------------------------------------------------------


def equivalence_gate(
    engine_adapters: list[EngineAdapter],
    model_dir: str,
    configs: dict | None = None,
    seed: int = 0,
    engine_hooks: dict | None = None,
) -> dict:
    """Run the 5 short greedy prompts through every engine SEQUENTIALLY (one
    engine on the GPU at a time) and compare token streams position by
    position against the first engine.

    Returns the transcript dict committed next to the results. ``all_match``
    True passes the gate outright; any divergence is listed with its position
    and both tokens and must be justified under the section-12c logit-margin
    protocol before measuring (the ``hf_margin`` slots are filled by
    ``hf_margin_review``; CLI: ``--equivalence-gate``).

    ``engine_hooks`` optionally maps an engine name to a zero-argument
    context-manager factory wrapped around that engine's whole turn
    (load -> run -> shutdown). The CLI uses it to record GPU idleness between
    engines and to launch/stop llama-server around the llamacpp turn so no
    two engines ever share the GPU.
    """
    workload = equivalence_workload(seed)
    configs = configs or {}
    engine_hooks = engine_hooks or {}
    streams: dict = {}
    versions: dict = {}
    order: list[str] = []
    for adapter in engine_adapters:
        hook = engine_hooks.get(adapter.name)
        with hook() if hook is not None else contextlib.nullcontext():
            adapter.load(model_dir, configs.get(adapter.name, {}))
            try:
                versions[adapter.name] = adapter.version_info()
                traces = adapter.run(workload)
                assert_forced_lengths(traces, workload)
                by_id = {t.request_id: t for t in traces}
                streams[adapter.name] = [
                    list(by_id[r.request_id].token_ids) for r in workload.requests
                ]
                order.append(adapter.name)
            finally:
                adapter.shutdown()

    comparisons = []
    all_match = True
    reference = order[0] if order else None
    for candidate in order[1:]:
        for request in workload.requests:
            ref_tokens = streams[reference][request.request_id]
            cand_tokens = streams[candidate][request.request_id]
            divergence = None
            for position, (x, y) in enumerate(zip(ref_tokens, cand_tokens)):
                if x != y:
                    divergence = {
                        "position": position,
                        "reference_token": x,
                        "candidate_token": y,
                        "hf_margin": None,  # filled by the section-12c margin review
                    }
                    break
            match = divergence is None and len(ref_tokens) == len(cand_tokens)
            all_match = all_match and match
            comparisons.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "request_id": request.request_id,
                    "match": match,
                    "first_divergence": divergence,
                }
            )

    return {
        "workload": {
            **_workload_summary(workload),
            "prompts": [list(r.prompt_token_ids) for r in workload.requests],
        },
        "engines": order,
        "engine_versions": versions,
        "streams": streams,
        "comparisons": comparisons,
        "all_match": all_match,
        "note": (
            "divergences require top1-top2 logit-margin justification "
            "(docs/DESIGN.md section 12c) before any measured trial"
        ),
    }


def hf_margin_review(transcript: dict, model_dir: str) -> dict:
    """Fill the ``hf_margin`` slot of every recorded divergence (section-12c
    protocol): teacher-force HF transformers FP16 over the shared prefix
    (prompt + reference stream up to the divergence position - identical for
    both engines at a FIRST divergence by construction) and record the
    top1-top2 logit margin at that position. A margin below
    ``EQUIVALENCE_MARGIN`` documents the divergence as an FP16 coin-flip.

    Runs only when divergences exist. Uses a raw forward pass (never
    ``generate``), so HF generation-config processors (e.g. the shipped
    repetition_penalty) cannot touch the logits. Heavy imports are lazy -
    the matched-stream path never needs torch/transformers.
    """
    divergences = [
        comparison
        for comparison in transcript["comparisons"]
        if not comparison["match"] and comparison["first_divergence"]
    ]
    if not divergences:
        return {"ran": False, "reason": "no divergences to review"}

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    except TypeError:  # releases that renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16)
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float16:
        raise RuntimeError(
            f"margin review requires FP16 weights (section 12c); got {parameter_dtype}"
        )
    model = model.to(device).eval()

    prompts = transcript["workload"]["prompts"]
    reference = transcript["comparisons"][0]["reference"]
    with torch.inference_mode():
        for comparison in divergences:
            request_id = comparison["request_id"]
            divergence = comparison["first_divergence"]
            prefix = list(prompts[request_id]) + list(
                transcript["streams"][reference][request_id][: divergence["position"]]
            )
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[0, -1].float()
            top = torch.topk(logits, 2)
            divergence["hf_margin"] = float(top.values[0] - top.values[1])
            divergence["hf_top1_token"] = int(top.indices[0])
            divergence["hf_top2_token"] = int(top.indices[1])
            divergence["hf_top1_logit"] = float(top.values[0])
            divergence["hf_top2_logit"] = float(top.values[1])

    review = {
        "ran": True,
        "protocol": "teacher-forced raw forward at each first-divergence position",
        "margin_threshold": EQUIVALENCE_MARGIN,
        "device": device,
        "dtype": "float16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "reviewed_divergences": len(divergences),
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return review


def gate_passes(transcript: dict, margin_threshold: float = EQUIVALENCE_MARGIN) -> bool:
    """Gate verdict: pass when every stream matched, or when every recorded
    divergence carries a reviewed HF margin strictly below the threshold
    (bench/FAIRNESS.md "Model and inputs"). An unreviewed divergence
    (``hf_margin`` None) or a length-only mismatch never passes.

    >>> gate_passes({"all_match": True, "comparisons": []})
    True
    >>> divergent = {"match": False, "first_divergence": {"hf_margin": 0.05}}
    >>> gate_passes({"all_match": False, "comparisons": [divergent]})
    True
    >>> unreviewed = {"match": False, "first_divergence": {"hf_margin": None}}
    >>> gate_passes({"all_match": False, "comparisons": [unreviewed]})
    False
    >>> wide = {"match": False, "first_divergence": {"hf_margin": 0.4}}
    >>> gate_passes({"all_match": False, "comparisons": [wide]})
    False
    """
    if transcript["all_match"]:
        return True
    for comparison in transcript["comparisons"]:
        if comparison["match"]:
            continue
        divergence = comparison["first_divergence"]
        if divergence is None:  # length-only mismatch: never a coin-flip
            return False
        margin = divergence.get("hf_margin")
        if margin is None or not margin < margin_threshold:
            return False
    return True


def _gpu_memory_used_mib() -> int | None:
    """Current GPU memory use (MiB) via nvidia-smi; None when unavailable.
    Recorded between the gate's engine turns (bench/FAIRNESS.md "Hardware
    and environment": the GPU is verified idle between engines)."""
    capture = _capture_cmd(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    if not capture["ok"] or not capture["output"]:
        return None
    try:
        return int(capture["output"].strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


@contextlib.contextmanager
def _managed_llama_server(cmd: str, log_path: Path):
    """Launch llama-server for the llamacpp gate turn and stop it afterwards,
    so no two engines ever hold the GPU together. The adapter's own load()
    polls /health, so no readiness wait is needed here."""
    log_handle = open(log_path, "ab")
    with log_handle:
        process = subprocess.Popen(
            shlex.split(cmd), stdout=log_handle, stderr=subprocess.STDOUT
        )
        try:
            yield
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)


def run_equivalence_gate_cli(args: argparse.Namespace) -> int:
    """CLI body for ``--equivalence-gate`` (bench/FAIRNESS.md "Model and
    inputs"): run the gate across ``--engines``, review any divergences with
    ``hf_margin_review``, write the transcript JSON to ``--out``, and exit
    0 only on a pass. Exit codes: 2 = a request violated its forced length
    (setup broken), 3 = streams diverged beyond documented FP16 coin-flips
    (misconfigured baseline or broken GGUF - fix it, do not benchmark it)."""
    if args.offline_crosscheck:
        raise SystemExit("--equivalence-gate and --offline-crosscheck are exclusive")
    if args.engine:
        raise SystemExit("--equivalence-gate takes --engines (comma list), not --engine")
    if args.engine_arg:
        raise SystemExit(
            "--engine-arg is not allowed in gate mode: the gate runs each "
            "engine's pinned configuration (bench/FAIRNESS.md)"
        )
    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in engines if name not in adapters_mod.ENGINE_NAMES]
    if unknown:
        raise SystemExit(f"unknown engine(s) {unknown}; expected {adapters_mod.ENGINE_NAMES}")
    if len(engines) != len(set(engines)):
        raise SystemExit(f"duplicate engine in --engines: {args.engines!r}")
    if set(adapters_mod.VLLM_ENGINE_NAMES) <= set(engines):
        raise SystemExit(
            "the two vLLM arms cannot share one gate run: a single "
            "interpreter has exactly one vllm installed, so 'vllm' and "
            "'vllm024' here would measure the same binary twice while "
            "claiming two arms - run each arm's gate under its own venv "
            "(bench/FAIRNESS.md 'Engine configuration')"
        )
    if len(engines) < 2:
        raise SystemExit("the gate compares at least two engines")
    if args.llamacpp_server_cmd and "llamacpp" not in engines:
        raise SystemExit("--llamacpp-server-cmd given but llamacpp is not in --engines")

    configs: dict = {}
    for name in engines:
        per_engine = argparse.Namespace(**{**vars(args), "engine": name, "engine_arg": []})
        configs[name] = build_engine_config(per_engine)
    if args.llamacpp_server_cmd and "llamacpp" in configs:
        # The managed server loads the GGUF from cold; give the adapter's
        # /health poll more room than the 30 s external-server default.
        configs["llamacpp"].setdefault("connect_timeout_s", 180.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_path = out_path.parent / "llama_server_gate.log"

    gpu_memory_readings: list[dict] = []

    def _make_turn_hook(name: str):
        @contextlib.contextmanager
        def _hook():
            gpu_memory_readings.append(
                {
                    "engine": name,
                    "stage": "before_load",
                    "gpu_memory_used_mib": _gpu_memory_used_mib(),
                }
            )
            if name == "llamacpp" and args.llamacpp_server_cmd:
                with _managed_llama_server(args.llamacpp_server_cmd, server_log_path):
                    yield
            else:
                yield

        return _hook

    adapters = [adapters_mod.resolve_adapter(name) for name in engines]
    hooks = {name: _make_turn_hook(name) for name in engines}
    environment = capture_environment()

    print(f"[gate] engines={engines} seed={args.seed}")
    try:
        transcript = equivalence_gate(
            adapters,
            args.model_dir,
            configs=configs,
            seed=args.seed,
            engine_hooks=hooks,
        )
    except ForcedLengthError as error:
        print(f"[gate] VOID (engine setup broken): {error}", file=sys.stderr)
        return 2
    gpu_memory_readings.append(
        {
            "engine": None,
            "stage": "after_final_shutdown",
            "gpu_memory_used_mib": _gpu_memory_used_mib(),
        }
    )

    review: dict = {"ran": False, "reason": "all streams matched"}
    if not transcript["all_match"]:
        try:
            review = hf_margin_review(transcript, args.model_dir)
        except ImportError as error:
            review = {"ran": False, "reason": f"margin review unavailable: {error!r}"}

    passed = gate_passes(transcript)
    result = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "mode": "equivalence_gate",
        "seed": args.seed,
        "model_dir": args.model_dir,
        "engine_requested_configs": configs,
        "environment": environment,
        "llamacpp_server_cmd": args.llamacpp_server_cmd,
        "llamacpp_server_log": (
            str(server_log_path) if args.llamacpp_server_cmd else None
        ),
        "gpu_memory_used_mib": gpu_memory_readings,
        "margin_review": review,
        "gate_pass": passed,
        **transcript,
    }
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    for comparison in transcript["comparisons"]:
        if comparison["match"]:
            continue
        divergence = comparison["first_divergence"]
        print(
            f"[gate] divergence {transcript['engines'][0]} vs "
            f"{comparison['candidate']} request {comparison['request_id']}: "
            f"{divergence}"
        )
    print(f"[gate] wrote {out_path}")
    if passed:
        detail = (
            "all token streams identical"
            if transcript["all_match"]
            else "every divergence reviewed as an FP16 coin-flip "
            f"(HF top1-top2 margin < {EQUIVALENCE_MARGIN})"
        )
        print(f"[gate] PASS: {detail}")
        return 0
    print(
        "[gate] FAIL: token streams diverge beyond documented FP16 coin-flips "
        "-- fix the engine configuration or conversion before any measured "
        "trial (bench/FAIRNESS.md 'Model and inputs')",
        file=sys.stderr,
    )
    return 3


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.run_bench",
        description="Cross-engine benchmark runner (rules: bench/FAIRNESS.md).",
    )
    parser.add_argument(
        "--engine",
        choices=list(adapters_mod.ENGINE_NAMES),
        default=None,
        help="engine to measure (required unless --equivalence-gate)",
    )
    parser.add_argument("--case", choices=sorted(CASES), default="primary")
    parser.add_argument("--trials", type=int, default=3, help="measured trials (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured warmup runs (default 1)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output JSON path")
    # Redline engine configuration (docs/DESIGN.md section 10; redline
    # defaults = dev preset - safe on the smallest supported GPU; pod runs
    # pass the bench preset explicitly and the full config is recorded in the
    # results JSON). --max-batch / --max-seq-len double as OPT-IN vLLM
    # overrides (max_num_seqs / max_model_len); the redline dev-preset
    # defaults are applied in build_engine_config for redline ONLY - a
    # default that silently capped a baseline below the workload's
    # concurrency would be a handicapped comparison (bench/FAIRNESS.md
    # "Engine configuration").
    parser.add_argument("--kv-gb", type=float, default=1.0)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="redline: max concurrent sequences (default 8, dev preset); "
        "vllm/vllm024: sets max_num_seqs only when given, else vLLM's own "
        "default stands",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="redline: max sequence length (default 2048, dev preset); "
        "vllm/vllm024: sets max_model_len only when given, else vLLM's own "
        "default stands",
    )
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--graphs", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--admission-policy", choices=("reserve_full", "watermark"), default="reserve_full"
    )
    # llama.cpp: where the externally launched llama-server listens.
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    # Engine-specific passthrough for the documented baseline tuning passes,
    # e.g. --engine-arg gpu_memory_utilization=0.9 (values parsed as JSON when
    # possible, else kept as strings). Everything lands in the results JSON.
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra engine config (repeatable)",
    )
    parser.add_argument(
        "--offline-crosscheck",
        action="store_true",
        help="client-overhead cross-check: run the case through the engine's "
        "offline batch entrypoint (vLLM LLM.generate) instead of the streaming "
        "client and write a mode='offline_crosscheck' JSON "
        "(bench/FAIRNESS.md 'Measured path')",
    )
    # Cross-engine output-equivalence gate (bench/FAIRNESS.md "Model and
    # inputs"): required to pass BEFORE any measured trial.
    parser.add_argument(
        "--equivalence-gate",
        action="store_true",
        help="run the 5 short greedy prompts through every --engines engine "
        "sequentially (one engine on the GPU at a time), write the transcript "
        "JSON to --out, and exit 0 only when the token streams match or every "
        "divergence is a reviewed FP16 coin-flip (HF top1-top2 margin "
        f"< {EQUIVALENCE_MARGIN}, docs/DESIGN.md section 12c); exit 3 otherwise",
    )
    parser.add_argument(
        "--engines",
        default="redline,vllm,llamacpp",
        help="gate mode: comma-separated engines, later ones compared against "
        "the first (default: redline,vllm,llamacpp - the vllm024 arm runs its "
        "own gate under its own venv, e.g. --engines redline,vllm024, and "
        "cannot share a gate process with the vllm arm)",
    )
    parser.add_argument(
        "--llamacpp-server-cmd",
        default=None,
        metavar="CMD",
        help="gate mode: launch this llama-server command for the llamacpp "
        "turn and stop it afterwards (keeps the GPU exclusive per engine); "
        "when absent, a server already listening at --server-url is expected",
    )
    return parser


def parse_engine_args(pairs: list[str]) -> dict:
    """``KEY=VALUE`` passthrough parser; VALUE is JSON when it parses.

    >>> parse_engine_args(["gpu_memory_utilization=0.9", "mode=full"])
    {'gpu_memory_utilization': 0.9, 'mode': 'full'}
    """
    out: dict = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--engine-arg expects KEY=VALUE, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


# Redline dev-preset defaults (docs/DESIGN.md section 10; they mirror
# bench/adapters/redline_a.DEFAULT_CONFIG) - applied to redline only, when
# the corresponding flag was not given.
REDLINE_DEV_MAX_BATCH = 8
REDLINE_DEV_MAX_SEQ_LEN = 2048


def build_engine_config(args: argparse.Namespace) -> dict:
    """Engine configuration passed to the adapter and recorded in the results
    JSON. Redline receives its full explicit configuration (dev-preset
    defaults where flags are absent). For the vLLM arms (vllm and vllm024,
    identical treatment) the shared --max-batch / --max-seq-len flags are
    OPT-IN overrides: when absent, no max_num_seqs / max_model_len key is
    emitted and the baseline's own defaults stand - inheriting redline's dev
    preset (max_batch 8) would silently strangle a 64-request wave, exactly
    the handicapped-baseline configuration bench/FAIRNESS.md "Engine
    configuration" prohibits."""
    if args.engine == "redline":
        config = {
            "kv_pool_gb": args.kv_gb,
            "max_batch": (
                REDLINE_DEV_MAX_BATCH if args.max_batch is None else args.max_batch
            ),
            "enable_cuda_graphs": bool(args.graphs),
            "max_seq_len": (
                REDLINE_DEV_MAX_SEQ_LEN if args.max_seq_len is None else args.max_seq_len
            ),
            "prefill_chunk": args.prefill_chunk,
            "admission_policy": args.admission_policy,
        }
    elif args.engine in adapters_mod.VLLM_ENGINE_NAMES:
        config = {}
        if args.max_seq_len is not None:
            config["max_model_len"] = args.max_seq_len
        if args.max_batch is not None:
            config["max_num_seqs"] = args.max_batch
    else:  # llamacpp
        config = {"server_url": args.server_url}
    config.update(parse_engine_args(args.engine_arg))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.equivalence_gate:
        return run_equivalence_gate_cli(args)
    if not args.engine:
        parser.error("--engine is required (or pass --equivalence-gate)")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")

    workload = CASES[args.case](args.seed)
    config = build_engine_config(args)

    # FAIRNESS.md "Engine configuration": no baseline runs deliberately
    # handicapped flags. An explicit vLLM concurrency cap below the
    # workload's request count would measure a strawman; refuse outright
    # (both vLLM arms).
    cap = config.get("max_num_seqs")
    if (
        args.engine in adapters_mod.VLLM_ENGINE_NAMES
        and isinstance(cap, int)
        and cap < len(workload.requests)
    ):
        raise SystemExit(
            f"refusing to measure: max_num_seqs={cap} caps vLLM below the "
            f"{len(workload.requests)} requests of case '{args.case}' -- a "
            "handicapped baseline (bench/FAIRNESS.md 'Engine configuration'). "
            "Omit --max-batch to let vLLM's own default stand, or pass a "
            "value >= the request count."
        )

    adapter = adapters_mod.resolve_adapter(args.engine)
    environment = capture_environment()

    print(f"[bench] engine={args.engine} case={args.case} seed={args.seed}")
    print(f"[bench] engine config: {config}")

    if args.offline_crosscheck:
        # Own measurement mode: no streaming client, no trials block. The
        # adapter's offline entrypoint manages the engine lifecycle itself.
        return run_offline_crosscheck(adapter, args, workload, config, environment)

    try:
        adapter.load(args.model_dir, config)
        warmups = []
        for index in range(args.warmup):
            print(f"[bench] warmup {index + 1}/{args.warmup} (unmeasured)")
            warmups.append(run_trial(adapter, workload)["metrics"])

        trials = []
        for index in range(args.trials):
            print(f"[bench] trial {index + 1}/{args.trials}")
            trial = run_trial(adapter, workload)
            trial["index"] = index
            trials.append(trial)

        headline_values = [t["metrics"][HEADLINE_METRIC] for t in trials]
        reran = False
        if spread_exceeds(headline_values):
            print(
                f"[bench] dispersion rule: {HEADLINE_METRIC} spread exceeds "
                f"{DISPERSION_FRACTION:.0%} of the median -- one re-run"
            )
            reran = True
            extra = run_trial(adapter, workload)
            extra["index"] = len(trials)
            extra["dispersion_rerun"] = True
            trials.append(extra)
            headline_values = [t["metrics"][HEADLINE_METRIC] for t in trials]
        # Persistence is judged over ALL measured trials including the re-run
        # (FAIRNESS.md commits every raw trial). Appending a trial can never
        # shrink max-min, so once the rule fires the number publishes flagged
        # unless the re-run shifted the median enough to absorb the spread -
        # conservative by construction: over-flags, never under-flags.
        flagged = spread_exceeds(headline_values)

        median_index = select_median_trial(headline_values)
        summary = summarize_metrics(trials, median_index)

        engine_stats = None
        stats_probe = getattr(adapter, "engine_stats", None)
        if callable(stats_probe):
            engine_stats = stats_probe()
        lt_algo_report = None
        algo_probe = getattr(adapter, "algo_report", None)
        if callable(algo_probe):
            lt_algo_report = algo_probe()

        result = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "engine": args.engine,
            "case": args.case,
            "seed": args.seed,
            "workload": _workload_summary(workload),
            "engine_config": adapter.version_info(),
            "environment": environment,
            "warmup": warmups,
            "trials": trials,
            "median_trial_index": median_index,
            "metrics": summary,
            "dispersion": {
                "metric": HEADLINE_METRIC,
                "threshold_fraction": DISPERSION_FRACTION,
                "values": headline_values,
                "reran": reran,
                "flag": flagged,
            },
            "forced_length_ok": True,  # violations raise before this point
            "engine_stats": engine_stats,
            "lt_algo_report": lt_algo_report,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")

        print(f"[bench] median trial: {median_index}")
        for name, entry in summary.items():
            flag = " [dispersion-flagged]" if flagged and name == HEADLINE_METRIC else ""
            print(
                f"[bench] {name}: {entry['value']:.6g} "
                f"(min {entry['min']:.6g}, max {entry['max']:.6g}){flag} -- {entry['label']}"
            )
        print(f"[bench] wrote {out_path}")
        return 0
    except ForcedLengthError as error:
        print(f"[bench] TRIAL VOID: {error}", file=sys.stderr)
        return 2
    finally:
        adapter.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
