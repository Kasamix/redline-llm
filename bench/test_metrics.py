"""CPU-only tests for the bench harness (workload, metrics, runner, report).

Everything here runs without a GPU or any engine installed: adapters are
exercised through fakes and synthetic timestamp fixtures. Run with::

    python -m pytest bench/test_metrics.py

Also executes every doctest in the harness modules (the metric formulas'
normative examples).
"""

from __future__ import annotations

import asyncio
import doctest
import json
import time
from pathlib import Path

import pytest

import bench.adapters as adapters_mod
import bench.report as report
import bench.run_bench as run_bench
import bench.workload as workload_mod
from bench.adapters import EngineAdapter, RequestTrace, drive_workload, resolve_adapter
from bench.run_bench import (
    ForcedLengthError,
    HEADLINE_METRIC,
    METRIC_LABELS,
    RESULTS_SCHEMA_VERSION,
    assert_forced_lengths,
    compute_metrics,
    equivalence_gate,
    parse_engine_args,
    percentile,
    select_median_trial,
    spread_exceeds,
    summarize_metrics,
)
from bench.workload import CASES, selftest_lines, synthetic_workload, token_id

# --------------------------------------------------------------------------
# Doctests (normative formula examples live next to the formulas)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [workload_mod, run_bench, report, adapters_mod], ids=lambda m: m.__name__
)
def test_doctests(module):
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0


# --------------------------------------------------------------------------
# Workload generator
# --------------------------------------------------------------------------

# First 8 token IDs for seed=0, req in {0,1}, pos in {0..3} - the exact values
# `redline_bench --selftest-workload` must print (cross-language identity;
# normative algorithm in bench/FAIRNESS.md "Workload generator").
GOLDEN_SELFTEST = [
    "seed=0 req=0 pos=0 id=40535",
    "seed=0 req=0 pos=1 id=72465",
    "seed=0 req=0 pos=2 id=39110",
    "seed=0 req=0 pos=3 id=83053",
    "seed=0 req=1 pos=0 id=31861",
    "seed=0 req=1 pos=1 id=23711",
    "seed=0 req=1 pos=2 id=79207",
    "seed=0 req=1 pos=3 id=9130",
]


def test_selftest_golden_ids():
    assert selftest_lines(0) == GOLDEN_SELFTEST


def test_selftest_cli(capsys):
    assert workload_mod.main(["--selftest"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == GOLDEN_SELFTEST


def test_token_ids_in_bpe_range():
    for request_id in (0, 7, 63):
        for position in (0, 1, 1023):
            value = token_id(0, request_id, position)
            assert 1000 <= value < 100_000


def test_workload_deterministic():
    a = synthetic_workload("x", 4, 16, 8, seed=0)
    b = synthetic_workload("x", 4, 16, 8, seed=0)
    assert [r.prompt_token_ids for r in a.requests] == [
        r.prompt_token_ids for r in b.requests
    ]
    c = synthetic_workload("x", 4, 16, 8, seed=1)
    assert a.requests[0].prompt_token_ids != c.requests[0].prompt_token_ids


@pytest.mark.parametrize(
    ("case", "num", "input_len", "output_len", "stagger"),
    [
        ("primary", 64, 1024, 256, 0.0),
        ("arrival", 64, 1024, 256, 0.100),
        ("longout", 64, 1024, 1024, 0.0),
        ("batch1", 1, 1024, 256, 0.0),
    ],
)
def test_cases_table(case, num, input_len, output_len, stagger):
    """The 4 cases exactly as bench/FAIRNESS.md 'Cases' specifies them."""
    workload = CASES[case](0)
    assert len(workload.requests) == num
    assert all(len(r.prompt_token_ids) == input_len for r in workload.requests)
    assert all(r.max_new_tokens == output_len for r in workload.requests)
    for request in workload.requests:
        assert request.arrival_time_s == pytest.approx(request.request_id * stagger)


# --------------------------------------------------------------------------
# Metric functions (synthetic timestamp fixtures)
# --------------------------------------------------------------------------


def _trace(request_id, enqueue, times, ids=None):
    return RequestTrace(
        request_id=request_id,
        enqueue_time=enqueue,
        token_times=list(times),
        token_ids=list(ids) if ids is not None else list(range(len(times))),
    )


def test_compute_metrics_hand_computed():
    # Request 0: submit 10.0, tokens at 11.0 + k*0.05 for k in 0..3
    # Request 1: submit 10.2, tokens at 12.2 + k*0.10 for k in 0..3
    t0 = _trace(0, 10.0, [11.0, 11.05, 11.10, 11.15])
    t1 = _trace(1, 10.2, [12.2, 12.3, 12.4, 12.5])
    metrics = compute_metrics([t0, t1])

    assert metrics["total_new_tokens"] == 8
    assert metrics["wall_window_s"] == pytest.approx(12.5 - 10.0)
    assert metrics["throughput_tok_s"] == pytest.approx(8 / 2.5)
    # TTFT: 1.0 and 2.0 -> p50 = 1.5, p99 = 1.99
    assert metrics["ttft_p50_s"] == pytest.approx(1.5)
    assert metrics["ttft_p99_s"] == pytest.approx(1.99)
    # Pooled ITLs: three 0.05s and three 0.1s
    assert metrics["itl_p50_s"] == pytest.approx(0.075)
    assert metrics["itl_p99_s"] == pytest.approx(0.1)
    # Decode rates: 3/0.15 = 20, 3/0.3 = 10 -> p50 = 15
    assert metrics["decode_rate_p50_tok_s"] == pytest.approx(15.0)
    assert metrics["decode_rate_min_tok_s"] == pytest.approx(10.0)
    assert metrics["decode_rate_max_tok_s"] == pytest.approx(20.0)
    assert metrics["num_requests"] == 2


def test_compute_metrics_single_token_request():
    """One-token requests have no ITL and no decode rate; nothing crashes and
    the pooled metrics simply exclude them."""
    metrics = compute_metrics([_trace(0, 0.0, [1.0])])
    assert metrics["throughput_tok_s"] == pytest.approx(1.0)
    assert "itl_p50_s" not in metrics
    assert "decode_rate_p50_tok_s" not in metrics


def test_compute_metrics_zero_span_decode_rate():
    """All tokens of a request sharing one timestamp (single-chunk delivery):
    no ZeroDivisionError; the request pools zero-valued ITLs but contributes
    no decode rate (zero decode span, the rate is undefined)."""
    burst = _trace(0, 0.0, [1.0, 1.0, 1.0])
    normal = _trace(1, 0.0, [1.0, 1.5, 2.0])
    metrics = compute_metrics([burst, normal])
    assert metrics["decode_rate_p50_tok_s"] == pytest.approx(2.0)  # normal only
    assert metrics["itl_p50_s"] == pytest.approx(0.25)  # pooled 0, 0, 0.5, 0.5
    solo = compute_metrics([burst])
    assert "decode_rate_p50_tok_s" not in solo


def test_compute_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_metrics([])
    with pytest.raises(ValueError):
        compute_metrics([RequestTrace(0, 0.0)])  # no tokens


def test_percentile_matches_definition():
    values = [4.0, 1.0, 3.0, 2.0]  # unsorted on purpose
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5


def test_forced_length_assertions():
    workload = synthetic_workload("t", 2, 4, 3, seed=0)
    good = [
        _trace(0, 0.0, [1.0, 1.1, 1.2]),
        _trace(1, 0.0, [1.0, 1.1, 1.2]),
    ]
    assert_forced_lengths(good, workload)  # no raise

    short = [good[0], _trace(1, 0.0, [1.0, 1.1])]
    with pytest.raises(ForcedLengthError, match="request 1 produced 2 tokens, expected 3"):
        assert_forced_lengths(short, workload)

    missing = [good[0]]
    with pytest.raises(ForcedLengthError, match="request 1 returned no trace"):
        assert_forced_lengths(missing, workload)

    unknown = good + [_trace(9, 0.0, [1.0, 1.1, 1.2])]
    with pytest.raises(ForcedLengthError, match="request 9 not in workload"):
        assert_forced_lengths(unknown, workload)


def test_median_and_dispersion_rules():
    assert select_median_trial([10.0, 30.0, 20.0]) == 2
    assert select_median_trial([10.0, 30.0, 20.0, 25.0]) == 2  # lower median of 4
    # 3% of median rule
    assert not spread_exceeds([100.0, 100.5, 101.0])
    assert spread_exceeds([100.0, 100.5, 104.0])
    trials = [
        {"metrics": {"throughput_tok_s": 10.0, "ttft_p50_s": 0.5}},
        {"metrics": {"throughput_tok_s": 12.0, "ttft_p50_s": 0.4}},
        {"metrics": {"throughput_tok_s": 11.0, "ttft_p50_s": 0.6}},
    ]
    summary = summarize_metrics(trials, 2)
    assert summary["throughput_tok_s"] == {
        "value": 11.0,
        "min": 10.0,
        "max": 12.0,
        "label": METRIC_LABELS["throughput_tok_s"],
    }
    assert summary["ttft_p50_s"]["value"] == 0.6


def test_parse_engine_args_rejects_bad_pairs():
    with pytest.raises(ValueError):
        parse_engine_args(["novalue"])
    assert parse_engine_args(["a=1", "b=true", "c=x"]) == {"a": 1, "b": True, "c": "x"}


def test_build_engine_config_baseline_caps_are_opt_in():
    """--max-batch/--max-seq-len are redline dev-preset knobs; for the vLLM
    arms (vllm and vllm024, identical treatment) they map to
    max_num_seqs/max_model_len ONLY when explicitly given, so a run without
    them (the benchmark-matrix command) leaves vLLM's own defaults in charge
    (FAIRNESS.md: no handicapped baselines)."""
    parser = run_bench.build_arg_parser()
    tail = ["--model-dir", "m", "--out", "o.json"]

    for arm in ("vllm", "vllm024"):
        config = run_bench.build_engine_config(parser.parse_args(["--engine", arm, *tail]))
        assert config == {}  # notably: no llamacpp server_url, no redline preset
        config = run_bench.build_engine_config(
            parser.parse_args(
                ["--engine", arm, "--max-batch", "64", "--max-seq-len", "4096", *tail]
            )
        )
        assert config["max_num_seqs"] == 64
        assert config["max_model_len"] == 4096

    # Redline keeps its dev-preset defaults when the flags are absent.
    config = run_bench.build_engine_config(
        parser.parse_args(["--engine", "redline", *tail])
    )
    assert config["max_batch"] == 8  # docs/DESIGN.md section 10 dev preset
    assert config["max_seq_len"] == 2048


# --------------------------------------------------------------------------
# Shared asyncio driver
# --------------------------------------------------------------------------


def test_drive_workload_honors_arrivals_and_orders_tokens():
    workload = synthetic_workload("t", 3, 4, 2, seed=0, arrival_stagger_s=0.05)

    async def stream_one(request, trace):
        for k in range(request.max_new_tokens):
            await asyncio.sleep(0.001)
            trace.record(request.prompt_token_ids[k])
        trace.finish_reason = "length"

    traces = asyncio.run(drive_workload(workload, stream_one))
    traces.sort(key=lambda t: t.request_id)
    assert [t.request_id for t in traces] == [0, 1, 2]
    # Arrival stagger: request 2 submits >= ~0.1 s after request 0 (allow timer slack).
    assert traces[2].enqueue_time - traces[0].enqueue_time >= 0.06
    for trace in traces:
        assert trace.num_tokens == 2
        assert trace.token_times == sorted(trace.token_times)
        assert trace.finish_reason == "length"
        assert trace.token_times[0] >= trace.enqueue_time


def test_resolve_adapter_constructs_without_engine_libs():
    """Instantiation must not import redline/vllm/aiohttp (lazy imports)."""
    assert adapters_mod.ENGINE_NAMES == ("redline", "vllm", "vllm024", "llamacpp")
    for name in adapters_mod.ENGINE_NAMES:
        adapter = resolve_adapter(name)
        assert adapter.name == name
        assert isinstance(adapter, EngineAdapter)
    with pytest.raises(ValueError):
        resolve_adapter("hf")


def test_vllm024_is_thin_vllm_variant():
    """The vllm024 arm shares the vllm adapter code path verbatim; only the
    registry NAME (and therefore the results JSON engine field, report row,
    and duplicate detection key) differs. The recorded vllm_version comes
    from whichever venv the invoking interpreter runs under - the inherited
    version_info re-records it at runtime."""
    from bench.adapters.vllm_a import Vllm024Adapter, VllmAdapter

    adapter = resolve_adapter("vllm024")
    assert type(adapter) is Vllm024Adapter
    assert isinstance(adapter, VllmAdapter)
    assert adapter.name == "vllm024"
    # Thin means thin: no measurement method is overridden.
    for method in ("load", "run", "_stream_one", "run_offline", "version_info", "shutdown"):
        assert getattr(Vllm024Adapter, method) is getattr(VllmAdapter, method)
    assert adapters_mod.VLLM_ENGINE_NAMES == ("vllm", "vllm024")


# --------------------------------------------------------------------------
# Redline adapter against a stub engine module (exercises the real pump loop)
# --------------------------------------------------------------------------


class _StubRedlineEngine:
    """Pure-Python stand-in for the pybind Engine: same call surface as
    docs/DESIGN.md section 10. Each request 'prefills' for two empty steps,
    then emits one token per step; tokens echo prompt-derived values so the
    client's id bookkeeping is checkable."""

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir
        self.kwargs = kwargs
        self._next_id = 100  # engine ids deliberately differ from workload ids
        self._active = {}
        self._steps = 0
        self.abort_first = False
        self.emit_unknown = False

    def add_request(self, token_ids, max_new_tokens, ignore_eos=False):
        assert token_ids, "empty prompt"
        assert ignore_eos, "bench must force exact lengths"
        engine_id = self._next_id
        self._next_id += 1
        self._active[engine_id] = {
            "prefill_left": 2,
            "emitted": 0,
            "max_new": max_new_tokens,
            "base": token_ids[0],
        }
        return engine_id

    def has_capacity(self):
        return True

    def step(self):
        self._steps += 1
        if self.emit_unknown and self._active:
            # Simulates a stale in-flight request from an earlier aborted
            # trial surfacing mid-pump (an id the adapter never mapped).
            return [(999_999, 1, False, "")]
        results = []
        for engine_id, state in list(self._active.items()):
            if state["prefill_left"] > 0:
                state["prefill_left"] -= 1
                continue
            if self.abort_first and engine_id == 100:
                results.append((engine_id, -1, True, "aborted"))
                del self._active[engine_id]
                self.abort_first = False
                continue
            token = (state["base"] + state["emitted"]) % 151_936
            state["emitted"] += 1
            finished = state["emitted"] >= state["max_new"]
            results.append((engine_id, token, finished, "length" if finished else ""))
            if finished:
                del self._active[engine_id]
        return results

    def stats(self):
        return {"steps": self._steps, "waiting": 0, "running": len(self._active)}


@pytest.fixture
def stub_redline(monkeypatch):
    import sys
    import types

    module = types.ModuleType("redline")
    module.Engine = _StubRedlineEngine
    monkeypatch.setitem(sys.modules, "redline", module)
    return module


def test_redline_adapter_pump_loop(stub_redline):
    from bench.adapters.redline_a import RedlineAdapter

    workload = synthetic_workload("t", 3, 8, 5, seed=0, arrival_stagger_s=0.01)
    adapter = RedlineAdapter()
    adapter.load("model-dir", {"max_batch": 4})
    try:
        traces = adapter.run(workload)
        assert_forced_lengths(traces, workload)
        traces.sort(key=lambda t: t.request_id)
        for trace, request in zip(traces, workload.requests):
            assert trace.finish_reason == "length"
            assert trace.token_times == sorted(trace.token_times)
            # Stub echoes base+k: proves engine-id -> workload-id routing.
            base = request.prompt_token_ids[0]
            assert trace.token_ids == [(base + k) % 151_936 for k in range(5)]
        # A second trial on the same engine must work (per-trial state resets).
        traces = adapter.run(workload)
        assert_forced_lengths(traces, workload)
        stats = adapter.engine_stats()
        assert stats["steps"] > 0
        assert adapter.algo_report() is None
        info = adapter.version_info()
        assert info["engine"] == "redline"
        assert info["config"]["max_batch"] == 4
        assert info["config"]["admission_policy"] == "reserve_full"  # dev default
    finally:
        adapter.shutdown()


def test_redline_adapter_surfaces_aborts_as_short_traces(stub_redline):
    from bench.adapters.redline_a import RedlineAdapter

    workload = synthetic_workload("t", 2, 8, 5, seed=0)
    adapter = RedlineAdapter()
    adapter.load("model-dir", {})
    try:
        adapter._engine.abort_first = True
        traces = adapter.run(workload)
        by_id = {t.request_id: t for t in traces}
        aborted = [t for t in traces if t.finish_reason == "aborted"]
        assert len(aborted) == 1
        assert aborted[0].num_tokens == 0  # no token recorded for the abort result
        assert by_id[1 - aborted[0].request_id].num_tokens == 5
        with pytest.raises(ForcedLengthError, match="aborted"):
            assert_forced_lengths(traces, workload)
    finally:
        adapter.shutdown()


def test_redline_adapter_forwards_every_config_key(stub_redline):
    """Recorded config == applied config: unknown keys (e.g. a redline ctor
    kwarg passed via --engine-arg) reach the constructor instead of being
    recorded in the results JSON but silently dropped (a genuinely unknown
    kwarg then raises TypeError from pybind11 loudly)."""
    from bench.adapters.redline_a import RedlineAdapter

    adapter = RedlineAdapter()
    adapter.load("model-dir", {"max_batch": 4, "pad_eager_to_bucket": True})
    try:
        assert adapter._engine.kwargs["max_batch"] == 4
        assert adapter._engine.kwargs["pad_eager_to_bucket"] is True
        assert adapter.version_info()["config"]["pad_eager_to_bucket"] is True
    finally:
        adapter.shutdown()


def test_redline_adapter_refuses_reuse_after_failed_trial(stub_redline):
    """A trial that dies mid-flight leaves unabortable requests inside the
    engine whose ids would corrupt the next trial's pump routing; the adapter
    poisons itself so a future retry loop fails loudly, and load() rebuilds."""
    from bench.adapters.redline_a import RedlineAdapter

    workload = synthetic_workload("t", 2, 8, 5, seed=0)
    adapter = RedlineAdapter()
    adapter.load("model-dir", {})
    try:
        adapter._engine.emit_unknown = True
        with pytest.raises(RuntimeError):
            adapter.run(workload)
        with pytest.raises(RuntimeError, match="rebuild"):
            adapter.run(workload)
        adapter.load("model-dir", {})  # rebuild clears the guard
        traces = adapter.run(workload)
        assert_forced_lengths(traces, workload)
    finally:
        adapter.shutdown()


# --------------------------------------------------------------------------
# Baseline adapters: config filtering and fairness guards (no engine libs)
# --------------------------------------------------------------------------


def test_vllm_filter_kwargs_drops_unknown_keys():
    import dataclasses

    from bench.adapters.vllm_a import _filter_kwargs

    @dataclasses.dataclass
    class Args:
        model: str = ""
        dtype: str = "auto"

    kept, dropped = _filter_kwargs(Args, {"model": "m", "dtype": "float16", "bogus": 1})
    assert kept == {"model": "m", "dtype": "float16"}
    assert dropped == ["bogus"]


def test_vllm_filter_kwargs_var_keyword_passthrough():
    """``LLM.__init__(..., **kwargs)`` forwards unknown keys to EngineArgs;
    filtering against its named parameters would silently drop engine-critical
    config (enable_prefix_caching=False is not a named LLM parameter)."""
    from bench.adapters.vllm_a import _filter_kwargs

    class LLMLike:
        def __init__(self, model, dtype="auto", **kwargs):
            pass

    kept, dropped = _filter_kwargs(
        LLMLike, {"model": "m", "enable_prefix_caching": False, "dtype": "float16"}
    )
    assert kept == {"model": "m", "enable_prefix_caching": False, "dtype": "float16"}
    assert dropped == []


def test_llamacpp_token_path_validation(monkeypatch):
    import bench.adapters.llamacpp_a as llamacpp_a

    adapter = llamacpp_a.LlamaCppAdapter()
    calls = {}

    def fake_http(url, payload=None, timeout=10.0):
        calls[url.rsplit("/", 1)[-1]] = payload
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"default_generation_settings": {"n_ctx": 4}, "total_slots": 2}
        if url.endswith("/completion"):
            return {"timings": {"prompt_n": len(payload["prompt"])}}
        raise AssertionError(url)

    monkeypatch.setattr(llamacpp_a, "_http_json", fake_http)
    adapter.load("unused", {"server_url": "http://host:1234/"})
    assert calls["completion"]["cache_prompt"] is False  # no prompt-cache advantage
    assert adapter.version_info()["total_slots"] == 2

    # Context arithmetic guard: 8 in + 5 out > n_ctx 4 per slot -> refuse.
    workload = synthetic_workload("t", 2, 8, 5, seed=0)
    with pytest.raises(RuntimeError, match="per-slot context"):
        adapter._check_context_fits(workload)

    # Token-array path must be verified, not assumed: mismatched count -> refuse.
    def broken_http(url, payload=None, timeout=10.0):
        if url.endswith("/completion"):
            return {"timings": {"prompt_n": 1}}  # server retokenized/truncated
        return fake_http(url, payload, timeout)

    monkeypatch.setattr(llamacpp_a, "_http_json", broken_http)
    with pytest.raises(RuntimeError, match="token-array prompt path"):
        adapter.load("unused", {"server_url": "http://host:1234"})


def test_llamacpp_refuses_fewer_slots_than_requests():
    """A server missing --parallel 64 would queue the wave server-side - a
    silently handicapped baseline; the adapter refuses to measure it."""
    import bench.adapters.llamacpp_a as llamacpp_a

    adapter = llamacpp_a.LlamaCppAdapter()
    workload = synthetic_workload("t", 3, 4, 2, seed=0)
    adapter._props = {"total_slots": 2}
    with pytest.raises(RuntimeError, match="parallel slots"):
        adapter._check_parallel_slots(workload)
    adapter._props = {"total_slots": 3}
    adapter._check_parallel_slots(workload)  # exact fit passes
    adapter._props = None
    adapter._check_parallel_slots(workload)  # unreported -> the FAIRNESS.md launch-line pin governs


# --------------------------------------------------------------------------
# End-to-end runner with a fake engine (CPU-only)
# --------------------------------------------------------------------------


class FakeAdapter(EngineAdapter):
    """Synthetic engine: exact forced lengths, deterministic timing. Each
    run() gets slower by `drift` so the dispersion rule can be exercised."""

    name = "fake"

    def __init__(self, drift=0.0, fail_short=False):
        self.drift = drift
        self.fail_short = fail_short
        self.runs = 0
        self.loaded = False
        self.shut_down = False

    def load(self, model_dir, config):
        self.loaded = True
        self.config = dict(config)

    async def _stream_one(self, request, trace):  # pragma: no cover - unused
        raise NotImplementedError

    def run(self, workload):
        delta = 0.010 * (1.0 + self.drift * self.runs)
        self.runs += 1
        traces = []
        for request in workload.requests:
            enqueue = request.arrival_time_s
            first = enqueue + 0.5
            count = request.max_new_tokens - (1 if self.fail_short else 0)
            times = [first + k * delta for k in range(count)]
            traces.append(
                RequestTrace(
                    request_id=request.request_id,
                    enqueue_time=enqueue,
                    token_times=times,
                    token_ids=[token_id(1, request.request_id, k) for k in range(count)],
                    finish_reason="length",
                )
            )
        return traces

    def shutdown(self):
        self.shut_down = True

    def engine_stats(self):
        return {"steps": 123, "graph_replays": 100, "eager_decodes": 23}

    def algo_report(self):
        return None


FAKE_ENVIRONMENT = {
    "timestamp_utc": "2026-01-01T00:00:00+00:00",
    "command_line": ["python", "-m", "bench.run_bench"],
    "python": "test",
    "platform": "test",
    "hostname": "test",
    "kernel": "test",
    "mem_total_kb": None,
    "nvidia_smi_q": {"cmd": ["nvidia-smi", "-q"], "ok": False, "output": None, "error": "n/a"},
    "lscpu": {"cmd": ["lscpu"], "ok": False, "output": None, "error": "n/a"},
    "nvcc_version": {"cmd": ["nvcc", "--version"], "ok": False, "output": None, "error": "n/a"},
    "gcc_version": {"cmd": ["gcc", "--version"], "ok": False, "output": None, "error": "n/a"},
    "pip_freeze": {"cmd": ["pip", "freeze"], "ok": False, "output": None, "error": "n/a"},
    "git_commit": {"cmd": ["git"], "ok": False, "output": None, "error": "n/a"},
}


def _run_main(monkeypatch, tmp_path, engine, adapter, extra_args=()):
    monkeypatch.setattr(run_bench.adapters_mod, "resolve_adapter", lambda name: adapter)
    monkeypatch.setattr(run_bench, "capture_environment", lambda: dict(FAKE_ENVIRONMENT))
    out = tmp_path / f"{engine}_batch1.json"
    code = run_bench.main(
        [
            "--engine",
            engine,
            "--case",
            "batch1",
            "--trials",
            "3",
            "--model-dir",
            "unused",
            "--out",
            str(out),
            *extra_args,
        ]
    )
    return code, out


def test_run_bench_end_to_end_stable(monkeypatch, tmp_path):
    adapter = FakeAdapter(drift=0.0)
    code, out = _run_main(monkeypatch, tmp_path, "redline", adapter)
    assert code == 0
    assert adapter.loaded and adapter.shut_down
    assert adapter.runs == 4  # 1 warmup + 3 trials, no dispersion re-run

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema_version"] == RESULTS_SCHEMA_VERSION
    assert result["engine"] == "redline"
    assert result["case"] == "batch1"
    assert result["workload"] == {
        "name": "batch1",
        "num_requests": 1,
        "input_len": 1024,
        "output_len": 256,
        "arrival_stagger_s": 0.0,
    }
    assert len(result["warmup"]) == 1
    assert len(result["trials"]) == 3
    assert result["dispersion"]["reran"] is False
    assert result["dispersion"]["flag"] is False
    assert result["forced_length_ok"] is True
    assert result["engine_stats"]["steps"] == 123
    assert result["lt_algo_report"] is None
    # Raw per-token data present, rebased to the trial's first submit.
    trial = result["trials"][0]
    request = trial["requests"][0]
    assert request["enqueue_time_s"] == 0.0
    assert len(request["token_times_s"]) == 256
    assert len(request["token_ids"]) == 256
    assert request["finish_reason"] == "length"
    # Headline label carries the prefill-inclusive wording.
    assert "prefill" in result["metrics"]["throughput_tok_s"]["label"]
    # min <= value <= max on every summarized metric.
    for entry in result["metrics"].values():
        assert entry["min"] <= entry["value"] <= entry["max"]


def test_run_bench_dispersion_rerun_and_flag(monkeypatch, tmp_path):
    adapter = FakeAdapter(drift=0.2)  # each run ~monotonically slower
    code, out = _run_main(monkeypatch, tmp_path, "redline", adapter)
    assert code == 0
    assert adapter.runs == 5  # 1 warmup + 3 trials + 1 dispersion re-run

    result = json.loads(out.read_text(encoding="utf-8"))
    assert len(result["trials"]) == 4
    assert result["trials"][3].get("dispersion_rerun") is True
    assert result["dispersion"]["reran"] is True
    assert result["dispersion"]["flag"] is True  # drift persists after the re-run
    values = result["dispersion"]["values"]
    assert result["trials"][result["median_trial_index"]]["metrics"][
        "throughput_tok_s"
    ] == sorted(values)[(len(values) - 1) // 2]


def test_run_bench_refuses_forced_length_violation(monkeypatch, tmp_path):
    adapter = FakeAdapter(fail_short=True)
    code, out = _run_main(monkeypatch, tmp_path, "redline", adapter)
    assert code == 2  # trial void, nothing written
    assert not out.exists()
    assert adapter.shut_down


@pytest.mark.parametrize("arm", ["vllm", "vllm024"])
def test_run_bench_refuses_handicapped_baseline_cap(tmp_path, arm):
    """An explicit vLLM concurrency cap below the workload's request count is
    a deliberately handicapped baseline (FAIRNESS.md 'Engine configuration');
    the runner refuses before touching any engine - on both vLLM arms."""
    argv = [
        "--engine", arm, "--case", "primary",
        "--model-dir", "unused", "--out", str(tmp_path / "x.json"),
    ]
    with pytest.raises(SystemExit, match="max_num_seqs=8"):
        run_bench.main([*argv, "--max-batch", "8"])
    # The --engine-arg passthrough is guarded identically.
    with pytest.raises(SystemExit, match="max_num_seqs=8"):
        run_bench.main([*argv, "--engine-arg", "max_num_seqs=8"])


# --------------------------------------------------------------------------
# Offline client-overhead cross-check (--offline-crosscheck)
# --------------------------------------------------------------------------


class OfflineFakeAdapter(FakeAdapter):
    """FakeAdapter with a vLLM-style offline batch entrypoint."""

    name = "vllm"

    def __init__(self, short=False):
        super().__init__()
        self.short = short
        self.offline_calls = []

    def run_offline(self, model_dir, workload, config=None, warmup=1):
        self.offline_calls.append(
            {"model_dir": model_dir, "config": dict(config or {}), "warmup": warmup}
        )
        lengths = [
            r.max_new_tokens - (1 if self.short else 0) for r in workload.requests
        ]
        expected = [r.max_new_tokens for r in workload.requests]
        return {
            "mode": "offline_llm_generate",
            "wall_s": 2.0,
            "total_new_tokens": sum(lengths),
            "throughput_tok_s": sum(lengths) / 2.0,
            "per_request_lengths": lengths,
            "forced_length_ok": lengths == expected,
            "warmup_runs": warmup,
            "warmup_wall_s": [2.0] * warmup,
            "dropped_engine_args": [],
        }


def test_run_bench_offline_crosscheck(monkeypatch, tmp_path):
    adapter = OfflineFakeAdapter()
    code, out = _run_main(
        monkeypatch, tmp_path, "vllm", adapter, ("--offline-crosscheck",)
    )
    assert code == 0
    assert adapter.offline_calls == [
        {"model_dir": "unused", "config": {}, "warmup": 1}
    ]
    assert not adapter.loaded  # offline path never starts the streaming engine

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema_version"] == RESULTS_SCHEMA_VERSION
    assert result["mode"] == "offline_crosscheck"
    assert "trials" not in result  # not a cross-engine result shape
    assert result["forced_length_ok"] is True
    assert result["offline"]["throughput_tok_s"] == pytest.approx(256 / 2.0)
    assert result["offline"]["warmup_runs"] == 1


def test_run_bench_offline_crosscheck_refusals(monkeypatch, tmp_path):
    # Forced-length violation voids the run: exit 2, nothing written.
    adapter = OfflineFakeAdapter(short=True)
    code, out = _run_main(
        monkeypatch, tmp_path, "vllm", adapter, ("--offline-crosscheck",)
    )
    assert code == 2
    assert not out.exists()

    # Engines without an offline batch entrypoint are refused.
    with pytest.raises(SystemExit, match="no offline path"):
        _run_main(
            monkeypatch, tmp_path, "redline", FakeAdapter(), ("--offline-crosscheck",)
        )

    # Staggered arrivals cannot be represented by a closed offline batch.
    with pytest.raises(SystemExit, match="staggers arrivals"):
        _run_main(
            monkeypatch,
            tmp_path,
            "vllm",
            OfflineFakeAdapter(),
            ("--case", "arrival", "--offline-crosscheck"),
        )


# --------------------------------------------------------------------------
# Equivalence gate
# --------------------------------------------------------------------------


class EchoAdapter(FakeAdapter):
    """Emits a deterministic function of (engine tag, request, position)."""

    def __init__(self, name, flip_at=None):
        super().__init__()
        self.name = name
        self.flip_at = flip_at  # (request_id, position) to diverge at

    def run(self, workload):
        traces = super().run(workload)
        for trace in traces:
            trace.token_ids = [
                token_id(7, trace.request_id, k) for k in range(trace.num_tokens)
            ]
            if self.flip_at and self.flip_at[0] == trace.request_id:
                trace.token_ids[self.flip_at[1]] += 1
        return traces


def test_equivalence_gate_transcript():
    matching = EchoAdapter("vllm")
    diverging = EchoAdapter("llamacpp", flip_at=(2, 5))
    reference = EchoAdapter("redline")
    transcript = equivalence_gate([reference, matching, diverging], "unused")

    assert transcript["engines"] == ["redline", "vllm", "llamacpp"]
    assert transcript["workload"]["num_requests"] == 5
    assert len(transcript["workload"]["prompts"]) == 5
    assert not transcript["all_match"]
    comparisons = {
        (c["candidate"], c["request_id"]): c for c in transcript["comparisons"]
    }
    assert all(comparisons[("vllm", r)]["match"] for r in range(5))
    bad = comparisons[("llamacpp", 2)]
    assert not bad["match"]
    assert bad["first_divergence"]["position"] == 5
    assert bad["first_divergence"]["hf_margin"] is None
    assert (
        bad["first_divergence"]["candidate_token"]
        == bad["first_divergence"]["reference_token"] + 1
    )
    # every engine was shut down after its sequential turn
    for adapter in (reference, matching, diverging):
        assert adapter.shut_down


def _run_gate_main(monkeypatch, tmp_path, adapters, extra_args=()):
    """Drive the `--equivalence-gate` CLI wiring with fakes."""
    monkeypatch.setattr(
        run_bench.adapters_mod, "resolve_adapter", lambda name: adapters[name]
    )
    monkeypatch.setattr(run_bench, "capture_environment", lambda: dict(FAKE_ENVIRONMENT))
    monkeypatch.setattr(run_bench, "_gpu_memory_used_mib", lambda: 7)
    out = tmp_path / "transcript.json"
    code = run_bench.main(
        [
            "--equivalence-gate",
            "--engines",
            ",".join(adapters),
            "--model-dir",
            "unused",
            "--out",
            str(out),
            *extra_args,
        ]
    )
    return code, out


def test_run_bench_equivalence_gate_cli_pass(monkeypatch, tmp_path):
    adapters = {name: EchoAdapter(name) for name in ("redline", "vllm", "llamacpp")}
    code, out = _run_gate_main(monkeypatch, tmp_path, adapters)
    assert code == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema_version"] == RESULTS_SCHEMA_VERSION
    assert result["mode"] == "equivalence_gate"
    assert result["gate_pass"] is True
    assert result["all_match"] is True
    assert result["engines"] == ["redline", "vllm", "llamacpp"]
    assert result["engine_versions"] == {
        name: {"engine": name} for name in ("redline", "vllm", "llamacpp")
    }
    # Matched streams never load HF; the review records why it did not run.
    assert result["margin_review"]["ran"] is False
    # Idle reading before every engine turn plus one after the last shutdown.
    assert [r["engine"] for r in result["gpu_memory_used_mib"]] == [
        "redline",
        "vllm",
        "llamacpp",
        None,
    ]
    # Per-engine configs recorded: redline full explicit config, vLLM its own
    # defaults (no silent caps), llamacpp just the server URL.
    assert result["engine_requested_configs"]["redline"]["max_batch"] == 8
    assert result["engine_requested_configs"]["vllm"] == {}
    assert "server_url" in result["engine_requested_configs"]["llamacpp"]
    for adapter in adapters.values():
        assert adapter.loaded and adapter.shut_down


def test_run_bench_equivalence_gate_cli_margin_review(monkeypatch, tmp_path):
    def diverging_pair():
        return {
            "redline": EchoAdapter("redline"),
            "vllm": EchoAdapter("vllm", flip_at=(1, 3)),
        }

    # Unreviewed divergence (review unavailable): gate fails, transcript kept.
    monkeypatch.setattr(
        run_bench,
        "hf_margin_review",
        lambda transcript, model_dir: {"ran": False, "reason": "test stub"},
    )
    code, out = _run_gate_main(monkeypatch, tmp_path, diverging_pair())
    assert code == 3
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["gate_pass"] is False
    assert result["all_match"] is False

    # The same divergence reviewed as an FP16 coin-flip (margin < 0.1): pass.
    def fill_margins(transcript, model_dir):
        for comparison in transcript["comparisons"]:
            if comparison["first_divergence"]:
                comparison["first_divergence"]["hf_margin"] = 0.05
        return {"ran": True, "margin_threshold": run_bench.EQUIVALENCE_MARGIN}

    monkeypatch.setattr(run_bench, "hf_margin_review", fill_margins)
    code, out = _run_gate_main(monkeypatch, tmp_path, diverging_pair())
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["gate_pass"] is True
    assert result["all_match"] is False
    divergence = next(
        c["first_divergence"] for c in result["comparisons"] if not c["match"]
    )
    assert divergence["position"] == 3
    assert divergence["hf_margin"] == 0.05

    # A reviewed margin at/above the threshold still fails the gate.
    def wide_margins(transcript, model_dir):
        for comparison in transcript["comparisons"]:
            if comparison["first_divergence"]:
                comparison["first_divergence"]["hf_margin"] = 0.1
        return {"ran": True}

    monkeypatch.setattr(run_bench, "hf_margin_review", wide_margins)
    code, _ = _run_gate_main(monkeypatch, tmp_path, diverging_pair())
    assert code == 3


def test_run_bench_equivalence_gate_cli_refusals(tmp_path):
    """Every refusal fires before any adapter is resolved or loaded."""
    argv = ["--equivalence-gate", "--model-dir", "unused", "--out", str(tmp_path / "t.json")]
    with pytest.raises(SystemExit, match="not --engine"):
        run_bench.main([*argv, "--engine", "redline"])
    with pytest.raises(SystemExit, match="--engine-arg is not allowed"):
        run_bench.main([*argv, "--engine-arg", "x=1"])
    with pytest.raises(SystemExit, match="unknown engine"):
        run_bench.main([*argv, "--engines", "redline,nope"])
    with pytest.raises(SystemExit, match="duplicate engine"):
        run_bench.main([*argv, "--engines", "redline,redline"])
    # One interpreter has exactly one vllm installed: a gate claiming both
    # arms would measure the same binary twice. Each arm gates in its own
    # venv (e.g. --engines redline,vllm024 under the 0.24 interpreter).
    with pytest.raises(SystemExit, match="own venv"):
        run_bench.main([*argv, "--engines", "redline,vllm,vllm024"])
    with pytest.raises(SystemExit, match="at least two"):
        run_bench.main([*argv, "--engines", "redline"])
    # The gate default stays the three co-installable engines; vllm024 is
    # opt-in precisely because it needs its own interpreter.
    assert run_bench.build_arg_parser().parse_args(
        ["--model-dir", "unused", "--out", "o.json"]
    ).engines == "redline,vllm,llamacpp"
    with pytest.raises(SystemExit, match="llamacpp is not in --engines"):
        run_bench.main(
            [*argv, "--engines", "redline,vllm", "--llamacpp-server-cmd", "llama-server"]
        )
    # Normal measured mode still requires --engine (argparse usage error).
    with pytest.raises(SystemExit):
        run_bench.main(["--model-dir", "unused", "--out", str(tmp_path / "x.json")])


# --------------------------------------------------------------------------
# report.py: validation, rendering, --check, README splice
# --------------------------------------------------------------------------


def _fake_trial(index, scale, num_requests=2, output_len=4):
    """One synthetic trial whose stored metrics ARE recomputable from its raw
    per-request record - validate_result's raw-data audit re-derives them, so
    fixtures must be internally consistent exactly like a genuine
    run_bench.py file. ``scale`` slows the whole timing schedule, moving the
    trial's headline throughput."""
    traces = [
        RequestTrace(
            request_id=rid,
            enqueue_time=0.0,
            token_times=[scale * (0.5 + rid * 0.05 + k * 0.1) for k in range(output_len)],
            token_ids=[token_id(3, rid, k) for k in range(output_len)],
            finish_reason="length",
        )
        for rid in range(num_requests)
    ]
    return {
        "index": index,
        "started_utc": "2026-01-01T00:00:00+00:00",
        "metrics": compute_metrics(traces),
        "requests": [run_bench._trace_to_json(t, 0.0) for t in traces],
    }


def _fake_result(
    engine="redline", case="primary", gpu=None, flag=False, vllm_version=None,
    driver="560.35.03",
):
    # Three trials; the flagged variant spreads its headline values > 3% of
    # the median so the stored dispersion flag survives the flag-vs-spread
    # recomputation of the raw-data audit.
    scales = (1.0, 1.08, 1.03) if flag else (1.0, 1.0008, 1.0004)
    trials = [_fake_trial(i, s) for i, s in enumerate(scales)]
    values = [t["metrics"][HEADLINE_METRIC] for t in trials]
    assert spread_exceeds(values) is flag  # fixture self-check
    median_index = select_median_trial(values)
    metrics = summarize_metrics(trials, median_index)
    engine_config = {"engine": engine}
    if vllm_version is not None:
        engine_config["vllm_version"] = vllm_version
    nvidia = dict(FAKE_ENVIRONMENT["nvidia_smi_q"])
    if gpu is not None:
        nvidia = {
            "cmd": ["nvidia-smi", "-q"],
            "ok": True,
            "output": f"    Product Name                          : {gpu}\n"
            f"    Driver Version                        : {driver}\n",
            "error": None,
        }
    environment = dict(FAKE_ENVIRONMENT)
    environment["nvidia_smi_q"] = nvidia
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "engine": engine,
        "case": case,
        "seed": 0,
        "workload": {
            "name": case,
            "num_requests": 2,
            "input_len": 1024,
            "output_len": 4,
            "arrival_stagger_s": 0.0,
        },
        "engine_config": engine_config,
        "environment": environment,
        "warmup": [],
        "trials": trials,
        "median_trial_index": median_index,
        "metrics": metrics,
        "dispersion": {
            "metric": HEADLINE_METRIC,
            "threshold_fraction": 0.03,
            "values": values,
            "reran": False,
            "flag": flag,
        },
        "forced_length_ok": True,
        "engine_stats": None,
        "lt_algo_report": None,
    }


def test_report_validates_and_renders(tmp_path):
    for engine, flag in (("redline", False), ("vllm", True)):
        path = tmp_path / f"{engine}_primary.json"
        path.write_text(json.dumps(_fake_result(engine=engine, flag=flag)), encoding="utf-8")

    results = report.load_results([str(tmp_path)])
    assert {r["engine"] for r in results} == {"redline", "vllm"}

    table = report.render_markdown(results)
    assert "### case `primary`" in table
    assert "| `redline` |" in table and "| `vllm` |" in table
    assert "incl. prefill" in table  # headline column label
    assert "†" in table  # vllm row is dispersion-flagged
    assert "Qwen2.5-1.5B" in table  # scope caption
    assert "no claim of general engine superiority" in table.lower()


def test_report_refuses_bad_files(tmp_path):
    result = _fake_result()
    del result["environment"]["nvidia_smi_q"]
    path = tmp_path / "bad_env.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(report.ReportError, match="environment capture missing"):
        report.load_results([str(path)])

    result = _fake_result()
    result["forced_length_ok"] = False
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(report.ReportError, match="forced_length_ok"):
        report.load_results([str(path)])

    result = _fake_result()
    result["schema_version"] = 999
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(report.ReportError, match="schema_version"):
        report.load_results([str(path)])

    result = _fake_result()
    result["metrics"]["throughput_tok_s"]["label"] = "output tok/s"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(report.ReportError, match="prefill"):
        report.load_results([str(path)])

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(report.ReportError, match="unreadable"):
        report.load_results([str(path)])


def test_report_refuses_mismatched_environments(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(_fake_result(engine="redline", gpu="NVIDIA GeForce RTX 4090")),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(_fake_result(engine="vllm", gpu="NVIDIA GeForce RTX 2060")),
        encoding="utf-8",
    )
    with pytest.raises(report.ReportError, match="mismatched environments"):
        report.render_markdown(report.load_results([str(a), str(b)]))


def test_report_renders_same_gpu_mixed_driver_with_disclosure(tmp_path, capsys):
    """The cross-session two-arm case: rows sharing the GPU model but
    measured under different driver versions (the vllm024 arm runs in its
    own pod session - bench/FAIRNESS.md "Engine configuration") render, and
    the footnote names each driver with the rows measured under it instead
    of faking a single-driver guarantee. Distinct GPU models stay refused
    (test_report_refuses_mismatched_environments)."""
    a = tmp_path / "redline_primary.json"
    b = tmp_path / "vllm024_primary.json"
    a.write_text(
        json.dumps(
            _fake_result(
                engine="redline", gpu="NVIDIA GeForce RTX 4090", driver="580.159.04"
            )
        ),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(
            _fake_result(
                engine="vllm024",
                vllm_version="0.24.0",
                gpu="NVIDIA GeForce RTX 4090",
                driver="580.159.03",
            )
        ),
        encoding="utf-8",
    )

    table = report.render_markdown(report.load_results([str(tmp_path)]))
    assert "NVIDIA GeForce RTX 4090" in table
    assert "580.159.04 (redline rows)" in table
    assert "580.159.03 (vllm 0.24.0 rows)" in table
    assert "separate same-GPU-model sessions" in table
    # A directory that renders must also pass --check (the results gate).
    assert report.main(["--check", str(tmp_path)]) == 0
    assert "0 failure(s)" in capsys.readouterr().out


def test_report_refuses_mixed_gpu_capture(tmp_path):
    """A result whose nvidia-smi capture failed is an environment wildcard:
    rendering it next to captured results would fake a same-GPU guarantee."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(_fake_result(engine="redline", gpu="NVIDIA GeForce RTX 4090")),
        encoding="utf-8",
    )
    b.write_text(json.dumps(_fake_result(engine="vllm")), encoding="utf-8")  # no capture
    with pytest.raises(report.ReportError, match="GPU capture"):
        report.render_markdown(report.load_results([str(a), str(b)]))


def test_report_refuses_duplicate_engine_case(tmp_path):
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(_fake_result()), encoding="utf-8")
    with pytest.raises(report.ReportError, match="duplicate results"):
        report.render_markdown(report.load_results([str(tmp_path)]))


def test_report_renders_both_vllm_arms(tmp_path):
    """A mixed directory holding both vLLM baseline arms renders one row per
    arm, labeled from each file's own recorded vllm_version, in deterministic
    ENGINE_ORDER, and the environment footnote lists both arms' pins."""
    fixtures = {
        "llamacpp_primary.json": _fake_result(engine="llamacpp"),
        "redline_primary.json": _fake_result(engine="redline"),
        "vllm024_primary.json": _fake_result(engine="vllm024", vllm_version="0.24.0"),
        "vllm_primary.json": _fake_result(engine="vllm", vllm_version="0.9.2"),
    }
    for name, result in fixtures.items():
        (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")

    table = report.render_markdown(report.load_results([str(tmp_path)]))
    # Both arms present, distinctly labeled by their recorded versions.
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    assert [row.split("`")[1] for row in rows] == [
        "redline",
        "vllm 0.9.2",
        "vllm 0.24.0",
        "llamacpp",
    ]
    # The environment footnote lists both arms' version pins, in order.
    assert "Engines: redline; vllm 0.9.2; vllm024 0.24.0; llamacpp." in table
    # The reproduction section keeps one recorded command line per arm.
    assert table.count("python -m bench.run_bench") >= 4


def test_report_check_treats_vllm_arms_as_distinct(tmp_path, capsys):
    """``--check``: (vllm, case) and (vllm024, case) are distinct pairs - a
    directory holding both arms passes - while duplicate detection still
    fires WITHIN an arm."""
    (tmp_path / "vllm_primary.json").write_text(
        json.dumps(_fake_result(engine="vllm", vllm_version="0.9.2")), encoding="utf-8"
    )
    (tmp_path / "vllm024_primary.json").write_text(
        json.dumps(_fake_result(engine="vllm024", vllm_version="0.24.0")), encoding="utf-8"
    )
    assert report.main(["--check", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "engine=vllm case=primary" in out
    assert "engine=vllm024 case=primary" in out

    (tmp_path / "vllm024_primary_rerun.json").write_text(
        json.dumps(_fake_result(engine="vllm024", vllm_version="0.24.0")), encoding="utf-8"
    )
    assert report.main(["--check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "duplicate results for engine=vllm024 case=primary" in out
    assert "would not render" in out


def test_report_refuses_ambiguous_arm_labels(tmp_path, capsys):
    """Two different engine names rendering the same row label - e.g. a
    preliminary engine="vllm" file measured under the 0.24 venv next to the
    formal vllm024 arm - would make the table ambiguous about what was
    measured; render refuses and --check fails identically."""
    (tmp_path / "a.json").write_text(
        json.dumps(_fake_result(engine="vllm", vllm_version="0.24.0")), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        json.dumps(_fake_result(engine="vllm024", vllm_version="0.24.0")), encoding="utf-8"
    )
    with pytest.raises(report.ReportError, match="ambiguous row label"):
        report.render_markdown(report.load_results([str(tmp_path)]))
    assert report.main(["--check", str(tmp_path)]) == 1
    assert "ambiguous row label" in capsys.readouterr().out


def test_report_check_mode(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_fake_result(flag=True)), encoding="utf-8")
    assert report.main(["--check", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DISPERSION-FLAGGED" in out
    assert "1 dispersion-flagged" in out

    bad = tmp_path / "bad.json"
    result = _fake_result(engine="vllm")
    result["forced_length_ok"] = False
    bad.write_text(json.dumps(result), encoding="utf-8")
    assert report.main(["--check", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_report_check_mode_flags_duplicates(tmp_path, capsys):
    """``--check`` must fail on duplicate engine/case pairs - the exact
    condition ``render_markdown`` refuses - so a directory that passes
    ``--check`` is guaranteed to render."""
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(_fake_result()), encoding="utf-8")
    assert report.main(["--check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "duplicate results for engine=redline case=primary" in out
    assert "would not render" in out


def _tampered(out, mutate):
    """Reload a runner-written results file, apply one mutation, write back."""
    data = json.loads(out.read_text(encoding="utf-8"))
    mutate(data)
    out.write_text(json.dumps(data), encoding="utf-8")


def test_report_check_recomputes_from_raw_per_token_data(monkeypatch, tmp_path, capsys):
    """``--check`` audits the data, not the self-reported flags: forced
    lengths and every metric are re-derived from the committed per-token
    record, so truncating tokens (with ``forced_length_ok`` left green),
    editing a published median, editing a per-trial metric, re-pointing the
    median index, or deleting the raw data all fail the check."""
    code, out = _run_main(monkeypatch, tmp_path, "redline", FakeAdapter())
    assert code == 0
    assert report.main(["--check", str(out)]) == 0  # the genuine file passes
    pristine = out.read_text(encoding="utf-8")
    capsys.readouterr()

    def check_fails(expected_fragment):
        assert report.main(["--check", str(out)]) == 1
        assert expected_fragment in capsys.readouterr().out
        out.write_text(pristine, encoding="utf-8")  # restore for the next tamper

    # (a) The exact demo this audit exists for: drop 3 tokens from one
    # request's token_ids AND token_times_s, leaving forced_length_ok true.
    def truncate(data):
        request = data["trials"][0]["requests"][0]
        request["token_ids"] = request["token_ids"][:-3]
        request["token_times_s"] = request["token_times_s"][:-3]
        assert data["forced_length_ok"] is True

    _tampered(out, truncate)
    check_fails("forced_length_ok is contradicted")

    # (b) Ragged tamper: token_ids shortened but timestamps left intact.
    def rag(data):
        request = data["trials"][0]["requests"][0]
        request["token_ids"] = request["token_ids"][:-1]

    _tampered(out, rag)
    check_fails("index-parallel")

    # (c) Edited summary value (the published number itself).
    _tampered(out, lambda d: d["metrics"]["throughput_tok_s"].update(
        value=d["metrics"]["throughput_tok_s"]["value"] * 1.01))
    check_fails("do not re-derive")

    # (d) Edited per-trial metric: contradicted by recomputation from raw data.
    def bump_trial(data):
        trial = data["trials"][data["median_trial_index"]]
        trial["metrics"]["throughput_tok_s"] *= 1.01
        # keep the summary/dispersion consistent with the edited trial so the
        # per-trial recomputation is what has to catch it
        data["metrics"]["throughput_tok_s"]["value"] = trial["metrics"]["throughput_tok_s"]
        data["dispersion"]["values"] = [
            t["metrics"]["throughput_tok_s"] for t in data["trials"]
        ]
        data["metrics"]["throughput_tok_s"]["min"] = min(data["dispersion"]["values"])
        data["metrics"]["throughput_tok_s"]["max"] = max(data["dispersion"]["values"])

    _tampered(out, bump_trial)
    check_fails("recomputed from the committed per-token data")

    # (e) Deleting the raw record is not an escape hatch.
    def drop_requests(data):
        data["trials"][0]["requests"] = []

    _tampered(out, drop_requests)
    check_fails("no raw per-request data")

    # (f) median_trial_index must actually be the median by the headline metric.
    def wrong_median(data):
        data["median_trial_index"] = (data["median_trial_index"] + 1) % len(data["trials"])
        # re-derive the summary from the re-pointed index so only the
        # median-selection audit can catch the swap
        data["metrics"] = run_bench.summarize_metrics(
            data["trials"], data["median_trial_index"]
        )

    _tampered(out, wrong_median)
    check_fails("not the median trial")


def test_report_check_dispersion_flag_recomputed(monkeypatch, tmp_path, capsys):
    """A dispersion-flagged file whose flag is quietly cleared (or a stable
    file gaining an unearned flag) fails: the flag is recomputed from the
    trials' spread, never trusted."""
    good = tmp_path / "flagged.json"
    good.write_text(json.dumps(_fake_result(flag=True)), encoding="utf-8")
    assert report.main(["--check", str(good)]) == 0
    capsys.readouterr()

    data = json.loads(good.read_text(encoding="utf-8"))
    data["dispersion"]["flag"] = False  # hide the variance flag
    good.write_text(json.dumps(data), encoding="utf-8")
    assert report.main(["--check", str(good)]) == 1
    assert "contradicts the spread rule" in capsys.readouterr().out


_COMMITTED_RAW_RESULTS = Path(__file__).resolve().parent / "results" / "raw"


@pytest.mark.skipif(
    not _COMMITTED_RAW_RESULTS.is_dir(),
    reason="no committed results directory in this tree",
)
def test_committed_results_survive_deep_check():
    """The committed benchmark artifacts must pass the full audit - schema,
    forced lengths and metrics recomputed from their own per-token data,
    summary/median/dispersion re-derivation, environment consistency. This
    runs in CI on every push, so a regression in the harness OR a tampered
    results file cannot merge green."""
    assert report.main(["--check", str(_COMMITTED_RAW_RESULTS)]) == 0


def test_report_readme_splice(tmp_path):
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps(_fake_result()), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Redline\n\nintro\n\n"
        f"{report.README_BEGIN}\nstale table\n{report.README_END}\n\ntail\n",
        encoding="utf-8",
    )
    assert report.main(["--results", str(result_path), "--write-readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "stale table" not in text
    assert "### case `primary`" in text
    assert text.startswith("# Redline")
    assert text.rstrip().endswith("tail")

    bare = tmp_path / "BARE.md"
    bare.write_text("# no markers\n", encoding="utf-8")
    with pytest.raises(report.ReportError, match="markers"):
        report.main(["--results", str(result_path), "--write-readme", str(bare)])


def test_report_end_to_end_with_runner_output(monkeypatch, tmp_path):
    """The full loop: run_bench writes a results file, report validates,
    checks, and renders it - the exact measure/check/render publication flow, CPU-only."""
    code, out = _run_main(monkeypatch, tmp_path, "redline", FakeAdapter())
    assert code == 0
    assert report.main(["--check", str(tmp_path)]) == 0
    table = report.render_markdown(report.load_results([str(out)]))
    assert "| `redline` |" in table
    assert "GPU capture unavailable" in table  # honest footnote without a GPU
