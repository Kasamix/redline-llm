"""Adapter for vLLM (baseline engine; two pinned arms).

Pinning and settings (normative rules in bench/FAIRNESS.md):
  - two baseline arms share this code path: engine ``vllm`` (``vllm==0.9.2``,
    frozen in bench/requirements.lock) and engine ``vllm024``
    (``vllm==0.24.0``, frozen in bench/requirements-vllm024.lock). Each arm
    runs under its own virtual environment - the interpreter invoking
    ``bench.run_bench`` selects the arm's venv - and ``version_info``
    re-records the venv's installed vllm version into the results JSON. The
    published pins, their relationship to the latest stable release at the
    lock date, and the per-arm gate-first protocol are disclosed in
    bench/FAIRNESS.md "Engine configuration" (a stale baseline reads as
    strawmanning even when innocent - stated there, not hidden here).
  - V1 engine public APIs only - the legacy pre-V1 ``AsyncLLMEngine`` is
    removed upstream. Streaming path (per-token timestamps) uses the V1 async
    engine (``AsyncLLM``); an offline ``LLM.generate`` run of the same
    workload (``run_offline``) bounds the streaming client's overhead and is
    reported alongside (FAIRNESS.md "Measured path"; invoked via
    ``python -m bench.run_bench --offline-crosscheck``).
  - dtype float16, greedy sampling (temperature=0), ``ignore_eos=True`` with
    forced ``max_tokens`` (the harness asserts exact output lengths);
    streaming outputs pinned to CUMULATIVE (``output_kind``) where the
    installed ``SamplingParams`` supports the field - the client's delta
    bookkeeping assumes cumulative ``token_ids``, and a DELTA-mode default
    on a release without the field would fail the forced-length assert
    loudly (exit 2), never record a silently short stream
  - CUDA-graph execution on (vLLM's default mode; ``enforce_eager=False``);
    the bounded tuning pass compares the default piecewise mode against the
    full-CUDA-graph compilation config via ``--engine-arg`` passthrough and
    publishes the best (FAIRNESS.md "Best-configuration duty")
  - prompts passed as token IDs (``TokensPrompt``) to bypass tokenization
  - prefix caching disabled: the trial protocol repeats identical prompts
    (1 warmup + 3 measured trials), so a prefix cache would measure cache
    hits instead of inference; Redline has no equivalent. Recorded in the
    engine config either way.

Version tolerance: exact kwargs drift between vLLM releases, and the real pin
happens at lock date. ``load()`` therefore filters its engine-arg dict against
the installed ``AsyncEngineArgs`` fields and records anything it had to drop
into ``version_info`` - a dropped key is visible in the results JSON, never
silent.

Loop affinity: the V1 async engine binds background tasks to the event loop
it first runs under, while the harness opens a fresh loop per trial. The
adapter therefore owns one persistent loop on a dedicated thread for the
engine's whole lifetime and submits each trial into it. Because that means
the engine is built off the main thread, ``VLLM_USE_V1=1`` is pinned into
the environment first: 0.9.x-era releases otherwise fall back to V0 for
off-main-thread builds ("engine in background thread is experimental") and
the V1 engine class then refuses to construct. V1 is the pinned execution
path (bench/FAIRNESS.md "Engine configuration"); the pin is vLLM's own
documented workaround, set with ``setdefault`` so an explicit operator
override wins, and harmless on releases without a V0 fallback.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from typing import Any

from bench.adapters import EngineAdapter, RequestTrace, drive_workload
from bench.workload import Workload, WorkloadRequest

DEFAULT_CONFIG = {
    "dtype": "float16",
    "enforce_eager": False,  # CUDA-graph execution on (default mode)
    "enable_prefix_caching": False,  # trials repeat identical prompts
    "disable_log_stats": True,
}


def _pin_v1_engine() -> None:
    """Pin the V1 engine selection (module docstring, "Loop affinity")."""
    os.environ.setdefault("VLLM_USE_V1", "1")


def _coerce_compilation_config(kwargs: dict) -> None:
    """Turn a dict-valued ``compilation_config`` (the ``--engine-arg`` JSON
    passthrough form used by the tuning pass) into a real ``CompilationConfig``.

    0.9.x-era ``EngineArgs.create_engine_config`` feeds a dict through
    ``str(dict)`` - Python repr, not JSON - into a JSON parser and fails; the
    offline ``LLM`` entrypoint handles dicts itself. Coercing here makes both
    paths take the already-an-object branch. Unknown fields still fail loudly
    from ``CompilationConfig`` itself."""
    value = kwargs.get("compilation_config")
    if isinstance(value, dict):
        from vllm.config import CompilationConfig

        kwargs["compilation_config"] = CompilationConfig(**value)


def _import_async_llm():
    """V1 async engine class across recent layouts."""
    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM
    except ImportError:
        # Releases where the V1 engine is re-exported under the historic name.
        from vllm import AsyncLLMEngine

        return AsyncLLMEngine


def _import_tokens_prompt():
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

        return TokensPrompt


def _filter_kwargs(cls, kwargs: dict) -> tuple[dict, list[str]]:
    """Keep only kwargs the installed class accepts; report the dropped ones.

    A ``**kwargs`` catch-all (``LLM.__init__`` forwards it to ``EngineArgs``)
    means every key is accepted: filtering against the named parameters there
    would silently drop engine-critical config (e.g.
    ``enable_prefix_caching=False``, which is not a named ``LLM`` parameter),
    and a genuinely unknown key then fails loudly inside the engine instead.
    """
    import dataclasses
    import inspect

    known: set[str] | None = None
    if dataclasses.is_dataclass(cls):
        known = {f.name for f in dataclasses.fields(cls)}
    else:
        try:
            params = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            params = None
        if params is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            known = set(params)
    if known is None:
        return dict(kwargs), []
    kept = {k: v for k, v in kwargs.items() if k in known}
    dropped = sorted(k for k in kwargs if k not in known)
    return kept, dropped


class VllmAdapter(EngineAdapter):
    name = "vllm"

    def __init__(self) -> None:
        self._engine: Any = None
        self._config: dict = {}
        self._dropped_args: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    # -- persistent loop -----------------------------------------------------

    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run, name="vllm-bench-loop", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread

    def _submit(self, coro):
        assert self._loop is not None, "load() must run before engine calls"
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # -- lifecycle ------------------------------------------------------------

    def load(self, model_dir: str, config: dict) -> None:
        _pin_v1_engine()  # before any engine-arg resolution (module docstring)
        from vllm.engine.arg_utils import AsyncEngineArgs

        async_llm_cls = _import_async_llm()

        self._config = dict(DEFAULT_CONFIG)
        self._config.update(config or {})

        engine_kwargs = {k: v for k, v in self._config.items() if k not in ("model_dir",)}
        engine_kwargs["model"] = model_dir
        engine_kwargs, self._dropped_args = _filter_kwargs(AsyncEngineArgs, engine_kwargs)
        _coerce_compilation_config(engine_kwargs)

        self._start_loop()

        async def _build():
            # Built on the persistent loop thread so every background task the
            # engine spawns is bound to the loop the trials will use.
            return async_llm_cls.from_engine_args(AsyncEngineArgs(**engine_kwargs))

        self._engine = self._submit(_build())

    def shutdown(self) -> None:
        if self._engine is not None:
            engine, self._engine = self._engine, None

            async def _stop():
                stop = getattr(engine, "shutdown", None)
                if callable(stop):
                    result = stop()
                    if asyncio.iscoroutine(result):
                        await result

            if self._loop is not None:
                self._submit(_stop())
            del engine
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=30)
            self._loop.close()
            self._loop = None
            self._loop_thread = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def version_info(self) -> dict:
        info = {"engine": self.name, "config": dict(self._config)}
        if self._dropped_args:
            info["dropped_engine_args"] = list(self._dropped_args)
        try:
            import vllm

            info["vllm_version"] = getattr(vllm, "__version__", None)
        except Exception:
            info["vllm_version"] = None
        return info

    # -- trial body -------------------------------------------------------------

    def run(self, workload: Workload) -> list[RequestTrace]:
        # One persistent loop for all trials (see module docstring).
        return self._submit(drive_workload(workload, self._stream_one))

    def _sampling_params(self, max_new_tokens: int, streaming: bool = True):
        from vllm import SamplingParams

        kwargs = dict(
            temperature=0.0,  # greedy
            max_tokens=max_new_tokens,
            ignore_eos=True,  # forced output lengths (bench/FAIRNESS.md "Cases")
            detokenize=False,  # token IDs only; text is outside the measured path
        )
        if streaming:
            # Pin cumulative streaming explicitly rather than relying on the
            # V1 default: _stream_one records the delta of each yield's
            # cumulative token_ids. Releases without the enum predate DELTA
            # mode and stream cumulatively anyway.
            try:
                from vllm.sampling_params import RequestOutputKind

                kwargs["output_kind"] = RequestOutputKind.CUMULATIVE
            except ImportError:
                pass
        # Kwarg drift across releases: retry without the newest extras the
        # installed SamplingParams rejects. Never silent in effect - a
        # DELTA-default release that also rejected output_kind would fail
        # the forced-length assert (exit 2).
        last_error: TypeError | None = None
        for drop in ((), ("output_kind",), ("output_kind", "detokenize")):
            attempt = {k: v for k, v in kwargs.items() if k not in drop}
            try:
                return SamplingParams(**attempt)
            except TypeError as error:
                last_error = error
        raise last_error  # not even the minimal kwargs were accepted

    async def _stream_one(self, request: WorkloadRequest, trace: RequestTrace) -> None:
        if self._engine is None:
            raise RuntimeError("VllmAdapter.load() must be called before run()")
        tokens_prompt_cls = _import_tokens_prompt()
        prompt = tokens_prompt_cls(prompt_token_ids=list(request.prompt_token_ids))
        params = self._sampling_params(request.max_new_tokens)

        seen = 0
        async for output in self._engine.generate(
            prompt, params, request_id=f"bench-{request.request_id}"
        ):
            now = time.monotonic()
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            # Cumulative streaming (pinned via output_kind where supported):
            # record only the delta. A multi-token delta gets one shared
            # receive timestamp (that is when the client saw them). If a
            # release streamed DELTA despite the pin, the recorded count
            # falls short and the forced-length assert voids the trial.
            for token in token_ids[seen:]:
                trace.record(token, now)
            seen = len(token_ids)
            if output.finished:
                trace.finish_reason = str(completion.finish_reason or "")

    # -- offline cross-check ------------------------------------------------------

    def run_offline(
        self,
        model_dir: str,
        workload: Workload,
        config: dict | None = None,
        warmup: int = 1,
    ) -> dict:
        """Client-overhead cross-check: the same workload through offline
        ``LLM.generate`` (no streaming client), reporting wall time and
        aggregate throughput only - no per-token timestamps exist on this
        path. Mirrors the streaming trial protocol with ``warmup`` unmeasured
        ``generate`` calls before the measured one. Self-contained (does not
        require ``load()``); run it as its own process invocation so the two
        engines never share the GPU. The result is reported next to the
        streaming numbers (FAIRNESS.md "Measured path"); invoked via
        ``python -m bench.run_bench --offline-crosscheck``.
        """
        _pin_v1_engine()  # same pinned V1 path as the streaming engine
        from vllm import LLM

        tokens_prompt_cls = _import_tokens_prompt()
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        self._config = dict(merged)  # version_info() reflects the offline config
        kwargs = {k: v for k, v in merged.items() if k not in ("model_dir",)}
        kwargs["model"] = model_dir
        kwargs, dropped = _filter_kwargs(LLM, kwargs)
        _coerce_compilation_config(kwargs)
        self._dropped_args = dropped

        llm = LLM(**kwargs)
        try:
            prompts = [
                tokens_prompt_cls(prompt_token_ids=list(r.prompt_token_ids))
                for r in workload.requests
            ]
            # streaming=False: no output_kind pin - the offline LLM.generate
            # entrypoint manages its own output mode.
            params = [
                self._sampling_params(r.max_new_tokens, streaming=False)
                for r in workload.requests
            ]
            warmup_walls: list[float] = []
            for _ in range(max(0, warmup)):
                t0 = time.monotonic()
                llm.generate(prompts, params)
                warmup_walls.append(time.monotonic() - t0)
            start = time.monotonic()
            outputs = llm.generate(prompts, params)
            wall_s = time.monotonic() - start
        finally:
            del llm
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        lengths = [len(o.outputs[0].token_ids) for o in outputs]
        expected = [r.max_new_tokens for r in workload.requests]
        return {
            "mode": "offline_llm_generate",
            "wall_s": wall_s,
            "total_new_tokens": sum(lengths),
            "throughput_tok_s": sum(lengths) / wall_s if wall_s > 0 else 0.0,
            "per_request_lengths": lengths,
            "forced_length_ok": lengths == expected,
            "warmup_runs": len(warmup_walls),
            "warmup_wall_s": warmup_walls,
            "dropped_engine_args": dropped,
        }


class Vllm024Adapter(VllmAdapter):
    """The second vLLM baseline arm: ``vllm==0.24.0`` (bench/FAIRNESS.md
    "Engine configuration", pin disclosure).

    Deliberately a thin variant of :class:`VllmAdapter` - same load, trial,
    streaming, and offline-crosscheck code path, no overridden measurement
    method. The arm is selected by the interpreter that invokes
    ``bench.run_bench`` (each arm has its own virtual environment; freeze
    files ``bench/requirements.lock`` for 0.9.2 and
    ``bench/requirements-vllm024.lock`` for 0.24.0), and the inherited
    ``version_info`` re-records that venv's installed ``vllm.__version__``
    into the results JSON at runtime. What this subclass adds is only the
    distinct engine NAME, which keeps the two arms apart end to end: the
    results JSON ``engine`` field, the report's per-arm rows ("vllm 0.9.2" /
    "vllm 0.24.0", labeled from the recorded version), and the duplicate
    (engine, case) detection all key on it.
    """

    name = "vllm024"
