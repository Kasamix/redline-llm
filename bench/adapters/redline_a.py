"""Adapter for the redline engine (in-process via the pybind11 module).

THIS adapter is Redline's measured path for every published cross-engine
number: all engines run through the same Python harness with the same asyncio
client structure, so client overhead is shared rather than engine-specific
(bench/FAIRNESS.md "Measured path"). The redline_bench C++ CLI is a profiling
tool only and its numbers never enter a comparison table.

Drive pattern (docs/DESIGN.md section 10 pump loop, adapted to the shared
asyncio client of ``bench.adapters``):

  - Every ``redline.Engine`` call (constructor aside, which is thread-safe to
    make anywhere) is serialized onto ONE dedicated executor thread - the
    engine is designed for a single driver thread, and ``step()`` /
    ``add_request()`` release the GIL, so the event loop keeps running while
    the engine works.
  - A pump task loops ``step()`` and routes each emitted
    ``(req_id, token, finished, finish_reason)`` to that request's asyncio
    queue; the per-request client task (shared ``drive_workload`` structure)
    records ``time.monotonic()`` when it receives the token - symmetric with
    the vLLM streaming and llama.cpp SSE consumers.
  - Submission honors ``has_capacity()`` backpressure (waiting-queue cap;
    KV admission happens inside ``step()``).
  - A trial that fails inside the adapter POISONS it: the engine may still
    hold in-flight requests (docs/DESIGN.md section 10 exposes no abort
    API), and their ids would surface in the next trial's pump as unknown,
    corrupting its routing. ``run()`` therefore refuses after a failed trial
    until ``load()`` rebuilds the engine. The shipped runner exits the
    process on any trial failure, so this guard protects future retry
    loops, not the normal path.

The module is importable without the built ``redline`` extension; the import
happens inside ``load()`` (build it first: ``PYTHONPATH=$BUILD``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
from typing import Any

from bench.adapters import EngineAdapter, RequestTrace, drive_workload
from bench.workload import Workload, WorkloadRequest

# Engine construction defaults = the dev preset (docs/DESIGN.md sections
# 10-11): the smallest supported GPU must be safe by default. The harness
# always passes its full configuration explicitly and records it in the
# results JSON (bench/FAIRNESS.md "Engine configuration").
DEFAULT_CONFIG = {
    "kv_pool_gb": 1.0,
    "max_batch": 8,
    "enable_cuda_graphs": True,
    "max_seq_len": 2048,
    "prefill_chunk": 1024,
    "admission_policy": "reserve_full",
}

# Type coercions for the known section-10 constructor keys. Every key of the
# merged config - known or not - is forwarded to the redline.Engine
# constructor, so the config recorded by version_info() is exactly what the
# engine was built with: an unsupported key raises TypeError from pybind11
# immediately instead of being recorded in the results JSON but silently
# ignored.
_CTOR_COERCIONS = {
    "kv_pool_gb": float,
    "max_batch": int,
    "enable_cuda_graphs": bool,
    "max_seq_len": int,
    "prefill_chunk": int,
    "admission_policy": str,
}

# Sentinel pushed into every per-request queue if the pump dies, so no client
# task can wait forever on a queue that will never be fed.
_PUMP_FAILED = object()


class RedlineAdapter(EngineAdapter):
    name = "redline"

    def __init__(self) -> None:
        self._engine: Any = None
        self._module: Any = None
        self._config: dict = {}
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Per-trial state (reset in _run_async).
        self._queues: dict[int, asyncio.Queue] = {}
        self._engine_to_request: dict[int, int] = {}
        self._total = 0
        self._submitted = 0
        self._finished = 0
        self._pump_error: BaseException | None = None
        # True after a trial failed with requests possibly still in flight
        # inside the engine; run() then refuses until load() rebuilds.
        self._poisoned = False

    # -- lifecycle ----------------------------------------------------------

    def load(self, model_dir: str, config: dict) -> None:
        import redline  # built pybind module; PYTHONPATH must include $BUILD

        if self._executor is not None:
            self.shutdown()  # rebuild path (e.g. after a poisoned trial)
        self._module = redline
        self._config = dict(DEFAULT_CONFIG)
        self._config.update(config or {})
        self._poisoned = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="redline-engine"
        )
        # Every merged config key is forwarded (typed coercion for the known
        # section-10 keys, verbatim otherwise) so recorded config == applied
        # config; pybind11 rejects a genuinely unknown kwarg loudly.
        kwargs = {
            key: (_CTOR_COERCIONS[key](value) if key in _CTOR_COERCIONS else value)
            for key, value in self._config.items()
        }
        # Construct on the engine thread as well, so init and every later
        # call share one thread.
        self._engine = self._executor.submit(
            redline.Engine, model_dir, **kwargs
        ).result()

    def shutdown(self) -> None:
        """Drop the engine (its destructor frees all device memory). The
        harness verifies GPU idleness between engines externally."""
        self._engine = None
        gc.collect()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def version_info(self) -> dict:
        info = {"engine": self.name, "config": dict(self._config)}
        if self._module is not None:
            info["module_file"] = getattr(self._module, "__file__", None)
            info["module_version"] = getattr(self._module, "__version__", None)
        return info

    # -- engine-thread helpers ------------------------------------------------

    async def _engine_call(self, fn, *args):
        """Run one engine call on the dedicated engine thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def engine_stats(self) -> dict | None:
        """`Engine.stats()` counter dump (docs/DESIGN.md section 10), recorded
        into the results JSON. Callable from sync code between trials."""
        if self._engine is None or self._executor is None:
            return None
        return self._executor.submit(self._engine.stats).result()

    def algo_report(self):
        """cuBLASLt algo-selection dump for the results JSON, if the engine
        exposes one (docs/DESIGN.md section 12d records Lt algo IDs to keep
        shape-equality arguments auditable). Returns None until an engine-side
        report hook exists."""
        if self._engine is None or self._executor is None:
            return None
        for attr in ("algo_report", "lt_algo_report", "algos"):
            probe = getattr(self._engine, attr, None)
            if callable(probe):
                try:
                    return self._executor.submit(probe).result()
                except Exception:
                    return None
        return None

    # -- trial body ----------------------------------------------------------

    async def _run_async(self, workload: Workload) -> list[RequestTrace]:
        if self._engine is None:
            raise RuntimeError("RedlineAdapter.load() must be called before run()")
        if self._poisoned:
            raise RuntimeError(
                "a previous trial failed with requests possibly still in "
                "flight inside the engine (no abort API, docs/DESIGN.md "
                "section 10); call load() again to rebuild the engine before "
                "the next trial"
            )
        self._queues = {r.request_id: asyncio.Queue() for r in workload.requests}
        self._engine_to_request = {}
        self._total = len(workload.requests)
        self._submitted = 0
        self._finished = 0
        self._pump_error = None

        pump = asyncio.create_task(self._pump())
        try:
            traces = await drive_workload(workload, self._stream_one)
            await pump  # normal exit: pump stops once every request finished
        except BaseException:
            # In-flight requests cannot be aborted; their ids would reach the
            # next trial's pump as unknown. Poison the adapter (see module
            # docstring) and let load() rebuild.
            self._poisoned = True
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):
                pass
            raise
        return traces

    async def _pump(self) -> None:
        """step() loop: route emitted tokens to per-request queues.

        Runs until every request in this trial has finished. When the engine
        is idle (nothing submitted yet, or between staggered arrivals) it
        naps briefly instead of spinning; while requests are in flight it
        calls step() back to back - an empty result then just means a prefill
        chunk that emitted no token yet.
        """
        try:
            while self._finished < self._total:
                results = await self._engine_call(self._engine.step)
                if results:
                    for engine_id, token, finished, reason in results:
                        request_id = self._engine_to_request.get(engine_id)
                        if request_id is None:
                            raise RuntimeError(
                                f"engine returned unknown request id {engine_id}"
                            )
                        self._queues[request_id].put_nowait((token, finished, reason))
                        if finished:
                            self._finished += 1
                elif self._submitted == self._finished:
                    await asyncio.sleep(0.001)
        except BaseException as error:
            self._pump_error = error
            for queue in self._queues.values():
                queue.put_nowait(_PUMP_FAILED)
            raise

    async def _stream_one(self, request: WorkloadRequest, trace: RequestTrace) -> None:
        # Backpressure per the section-10 pump pattern: submit while
        # has_capacity(). The waiting-queue cap (1024) is far above any case
        # here, so this loop is expected to pass immediately.
        while not await self._engine_call(self._engine.has_capacity):
            await asyncio.sleep(0.001)
        engine_id = await self._engine_call(
            self._engine.add_request,
            list(request.prompt_token_ids),
            request.max_new_tokens,
            True,  # ignore_eos: forced output lengths (bench/FAIRNESS.md "Cases")
        )
        # Safe against the pump seeing this id first: add_request and step()
        # are serialized on one engine thread, and their loop-side completions
        # are delivered in submission order.
        self._engine_to_request[engine_id] = request.request_id
        self._submitted += 1

        queue = self._queues[request.request_id]
        while True:
            item = await queue.get()
            if item is _PUMP_FAILED:
                raise RuntimeError("redline pump loop failed") from self._pump_error
            token, finished, reason = item
            if reason != "aborted":
                trace.record(token)
            if finished:
                trace.finish_reason = reason
                return
