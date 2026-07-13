"""Engine adapters: one class per engine, one shared interface.

Each adapter drives its engine over a ``bench.workload.Workload`` and returns
raw per-token wall-clock timestamps. All metric math happens engine-
agnostically in bench/run_bench.py so no adapter can flatter its engine.

Shared client structure (bench/FAIRNESS.md "Measured path"): every adapter
runs the same asyncio driver - one task per request that (1) sleeps until the
request's ``arrival_time_s``, (2) records the submit timestamp, (3) awaits an
engine-specific per-token stream, recording one ``time.monotonic()`` per
generated token *at the moment the client receives it*. Redline consumes an
in-process pump loop, vLLM its V1 async streaming API, llama.cpp an HTTP/SSE
stream - but the measured client path around them is identical code
(``drive_workload``), so client overhead is shared structure rather than a
per-engine advantage.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field

from bench.workload import Workload, WorkloadRequest

ENGINE_NAMES = ("redline", "vllm", "vllm024", "llamacpp")

# The two vLLM baseline arms (bench/FAIRNESS.md "Engine configuration", pin
# disclosure): identical adapter code path, distinct engine names. The arm is
# selected by the virtual environment of the interpreter invoking
# bench.run_bench - "vllm" is the pinned vllm==0.9.2 venv
# (bench/requirements.lock), "vllm024" the vllm==0.24.0 venv
# (bench/requirements-vllm024.lock) - and the adapter re-records the venv's
# installed vllm version into every results JSON.
VLLM_ENGINE_NAMES = ("vllm", "vllm024")


@dataclass
class RequestTrace:
    """Per-request measurement: monotonic timestamps, one per generated token.

    ``enqueue_time`` is the client-side submit intent (taken immediately
    before the engine-specific submit call); ``token_times[k]`` is when the
    client received generated token k; ``token_ids[k]`` is that token. The
    two lists are index-parallel. All timestamps are ``time.monotonic()``.
    """

    request_id: int
    enqueue_time: float
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""

    def record(self, token_id: int, now: float | None = None) -> None:
        """Append one generated token with its client receive timestamp."""
        self.token_times.append(time.monotonic() if now is None else now)
        self.token_ids.append(token_id)

    @property
    def num_tokens(self) -> int:
        return len(self.token_times)


async def drive_workload(
    workload: Workload,
    stream_one,
) -> list[RequestTrace]:
    """The shared asyncio client: one task per request.

    ``stream_one(request, trace)`` is the engine-specific coroutine; it must
    submit the request and call ``trace.record(...)`` once per generated
    token as it arrives, setting ``trace.finish_reason`` at the end. Arrival
    offsets are honored against a common epoch taken at call time.
    """

    epoch = time.monotonic()

    async def _one(request: WorkloadRequest) -> RequestTrace:
        delay = epoch + request.arrival_time_s - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        trace = RequestTrace(request_id=request.request_id, enqueue_time=time.monotonic())
        await stream_one(request, trace)
        return trace

    tasks = [asyncio.create_task(_one(r)) for r in workload.requests]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        raise


class EngineAdapter(abc.ABC):
    """Common driver interface. Engines load and shut down once per bench run."""

    name: str = "base"

    @abc.abstractmethod
    def load(self, model_dir: str, config: dict) -> None:
        """Start the engine: weights on GPU, ready to serve."""

    @abc.abstractmethod
    async def _stream_one(self, request: WorkloadRequest, trace: RequestTrace) -> None:
        """Submit one request and record its per-token timestamps."""

    async def _run_async(self, workload: Workload) -> list[RequestTrace]:
        """Default trial body: just the shared driver. Adapters that need a
        background task (redline's pump) or per-trial resources (llama.cpp's
        HTTP session) override this and still call ``drive_workload``."""
        return await drive_workload(workload, self._stream_one)

    def run(self, workload: Workload) -> list[RequestTrace]:
        """Run every request to completion (greedy decoding, forced lengths)."""
        return asyncio.run(self._run_async(workload))

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release the GPU entirely (verified idle between engines)."""

    def version_info(self) -> dict:
        """Engine version/commit pins recorded into the results JSON."""
        return {"engine": self.name}


def resolve_adapter(name: str) -> EngineAdapter:
    """Instantiate an adapter by engine name.

    Imports are lazy so that resolving one engine never requires the other
    engines' libraries to be installed.
    """
    if name == "redline":
        from bench.adapters.redline_a import RedlineAdapter

        return RedlineAdapter()
    if name == "vllm":
        from bench.adapters.vllm_a import VllmAdapter

        return VllmAdapter()
    if name == "vllm024":
        from bench.adapters.vllm_a import Vllm024Adapter

        return Vllm024Adapter()
    if name == "llamacpp":
        from bench.adapters.llamacpp_a import LlamaCppAdapter

        return LlamaCppAdapter()
    raise ValueError(f"unknown engine '{name}' (expected one of {ENGINE_NAMES})")
