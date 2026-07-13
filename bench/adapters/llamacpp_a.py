"""Adapter for llama.cpp (baseline engine, driven over llama-server HTTP/SSE).

Pinning and settings (normative rules in bench/FAIRNESS.md):
  - llama.cpp pinned to a documented commit hash, recorded in the results JSON
    (``version_info`` snapshots the server's /props payload)
  - model converted with convert_hf_to_gguf.py --outtype f16 (FP16 GGUF, no
    quantization); exact conversion command + GGUF hash documented with
    results; the cross-engine output-equivalence gate is what proves the
    conversion did not silently change the model
  - llama-server is launched EXTERNALLY (the recorded launch line lives in
    bench/FAIRNESS.md "Engine configuration"):
    -ngl 99 (full offload), flash attention on, --parallel 64
    --cont-batching, and the MANDATORY context arithmetic - the server splits
    -c across parallel slots, so -c >= 64 * (1024 + 256) = 81920 for
    primary/arrival and -c >= 64 * 2048 = 131072 for longout; defaults would
    truncate every prompt and invalidate the run. ``run()`` re-checks the
    per-slot context against the workload and refuses to measure when the
    server reports one that cannot fit prompt + output; it likewise refuses
    when /props reports fewer parallel slots than the workload's requests
    (the excess would queue server-side - a silently handicapped baseline).
  - -b/-ub (logical/physical batch) chosen by the bounded, documented tuning
    sweep; best published, losing configs committed with raw results
  - greedy sampling as temperature 0 PLUS top_k 1 - llama.cpp's cheap-greedy
    idiom. temperature=0 alone still runs the default filter chain
    (top-k 40 -> top-p -> min-p) over the 151936-entry candidate array per
    token per slot on the CPU; the other engines' greedy paths pay no such
    filter cost, so the default chain would be a small hidden handicap.
    top_k=1 was proven token-stream-identical to the default chain on the
    full primary workload (16384 tokens, stream and non-stream;
    bench/results/tuning/llamacpp_sampler_chain/diag_decomposition.json).
    Measured effect is small: ~2% on the non-stream path, within run-to-run
    variance on the streaming path (record in bench/FAIRNESS.md)
  - fixed n_predict per request plus ignore_eos so EOS cannot shorten a
    request (forced lengths; the harness asserts exact output token counts)
  - prompts as token arrays via the server API, VALIDATED at load() by a
    1-token probe whose reported prompt token count must equal the array
    length. If that path ever fails, the ONLY permitted fallback preserves
    token identity: detokenize + re-encode with a hard assert that the
    re-encoded ID sequence equals the original. Text fallback without that
    assert is prohibited - this adapter implements no text fallback at all.
  - prompt caching disabled per request (``cache_prompt: false``): trials
    repeat identical prompts, and a warm prompt cache would skip prefill on
    later trials, measuring cache hits instead of inference
  - HTTP/SSE client overhead bounded via a llama-batched-bench cross-check,
    reported next to the results (FAIRNESS.md "Measured path")

The streaming client needs ``aiohttp`` (imported lazily inside the trial so
the rest of the harness works without it); load()-time probes use urllib.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from bench.adapters import EngineAdapter, RequestTrace, drive_workload
from bench.workload import Workload, WorkloadRequest, token_id

DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8080",
    "connect_timeout_s": 30.0,
    # No total timeout on streaming reads: a longout trial legitimately runs
    # for minutes. Per-read inactivity is bounded instead.
    "read_timeout_s": 600.0,
}


def _http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    """Small sync JSON-over-HTTP helper (load()-time probes only)."""
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class LlamaCppAdapter(EngineAdapter):
    name = "llamacpp"

    def __init__(self) -> None:
        self._config: dict = {}
        self._server_url: str = DEFAULT_CONFIG["server_url"]
        self._props: dict | None = None
        self._session: Any = None  # aiohttp.ClientSession, per-trial (loop-bound)

    # -- lifecycle ----------------------------------------------------------

    def load(self, model_dir: str, config: dict) -> None:
        """Verify the externally launched llama-server is up, snapshot its
        /props (build info, slot count, per-slot context), and validate the
        token-array prompt path. ``model_dir`` identifies the checkpoint for
        the record; the GGUF the server actually serves is pinned in
        bench/FAIRNESS.md."""
        del model_dir  # server already holds the (converted) model
        self._config = dict(DEFAULT_CONFIG)
        self._config.update(config or {})
        self._server_url = str(self._config["server_url"]).rstrip("/")

        deadline = time.monotonic() + float(self._config["connect_timeout_s"])
        last_error: Exception | None = None
        while True:
            try:
                health = _http_json(f"{self._server_url}/health")
                if health.get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError) as error:
                last_error = error
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"llama-server not reachable at {self._server_url} "
                    f"(launch it first; see bench/FAIRNESS.md engine table)"
                ) from last_error
            time.sleep(0.5)

        try:
            self._props = _http_json(f"{self._server_url}/props")
        except (urllib.error.URLError, OSError, ValueError):
            self._props = None  # older servers; recorded as unavailable

        self._validate_token_prompt_path()

    def shutdown(self) -> None:
        """The server is external and owns all GPU state; the client holds
        nothing between trials (sessions are per-trial)."""

    def version_info(self) -> dict:
        info = {"engine": self.name, "server_url": self._server_url}
        if self._props is not None:
            for key in ("build_info", "version", "total_slots", "model_path"):
                if key in self._props:
                    info[key] = self._props[key]
            info["props"] = self._props
        return info

    # -- fairness guards ------------------------------------------------------

    def _validate_token_prompt_path(self) -> None:
        """FAIRNESS.md "Model and inputs": token arrays must reach the model
        with exact token counts. Send a tiny completion with a known token
        array and require the server's reported prompt token count to equal
        the array length. Hard error otherwise - no silent text fallback."""
        probe = [token_id(0, 0, position) for position in range(8)]
        response = _http_json(
            f"{self._server_url}/completion",
            {
                "prompt": probe,
                "n_predict": 1,
                "temperature": 0.0,
                "top_k": 1,
                "stream": False,
                "cache_prompt": False,
                "return_tokens": True,
            },
            timeout=120.0,
        )
        reported = None
        timings = response.get("timings")
        if isinstance(timings, dict) and "prompt_n" in timings:
            reported = timings["prompt_n"]
        elif "tokens_evaluated" in response:
            reported = response["tokens_evaluated"]
        if reported != len(probe):
            raise RuntimeError(
                "llama-server token-array prompt path failed validation: sent "
                f"{len(probe)} token IDs, server reports {reported!r} prompt "
                "tokens. The only permitted fallback must preserve the exact "
                "token sequence (bench/FAIRNESS.md); refusing to measure."
            )

    def _slot_context(self) -> int | None:
        if not isinstance(self._props, dict):
            return None
        settings = self._props.get("default_generation_settings")
        if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
            return settings["n_ctx"]
        return None

    def _check_context_fits(self, workload: Workload) -> None:
        """Mandatory context arithmetic (FAIRNESS.md engine table): the
        server splits -c across --parallel slots; every slot must fit
        prompt + forced output."""
        slot_ctx = self._slot_context()
        if slot_ctx is None:
            return  # server did not report it; the launch command is pinned in FAIRNESS.md
        need = max(len(r.prompt_token_ids) + r.max_new_tokens for r in workload.requests)
        if slot_ctx < need:
            raise RuntimeError(
                f"llama-server per-slot context {slot_ctx} < {need} required by "
                f"workload '{workload.name}' (prompt + forced output). Relaunch "
                "with -c >= slots * (input + output); see bench/FAIRNESS.md."
            )

    def _check_parallel_slots(self, workload: Workload) -> None:
        """A server with fewer parallel slots than the workload's requests
        queues the excess server-side - a silently handicapped baseline
        (bench/FAIRNESS.md pins ``--parallel 64`` for the measured cases).
        Skipped when /props did not report a slot count (the launch command
        is pinned in bench/FAIRNESS.md either way)."""
        slots = self._props.get("total_slots") if isinstance(self._props, dict) else None
        if isinstance(slots, int) and 0 < slots < len(workload.requests):
            raise RuntimeError(
                f"llama-server reports {slots} parallel slots but workload "
                f"'{workload.name}' has {len(workload.requests)} requests; "
                "relaunch with --parallel >= the request count "
                "(bench/FAIRNESS.md 'Engine configuration')."
            )

    # -- trial body -------------------------------------------------------------

    async def _run_async(self, workload: Workload) -> list[RequestTrace]:
        try:
            import aiohttp
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "the llama.cpp adapter needs aiohttp for HTTP/SSE streaming "
                "(pip install aiohttp)"
            ) from error

        self._check_context_fits(workload)
        self._check_parallel_slots(workload)
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=float(self._config["connect_timeout_s"]),
            sock_read=float(self._config["read_timeout_s"]),
        )
        connector = aiohttp.TCPConnector(limit=0)  # one socket per concurrent request
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            self._session = session
            try:
                return await drive_workload(workload, self._stream_one)
            finally:
                self._session = None

    async def _stream_one(self, request: WorkloadRequest, trace: RequestTrace) -> None:
        payload = {
            "prompt": list(request.prompt_token_ids),  # token array, never text
            "n_predict": request.max_new_tokens,  # fixed forced length
            "temperature": 0.0,  # greedy...
            "top_k": 1,  # ...via the cheap argmax chain (module docstring;
            # token-stream-identical to the default chain, measured)
            "ignore_eos": True,  # EOS must not shorten the forced length
            "cache_prompt": False,  # no cross-trial prompt-cache advantage
            "return_tokens": True,  # token IDs in every streamed chunk
            "stream": True,
        }
        async with self._session.post(
            f"{self._server_url}/completion", json=payload
        ) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue  # SSE keep-alives / blank separators
                now = time.monotonic()
                chunk = json.loads(line[len("data:") :].strip())
                tokens = chunk.get("tokens")
                if tokens:
                    for token in tokens:
                        trace.record(int(token), now)
                elif chunk.get("content") and not chunk.get("stop"):
                    raise RuntimeError(
                        "llama-server streamed content without token IDs; "
                        "return_tokens appears unsupported by this build - "
                        "token identity cannot be verified, refusing to measure"
                    )
                if chunk.get("stop"):
                    # Current llama-server builds report stop_type
                    # ("limit"/"eos"/"word"/"none"); older builds the
                    # stopped_limit/stopped_eos booleans. Map both; nothing
                    # gates on finish_reason (the forced-length assert is
                    # count-based) but the record should be faithful.
                    stop_type = chunk.get("stop_type")
                    if stop_type == "limit" or chunk.get("stopped_limit"):
                        trace.finish_reason = "length"
                    elif stop_type == "eos" or chunk.get("stopped_eos"):
                        trace.finish_reason = "eos"
                    elif isinstance(stop_type, str) and stop_type:
                        trace.finish_reason = stop_type
                    else:
                        trace.finish_reason = "stop"
                    return
        raise RuntimeError(
            f"llama-server stream for request {request.request_id} ended "
            "without a stop chunk"
        )
