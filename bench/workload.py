"""Fixed-seed synthetic workloads.

Every engine adapter receives byte-identical prompts as raw token IDs, keeping
tokenizer differences out of the measured path entirely. The token-id
generator is the normative splitmix64 algorithm of bench/FAIRNESS.md
("Workload generator") and is mirrored bit-for-bit by src/bench_main.cpp -
which is why it is hand-rolled here instead of using numpy (numpy and C++
stdlib RNGs can never produce identical streams).

Cases (bench/FAIRNESS.md "Cases"):
    primary   64 concurrent requests, 1024 in / 256 out, all at t=0
    arrival   64 requests, 1024 in / 256 out, request i submitted at i*100 ms
    longout   64 concurrent requests, 1024 in / 1024 out, all at t=0
    batch1    a single request, 1024 in / 256 out (latency)

Selftest (cross-language identity check)::

    python -m bench.workload --selftest

prints the first 8 token IDs - (seed=0, req in {0,1}, pos in {0..3}), one per
line, req-major - in the normative line format ``seed=S req=R pos=P id=I``.
``redline_bench --selftest-workload`` prints the byte-identical lines, so
``diff`` between the two outputs proves the C++ and Python generators match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

QWEN25_VOCAB_SIZE = 151_936

# Uniform over [1000, 100000): ordinary BPE tokens only, far below the
# special-token range (>= 151643; docs/MODEL_SPEC.md section 6).
TOKEN_ID_LOW = 1000
TOKEN_ID_RANGE = 99_000

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """splitmix64 mixer - bit-identical to src/bench_main.cpp::SplitMix64."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def token_id(seed: int, request_id: int, position: int) -> int:
    """Normative token-id generator (bench/FAIRNESS.md)."""
    x = ((seed << 40) ^ (request_id << 20) ^ position) & _MASK64
    return TOKEN_ID_LOW + _splitmix64(x) % TOKEN_ID_RANGE


@dataclass(frozen=True)
class WorkloadRequest:
    """One request: prompt token IDs, forced generation length, arrival time.

    Forced length means EOS is ignored by every engine (Redline `ignore_eos`,
    vLLM `ignore_eos=True`, llama.cpp fixed `n_predict`); the harness asserts
    exactly `max_new_tokens` tokens came back.
    """

    request_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_time_s: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    requests: list[WorkloadRequest]


def synthetic_workload(
    name: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    *,
    seed: int = 0,
    arrival_stagger_s: float = 0.0,
) -> Workload:
    """Deterministic random-token workload derived from ``seed`` (default 0,
    the single seed used for all published runs)."""
    requests = [
        WorkloadRequest(
            request_id=r,
            prompt_token_ids=[token_id(seed, r, p) for p in range(input_len)],
            max_new_tokens=output_len,
            arrival_time_s=r * arrival_stagger_s,
        )
        for r in range(num_requests)
    ]
    return Workload(name=name, requests=requests)


def primary_case(seed: int = 0) -> Workload:
    """Headline throughput: 64 concurrent requests, 1024/256, closed wave."""
    return synthetic_workload("primary", 64, 1024, 256, seed=seed)


def arrival_case(seed: int = 0) -> Workload:
    """Prefill/decode interference: request i arrives at t = i * 100 ms, so
    prefills land mid-decode instead of being front-loaded - the regime a
    dedicated-prefill scheduler finds hardest (measured, not hidden)."""
    return synthetic_workload("arrival", 64, 1024, 256, seed=seed, arrival_stagger_s=0.100)


def longout_case(seed: int = 0) -> Workload:
    """Wider decode window (~4x the primary case's measured decode span)."""
    return synthetic_workload("longout", 64, 1024, 1024, seed=seed)


def batch1_case(seed: int = 0) -> Workload:
    """Latency case: one request, 1024/256 - same shape as primary so it
    isolates concurrency, not prompt length."""
    return synthetic_workload("batch1", 1, 1024, 256, seed=seed)


def equivalence_workload(seed: int = 0) -> Workload:
    """The 5 short greedy prompts of the cross-engine output-equivalence gate
    (bench/FAIRNESS.md "Model and inputs"): 5 requests, 64 tokens in / 32 out,
    all at t=0. Not a measured case - token streams from all engines are
    compared before any measured trial."""
    return synthetic_workload("equivalence", 5, 64, 32, seed=seed)


CASES = {
    "primary": primary_case,
    "arrival": arrival_case,
    "longout": longout_case,
    "batch1": batch1_case,
}

# --- selftest -------------------------------------------------------------

# The (req, pos) grid printed by --selftest; redline_bench --selftest-workload
# prints the identical grid in the identical line format.
SELFTEST_REQUESTS = (0, 1)
SELFTEST_POSITIONS = (0, 1, 2, 3)


def selftest_lines(seed: int = 0) -> list[str]:
    """Normative selftest output, one line per (req, pos), req-major.

    >>> selftest_lines()[0]
    'seed=0 req=0 pos=0 id=40535'
    >>> len(selftest_lines())
    8
    """
    return [
        f"seed={seed} req={r} pos={p} id={token_id(seed, r, p)}"
        for r in SELFTEST_REQUESTS
        for p in SELFTEST_POSITIONS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.workload",
        description="Fixed-seed synthetic workload generator (see module docstring).",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="print the first 8 token IDs (seed, req in {0,1}, pos in {0..3}) "
        "for comparison against `redline_bench --selftest-workload`",
    )
    parser.add_argument("--seed", type=int, default=0, help="workload seed (default 0)")
    args = parser.parse_args(argv)
    if not args.selftest:
        parser.error("nothing to do: pass --selftest")
    for line in selftest_lines(args.seed):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
