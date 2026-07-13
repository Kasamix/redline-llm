"""Suite (f): layer-0 parity fixture (docs/DESIGN.md §12f).

Compares the engine's layer-0 activations for ONE fixed prompt - captured by
the ``debug_dump_dir`` hook on the first prefill chunk - against the HF
capture produced by ``gen_layer0_fixture.py``, stage by stage and in
computation order:

  1. ``normed``    post-``input_layernorm``
  2. ``qkv``       fused QKV projection output (pre-RoPE, bias applied);
                   columns [q 0:1536 | k 1536:1792 | v 1792:2048]
  3. ``attn_out``  attention block output (post o-GEMM, pre residual add)
  4. ``mlp_out``   MLP block output (pre residual add)

The point of the ordering: the FIRST stage whose max-abs difference exceeds
its tolerance names the component that broke - a diverging ``qkv`` with a
clean ``normed`` is a fused-weight/bias packing bug (K/V row-range swap,
bias on the wrong axis), a diverging ``attn_out`` over a clean ``qkv`` is
RoPE/scatter/attention, and so on. This validates the fused layouts and the
cuBLASLt BIAS epilogue directly against HF numbers, not only against unit
references (§12f), and runs BEFORE any full-e2e debugging.

Both sides of the comparison are FP16 with FP32 accumulation but different
GEMM tilings, so stages match closely yet not bitwise. The gate is
per-element: ``|engine - ref| <= max(2e-2, one FP16 ULP at the reference
element)``. The 2e-2 absolute floor is the §12f start value and governs
everywhere below magnitude 32; the ULP floor is the FP16 representability
limit, needed because Qwen2.5-1.5B's K projection carries attention-sink
outliers up to |k| ~ 318 where one FP16 ULP is 0.25 - two correctly-rounded
implementations with different (equally valid) FP32 reduction orders can
land on adjacent FP16 values there, and no absolute 2e-2 gate is satisfiable
by ANY correct FP16 arithmetic at such magnitudes. Measured on the dev GPU (RTX
2060): engine q output bit-identical to HF; engine and HF k outputs each
exactly 0.1241 max-abs from an FP64 recomputation (equidistant - neither
side is "more correct"), disagreeing with each other by at most 1 ULP; a
layout/packing bug (§12f's target class) produces O(1)-to-O(100) errors on
small elements and cannot hide under either floor. Mean-abs is reported per
stage but does not gate.

Run (fixture generation is a SEPARATE OS process, §12c/§12f):
    python tests/e2e/gen_layer0_fixture.py --model $MODEL --out ~/redline_ref/layer0.npz
    PYTHONPATH=$BUILD pytest tests/e2e/test_layer0.py --model $MODEL \
        --ref ~/redline_ref/layer0.npz -v
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

np = pytest.importorskip("numpy", reason="suite (f) reads raw FP16 dumps via numpy")

# Computation order is load-bearing: the first diverging stage localizes the
# bug (module docstring).
STAGE_ORDER = ("normed", "qkv", "attn_out", "mlp_out")

# Per-stage ABSOLUTE tolerance floors (engine FP16 vs HF FP16), §12f start
# values. The effective per-element allowance is max(floor, 1 FP16 ULP of the
# reference element) - see the module docstring for the dev-GPU measurements
# motivating the ULP term (attention-sink K outliers at |k| ~ 318, where one
# FP16 ULP is 0.25 and an absolute 2e-2 cannot be met by correct arithmetic).
STAGE_TOLERANCES = {
    "normed": 2e-2,
    "qkv": 2e-2,
    "attn_out": 2e-2,
    "mlp_out": 2e-2,
}

# Cross-language contract with ModelExecutor::DumpLayer0Activations
# (src/core/model_executor.cpp): the descriptor names every data file, so only
# the descriptor's own name is hardcoded here.
DUMP_META_FILE = "layer0_meta.json"
DUMP_SCHEMA_VERSION = 1

# Engine ctor default for prefill_chunk (docs/DESIGN.md section 10); the dump
# covers the FIRST prefill chunk only, so the prompt must fit one chunk.
DEFAULT_PREFILL_CHUNK = 1024

_NPZ_MAGIC = b"PK"  # .npz is a zip archive


# ---------------------------------------------------------------------------
# Reference fixture (.npz from gen_layer0_fixture.py)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def layer0_ref(pytestconfig: pytest.Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads and validates the HF layer-0 fixture; (arrays, meta)."""
    path = pytestconfig.getoption("--ref")
    if not path:
        pytest.skip("--ref not provided (run gen_layer0_fixture.py first)")
    if not os.path.isfile(path):
        # The forgot-to-generate case: point at the generator instead of
        # surfacing a raw FileNotFoundError from open().
        pytest.skip(
            f"--ref {path!r} does not exist - generate the layer-0 fixture first: "
            f"python tests/e2e/gen_layer0_fixture.py --model $MODEL --out {path}"
        )
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != _NPZ_MAGIC:
        # Shared --ref plumbing: a whole-directory e2e run passes suite (c)'s
        # JSON reference; suite (f) needs its own .npz (see the module docstring).
        pytest.skip(
            f"--ref {path!r} is not a layer-0 .npz fixture (gen_layer0_fixture.py); "
            "run suite (f) with its own --ref"
        )

    data = np.load(path, allow_pickle=False)
    required = {"prompt_token_ids", "normed", "qkv", "attn_out", "mlp_out", "meta_json"}
    missing = required - set(data.files)
    if missing:
        raise pytest.UsageError(f"layer-0 fixture {path!r} is missing keys {sorted(missing)}")

    meta = json.loads(data["meta_json"].item())
    if meta.get("schema_version") != 1:
        raise pytest.UsageError(
            f"layer-0 fixture schema_version {meta.get('schema_version')!r} unsupported (want 1)"
        )
    if meta.get("qkv_rope_applied") is not False:
        raise pytest.UsageError(
            "layer-0 fixture qkv must be captured PRE-rotary (qkv_rope_applied: false); "
            "regenerate with gen_layer0_fixture.py"
        )

    arrays = {name: data[name] for name in required - {"meta_json"}}
    for name in STAGE_ORDER:
        if arrays[name].dtype != np.float16:
            raise pytest.UsageError(
                f"layer-0 fixture array {name!r} has dtype {arrays[name].dtype}, want float16"
            )
    return arrays, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_dump(dump_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads the engine dump directory; (arrays keyed like meta['files'], meta)."""
    meta_path = os.path.join(dump_dir, DUMP_META_FILE)
    if not os.path.isfile(meta_path):
        pytest.fail(
            f"engine wrote no layer-0 dump ({meta_path} missing): the debug_dump_dir hook "
            "did not run - check that the built module's ctor forwards debug_dump_dir into "
            "EngineOptions (src/api/pybind.cpp) and that a prefill chunk executed"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("schema_version") == DUMP_SCHEMA_VERSION, (
        f"dump schema_version {meta.get('schema_version')!r} != {DUMP_SCHEMA_VERSION}"
    )
    assert meta.get("byte_order") == "little"

    dtypes = {"float16": np.float16, "int32": np.int32}
    arrays: dict[str, Any] = {}
    for name, entry in meta["files"].items():
        path = os.path.join(dump_dir, entry["file"])
        assert os.path.isfile(path), f"dump meta names {entry['file']} but the file is absent"
        shape = tuple(entry["shape"])
        dtype = dtypes[entry["dtype"]]
        raw = np.fromfile(path, dtype=dtype)
        expected = int(np.prod(shape))
        assert raw.size == expected, (
            f"dump file {entry['file']} holds {raw.size} {entry['dtype']} values, "
            f"expected {expected} for shape {shape}"
        )
        arrays[name] = raw.reshape(shape)
    return arrays, meta


def _stats(engine_arr: Any, ref_arr: Any) -> tuple[float, float]:
    diff = np.abs(engine_arr.astype(np.float64) - ref_arr.astype(np.float64))
    return float(diff.max()), float(diff.mean())


def _gate_ratio(engine_arr: Any, ref_arr: Any, tol_abs: float) -> float:
    """Worst per-element ratio |engine - ref| / max(tol_abs, ulp16(ref)).

    <= 1.0 passes. ``np.spacing`` on the FP16 reference is the local FP16
    ULP (subnormal-tiny near zero, where the absolute floor governs; 0.25 at
    the |k| ~ 318 attention-sink outliers, where the floor is unreachable by
    any correct FP16 rounding). NaN/inf on the engine side yields a NaN
    ratio, and ``NaN <= 1.0`` is False - non-finite dumps always fail.
    """
    diff = np.abs(engine_arr.astype(np.float64) - ref_arr.astype(np.float64))
    allowance = np.maximum(tol_abs, np.abs(np.spacing(ref_arr)).astype(np.float64))
    return float((diff / allowance).max())


# ---------------------------------------------------------------------------
# The suite-(f) test
# ---------------------------------------------------------------------------
def test_layer0_stages_match_hf(
    layer0_ref: tuple[dict[str, Any], dict[str, Any]],
    engine_kwargs: dict[str, Any],
    make_engine,
    tmp_path,
    report_sink: list,
) -> None:
    ref_arrays, ref_meta = layer0_ref
    prompt_ids = [int(t) for t in ref_arrays["prompt_token_ids"]]
    num_tokens = len(prompt_ids)

    # The dump covers the FIRST prefill chunk; a prompt longer than the chunk
    # would leave the fixture comparing a truncated extent.
    prefill_chunk = int(engine_kwargs.get("prefill_chunk", DEFAULT_PREFILL_CHUNK))
    assert num_tokens <= prefill_chunk, (
        f"fixture prompt has {num_tokens} tokens but prefill_chunk={prefill_chunk}: the layer-0 "
        "dump covers only the first chunk - shorten the prompt or raise prefill_chunk"
    )

    # -- run the engine with the dump hook armed ----------------------------
    dump_dir = tmp_path / "layer0_dump"
    try:
        engine = make_engine(debug_dump_dir=str(dump_dir))
    except TypeError as err:
        pytest.fail(
            f"redline.Engine does not accept debug_dump_dir ({err}): the built module predates "
            "the pybind debug-dump wiring (src/api/pybind.cpp) - rebuild it from the current tree"
        )
    except RuntimeError as err:
        # A module built from an older interim tree carries a placeholder
        # guard that rejects any non-empty debug_dump_dir. Match its sentinel
        # text narrowly: every other RuntimeError (e.g. the memory-preflight
        # budget) is a real engine failure and must propagate untouched.
        if "not wired up" not in str(err):
            raise
        pytest.fail(
            f"redline.Engine rejected debug_dump_dir ({err}): the built module still carries "
            "an interim placeholder guard - rebuild it from the current tree "
            "(src/api/pybind.cpp forwards debug_dump_dir into EngineOptions)"
        )
    request_id = engine.add_request(prompt_ids, 1)  # max_new=1: prefill + first token only
    finished = False
    for _ in range(32):
        for rid, _token, fin, _reason in engine.step():
            if rid == request_id and fin:
                finished = True
        if finished:
            break
    assert finished, "request did not finish within 32 steps (max_new_tokens=1)"

    dump_arrays, dump_meta = _load_dump(str(dump_dir))

    # -- the two captures must describe the same computation ----------------
    assert dump_meta["rows"] == num_tokens, (
        f"dump covered {dump_meta['rows']} rows, fixture prompt has {num_tokens} tokens"
    )
    assert dump_meta["chunk_start"] == 0, "dump did not capture the prompt's first chunk"
    assert dump_meta["ctx"] == num_tokens
    assert dump_meta["qkv_rope_applied"] is False, "engine qkv dump must be pre-RoPE"
    for key in ("hidden_size", "qkv_width", "num_q_heads", "num_kv_heads", "head_dim"):
        assert dump_meta[key] == ref_meta[key], (
            f"geometry mismatch on {key}: engine dump {dump_meta[key]} vs fixture {ref_meta[key]}"
        )
    assert np.array_equal(
        dump_arrays["input_ids"], np.asarray(prompt_ids, dtype=np.int32)
    ), "engine consumed different token ids than the fixture captured - comparison is void"

    # -- stage-by-stage comparison, computation order ------------------------
    hidden = int(dump_meta["hidden_size"])
    kv_width = int(dump_meta["num_kv_heads"]) * int(dump_meta["head_dim"])
    qkv_slices = {  # feature-column ranges inside the fused qkv buffer
        "q": (0, hidden),
        "k": (hidden, hidden + kv_width),
        "v": (hidden + kv_width, hidden + 2 * kv_width),
    }

    lines = [
        f"layer-0 parity vs HF (§12f): T={num_tokens}, "
        f"prefill_gemm_mode={dump_meta.get('prefill_gemm_mode')}, "
        f"hf_attn={ref_meta.get('attn_implementation')}, "
        f"hf_gpu={ref_meta.get('gpu_name')}",
        f"{'stage':<12} {'shape':<14} {'max_abs':>12} {'mean_abs':>12} {'tol':>10} "
        f"{'x-allow':>8}  status",
    ]
    first_divergent: str | None = None
    stage_stats: dict[str, tuple[float, float, float]] = {}

    for stage in STAGE_ORDER:
        engine_arr = dump_arrays[stage]
        ref_arr = ref_arrays[stage]
        assert engine_arr.shape == tuple(ref_arr.shape), (
            f"stage {stage}: engine dump shape {engine_arr.shape} != fixture {ref_arr.shape}"
        )
        max_abs, mean_abs = _stats(engine_arr, ref_arr)
        tol = STAGE_TOLERANCES[stage]
        # Gate: worst |diff| relative to the per-element allowance
        # max(tol, 1 FP16 ULP of the reference element); <= 1.0 passes.
        excess = _gate_ratio(engine_arr, ref_arr, tol)
        stage_stats[stage] = (max_abs, mean_abs, excess)
        ok = excess <= 1.0
        if not ok and first_divergent is None:
            first_divergent = stage
        lines.append(
            f"{stage:<12} {str(engine_arr.shape):<14} {max_abs:>12.5e} {mean_abs:>12.5e} "
            f"{tol:>10.1e} {excess:>8.4f}  {'ok' if ok else 'FAIL'}"
        )

        if stage == "qkv":
            # Informational per-slice localization: a clean q with dirty k/v
            # is a K/V packing problem, not a GEMM/norm problem.
            for slice_name, (lo, hi) in qkv_slices.items():
                s_max, s_mean = _stats(engine_arr[:, lo:hi], ref_arr[:, lo:hi])
                lines.append(
                    f"  {slice_name + ' slice':<10} {str((engine_arr.shape[0], hi - lo)):<14} "
                    f"{s_max:>12.5e} {s_mean:>12.5e} {'':>10}  (informational)"
                )
            # Direct K/V-swap probe (§12f's marquee failure mode): if the
            # straight slices diverge but the crossed slices agree, say so.
            k_lo, k_hi = qkv_slices["k"]
            v_lo, v_hi = qkv_slices["v"]
            k_max, _ = _stats(engine_arr[:, k_lo:k_hi], ref_arr[:, k_lo:k_hi])
            v_max, _ = _stats(engine_arr[:, v_lo:v_hi], ref_arr[:, v_lo:v_hi])
            if k_max > tol and v_max > tol:
                kx_max, _ = _stats(engine_arr[:, k_lo:k_hi], ref_arr[:, v_lo:v_hi])
                vx_max, _ = _stats(engine_arr[:, v_lo:v_hi], ref_arr[:, k_lo:k_hi])
                if kx_max <= tol and vx_max <= tol:
                    lines.append(
                        "  >> engine K matches HF V and vice versa: the K/V row ranges inside "
                        "w_qkv/b_qkv appear SWAPPED (docs/DESIGN.md section 12f, section 4 "
                        "fusion table)"
                    )

    report = "\n".join(lines)
    report_sink.append(report)
    print(report)  # visible with -s / on failure even without the summary hook

    if first_divergent is not None:
        max_abs, mean_abs, excess = stage_stats[first_divergent]
        pytest.fail(
            f"first diverging stage: {first_divergent!r} "
            f"(worst diff = {excess:.4f}x its allowance max(tol "
            f"{STAGE_TOLERANCES[first_divergent]:.1e}, 1 FP16 ULP of ref); "
            f"max_abs {max_abs:.5e}, mean_abs {mean_abs:.5e}); stages upstream of it are "
            f"clean, so debug there first.\n{report}"
        )
