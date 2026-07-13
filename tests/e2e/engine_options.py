"""Single source of truth for the e2e suites' option parsing.

One implementation, three consumers: ``tests/e2e/conftest.py`` exposes these
helpers as the shared ``model_dir``/``engine_kwargs`` fixtures, and the
suite-(d)/(e) modules (``test_invariance.py``, ``test_graph_equivalence.py``)
call them directly when building their dev-preset overlays. Historically each
consumer carried its own parser and the semantics drifted - e.g.
``enable_cuda_graphs=on`` coerced to ``True`` through the conftest table but
stayed the string ``"on"`` through the suite-local parsers (a pybind
``TypeError`` at the ctor), and ``pad_eager_to_bucket=8`` failed the
conftest's boolean coercion despite the kwarg being integer-valued
(src/core/engine.hpp). This module keeps the documented section-10
constructor semantics in exactly one place.

Coercion table = the docs/DESIGN.md section-10 constructor keys plus the two
debug kwargs. Unlisted keys are forwarded verbatim as strings so a genuinely
unknown key fails loudly at the pybind constructor instead of being silently
dropped.
"""

from __future__ import annotations

import os


def parse_bool(text: str) -> bool:
    """``1/true/yes/on`` -> True, ``0/false/no/off`` -> False (case-
    insensitive); anything else raises ValueError."""
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"cannot parse boolean from {text!r}")


def parse_pad_eager_to_bucket(text: str) -> int | bool:
    """``pad_eager_to_bucket`` is integer-valued (src/core/engine.hpp: 0 =
    off; value ``b`` pads every eager decode batch to the smallest configured
    bucket >= max(live batch, b); ``True == 1``), so ``"8"`` must parse as 8.
    Boolean spellings stay accepted for the documented ``True == 1``
    equivalence."""
    try:
        return int(text.strip())
    except ValueError:
        return parse_bool(text)


# Type coercions for the documented section-10 constructor keys plus the two
# debug kwargs (docs/DESIGN.md section 10).
CTOR_COERCIONS = {
    "kv_pool_gb": float,
    "max_batch": int,
    "enable_cuda_graphs": parse_bool,
    "max_seq_len": int,
    "prefill_chunk": int,
    "admission_policy": str,
    "pad_eager_to_bucket": parse_pad_eager_to_bucket,
    "debug_dump_dir": str,
}


def parse_engine_kwargs(spec: str | None) -> dict[str, object]:
    """Parse a comma-separated ``key=value`` overlay for the
    ``redline.Engine`` constructor (the ``--engine-kwargs`` option /
    ``REDLINE_ENGINE_KWARGS`` environment fallback). Known keys are coerced
    through ``CTOR_COERCIONS``; unknown keys pass through as raw strings.
    Raises ValueError naming the offending entry on malformed input."""
    if not spec:
        return {}
    out: dict[str, object] = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if not sep or not key:
            raise ValueError(f"--engine-kwargs entry {item!r} is not in key=value form")
        coerce = CTOR_COERCIONS.get(key)
        try:
            out[key] = coerce(value) if coerce is not None else value
        except ValueError as error:
            raise ValueError(f"--engine-kwargs entry {key}={value!r}: {error}") from error
    return out


def get_option(config, name: str, env_var: str, default=None):
    """Read a pytest CLI option with an environment-variable fallback,
    tolerating an unregistered option (so a suite copied out of this tree,
    away from the registering conftest, still works via the env vars)."""
    try:
        value = config.getoption(name, default=None)
    except ValueError:
        value = None
    if value in (None, ""):
        value = os.environ.get(env_var)
    return default if value in (None, "") else value
