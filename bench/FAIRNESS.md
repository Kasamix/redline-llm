# Benchmark fairness rules

The reviewer-facing contract for every published Redline-vs-baseline number. Results are
published only after they are measured under these rules on documented hardware - this file
ships **complete**, with no open policy questions, before any numbers do. A rule that is not
written here is not relied on, and a number that violates a rule here is not published.

## Scope of claims

- Results are specific to one model and regime: **Qwen2.5-1.5B-Instruct, FP16, single GPU,
  greedy decoding, the workloads below**. A 1.5B model at high batch is the regime where
  host-side per-step overhead is proportionally largest - the most favorable setting for a
  lean C++ engine - so no claim of general engine superiority is made or implied. The README
  results table carries this scope statement in its caption.
- Every published claim names its workload case. Numbers from the closed single wave are never
  quoted as if they covered arrival-driven serving; the `arrival` case exists precisely
  because a dedicated-prefill scheduler is weakest under mid-decode arrivals (Redline's
  documented tradeoff, DESIGN section 16) and that regime is measured rather than hidden.
- Redline's composite (cuBLAS) prefill is expected to lose TTFT to fused-prefill baselines -
  roughly 350-400 ms of serialized prefill across the 64-prompt wave (arithmetic in DESIGN
  section 6.3). This expectation is stated before measurement; the TTFT numbers are published
  regardless. (Measured outcome: the wave serializes ~1.55 s of prefill - the design estimate
  was ~4x optimistic because dense-GEMM time went uncounted; the correction is recorded in
  docs/PROFILING.md section 5.)

## Hardware and environment

- All engines run on the same machine, same GPU, same driver, back to back. **Recorded
  scope amendment (2026-07-10):** this holds within each benchmark session; the `vllm024`
  arm was measured in its own later session on a different rented pod instance of the same
  GPU model and driver line (RTX 4090; driver 580.159.03 vs 580.159.04 for the session-B
  rows - rented pods are not driver-identical across instances). The cross-session
  comparability of those rows is disclosed with a same-pod redline re-measurement rather
  than assumed - full record under "Engine configuration" (pin disclosure), and the README
  footnote names each driver with the rows measured under it.
- Captured into every results JSON: full `nvidia-smi -q` (GPU model, driver, CUDA version,
  clocks, power limits, temperature), `lscpu` and total RAM (client-side timestamps are
  CPU-sensitive on rented pods), kernel version, `nvcc`/`gcc` versions, `pip freeze`, engine
  versions/commits, and the exact command lines. (The redline git-commit capture failed on
  the benchmark pod and is recorded as-is in every results JSON - see the redline pin
  amendment under "Engine configuration".)
- GPU clocks are locked with `nvidia-smi -lgc` where the pod permits; otherwise opportunistic
  clocks are recorded and the results are labeled as unlocked-clock runs.
- **Recorded clock-lock status (benchmark session, RTX 4090, 2026-07-10):** the lock attempt
  `nvidia-smi -lgc 3120` (= `clocks.max.sm`) was refused by the rented pod - "The current
  user does not have permission to change clocks" (unprivileged container) - so the published
  results are **unlocked-clock runs**. Opportunistic clocks are captured in the full
  `nvidia-smi -q` embedded in every results JSON, and clock snapshots were taken immediately
  before and after the benchmark matrix (session-time observations; those standalone
  snapshots are **not committed as artifacts** - the committed, auditable clock evidence is
  each run's own start-of-run `nvidia-smi -q` capture inside its results JSON).
- **Recorded clock-lock status (0.24.0 arm's session, RTX 4090, 2026-07-10):** the same
  attempt (`nvidia-smi -lgc 3105` = that pod's `clocks.max.sm`) was refused on the 0.24.0
  arm's pod with the identical permission error, so the `vllm 0.24.0` rows are
  **unlocked-clock runs** as well; opportunistic clocks are in each results JSON, with
  snapshots again taken immediately before and after that session's matrix (same caveat:
  not committed standalone - the per-run in-JSON captures are the committed evidence).
- The GPU is verified idle between engines: no engine process survives its turn, and the
  committed captures carry the evidence - every measured run's results JSON opens with a
  start-of-run `nvidia-smi -q` showing 1 MiB used (the llamacpp rows show only the measured
  `llama-server`'s own weights+KV footprint), and the equivalence-gate transcripts record a
  `memory.used` reading before every engine turn. **Precision amendment (2026-07-10):** an
  earlier revision said "memory fully released"; inside the single gate process the
  committed readings show ~428 MiB persisting after the first in-process engine turn and
  after final shutdown (`gpu_memory_used_mib` in `transcript.json` /
  `transcript-vllm024.json`) - the gate driver's own CUDA context, which lives as long as
  its process. "Fully released" holds between measured runs (each is a fresh process), not
  within the gate process; the readings are committed as-is rather than the claim being
  left imprecise.
- One unmeasured **warmup run** per engine/case precedes the measured trials.

## Model and inputs

- Same checkpoint everywhere: Qwen2.5-1.5B-Instruct at the revision pinned in
  `docs/MODEL_SPEC.md`, FP16 weights. llama.cpp uses an FP16 GGUF conversion
  (`convert_hf_to_gguf.py --outtype f16`; the exact command and the GGUF file hash are
  documented with the results; no quantization).
- Prompts are delivered as **raw token IDs** to every engine; tokenization sits outside the
  measured path. llama.cpp receives token arrays through its API (verified before measuring).
  If that path fails validation, the **only permitted fallback** is one that provably
  preserves the token sequence - e.g. detokenize then re-encode with a **hard assert that the
  re-encoded ID sequence equals the original**. A text fallback without that assert is
  prohibited: re-tokenization can silently change prompt lengths and void the
  identical-inputs guarantee.
- **Output-equivalence gate:** before any measured trial, the same 5 short greedy prompts run
  through every engine being measured - one gate run covers the co-installed engines (redline,
  vllm, llamacpp); the vllm024 arm cannot share an interpreter with the vllm arm, so it gates
  separately under its own venv against the redline reference (per-arm protocol in the pin
  disclosure below). Token streams must match up to documented FP16 coin-flips (each
  divergence justified with top1-top2 logit margins, the same protocol as the engine's
  HF-parity suite, DESIGN section 12c). The transcript is committed next to the results JSON.
  This is the evidence that all engines compute the same function **on the gate workload**
  (5 prompts, 64 input / 32 output tokens, batch 5) - a silently broken GGUF conversion or a
  misconfigured baseline surfaces here as divergent streams, so "all engines served the same
  model" is never an unverified assertion. **Scope note (binding on how the gate may be
  quoted):** the gate does not execute the measured shape - 1024-token prompts at batch 64
  mean 64 KV blocks per request (vs 4 here) and 64-way batch numerics - so a divergence that
  only manifests past 64 output tokens or at long context would pass this gate. Cross-engine
  function identity at the measured shape is therefore *supported but not gated*: the
  llama.cpp token-array path is validated at 1024 tokens (server-reported `prompt_n`, pin
  record below) and redline's own HF teacher-forced parity suite (DESIGN section 12c) covers
  prompts up to ~500 tokens with 128-token continuations. A same-shape cross-engine gate was
  not run and would require a new pod session.
- **Recorded gate outcome (benchmark session, RTX 4090, 2026-07-10):** the gate ran before
  any measured trial and PASSED with zero divergences - both baselines produced token
  streams bit-identical to the redline reference on every comparison, across all 5 prompts
  and all 32 greedy positions of each, so the logit-margin protocol had nothing to justify.
  Transcript (including per-engine version pins and the one-engine-on-GPU exclusivity
  record for each engine turn): `bench/results/equivalence/transcript.json`.
- **Recorded gate outcome (0.24.0 arm's own session, RTX 4090, 2026-07-10):** the `vllm024`
  arm's own gate (`--engines redline,vllm024`, run under that arm's virtual environment on
  its session's pod) ran before any measured trial of the arm and PASSED with zero
  divergences - token streams bit-identical to the redline reference on all 5 prompts and
  all 32 greedy positions, so the logit-margin protocol again had nothing to justify.
  Transcript (per-engine version pins and the per-turn GPU-exclusivity record):
  `bench/results/equivalence/transcript-vllm024.json`.

## Workload generator (normative)

Token IDs come from splitmix64 so the C++ CLI (`src/bench_main.cpp`) and the Python harness
(`bench/workload.py`) produce **bit-identical** streams (numpy and C++ `<random>` never
would). All arithmetic is modulo 2^64:

```
splitmix64(x):
    x = x + 0x9E3779B97F4A7C15
    x = (x xor (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x xor (x >> 27)) * 0x94D049BB133111EB
    return x xor (x >> 31)

token_id(seed, request_id, position) =
    1000 + splitmix64((seed << 40) xor (request_id << 20) xor position) mod 99000
```

IDs are uniform over [1000, 100000) - ordinary BPE tokens only, far below the special-token
range (>= 151643, see MODEL_SPEC section 6). The seed is **0** for all published runs.

## Cases

| case      | requests | input / output | arrival                          |
| --------- | -------- | -------------- | -------------------------------- |
| `primary` | 64       | 1024 / 256     | all at t=0 (closed single wave)  |
| `arrival` | 64       | 1024 / 256     | request i at t = i * 100 ms      |
| `longout` | 64       | 1024 / 1024    | all at t=0                       |
| `batch1`  | 1        | 1024 / 256     | t=0                              |

- Greedy decoding everywhere (temperature/sampling disabled per engine idiom).
- **Forced output lengths - EOS ignored on every engine**: Redline `ignore_eos=True` (a
  first-class request flag), vLLM `ignore_eos=True`, llama.cpp fixed `n_predict`. The harness
  **asserts** that every request in every trace produced exactly its forced output length on
  every engine; a violation voids the trial. Without this, random-token prompts emitting an
  early EOS would silently shrink one engine's workload.

## Engine configuration

| engine    | version pinning                                                                     | execution path                                                                                     |
| --------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| redline   | module version recorded in results JSON; source-revision amendment below             | CUDA graphs on; kv pool / max batch / seq len as published                                          |
| vLLM (0.9.2 arm, engine `vllm`) | `vllm==0.9.2`, pinned in `bench/requirements.lock` and re-recorded at runtime - **not the latest stable at the lock date**; pin disclosure below | V1 engine, FP16, CUDA-graph execution on; prompts as token IDs; `ignore_eos=True`                   |
| vLLM (0.24.0 arm, engine `vllm024`) | `vllm==0.24.0`, pinned in `bench/requirements-vllm024.lock` and re-recorded at runtime - measured under the full per-arm protocol in its own benchmark session; two-arm pin disclosure below | same adapter code path and measured path as the `vllm` arm, under its own virtual environment       |
| llama.cpp | documented commit hash, recorded in results JSON                                     | FP16 GGUF; `-ngl 99`; flash attention on; `--parallel 64 --cont-batching`; context sizing below     |

- **Recorded redline pin (amendment):** every results JSON records the module as
  `redline 0.1.0` (`engine_config.module_version`) together with a **failed git-commit
  capture** - the recorded `git rev-parse HEAD` attempt returned "fatal: not a git
  repository", because the benchmark pod ran from a source tree synced without `.git`. The
  synced tree was checksum-verified against the local source at sync time and is the engine
  source published in this repository; the harness files are the ones committed alongside
  these results. This amendment records what the JSONs actually contain: the missing
  in-JSON commit hash is a capture gap, not an unpinned build.
- **vLLM pin disclosure - two baseline arms (read before quoting any vLLM comparison):**
  the design-time policy (DESIGN.md section 13, milestone M5) was to measure the **latest
  stable vLLM at benchmark lock**, precisely because a stale baseline reads as strawmanning
  even when innocent. The original comparison rows do not meet that bar on their own: they
  measure `vllm==0.9.2` - the newest release of the 0.9.x line this adapter was developed
  and validated against - while PyPI's latest stable at the lock date (2026-07-09) was
  **0.24.0**, roughly twelve months (~24 releases) newer. The harness therefore carries
  **two vLLM baseline arms as distinct engines**, end to end (own `--engine` name, results
  JSON `engine` field, report table row, duplicate detection), and both are published:
  - arm `vllm` - `vllm==0.9.2`, frozen in `bench/requirements.lock`;
  - arm `vllm024` - `vllm==0.24.0`, frozen in `bench/requirements-vllm024.lock`.

  Both arms run the same adapter code path and the same measured path; each runs under its
  own virtual environment (the interpreter invoking `bench/run_bench.py` selects the arm's
  venv - one interpreter holds exactly one vllm, so the arms never share a process, and the
  equivalence gate refuses an invocation naming both), and each re-records its venv's
  installed vllm version into every results JSON at runtime. `bench/report.py` labels each
  table row from that recorded version ("vllm 0.9.2" / "vllm 0.24.0"), never from a file
  name, and refuses to render two arms whose labels would collide.
  **Gate-first protocol (restated, binding per arm):** no measured trial of an arm is
  publishable unless that arm's own fresh output-equivalence gate transcript was recorded
  *before* its measured trials (e.g. `--engines redline,vllm024` under the 0.24 venv),
  then the standard warmup + 3-trial protocol with the dispersion rule, raw per-token
  JSON committed; each arm also gets its own bounded tuning pass (best-configuration duty
  above). **Both arms meet the binding gate-before-measured-trials ordering:** the 0.9.2
  arm gated in the session that measured the redline and llama.cpp rows (transcript
  2026-07-10T00:58 UTC; that session's measured trials start 01:01 UTC), and the 0.24.0
  arm in its own later session - own venv, own gate transcript
  (`bench/results/equivalence/transcript-vllm024.json`, PASS with zero divergences,
  recorded above; 07:15 UTC), own bounded tuning pass (07:16-07:18 UTC, record below),
  then the 4-case matrix published as the `vllm 0.24.0` rows (07:19 UTC onward).
  **Sequence correction (recorded 2026-07-10):** an earlier revision of this paragraph
  stated the ordering as gate, *then* tuning, then trials, met by both arms on
  2026-07-10. The committed timestamps contradict that for the first session: the 0.9.2
  arm's tuning record (`bench/results/tuning/vllm_primary_default.json`) is stamped
  2026-07-09T23:07 UTC and the llama.cpp `-b`/`-ub` sweep 2026-07-10T00:18-00:20 UTC -
  both *before* that session's gate transcript (00:58 UTC). Only the 0.24.0 arm followed
  gate → tuning → trials as stated. The deviation does not touch what the gate protects -
  no measured, published trial of any arm precedes its gate, and the 0.9.2 arm's tuning
  outcome was decided by the alternate configuration failing to initialize, not by a
  performance readout - but the original sequence claim was wrong and is corrected here
  against the timestamps rather than restated. Preliminary `vllm==0.24.0` trials taken
  earlier the same day, before that arm's gate existed, are superseded by the formal arm
  and are not published as rows; their raw files are retained in the repository under
  `bench/results/addendum-vllm024/` so the supersession is auditable from a clone -
  preliminary medians 5167.94 (`primary`) / 2180.73 (`arrival`) / 7021.30 (`longout`) /
  219.70 (`batch1`) vs the published 5195.40 / 2180.78 / 7025.10 / 220.24; the formal
  re-measurement moved every case by well under 1%. (Those preliminary files record
  `engine: "vllm"` under the 0.24 venv - exactly the ambiguous-label shape
  `bench/report.py` refuses to render next to the formal arm, so they cannot leak into
  the table.)
  **Cross-session environment disclosure (binding on every 0.24.0 comparison):** the
  0.24.0 arm's session ran on a different rented pod instance of the same GPU model -
  RTX 4090, driver 580.159.03 vs 580.159.04 for the other rows (scope amendment under
  "Hardware and environment"; the README footnote names each driver with its rows). To
  anchor cross-session comparability with a measurement instead of an assumption, the
  published redline `primary` invocation was re-run verbatim on the 0.24.0 arm's pod in
  the same session: **4849.36 tok/s** median [4840.99, 4857.51] vs the published 4748.20
  [4721.32, 4758.29] - the 0.24.0 arm's pod runs redline **+2.1% faster** than the pod
  behind the published rows, so comparing the `vllm 0.24.0` rows against the published
  redline rows overstates redline's deficit by roughly that pod-to-pod margin, never the
  reverse. Committed:
  `bench/results/crosscheck/redline_primary_serving_reanchor_vllm024_session.json`
  (a cross-check record, never a table row).
  **Reading the two arms:** the `vllm 0.24.0` rows are published alongside - not
  replacing - the `vllm 0.9.2` rows. Measured outcome: **vLLM 0.24.0 is faster than
  redline on every case** - `primary` 5195.40 vs 4748.20 (redline at 91.4%; 93.3%
  against the same-pod redline re-anchor), `arrival` 2180.78 vs 2023.23 (92.8%),
  `longout` 7025.10 vs 6546.91 (93.2%), `batch1` 220.24 vs 156.96 (71.3%) - and faster
  than its own 0.9.2 pin on every case (+17.5% / +5.1% / +22.2% / +15.3%). Redline's
  primary/longout wins hold **against the pinned 0.9.2 arm only**; **no claim of beating
  current vLLM is made - the measured evidence is that redline does not.** The adapter
  targets the V1 engine's public APIs (the legacy pre-V1 `AsyncLLMEngine` is removed
  upstream): offline `LLM.generate` for the client-overhead cross-check, the V1 async
  streaming API for per-token timestamps.
- **llama.cpp context arithmetic (mandatory):** `llama-server` splits `-c` across `--parallel`
  slots, so each slot must fit prompt + output:
  `-c >= 64 * (1024 + 256) = 81920` for `primary`/`arrival`, `-c >= 64 * 2048 = 131072` for
  `longout`. Defaults would truncate every prompt and invalidate the run.
- **Best-configuration duty:** each baseline gets a bounded, documented tuning pass at
  benchmark time - vLLM: the pinned release's default CUDA-graph compilation mode vs its
  alternate (0.9.2 arm: piecewise vs `full_cuda_graph`, where the alternate failed to
  initialize on sm_89 and the pass reduced to release-default-only - record below; 0.24.0
  arm: `FULL_AND_PIECEWISE` vs `PIECEWISE`, a genuine measured comparison); llama.cpp:
  `-b`/`-ub` (logical/physical batch) sweep - the best result is published and the losing
  configs are committed with the raw results. No baseline runs deliberately handicapped
  flags, and no hidden tuning is applied to one engine only.
- **Recorded vLLM pin, 0.9.2 arm (lock date 2026-07-09):** `vllm==0.9.2` in a dedicated
  virtual environment, with the dependency set its own resolution selects: `torch==2.7.0` (cu126
  wheels), `triton==3.3.0`, `xformers==0.0.30`, Python 3.12.3. `transformers` is held at
  `4.53.3`, the newest release of the line contemporary with this vLLM version: later
  transformers releases ship an `aimv2` config themselves, and vLLM 0.9.x's own registration
  of that name then fails at import. Full freeze: `bench/requirements.lock`. The V1 engine is
  pinned explicitly (`VLLM_USE_V1=1`, set by the adapter): this release otherwise falls back
  to V0 when the engine is constructed outside the main thread, which the adapter's
  persistent-loop architecture does by design.
- **Recorded vLLM pin, 0.24.0 arm (lock date 2026-07-09; measured 2026-07-10):**
  `vllm==0.24.0` - PyPI's latest stable at the lock date - in its own dedicated virtual
  environment, with the dependency set its own resolution selects: `torch==2.11.0` (cu130
  wheels), `triton==3.6.0`, `transformers==5.13.0`, `numpy==2.3.5`, Python 3.12.3. Full
  freeze: `bench/requirements-vllm024.lock`. This release line is V1-only (there is no V0
  engine to fall back to), so the adapter's `VLLM_USE_V1=1` pin is inert here; the adapter
  changes the same two settings as on the 0.9.2 arm (FP16 dtype per the contract, prefix
  caching off per the bullet below) and everything else is the release's own default -
  including its `FULL_AND_PIECEWISE` CUDA-graph mode, asynchronous scheduling, and
  chunked prefill at `max_num_batched_tokens=2048`. **Recording correction:** an earlier
  revision said those three defaults were "all recorded in the results JSONs"; they are
  not. `engine_config` in the results JSONs records the adapter's explicit settings
  (`dtype`, `enforce_eager`, `enable_prefix_caching`, `disable_log_stats`) plus the
  runtime-recorded `vllm_version` - no `compilation_config`/`cudagraph_mode`, scheduling,
  or `max_num_batched_tokens` key appears anywhere in them. The three defaults are
  *implied* by the recorded pin (`vllm==0.24.0` release documentation) together with the
  recorded absence of any override, which is what makes the measured configuration the
  release default; they are not individually captured, and this file now says exactly
  that.
  The client-overhead cross-check record below was measured on the 0.9.2 arm; both arms
  run the identical streaming client structure ("Measured path"), so it bounds the shared
  client, not a per-arm difference.
- **vLLM prefix caching is explicitly disabled** (`enable_prefix_caching=False` in the
  adapter's `DEFAULT_CONFIG`, recorded in every results JSON's engine config): the trial
  protocol repeats the identical prompt set across the warmup and all measured trials, so
  vLLM's default prefix cache would serve repeat prefills from cache and measure a workload
  no other engine runs - the same reasoning as llama.cpp's `cache_prompt: false` below;
  redline has no prompt cache. Besides the contract-mandated FP16 dtype and silencing stats
  logging, this is the only setting the adapter changes from vLLM's engine defaults.
- **vLLM tuning-pass record, 0.9.2 arm (RTX 4090, sm_89):** the pass had exactly one
  alternative configuration - `compilation_config={"full_cuda_graph": true}` vs the
  release-default piecewise CUDA-graph mode - and that alternative **failed to initialize**
  on this GPU at this version: the FlashAttention backend requires AoT scheduling for full
  CUDA graphs (`ValueError: AoT scheduling is required for full cuda graph`), available on
  its FA3 path (Hopper) but not the FA2 path used on sm_89. The failure transcript is the
  committed raw result for that configuration:
  `bench/results/tuning/vllm_primary_full_cudagraph_failed.log`. **Plainly: no measured
  two-configuration comparison happened on this arm - the published `vllm 0.9.2` rows are
  release-default-only** (piecewise mode, the only configuration that initializes; its
  tuning-window run is committed as `bench/results/tuning/vllm_primary_default.json`), not
  the winner of a performance readout. The 0.24.0 arm's pass (record below) was a genuine
  two-configuration comparison with both arms measured; its release default won.
- **vLLM tuning-pass record, 0.24.0 arm (RTX 4090, sm_89):**
  `compilation_config.cudagraph_mode` - the release default **FULL_AND_PIECEWISE** vs
  **PIECEWISE** (the 0.9.2-era behavior), 1-trial arms at the primary shape through the
  harness: default 5186.43 tok/s, PIECEWISE 5045.52 tok/s. **Winner: the release-default
  FULL_AND_PIECEWISE mode** (+2.8%); the published trials run it (no
  `compilation_config` override, so the measured configuration is the release's own
  default, like the 0.9.2 arm's). Both arms' raw results are committed under
  `bench/results/tuning/` (`vllm024_primary_default.json`,
  `vllm024_primary_piecewise.json`).
- **vLLM client-overhead cross-check record (see "Measured path"):** offline `LLM.generate`
  vs the published serving numbers - three independent offline processes per applicable
  case, identical adapter configuration as the serving runs, same pod session as the
  matrix. Median offline vs published serving median: `primary` 4648.48 vs 4420.39 tok/s
  (offline +5.16%), `longout` 5951.19 vs 5751.08 (+3.48%), `batch1` 195.02 vs 190.94
  (+2.13%, and the offline and serving min-max bands overlap). `arrival` is refused by
  design (`--offline-crosscheck` exits nonzero: a closed offline batch cannot represent
  staggered arrivals). The offline path is uniformly the faster one - it has no per-token
  streaming client and a differently-anchored wall window - so the per-token streaming
  client costs at most ~5% of offline throughput on these workloads; published numbers
  remain serving-path numbers for all engines. A fresh 1-trial serving re-anchor measured
  in the same session lands +0.36% from the published serving median, tying the two
  measurement windows together on unlocked clocks. Raw:
  `bench/results/crosscheck/vllm_{primary,longout,batch1}_offline_crosscheck_r{1,2,3}.json`
  and `bench/results/crosscheck/vllm_primary_serving_reanchor.json`.
- **Recorded llama.cpp pin (lock date 2026-07-09):** commit
  `049326a00025d00b08cc188ed716b681e984a3f8` (master of 2026-07-09, build number 9949,
  `git describe` b9946-3-g049326a00), built with
  `cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)`
  (CUDA 12.9.86, gcc 13.3.0, `CMAKE_CUDA_ARCHITECTURES=89-real` by native detection).
  FP16 GGUF produced by the pinned tree's own converter:
  `python3 convert_hf_to_gguf.py <model_dir> --outtype f16 --outfile qwen2.5-1.5b-f16.gguf`
  (gguf-py 0.19.0, GGUF format v3, no quantization); sha256
  `9179b69919760407f6765313ee3efc438bab3cdeb2e63f9852076305fc183f03`
  (3,093,668,960 bytes). Measured launch line: `llama-server -m qwen2.5-1.5b-f16.gguf
  -ngl 99 -fa on -c 81920 --parallel 64 --cont-batching -b 8192 -ub 2048` (context dump
  confirms `flash_attn = enabled`, 29/29 layers offloaded, 64 slots x 1280 ctx per slot).
  Token-array prompt path validated before measuring: server-reported `prompt_n` equals the
  sent array length at 8 and 1024 tokens, streamed chunks carry token IDs
  (`return_tokens`), and `cache_prompt: false` demonstrably re-evaluates the full prompt on
  repeats (server-log evidence).
- **llama.cpp greedy idiom (evidence-backed):** requests pin `temperature: 0` **and**
  `top_k: 1`. Temperature 0 alone still runs llama.cpp's default filter chain
  (top-k 40 -> top-p 0.95 -> min-p 0.05) over the 151,936-entry CPU candidate array per
  token per slot - a filter cost no other engine's greedy path pays. `top_k: 1` measured
  token-stream-identical to the default chain over the full primary workload (16,384
  tokens, stream and non-stream). The measured effect is small (~2% on the non-stream
  path; within run-to-run variance on the streaming path); it is adopted for idiom
  symmetry, not as a speedup. Raw runs for both configurations are committed under
  `bench/results/tuning/llamacpp_sampler_chain/`.
- **llama.cpp tuning-pass record (RTX 4090, sm_89):** bounded `-b`/`-ub` sweep at the
  primary shape, 1-trial arms through the harness: `-b 2048 -ub 512` (build defaults)
  1116.7, `-b 4096 -ub 1024` 1172.5, `-b 8192 -ub 2048` 1182.6, `-b 2048 -ub 2048`
  1148.4 tok/s. The four arms sit within ~6% and their ordering swaps between runs (an
  identical sweep under the default sampler chain measured 1121.5 / 1191.6 / 1185.5 /
  1160.4 tok/s with `-b 4096 -ub 1024` ahead) - the published trials (3-trial protocol +
  dispersion rule) absorb this variance. **Published config: `-b 8192 -ub 2048`**, the
  best arm of the final sweep; every arm of both sweeps is committed under
  `bench/results/tuning/`.
- **llama.cpp client-overhead cross-check record (FAIRNESS "Measured path"):**
  `llama-batched-bench` at the matched shape (64 x 1024/256, winner flags, second of two
  back-to-back passes) finishes the wave in 4.14 s = 3957 tok/s generated-token
  equivalent, vs 1205 tok/s through llama-server + HTTP streaming (-69.5%). The
  decomposition shows the HTTP transport and the shared asyncio client are NOT the
  distorting element: (a) non-stream requests through the same server reach only
  1703-1732 tok/s, so per-token SSE emission+delivery accounts for ~23-32% of the
  non-stream rate and everything else is inside the server; (b) server-side per-slot
  eval rates printed by llama-server itself across the measured primary waves -
  37.8-45.5 ms/token; every `slot print_timing` line of that serving session is
  committed as `bench/results/crosscheck/llamacpp_server_print_timing_primary.log` -
  run at the same per-token pace as the client-observed ITL p50 of the published trials
  (35.9-36.3 ms across trials): the client consumes tokens essentially as fast as the
  server emits them; (c) batched-bench never samples or serves (it feeds synthetic
  tokens; no sampler call sites in `tools/batched-bench`), while llama-server samples
  each slot sequentially on the CPU per step (`tools/server/server.cpp`
  `common_sampler_sample` loop), so the batched-bench figure is a no-serving upper bound
  that no HTTP serving path reaches. Published llama.cpp numbers are therefore labeled
  as llama-server serving numbers; raw decomposition data:
  `bench/results/tuning/llamacpp_sampler_chain/diag_decomposition.json`. The serving-side
  operand of the -69.5% delta (1205.10 tok/s) is the committed 1-trial bring-up run
  `bench/results/crosscheck/llamacpp_bringup.json`; the delta arithmetic itself is
  `bench/results/crosscheck/llamacpp_crosscheck_delta.json`, and the raw
  `llama-batched-bench` output is
  `bench/results/crosscheck/llamacpp_batched_bench_crosscheck.txt`.
- **Cross-check artifact index (every file under `bench/results/crosscheck/`, so no
  committed artifact sits uncited):**
  `vllm_{primary,longout,batch1}_offline_crosscheck_r{1,2,3}.json` and
  `vllm_primary_serving_reanchor.json` - the offline cross-check series and serving
  re-anchor cited in the vLLM client-overhead record above;
  `redline_primary_serving_reanchor_vllm024_session.json` - the cross-session redline
  re-anchor cited in the two-arm pin disclosure;
  `vllm_primary_offline_crosscheck.json` - a single earlier offline run
  (2026-07-09T23:09 UTC, the same pre-gate bring-up window as the 0.9.2 tuning pass),
  superseded by the post-gate `_r{1,2,3}` series (2026-07-10T01:49 UTC onward) and cited
  as evidence nowhere - retained rather than deleted so the superseded run stays
  auditable;
  `llamacpp_bringup.json` - the 1-trial llama-server serving run behind the 1205.10 tok/s
  serving-side figure of the batched-bench decomposition above;
  `llamacpp_crosscheck_delta.json` - that decomposition's arithmetic record (3957.49
  gen-equivalent vs 1205.10 serving, -69.5%);
  `llamacpp_batched_bench_crosscheck.txt` - the raw `llama-batched-bench` output;
  `llamacpp_server_print_timing_primary.log` - the per-slot server timings cited in the
  decomposition above. Cross-check records never render as table rows (`bench/report.py`
  refuses their shapes or, for the full-shape re-anchors, their duplicate engine/case
  pairs against the published files).

## Measured path (client symmetry)

- **All published cross-engine numbers come from the same Python harness**
  (`bench/run_bench.py`) with the same asyncio client structure: Redline in-process via its
  pybind module, vLLM in-process via its V1 streaming API, llama.cpp over `llama-server`'s
  HTTP/SSE API. Client-side overhead is therefore shared structure, not a per-engine
  advantage.
- The pure-C++ `redline_bench` CLI is a **profiling tool only**; its numbers never appear in
  a cross-engine table.
- Client overhead is bounded and reported per baseline: the vLLM streaming adapter is
  cross-checked against an offline `LLM.generate` run of the same workload, and the llama.cpp
  HTTP path against `llama-batched-bench`. Residual asymmetry (e.g. per-token SSE cost) is
  quantified in the report rather than ignored.

## Metrics (definitions are normative)

Per-token client-side monotonic timestamps. For request `i`: submit time `t_sub_i`, token
times `t_i,0 .. t_i,n-1`.

- **TTFT** = `t_i,0 - t_sub_i`; p50/p99 across requests.
- **ITL** = successive token deltas within a request, first token excluded; p50/p99 pooled
  across requests. Pooling is not neutral to scheduler style on the closed-wave cases:
  redline runs dedicated prefill steps, so a request whose prefill finishes early in a
  closed wave waits out the rest of the wave's prefills before its second token - one long
  first-to-second-token gap per such request. In the published `primary` median trial,
  63 of 16,320 pooled redline ITLs exceed 20 ms, ranging up to 1.54 s, and all of them lie
  beyond the pooled p99 cut - invisible in the ITL p99 column - while vLLM's single worst
  ITL in the same trial is 48.2 ms and is caught at p99. Redline's small closed-wave ITL
  p99 must therefore be read together with its TTFT p99 and the decode-rate distribution,
  never as a general tail-latency claim; the README caption repeats this next to the table.
- **Output token throughput** (headline) =
  `sum_i n_i / (max_i t_i,last - min_i t_sub_i)` - total generated tokens over the full wall
  window. This window **includes all prefill work** and is labeled as such. No metric in this
  harness claims to "exclude prefill" over a shared wall-clock window: with concurrent
  requests, nearly all prefill executes inside any such window, so that label would be false.
- **Per-request decode rate** (secondary; genuinely prefill-free by construction) =
  `(n_i - 1) / (t_i,last - t_i,0)` per request - the reciprocal of its mean ITL - reported as
  a distribution (p50 across requests). Other requests' prefills stalling a decode lower this
  number; that is measured interference, not measurement error.

## Trials, dispersion, reporting

- 3 measured trials per engine/case after the warmup run. The **median** trial is reported,
  always alongside **min-max**. When a dispersion re-run leaves an even trial count, the
  reported "median" is the **lower-middle** of the sorted headline values
  (`bench/run_bench.select_median_trial`): one real measured trial is always published,
  never an interpolated midpoint. This mattered only on the three flagged llama.cpp cases -
  e.g. `primary` publishes 1319.97 from sorted trials [1203.29, 1319.97, 1338.00, 1357.33],
  where the two-middle interpolation would read 1329.0 - and the lower-middle choice sits
  at or below the midpoint by construction, i.e. on those rows it shades the *baseline's*
  published number down, not redline's; stated here so the selection rule is auditable
  instead of silently favorable.
- **Dispersion rule:** if (max − min) > 3% of the median on a headline metric, the case is
  re-run once; if the spread persists, the number is published with an explicit variance
  flag. Raw per-token JSON for every trial (including re-runs) is committed under
  `bench/results/`.
- **Recorded dispersion outcomes (benchmark matrix, 2026-07-10):** every redline and vLLM
  case sits far inside the threshold (no re-runs). The rule fired on three llama.cpp cases
  (`primary`, `arrival`, `longout`); each re-ran once, the spread persisted, and the
  numbers are published with the variance flag (†) - the per-trial headline values are in
  each results JSON's `dispersion` block. The slow trials are wave-level jitter on the
  llama-server serving path (TTFT-shaped: per-token decode rates stay flat across trials
  while a whole wave occasionally starts late), consistent with the run-to-run arm
  reordering already documented in the `-b`/`-ub` tuning record above.
- **Recorded dispersion outcomes (0.24.0 arm's matrix, 2026-07-10):** every `vllm024`
  case sits far inside the threshold - the largest spread is 0.23% of the median
  (`primary`); no re-runs, no flags.
- `bench/report.py` renders committed results JSON into the README table, its scope caption
  and environment footnote (GPU, driver, engine pins, measurement date), and the
  reproduction section (the exact recorded command line of every published run); no number
  appears anywhere that is not generated from a committed results file.
