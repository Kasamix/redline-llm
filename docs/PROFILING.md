# Profiling notes - Redline on RTX 4090 (sm_89)

<!-- Provenance: every measured number below is annotated with an HTML comment naming the
     exact committed source file(s) it was read from - raw benchmark JSONs under
     bench/results/raw/, trace-analyzer outputs and tuning-harness logs under
     bench/results/profiling/, cross-check records under bench/results/crosscheck/. Where a
     number's only record is the measurement session's logs or binary traces (archived
     outside the repository), the annotation says "archived" and names the source; such
     numbers are not independently auditable from this repository and are marked so
     deliberately rather than laundered. Comments cite; the visible text carries only the
     numbers. -->

Measured profiling record for the decode and prefill paths, following the six-section
skeleton in `docs/DESIGN.md` section 14. Every number in this document comes from a named
Nsight Systems trace, a CUDA-event microharness log, or a benchmark results JSON. The
trace-analyzer outputs, microharness logs, and A/B run JSONs backing sections 2-6 are
committed under `bench/results/profiling/`; the binary `.nsys-rep` traces themselves are
referenced by basename and attached as downloadable assets to the
[v0.1.0 release](https://github.com/Kasamix/redline-llm/releases/tag/v0.1.0), so every
timeline claim can be re-opened in Nsight Systems directly. Nothing here is hand-estimated
unless explicitly labeled as a model (analytic) value.

## 1. Environment and method

- **Hardware / software.** NVIDIA GeForce RTX 4090 (sm_89, 128 SMs, 24564 MiB), driver
  580.159.04, CUDA toolkit 12.9 (nvcc 12.9.86), Ubuntu 24.04, gcc 13.3.0, Python 3.12.3,
  Nsight Systems 2024.2.3. Two rented single-GPU pod sessions on the same driver: a
  kernel-tuning session (traces `decode500`, `decode500_tuned`, `prefill`, `prefill2048`,
  `nsys_v4final_node`) and a benchmark session (traces `p406_redline_primary`,
  `p406_redline_primary_eager`, `p406_redline_harness`, `p406_vllm_primary`; all committed
  `bench/results/raw/*.json` numbers).
  <!-- src: bench/results/raw/redline_primary.json environment block (nvidia-smi -q:
       RTX 4090, driver 580.159.04, 24564 MiB; nvcc 12.9.86; gcc 13.3.0-6ubuntu2~24.04.1;
       Python 3.12.3; kernel 6.8.0-124); 128 SMs: device header line of
       bench/results/profiling/harness_v4final.txt ("SMs 128"); nsys 2024.2.3: archived
       nsys install log; trace inventory: archived (analyzer outputs for each trace are
       committed under bench/results/profiling/an_*.json and are the numbers cited below) -->
- **Engine configuration.** sm_89 Release build; bench preset `kv_pool_gb=8, max_batch=64,
  max_seq_len=4096, prefill_chunk=2048`, CUDA graphs on (CLI diagnostics that use the
  engine defaults `max_seq_len=2048, prefill_chunk=1024` are noted where it matters - for
  1024-token prompts the chunk setting is execution-identical, measured in section 5).
  Model: Qwen2.5-1.5B-Instruct, FP16, greedy, forced output lengths.
  <!-- src: bench/results/raw/redline_primary.json engine_config + command_line;
       engine-default chunk 1024: src/bench_main.cpp defaults -->
- **Environment constraints (recorded, not worked around silently).** GPU clock locking is
  refused on the rented pods (`nvidia-smi -lgc`: no permission in the unprivileged
  container) - all numbers are unlocked-clock runs; observed SM clocks under load were
  2520-2790 MHz, snapshots embedded in the results JSONs. GPU hardware performance counters
  are administratively blocked on the pods (`RmProfilingAdminOnly: 1`, no `CAP_SYS_ADMIN`),
  so Nsight Compute cannot collect `dram__bytes.sum` or occupancy metrics. DRAM-traffic
  statements therefore use analytic byte models validated two ways: linearity of measured
  time in the modeled bytes, and an absolute byte ceiling (kernel time x same-session
  measured peak bandwidth bounds the bytes any kernel could have moved). Measured device
  peaks on this GPU: read-only kernel 956.6 GB/s, device-to-device memcpy 919-920 GB/s,
  rated 1008 GB/s.
  <!-- src: lgc refusal recorded in bench/FAIRNESS.md "Hardware and environment" + the
       nvidia-smi -q snapshots in bench/results/raw/*.json (clock range); counter block
       (RmProfilingAdminOnly:1, no CAP_SYS_ADMIN, ERR_NVGPUCTRPERM): archived session log.
       Device peaks: the 956.6 GB/s figure is the session's dedicated peak-measurement run
       (archived); the committed harness/probe headers reproduce it at 957.7-957.8 GB/s
       read-kernel, 919.1-919.7 memcpyD2D, rated 1008.1 = 2 x 10501 MHz x 384 bit
       (bench/results/profiling/harness_v4final.txt + probe_nocompute.txt headers) -->
- **Tracing method.** `nsys profile --trace=cuda,nvtx` around `redline_bench` via
  `scripts/profile_decode.sh`, capture window opened by `cudaProfilerStart()` after engine
  init + warmup (`--capture-range=cudaProfilerApi`), so weight upload, allocation, cuBLASLt
  probing, and graph capture sit outside every capture. CUDA-graph tracing runs at node
  level (`--cuda-graph-trace=node`): default graph-level tracing hides in-graph kernels and
  would hide in-graph bubbles from gap accounting. NVTX taxonomy: `step`,
  `prefill[len,chunk]`, `decode[b]`, `graph_replay[b]`, `sample`, `h2d_inputs`,
  `d2h_tokens`.
  <!-- src: scripts/profile_decode.sh (repo); the node-level-tracing-required finding:
       archived session log -->
- **Step accounting.** Decode steps are segmented by the once-per-step sampling kernel
  (argmax); per-step GPU busy = union of all kernel/memcpy/memset intervals in the step;
  host gap = wall − busy; gaps split into micro bubbles (< 30 µs, launch-pipeline spacing)
  and macro holes (≥ 30 µs, host-side work between GPU bursts). The NVTX `step` ranges
  cross-check the anchor segmentation on redline traces (255/255 decode steps, p50 within
  1%). Cross-engine traces run the identical serving harness (`bench/run_bench.py`,
  `--case primary --trials 1`) for both engines on the same pod, back to back, GPU verified
  idle between engines.
  <!-- src: bench/results/profiling/an_redline_cli.json nvtx_crosscheck block (255/255,
       step-wall p50 6797.4 vs 6859.5 us = 0.9%); analyzer method + idle gates: archived
       session log -->
- **Tracing overhead (measured).** redline: 4993.0 tok/s traced vs 5037.6 untraced (CLI,
  −0.9%); through the harness 4748.3 traced vs 4748.2 matrix median (−0.0%). vLLM: 4328.8
  traced vs 4420.4 matrix median (−2.1%). Overhead is carried in the section-3 robustness
  notes.
  <!-- src: bench/results/profiling/redline_primary_traced_cli.bench.json (4993.01);
       bench/results/profiling/redline_primary_graphs_r{1..3}.json (untraced median
       5037.59); bench/results/profiling/redline_primary_harness_traced.json (4748.27) vs
       bench/results/raw/redline_primary.json (4748.20);
       bench/results/profiling/vllm_primary_traced.json (4328.81) vs
       bench/results/raw/vllm_primary.json (4420.39) -->

## 2. Kernel-time breakdown per decode step

Primary decode shape (batch 64, context 1025-1280, graphs on; trace `p406_redline_primary`,
steady window of 200 consecutive bucket-64 decode steps; per-step GPU busy 6694 µs mean of
6854 µs wall; 312 kernels per step, min = max across the window):

<!-- src (whole table + class totals): bench/results/profiling/an_redline_cli.json tail
     block (top_kernels, class_us_per_step, per_step; busy mean 6694.1, wall mean 6854.4,
     kernels min=max=312) -->

| kernel | inst/step | avg µs | µs/step | share of GPU busy |
|---|---|---|---|---|
| cuBLASLt cutlass fp16 tensorop GEMMs (`Kernel2*`) | 85 | 32.80 | 2787.6 | 41.6% |
| `PagedAttentionDecodeKernel<256>` | 28 | 93.31 | 2612.7 | 39.0% |
| `ampere_fp16_s16816gemm_fp16_64x64` (Lt-selected) | 28 | 34.24 | 958.8 | 14.3% |
| `RmsNormResidualVecKernel<512,2>` | 56 | 1.88 | 105.5 | 1.6% |
| `SiluMulHalf2Kernel` | 28 | 2.03 | 56.9 | 0.85% |
| `RopeInplaceSharedKernel` | 28 | 1.55 | 43.3 | 0.65% |
| `splitKreduce_kernel` | 28 | 1.46 | 40.8 | 0.61% |
| `KvScatterKernel` | 28 | 1.22 | 34.1 | 0.51% |
| `GreedyArgmaxVecKernel<1024>` | 1 | 20.37 | 20.4 | 0.30% |
| `RmsNormVecKernel` + `EmbedGatherVec8Kernel` | 1 + 1 | ~1.6 | 3.2 | 0.05% |
| Lt split-k workspace memsets | 28 | 0.91 | 25.4 | 0.38% |
| H2D/D2H copies (pinned, section 3) | 5 | ~1.1 | 5.4 | 0.08% |

Class totals at this shape: GEMM 3787 µs (56.6%), decode attention 2613 µs (39.0%),
norm/activation/rope/embed/KV-write/argmax 264 µs (3.9%), memset + memcpy 31 µs (0.46%).

Short-context companion (trace `decode500_tuned`, 64 x 128/512 wave, context 129-640, 511
decode steps): GEMM 76.5% (3947 µs/step), attention 17.7% (32.61 µs avg/launch), small
kernels 5.2%, memsets 0.5%; per-step GPU work 5158.5 µs of 5326.3 µs wall. The two tables
bracket the decode regime: GEMM time is context-independent while the attention kernel
scales with context (32.6 → 93.3 µs avg between the two shapes), exactly the KV-traffic
proportionality validated in section 4.
<!-- src: kernel sums and step table of the decode500_tuned window:
     bench/results/profiling/after_kern_sum.txt + after_step_table.txt (GEMM 2017.214 ms
     over the 511-step window = 76.53%, attention 466.580 ms / 17.70% = 32.6 us avg, small
     5.24%, memset 0.53%, 5158.5/5326.3 us per step); class-share arithmetic from those
     committed sums -->

### 2.1 Small-kernel pass: per-kernel before/after (same 500-step workload)

The five non-GEMM, non-attention kernels were reworked between the `decode500` (before)
and `decode500_tuned` (after) traces - identical workload (64 x 128/512, seed 0, graphs
on) and identical execution structure (518 replays, 0 eager decodes, 0 aborts, bucket
histogram 1:7 64:511):

<!-- src (whole table): in-trace decode-window kernel sums,
     bench/results/profiling/before_kern_sum.txt + after_kern_sum.txt (before/after
     averages per kernel; the five-kernel and control rows recompute from these files);
     replay/bucket structure: archived session log -->

| kernel (inst/step) | before avg µs | after avg µs | delta | accepted change |
|---|---|---|---|---|
| `RmsNormResidual` (56) | 3.274 | 1.896 | −42.1% | half2 loads + register-cached pass 2 (no second global read) + 1-sync shuffle-tree reduction + block 512 (sm_80+) |
| `RmsNorm` plain (1) | 2.580 | 1.790 | −30.6% | same rework |
| `Rope` (28) | 1.788 | 1.613 | −9.8% | shared-sincos mapping, 7 heads/block (sm_80+): 8192 instead of 57344 sincosf per launch |
| `GreedyArgmax` (1) | 23.965 | 20.533 | −14.3% | uint4 scan + shuffle reduction + block 1024 (sm_80+) |
| `EmbedGather` (1) | 1.999 | 1.562 | −21.9% | uint4 tier, 192-thread blocks |
| `SiluMul` (28) | 2.135 | 2.132 | −0.1% (noise) | none - the 16-byte variant measured −16% in the harness and was rejected |
| **five-kernel total µs/step** | **321.7** | **234.9** | **−27.0%** | (`KvScatter`, untouched control: 1.272 → 1.267) |

Non-GEMM/non-attention share of decode-step GPU work: 7.34% → 5.77% (384.4 → 297.9
µs/step). Every accepted change cleared the ≥ 3%-of-its-kernel-time gate (smallest: rope
at 9.8%); the pass stopped when the sum of every remaining identified win measured
0.3-0.4% of step time (< 1%). The decode-attention journey (416.80 → 93.77 µs/launch) is
section 4's table; engine-level effects of both passes are section 6.
<!-- src: share and stop-rule arithmetic from the committed kernel sums
     (before_kern_sum.txt / after_kern_sum.txt); the silu 16-byte probe (3.297 vs
     2.829 us, rejected): archived harness log -->

### 2.2 Occupancy and vectorization notes (analytic; hardware counters blocked, section 1)

ptxas of the final build reports zero spill loads/stores on all five kernels; occupancy is
computed against Ada sm_89 limits (1536 threads / 48 warps / 65536 regs / 24 blocks per
SM), with achieved bandwidth measured against the section-1 device peaks:

<!-- src (whole table): bench/results/profiling/h_final_ptxas.txt (per-kernel register
     counts, zero spill loads/stores); occupancy columns are analytic (sm_89 limits);
     argmax 908 GB/s in-engine (= 95% of the measured read peak): archived session log -->

| kernel (final config) | regs | block, grid @ decode 64 | theoretical occupancy | binding observation |
|---|---|---|---|---|
| `RmsNormResidualVecKernel<512,2>` | 32 | 512, 64 | 100% (3 blocks/SM) | grid < #SMs - wins came from 16 warps/block and killing pass-2 re-reads (was ~25% of its traffic) |
| `RopeInplaceSharedKernel` | 20 | 448, (64,2) = 128 | 87.5% | exactly 1 block/SM; shared sincos removes 7x redundant transcendentals |
| `GreedyArgmaxVecKernel<1024>` | 28 | 1024, 64 | 66.7% | DRAM-bound: 908 GB/s in-engine = 95% of measured device peak |
| `EmbedGatherVec8Kernel` | 39 | 192, 64 | 100% (8 blocks/SM) | latency-bound random gather (3 KiB rows from a 445 MiB table) |
| `SiluMulHalf2Kernel` (unchanged) | 27 | 256, (18,64) | 100% (6 blocks/SM) | ~1.22 TB/s effective (L2-fed); wider 16-byte loads regressed - fewer threads in flight |

At decode batch 64 these row-per-block kernels place 64-128 blocks on 128 SMs, so
occupancy in the classical sense is capped by grid geometry, not by register or shared-mem
budgets; the measured wins came from wider loads (half2/uint4), more warps per block, and
removing redundant work, not from shaving register counts.

## 3. Timeline gap analysis: graphs, eager, and baseline serving paths

### 3.1 Host gaps with graphs on (the design gate)

500-step trace `decode500` (bucket-64 decode, node-level): decode wall 2763.464 ms, GPU
busy 2680.846 ms, **host gap 82.618 ms = 2.99% of decode wall** - under the 5% gate with
every component identified: stream-sync return tail inside `d2h_tokens` 1.891%, graph
launch submission 0.453%, H2D spacing 0.311%, mirror-fill host work 0.294%, inter-step
turnaround 0.041%. Largest single holes: two isolated ~1.6 ms OS-scheduler stalls in 511
steps (shared-CPU pod). Copy discipline, visible per item in the trace: exactly 4 async
H2D per step from pinned mirrors (3 x 256 B + one 32 KiB block-table slab) + 1 pinned
256 B D2H; 2811 memcpys in the capture, zero pageable; zero allocation API calls in the
capture range. At the primary shape (trace `p406_redline_primary`) the same engine floor
measures: gap total 2.34% of the 200-step window wall (per-step p50 148.9 µs on a
6859.5 µs step), of which 146.3 µs/step mean is sub-30-µs inter-kernel bubbles and only
13.9 µs mean is macro holes.
<!-- src: bench/results/profiling/gap_analysis_decode500.json (wall 2763.464 / busy
     2680.846 / gap 82.618 ms = 2.990%; gap_by_class_pct_of_wall 1.891/0.453/0.311/
     0.294/0.041; top_holes 1621.7/1580 us; copies: h2d exactly 4/step, hist 1533 x 256 B
     + 511 x 32768 B, d2h exactly 1/step x 256 B, memcpys_total_capture 2811,
     pageable_transfers_total_capture 0); zero-allocation grep of the capture: archived
     session log; primary-shape floor: bench/results/profiling/an_redline_cli.json
     (gap 2.34%, p50 148.9 us, micro mean 146.3, macro mean 13.9) -->

### 3.2 Eager vs graph replay (same binary, primary shape)

<!-- src (table): untraced medians of 3 -
     bench/results/profiling/redline_primary_{graphs,eager}_r{1..3}.json (per-metric
     median across r1..r3); traced gaps -
     bench/results/profiling/an_redline_cli.json (graphs: gap p50 148.9, micro p50 145.4)
     and an_redline_primary_eager.json (eager: gap p50 307.9, window 4.73%, micro p50
     300.2); API counts - api_per_step in the same two analyzer files -->

| mode | tok/s (median of 3, untraced) | step p50 ms | ITL p50 / p99 ms | per-step host gap p50 (traced) | API submissions per step |
|---|---|---|---|---|---|
| graphs on | 5037.59 | 6.752 | 6.686 / 6.943 | 148.9 µs (window total 2.34%) | 1 graph launch + 5 memcpy + 1 sync |
| eager | 4953.49 | 6.948 | 6.878 / 7.149 | 307.9 µs (window total 4.73%) | 312 kernel launches + 85 memsetAsync + 5 memcpy + 1 sync |
| delta | +1.70% | −196 µs (−2.8%) | −2.8% / −2.9% | −159 µs | ~400 → 7 |

Sources: `redline_primary_{graphs,eager}_r{1..3}.json` (untraced),
`p406_redline_primary(_eager)` (traced). Replaying one captured graph instead of ~312
per-kernel submissions removes 159 µs/step of host gap (micro-bubble p50 300.2 → 145.4 µs)
and 54 µs/step of GPU-side spacing - the traced total matches the untraced −196 µs step
delta. The eager path stays a tested fallback; graphs are the shipped default.
<!-- src: GPU-side spacing = busy p50 6746.5 (eager) vs 6692.4 (graphs) in
     bench/results/profiling/an_redline_primary_eager.json / an_redline_cli.json -->

### 3.3 Serving-path decode timelines vs the baselines (primary case)

Scope: every vLLM step-time and host-gap comparison in this section was traced against the
pinned `vllm==0.9.2` arm specifically (the version in the traced results' own `pip_freeze`);
it is not a claim about any newer vLLM release (pin disclosure and the second, `vllm==0.24.0`
arm: bench/FAIRNESS.md "Engine configuration").

Published medians (committed `bench/results/raw/*_primary.json`, same harness, same pod,
1 warmup + 3 trials): redline **4748.20 tok/s** [4721.32, 4758.29], vLLM 0.9.2 **4420.39**
[4415.42, 4421.30] (redline +7.4%), llama.cpp b9949 1319.97 [1203.29, 1357.33]
(dispersion-flagged; its deficit is dominated by 36.3 ms decode ITL plus wave-level serving
jitter with flat per-slot prompt-eval times - decomposed in the crosscheck records next to
the results, not re-traced here).
<!-- src: bench/results/raw/redline_primary.json (4748.20 [4721.32, 4758.29]),
     vllm_primary.json (4420.39 [4415.42, 4421.30]; vllm==0.9.2 in its pip_freeze),
     llamacpp_primary.json (1319.97 [1203.29, 1357.33], dispersion flag true, ITL p50
     36.279 ms); llama.cpp pin b9949: bench/FAIRNESS.md "Engine configuration"; jitter
     decomposition: bench/results/crosscheck/llamacpp_crosscheck_delta.json; flat
     per-slot prompt-eval evidence: the committed llama-server timing excerpt
     bench/results/crosscheck/llamacpp_server_print_timing_primary.log -->

**Wall decomposition from the published per-token records.** redline wall 3.4506 s =
1.5523 s prefill wave (TTFT p99) + 255 decode steps x 7.364 ms ITL (1.878 s) + ~20 ms tail.
vLLM wall 3.7065 s = 1.4386 s prefill phase (chunked prefill, fused prefill kernels -
**114 ms faster than redline's composite-GEMM prefill**, the deficit pre-stated in
DESIGN.md sections 6.3/16) + 255 x 8.987 ms (2.292 s, front edge overlapping its prefill
phase by ~24 ms). Net +255.9 ms wall = the +7.4% headline delta: redline gives back 114 ms
in the prefill wave and takes 414 ms back across the decode phase.
<!-- src: bench/results/raw/redline_primary.json median trial (wall_window_s 3.45057,
     ttft_p99_s 1.55227, itl_p50_s 7.364 ms) and vllm_primary.json (wall 3.70646, ttft_p99
     1.4386, itl_p50 8.987 ms); the decomposition itself is arithmetic over those
     committed values (delta wall 255.9 ms; decode +413.9 ms; prefill -113.7 ms; ~24 ms
     overlap; ~20 ms tail; both sides close to +-25 ms) -->

**Traced attribution of the decode-step delta** (traces `p406_redline_harness` /
`p406_vllm_primary`; both engines driven by the identical harness; steady window = last 200
pure-decode steps of the measured wave; vLLM's V1 EngineCore child process traced via its
spawn mode, startup-only knob):

<!-- src (table): bench/results/profiling/an_redline_harness.json (wall 7461.0/7433.6,
     busy 6704.8/6705.0, gap 791.7/728.5, gap p99 1114.2, 9.80%, kernels 312 min=max,
     api 1 cudaGraphLaunch + 5 memcpy) and an_vllm_primary.json (wall 9014.0/9097.9, busy
     7178.6/6999.3, gap 1816.9/2098.6, gap p99 4009.0, 23.07%, kernels mean 488.8 min 459
     max 543, api 29 cudaGraphLaunch + 177 cudaLaunchKernel + 1 cuLaunchKernel,
     memcpys 7) -->

| per decode step | redline | vLLM 0.9.2 | delta |
|---|---|---|---|
| wall p50 (mean) µs | 7461.0 (7433.6) | 9014.0 (9097.9) | +1553 (+1664) |
| GPU busy p50 (mean) µs | 6704.8 (6705.0) | 7178.6 (6999.3) | +474 (+294) |
| host gap p50 (mean) µs | 791.7 (728.5) | 1816.9 (2098.6) | +1025 (+1370) |
| host gap p99 µs | 1114 | 4009 | |
| gap share of wall | 9.8% | 23.1% | |
| GPU kernels per step | 312 (min = max) | 489 mean (459-543) | |
| launch API calls per step | 1 graph launch | 29 graph launches + 178 kernel launches | |
| memcpys per step | 5 | 7 | |

- **Kernel time: +294 µs/step (mean).** By class: the attention path is a wash -
  redline `PagedAttentionDecode` + `KvScatter` 2648.4 µs vs vLLM `flash_fwd_splitkv` +
  `combine` + `reshape_and_cache_flash` 2645.0 µs (−0.1%); GEMMs +78 µs (3871 vs 3794);
  norm/activation +22 µs; on-GPU memcpys +22 µs; sampler −7 µs and memsets −13 µs in
  vLLM's favor; the largest single term is **+195 µs/step of layout-glue kernels**
  (triton `cat`/`view`/copy and `elementwise_kernel`, ~140 instances/step) that redline
  does not run because its qkv/gate-up projections are consumed through strided views
  in place (DESIGN.md section 6.3, "no repack" rule).
  <!-- src: class_us_per_step in bench/results/profiling/an_redline_harness.json
       (attention 2614.06 + kv_cache 34.35 = 2648.4; gemm 3793.7; norm_act 209.6; memcpy
       7.2; sampler 20.1; memset 26.0) vs an_vllm_primary.json (attention 2645.0 =
       flash_fwd_splitkv 2504.4 + combine 96.3 + reshape_and_cache 44.3; gemm 3871.4;
       norm_act 231.6; memcpy 29.2; sampler 13.5; memset 13.2; other/layout-glue 195.2
       over 140 inst); the class deltas sum to the +294.0 busy-mean delta exactly
       (arithmetic over the two committed analyzer files) -->
- **Scheduling: +1370 µs/step (mean).** redline's 728.5 µs step gap = the 149 µs engine
  floor (section 3.1) plus ~580 µs of the shared Python serving client, which lives in
  the same process as the engine's pump loop. vLLM's 2098.6 µs mean gap (p99 4.0 ms) is
  GPU idle time inside its EngineCore process between steps - per-step scheduling, input
  preparation, sampled-token handling - while the streaming client runs in a separate
  process. Both engines pay the same client protocol; the gap difference is engine-side.
  <!-- src: arithmetic over committed values - 728.5 (an_redline_harness.json) - 148.9
       (an_redline_cli.json) ~= 580 us client share; untraced cross-check: CLI ITL 6.686
       (redline_primary_graphs_r{1..3}.json median) vs harness ITL 7.364 ms
       (bench/results/raw/redline_primary.json) = +678 us -->
- **Launch overhead (contained in the gap):** micro-bubbles 496 vs 209 µs/step (+287 µs)
  - 29 piecewise graph launches + 178 individual kernel launches per step vs one
  whole-step graph.
  <!-- src: micro_gap mean 496.3 (an_vllm_primary.json) vs 208.9 us
       (an_redline_harness.json), both under bench/results/profiling/ -->
- **Robustness.** Tracing costs vLLM ~2.1% and redline ~0.0% at the wave level, and the
  untraced published ITLs (7.364 vs 8.987 ms) independently give +1.62 ms/step - within
  2.5% of the traced wall delta. Even if the entire vLLM tracing overhead were charged
  against its gap, host-side time would still account for roughly three quarters of the
  decode-step difference.
  <!-- src: overhead numbers section 1 (traced-vs-matrix files under
       bench/results/profiling/); ITLs bench/results/raw/{redline,vllm}_primary.json;
       the a-fortiori charge-off is arithmetic over those committed values -->

Other cases, same mechanics, measured in the committed results: `arrival` inverts the
outcome (redline 2023.23 vs vLLM 2075.63 tok/s, −2.5%): with requests landing mid-decode,
redline's dedicated prefill steps stall running decodes (ITL p99 30.67 vs 26.32 ms) - the
scheduler tradeoff documented in DESIGN.md section 16 and quantified in section 5 here.
`longout` (64 x 1024/1024) runs 4x the decode steps of `primary` and widens the redline
lead (6546.91 vs 5751.08 tok/s, +13.8%; ITL p50 8.242 vs 9.754 ms; llama.cpp 1519.33,
dispersion-flagged) - the per-step decode advantage compounding as decode share grows.
`batch1` (156.96 vs 190.94 vs llama.cpp 227.83 tok/s) is a single-stream latency regime
this design does not optimize: the bucket-1 graph still launches full-width GEMM grids
(ITL p50 6.30 vs 5.15 / 4.28 ms). Both ship as measured.
<!-- src: bench/results/raw/{redline,vllm,llamacpp}_arrival.json (2023.23 / 2075.63 /
     1176.15; ITL p99 30.666 / 26.324 ms), *_longout.json (6546.91 / 5751.08 / 1519.33
     flagged; ITL p50 8.242 / 9.754 / 37.123 ms), *_batch1.json (156.96 / 190.94 / 227.83;
     ITL p50 6.295 / 5.147 / 4.276 ms) -->

## 4. Decode-attention kernel time vs its DRAM-traffic bound

Traffic model (each K/V byte read exactly once; batch 64, 2 KV heads, 16-token pages):
`bytes(ctx) = 128 * ceil(ctx/16) * 8192 B (K+V tiles) + 393 kB Q + 393 kB out + ~33 kB
tables`, so per launch 67.52 MB at ctx 1025 → 75.91 MB at ctx 1152 → 84.30 MB at ctx 1279;
x28 layers ≈ **1.9-2.4 GB per decode step** (KV-head-centric scale). A per-Q-head
streaming kernel would read K/V six times: ~449 MB/launch ≈ 12.6 GB/step.
<!-- src: analytic model (formula above); the per-ctx model bytes are tabulated in the
     committed harness logs' modelMB column
     (bench/results/profiling/harness_v4final.txt); the per-Q-head 6x figure is the same
     model with K/V re-read per Q head (arithmetic) -->

Validation without hardware counters (ncu blocked, section 1):

- **Byte ceiling.** The tuned kernel runs ctx 1152 in 93.77 µs; at the measured device
  peak it could move at most 93.77 µs x 956.6 GB/s = **89.7 MB < 449 MB** - per-Q-head
  traffic is physically impossible; actual traffic is within 18% of the KV-centric model.
  (True already for the pre-tuning kernel: 416.8 µs x 956.6 GB/s = 398.7 MB < 449 MB.)
- **Linearity.** Measured time is linear in `ceil(ctx/16)` at constant achieved GB/s
  across ctx 256-2048 (sweep below), matching tile-proportional traffic; intercept ~19 µs.
<!-- src: ceiling and linearity are arithmetic over the committed sweep
     (bench/results/profiling/harness_v4final.txt: 93.77 us at ctx 1152, ctx 256-2048
     rows with per-ctx GB/s); the 956.6 GB/s peak: section 1 note (archived peak run,
     corroborated by the committed harness headers at 957.7-957.8) -->

Tuning trail at ctx 1152, batch 64 (CUDA-event harness, 28 rotating layer pools so L2
stays cold; oracle cross-check max|diff| 3e-5 before every timing):

<!-- src (table): the harness logs of every revision, committed as
     bench/results/profiling/harness_{baseline,v1,v2,v3,v4,v5,v4final}.txt (us/launch and
     GB/s per ctx row; oracle cross-check line printed by each run); traffic-bound column
     = 79.4 us = 75.91 MB / 956.6 GB/s (arithmetic) -->

| rev | change | µs/launch | GB/s | x traffic bound (79.4 µs) |
|---|---|---|---|---|
| baseline | 128 thr, scalar staging, smem accumulator | 416.80 | 182.1 | 5.25x |
| V1 | 16-byte vectorized K/V tile staging | 376.21 | 201.8 | 4.74x |
| V2 | 256 threads (sm_80+) + 2-stage cp.async pipeline | 194.05 | 391.2 | 2.44x |
| V3 | FP32 Q smem, vector smem reads, register accumulator | 146.40 | 518.5 | 1.84x |
| V4 | 8-lane score groups, fused softmax phases, unrolled fold | 94.90 | 799.9 | 1.20x |
| V5 | 3-stage pipeline probe | 95.41 | 795.6 | reverted (−0.5%) |
| final | V4 after revert + rebuild | **93.77** | **809.5** | **1.18x** |

Final state: **1.18x the measured-peak traffic bound** (1.25x rated peak) and 1.03x the
kernel's own staged-load floor (staging-only probe: 91.03 µs = 833.9 GB/s), so remaining
compute-side headroom is under 5% - stopped there. Context sweep (µs/launch):
256→23.66, 512→43.56, 768→63.85, 1024→84.26, 1152→93.77, 1280→104.41, 1536→121.71,
2048→158.65 (848.7 GB/s). In-engine confirmation (node-level traces): 7560 instances avg
89.8 µs across ctx 1025-1279 (`nsys_v4final_node`), and 93.31 µs avg at the same shape in
the benchmark session (`p406_redline_primary`) - harness and engine agree within noise.
Engine-level effect of this kernel alone: 2851.69 → 4851.64 tok/s (+70.1%) on the primary
shape, ITL p50 16.226 → 6.857 ms, attention cost per decode step 11.67 → 2.63 ms, zero
VRAM delta.
<!-- src: staging-only probe: bench/results/profiling/probe_nocompute.txt (91.03 us =
     833.9 GB/s at ctx 1152); ctx sweep: bench/results/profiling/harness_v4final.txt;
     1.25x rated = arithmetic vs the 1008.1 GB/s header value; 7560 instances avg
     89.8 us: bench/results/profiling/kern_sum_v4final_node.txt (89802.6 ns avg);
     benchmark-session 93.31 us avg: bench/results/profiling/an_redline_cli.json.
     Engine-level effect: "after" side committed as
     bench/results/profiling/bench_v4final.json (4851.64 tok/s, ITL p50 6.857 ms);
     attention ms/step 11.671 -> 2.626 is the ms/step28 column of the committed
     harness_baseline.txt / harness_v4final.txt at ctx 1152; the "before" engine run
     (2851.69 tok/s, ITL 16.226 ms, VRAM 11814 MiB both sides) is the archived
     pre-rework baseline log -->

## 5. Prefill cost accounting vs the DESIGN 6.3 model

Composite prefill (kv_gather → 2 strided-batched score GEMMs per layer with K-operand
batch-stride 0 and FP32 D → causal softmax → 2 strided-batched PV GEMMs with D strided
into the attention output at ldd 1536) measured at the bench preset, graphs on (prefill is
eager by design, so every kernel is individually visible; traces `prefill` (chunk 1024)
and `prefill2048` (chunk 2048), 8 x 4096/8 workload):
<!-- src: per-chunk trace analyses committed as
     bench/results/profiling/analysis_chunk1024.json + analysis_chunk2048.json -->

**2048-token chunk at 4K context** (the model's stress point): step wall **79.45 ms**
(min 78.95, max 81.21 over 8), GPU busy 78.97 ms = **99.4% of wall**. Decomposition:
score GEMMs 15.07 ms + causal softmax 15.68 ms + PV GEMMs 9.80 ms = **softmax path
40.55 ms**, which moves the DESIGN 6.3 model's 33.82 GB at **834 GB/s effective** - 87% of
the measured device read peak, confirming the "~32 GB softmax-path traffic" expectation
(byte ceiling: 40.55 ms x 956.6 GB/s = 38.8 GB movable at most, so any ≥1.15x-model
traffic is excluded). Dense GEMMs (qkv/o/gate-up/down): 33.82 ms for 5.366 TFLOP =
**158.7 TFLOPS**, ~96% of the nominal fp16/fp32-acc dense rate - the chunk is
compute-saturated. Small kernels ~4.6 ms (silu 2.83, rmsnorm 0.62, rope 0.33, gather 0.19,
scatter 0.08, final-chunk lm_head GEMV 0.50 at ~930 GB/s, argmax 0.007).
<!-- src: bench/results/profiling/analysis_chunk2048.json group
     prefill[len=4096,chunk=2048] (step_wall_ms mean 79.450, min 78.945, max 81.211;
     gpu_busy 78.974 = 99.4%; scores_ms 15.070 + softmax_ms 15.677 + pv_ms 9.799 =
     path_ms 40.547; model_path_gb 33.82 -> implied 834.2 GB/s; dense_gemm_ms 33.822);
     the 5.366 TFLOP figure is the analytic FLOP count at this shape over the committed
     dense_gemm_ms; the small-kernel per-name split (silu 2.83, rmsnorm 0.63, rope 0.34,
     gather 0.19, scatter 0.08, lm_head-in-"other" 0.50, argmax 0.008) is the same
     analysis file's per-chunk cls_time_ms blocks -->

**Chunk-cost ladder** (chunk 1024, per-chunk wall as context grows): 24.91 ms at ctx 1024
→ 30.77 → 36.91 → 42.87 ms at ctx 4096; linear, slope ~6 ms per 1024 context = the
softmax-path increment; dense floor constant at 17.8-17.9 ms. The second pre-stated model
point lands: 1024-token chunk at 1K ctx = 4.23 GB softmax-path bytes in 4.53 ms at
934.5 GB/s (model said ~4 GB / ~5 ms). Whole 4096-token prompt: 135.5 ms (chunk 1024) vs
138.2 ms (chunk 2048) - finer chunks trade ~17% fewer softmax-path bytes and half the
per-chunk decode stall against ~5% dense-GEMM efficiency; net −2.0%.
<!-- src: bench/results/profiling/analysis_chunk1024.json groups
     prefill[len={1024,2048,3072,4096},chunk=1024] (step_wall_ms means 24.910 / 30.771 /
     36.905 / 42.866; path_ms 4.525 at 934.5 GB/s implied for the ctx-1024 group;
     dense_gemm_ms 17.760-17.895 across groups); whole-prompt totals 135.45 vs 138.24 ms
     = sums over the committed per-chunk groups of both analysis files -->

**What dedicated prefill steps cost the tails** (the scheduler's documented tradeoff):
tails are sums of chunk costs. The 64-prompt primary wave serializes 64 single-chunk
prefills: 64 x 24.2 ms measured mean chunk (prefill phase 1546.8 ms at 98.4% GPU busy in
`p406_redline_primary`) ≈ the published TTFT p99 1.552 s. On 8 x 4096/8 the first-finished
request waits out 28 trailing chunks of other requests: ITL p99 881.87 ms. The DESIGN 6.3
byte model is accurate; its "350-400 ms" wave-serialization estimate was ~4x optimistic
because dense GEMM time (72% of a ctx-1024 chunk) was not counted - recorded here as the
correction, and visible in the baseline comparison: vLLM's fused chunked prefill finishes
the same wave in 1.4386 s vs redline's 1.5523 s (TTFT p99, +7.9%; TTFT p50 vLLM 0.7216 vs
redline 0.7925 s, +9.8%).
<!-- src: prefill phase 1546.791 ms / busy 1522.149 (98.4%) over 64 prefills = 24.2 ms
     mean: bench/results/profiling/an_redline_cli.json wave block; TTFT p99 1.5523 s +
     TTFT p50 0.7925: bench/results/raw/redline_primary.json; vLLM 1.4386 / 0.7216:
     bench/results/raw/vllm_primary.json; the dense-share 72% correction is arithmetic
     over the committed ladder (dense 17.8-17.9 of a 24.9 ms ctx-1024 chunk); the
     8x4096/8 ITL p99 881.87 ms: archived diagnostic run of that shape -->

**Knob and configuration audit.** `prefill_chunk` 1024 vs 2048 on the `arrival` case is a
measured wash (throughput 2018.4 vs 2020.4 tok/s means, TTFT/ITL p99 deltas ≤ 0.1%, inside
the ±0.6% run-to-run band; mechanism: 1024-token prompts prefill in one chunk under both
settings) - defaults kept; the only real difference is scratch memory (700.6 MiB at 2048
vs 361.6 MiB at 1024). cuBLASLt on sm_89 accepts every probed prefill config natively
(59/59 at chunk 2048, 55/55 at chunk 1024): scores strided-batched algo id 23 (tile 23,
stages 9) at T=2048 / id 24 (tile 18) at T=1024, PV id 24 (tile 15); the per-head
fallbacks are probed-available but never selected; the repack stage is never reached.
`arrival` ITL p99 ≈ 31 ms = one ctx-1024 chunk (24.9 ms) + one decode step between two
tokens of a running request - the per-request cost of a mid-decode admission.
<!-- src: the 2x2 chunk-sensitivity runs, committed summary
     bench/results/profiling/arrival_summary.txt (2018.37/2018.42 vs 2020.18/2020.56
     tok/s; ITL p99 31.52-31.98 ms across arms; plus the strict-defaults control run);
     run-to-run band and scratch sizes (700.6 vs 361.6 MiB), Lt probe counts (59/59 +
     55/55) and algo ids/tiles/stages: archived session log + the lt_algo_report blocks
     embedded in the archived arrival run JSONs -->

## 6. Top-3 tuning actions with before/after measurements

<!-- src (row 1): kernel level - bench/results/profiling/harness_baseline.txt (416.80 us,
     182.1 GB/s) vs harness_v4final.txt (93.77 us, 809.5 GB/s); in-engine confirmation
     kern_sum_v4final_node.txt; engine level - "after" bench_v4final.json (4851.64 tok/s,
     ITL 6.857 ms), "before" (2851.69, 16.226 ms, VRAM 11814 MiB both sides) archived
     pre-rework baseline log.
     (row 2): kernel level - before_kern_sum.txt / after_kern_sum.txt (five-kernel 321.7
     -> 234.9 us/step, share 7.34% -> 5.77%); engine level (step p50 6.904 -> 6.810 ms;
     4870.3 -> 4931.1 tok/s; drift 0.253%): archived A/B run logs; the silu -16% and
     split-row-argmax rejections: archived harness logs.
     (row 3): section 3.2 sources - redline_primary_{graphs,eager}_r{1..3}.json +
     an_redline_cli.json + an_redline_primary_eager.json (gap 307.9 -> 148.9 us p50;
     micro 300.2 -> 145.4; ~400 -> 7 API; 6.948 -> 6.752 ms; 4953.49 -> 5037.59 tok/s),
     all under bench/results/profiling/. -->

| # | action | kernel level (before → after) | engine level (before → after) | evidence |
|---|---|---|---|---|
| 1 | Decode-attention rework: vectorized K/V staging, 256 threads + cp.async double buffering (sm_80+), register accumulator, fused softmax phases | 416.80 → **93.77 µs**/launch at ctx 1152 (4.44x; 182 → 809.5 GB/s, final 1.18x traffic bound) | 2851.69 → **4851.64 tok/s** (+70.1%); ITL p50 16.226 → 6.857 ms; VRAM unchanged (11814 MiB) | harness logs + `nsys_v4final_node`; section 4 |
| 2 | Small-kernel pass: rmsnorm half2+register pass-2 (−42% residual variant), rope shared-sincos 7 heads/block (−10%), argmax uint4+shuffle (−14%), embed uint4 tier (−22%); silu 16-byte probe measured −16% and rejected | five kernels 321.7 → **234.9 µs**/step (−27.0%); non-GEMM/non-attention share 7.34% → 5.77% | step p50 6.904 → 6.810 ms (−1.4%); 4870.3 → 4931.1 tok/s (+1.3%); p50 drift across runs 0.25% | `decode500` vs `decode500_tuned`; section 2.1 |
| 3 | Whole-step CUDA-graph replay for decode (captured per bucket at init; pinned-mirror batched copies) | per-step host gap 307.9 → **148.9 µs** p50; API submissions ~400 → 7; micro-bubbles 300 → 145 µs | step p50 6.948 → 6.752 ms (−2.8%); 4953.49 → **5037.59 tok/s** (+1.7%) | `p406_redline_primary(_eager)` + A/B runs; section 3.2 |

Measured-and-rejected changes are part of the record: the silu_mul 16-byte variant (−16%
vs half2, kept half2), a 3-stage attention pipeline (−0.5%, reverted), a split-row argmax
(+2 µs/launch potential but a device-scratch handshake inside the graph for ~0.04% of
step). Stop rule at the end of the pass: every remaining identified win measured at
0.3-0.4% of step time combined.
<!-- src: the 3-stage pipeline probe (-0.5%, reverted):
     bench/results/profiling/harness_v5.txt vs harness_v4.txt; the silu vec8 probe and
     split-row argmax estimate, and the residual-wins stop-rule sum (0.3-0.4% of step):
     archived harness logs -->
