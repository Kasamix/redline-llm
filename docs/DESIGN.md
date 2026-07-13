# Redline - Engineering Design

Status: **shipped (v1 implemented and measured).** This document was written pre-implementation
and is kept as the engineering record: where the measured outcome deviated from the design, the
deviation is annotated inline (*Shipped configuration note* / *Measured outcome* markers) rather
than the prediction being rewritten. Constants below are the expected values for
Qwen/Qwen2.5-1.5B-Instruct; `docs/MODEL_SPEC.md` is the authoritative, HF-verified copy, and the
loader re-validates every value from `config.json` at startup (hard error on mismatch, never
silent fallback).

## 0. Scope

Redline is a single-GPU LLM inference engine written from scratch in C++20/CUDA that serves
Qwen2.5-1.5B-Instruct in FP16 with a paged KV cache, iteration-level continuous batching, and
CUDA-graph decode. It exposes a Python module (`redline`) and a pure-C++ benchmark CLI
(`redline_bench`). Paged KV caching follows the concepts of the PagedAttention paper
(Kwon et al., 2023); all code is an original implementation.

Non-goals for v1: quantization, multi-GPU, speculative decoding, sampling other than greedy,
preemption/swapping, sliding-window attention, server/HTTP layer, tokenization (the engine
consumes and produces token IDs; tokenization stays client-side).

Target hardware (both must work from one binary via runtime config):

| Role  | GPU               | Arch  | VRAM  | Notes                                   |
|-------|-------------------|-------|-------|-----------------------------------------|
| dev   | RTX 2060          | sm_75 | 6 GiB | WSL2 Ubuntu, shares VRAM with display   |
| bench | RTX 4090 (rented) | sm_89 | 24 GiB| headless Linux pod                      |

## 1. Model constants and derived sizes

| Constant                | Value        | Derived                                  |
|-------------------------|--------------|------------------------------------------|
| hidden_size H           | 1536         |                                          |
| layers L                | 28           |                                          |
| attention heads         | 12           | head_dim D = 1536/12 = 128               |
| KV heads                | 2            | GQA group = 12/2 = 6 Q heads per KV head |
| intermediate_size I     | 8960         | gate+up fused width = 17920              |
| vocab V                 | 151936       |                                          |
| max_position_embeddings | 32768        | engine `max_seq_len` is far below this   |
| rope_theta              | 1e6          | NeoX-style half rotation (Section 6.3)   |
| rms_norm_eps            | 1e-6         |                                          |
| tie_word_embeddings     | true         | lm_head aliases embedding matrix         |
| attention bias          | Q,K,V only   | o_proj and MLP have no bias              |
| sliding window          | disabled     | full causal attention                    |
| checkpoint dtype        | BF16         | converted to FP16 at load (Section 4)    |
| EOS token IDs           | 151645,151643| read from generation_config.json         |

Parameter count (FP16 = 2 bytes each):

- Embedding: 151936 x 1536                      = 233,373,696
- Per layer: QKV 3,145,728 (+2,048 bias) + o_proj 2,359,296 + MLP 3 x 13,762,560 + 2 norms 3,072
                                                = 46,797,824
- Total: 233,373,696 + 28 x 46,797,824 + 1,536  = 1,543,714,304 params
- **Weights in FP16: 3,087,428,608 B = 2944 MiB = 2.88 GiB** (no lm_head copy; tied)

KV cache cost (FP16):

- Per token per layer: 2(K,V) x 2 heads x 128 x 2 B = 1 KiB
- Per token, all layers: **28 KiB**
- Per block (16 tokens), all layers: **448 KiB** (16 KiB per layer)
- `num_blocks = floor(kv_pool_bytes / 458752)`

## 2. System overview

```
Python (tokenize, drive)          C++ Engine (one thread, one CUDA stream)
  add_request(ids, max_new) --->  Scheduler ----> waiting FIFO
  step()  <-- (id,tok,fin) ---        |            admission (block reservation)
                                      v
                                 ModelExecutor
                                   prefill(seq, chunk)   eager, chunked (preset knob)
                                   decode(batch)         CUDA graph replay (or eager)
                                      |
                                 KV BlockAllocator (host free list)  +  paged pool (device)
```

Threading: the engine is single-threaded by contract - `add_request`/`step`/`stats` must be
called from one thread (debug-asserted). `step()` releases the GIL. All device work goes on one
non-default stream `s0`; no locks, no background threads in v1.

Repository layout:

```
src/core/       engine, scheduler, block allocator/table, config, types
src/kernels/    custom CUDA kernels (*.cu) + launch contracts (kernels.cuh)
src/gemm/       cuBLASLt wrapper
src/loader/     safetensors reader + BF16->FP16 conversion
src/api/        pybind11 module (redline)
src/bench_main.cpp  redline_bench CLI (profiling driver)
python/         redline_client (client-side tokenizer glue)
tests/          gtest units + Python e2e (tests/e2e)
bench/          multi-engine harness, adapters, FAIRNESS.md, results/
scripts/        bootstrap, fetch_model.py, runpod/, profile_decode.sh
docs/           DESIGN.md, MODEL_SPEC.md, PROFILING.md (created at M4)
```

## 3. Build

- CMake >= 3.26 + Ninja; `CMAKE_CUDA_ARCHITECTURES="75;89"`; host and device C++20 (CUDA 12.6
  supports `--std=c++20`); `-O3`; **no** `--use_fast_math` (correctness first - fast-math
  degrades `sinf`/`expf` used by RoPE/softmax; may be re-enabled per-kernel later only with
  passing tolerance tests).
- Targets: `redline_core` (static lib), `redline` (pybind11 module), `redline_bench` (CLI),
  `redline_tests` (gtest).
- FetchContent (pinned tags): googletest, pybind11, nlohmann/json. nlohmann/json is a deliberate
  addition to the seed dependency list: both the safetensors header and `config.json` are JSON,
  and a hand-rolled parser is risk without value. Link `cublasLt` and `cudart`; NVTX via the
  `nvtx3` headers shipped with CUDA 12 (no extra dependency).
- Linux-only runtime (WSL2 and pods); POSIX `mmap` in the loader; no MSVC support in v1.

## 4. Weight loading

Custom safetensors reader: read little-endian u64 header length, parse the JSON header
(`name -> {dtype, shape, data_offsets}`), `mmap` the file read-only, and upload tensor by tensor.
Both single-file `model.safetensors` and sharded `model.safetensors.index.json` are supported
(Qwen2.5-1.5B ships a single file).

BF16 -> FP16 conversion happens on CPU into a reusable pinned staging buffer (bounded size,
streamed per tensor): expand BF16 to FP32 (`u32 = u16 << 16`), then round to FP16 with
**round-to-nearest-even and subnormals preserved** - exactly what `torch.Tensor.to(float16)`
does when HF loads this BF16 checkpoint as FP16, so converted weights are bit-identical to the
reference's. No clamping and no flush-to-zero: any value outside FP16 range (|x| > 65504) makes
the loader **fail hard immediately**. The expected overflow count for this checkpoint is zero,
so failing is free - a single hit means a corrupt file or the wrong checkpoint, and clamping
would have silently diverged from the HF reference the e2e suite compares against. The
conversion routine is unit-tested exhaustively: all 65,536 BF16 bit patterns against a
torch-generated golden table (Section 12a).

Load-time weight preparation (fusions are layout-only, done once):

| Device tensor    | Shape [rows,cols] | Built from HF tensors                                   |
|------------------|-------------------|---------------------------------------------------------|
| `embed`          | [151936, 1536]    | `model.embed_tokens.weight`                             |
| `w_qkv` / `b_qkv`| [2048, 1536]/[2048]| q_proj rows 0:1536, k_proj 1536:1792, v_proj 1792:2048 |
| `w_o`            | [1536, 1536]      | `o_proj.weight`                                         |
| `w_gateup`       | [17920, 1536]     | gate_proj rows 0:8960, up_proj rows 8960:17920          |
| `w_down`         | [1536, 8960]      | `down_proj.weight`                                      |
| norms            | [1536] x (2L + 1) | input/post_attention layernorm + final `model.norm`     |
| `lm_head`        | alias of `embed`  | if `lm_head.weight` present in file, use it instead     |

All weights keep the HF `nn.Linear` convention `[out_features, in_features]`, row-major.

## 5. Device tensor layouts and index math

All activations are FP16 row-major unless stated; all reductions accumulate in FP32.

**Paged KV pool** - one `cudaMalloc` at init, logically
`pool[L=28][num_blocks][2 (0=K,1=V)][kv_heads=2][block_size=16][head_dim=128]`, FP16.
Element address (in elements) for layer `l`, physical block `b`, K/V `kv`, head `h`, slot `s`,
dim `d`:

```
idx = ((((l * num_blocks + b) * 2 + kv) * 2 + h) * 16 + s) * 128 + d
```

The layer-outer layout means one physical block ID is valid for every layer (one block table per
sequence, shared across layers), and the innermost `[16,128]` tile gives the decode kernel
coalesced 256 B rows per (head, token).

**Block tables** - device `int32 [max_batch][max_blocks_per_seq]`,
`max_blocks_per_seq = ceil(max_seq_len/16)` (dev 2048/16 = 128; bench 4096/16 = 256). Row `i`
holds physical block IDs of the sequence in batch slot `i`. Token position `p` lives in physical
block `block_table[i][p >> 4]`, slot `p & 15`.

**Per-step inputs** (persistent device buffers, `int32`), sized for their largest consumer -
prefill pushes up to `prefill_chunk` rows through the same buffers decode uses, so sizing them
at `max_batch` would be a silent device buffer overrun during every prefill chunk:

| Buffer       | Elements                         | Contents                                       |
|--------------|----------------------------------|------------------------------------------------|
| `input_ids`  | `max(prefill_chunk, max_bucket)` | token id per row                               |
| `positions`  | `max(prefill_chunk, max_bucket)` | absolute token position per row                |
| `seq_lens`   | `max_batch`                      | context length incl. current token; 0 = padded |
| block tables | `max_batch * max_blocks_per_seq` | described above                                |

There is **no** separate slot-index input: `kv_scatter` derives its pool slot in-kernel from
`positions[r]` and the block table (decode: table row `r`; prefill: all chunk rows share the
one sequence's row, passed with row-stride 0), so the per-step upload set is exactly the four
arrays above (Section 9).

Every per-step input - and `argmax_out` on the way back - has a **persistent pinned host
mirror** (`cudaHostAlloc` at init). The host fills mirrors, then issues one contiguous
`cudaMemcpyAsync` per array, never per-row copies; pageable or fragmented transfers silently
degrade to staged synchronous copies and refund the CUDA-graph latency win (Section 9).

**Activation scratch** (persistent, sized at init for `max(prefill_chunk, max_bucket)` rows):
`x` (residual), `normed`, `qkv_out [rows,2048]`, `attn_out`, `gateup [rows,17920]`,
`mlp_out`, `logits [max_batch,151936]`, `argmax_out int32[max_batch]`, prefill-only
`scores fp32 [12, chunk, max_seq_len]`, `probs fp16` (same shape), and a KV gather buffer
`khat/vhat [2 kv_heads, max_seq_len, 128]`. Within `qkv_out`, row layout is
`Q(12x128) | K(2x128) | V(2x128)`; Q head `h` of row `r` starts at `qkv_out + r*2048 + h*128`.

**Strided-view rule:** downstream kernels consume Q/K/V *directly as views into `qkv_out`* -
`q = qkv_out + 0`, `k = qkv_out + 1536`, `v = qkv_out + 1792`, each with **row stride 2048**
elements - so no split/repack kernel exists anywhere on the decode path (that would add ~3
launches per layer, +84 per step, defeating the fused QKV GEMM). Consequently every kernel
launcher takes explicit row-stride parameters (`src/kernels/kernels.cuh` is the contract);
dense-packed tensors simply pass `stride == row width`. No allocation ever happens after init
(required for graph capture, Section 9).

## 6. Forward pass

### 6.1 GEMMs (cuBLASLt)

All matmuls use `cublasLtMatmul` with FP16 A/B/D and `CUBLAS_COMPUTE_32F`. Logically every
linear layer is `Y[T, n_out] = X[T, in] @ W[n_out, in]^T (+ b[n_out])` over row-major buffers.
The descriptors, however, are specified in the **column-major dual** - the canonical TN
formulation that Lt's heuristics and epilogues cover best. (An earlier draft used
`CUBLASLT_ORDER_ROW` descriptors with a BIAS epilogue; rejected at review: Lt defines the
epilogue bias length as the number of *rows of D*, which under a row-major description of
`[T, n_out]` is `T` - a per-token bias on the wrong axis - and row-order + epilogue is the
thinnest-covered corner of Lt's algo space.)

The dual, with zero data movement: a row-major buffer `M[r, c]` with leading dimension `c` *is*
the column-major matrix `M^T [c, r]` with `ld = c`. So compute `Y^T = W @ X^T`:

- `A` = weight buffer (row-major `[n_out, in]`) read as col-major `[in, n_out]`, `opA = T`,
  `lda = in`;
- `B` = activation buffer (row-major `[T, in]`) read as col-major `[in, T]`, `opB = N`,
  `ldb = in`;
- `D` = output buffer (row-major `[T, n_out]`) read as col-major `[n_out, T]`, `ldd = n_out`;
- `m = n_out`, `n = T`, `k = in` - a plain TN GEMM.

`CUBLASLT_EPILOGUE_BIAS` adds a length-`m` vector broadcast across columns; here `m = n_out`,
so the bias is per-output-feature broadcast across token columns - the intended semantics. The
bias dtype for FP16-D/32F-compute (FP16, matching stored `b_qkv`) is probed at init and
recorded; if the heuristic returns zero algorithms for the BIAS epilogue on either arch, the
fallback is a plain GEMM plus a bias add folded into the RoPE kernel (Section 16).

Heuristics are queried once per (shape, epilogue, arch) at init - during bucket warmup, never
during capture - and cached; selected algo IDs are dumped into test/bench result JSON so the
shape-equality arguments of Section 12 stay auditable. One 32 MiB workspace is shared
(allocated at init).

| GEMM     | m (= n_out) | n (= T) | k (= in) | weight (A) | Epilogue        |
|----------|-------------|---------|----------|------------|-----------------|
| qkv      | 2048        | T       | 1536     | w_qkv      | BIAS (`b_qkv`)  |
| o        | 1536        | T       | 1536     | w_o        | -               |
| gateup   | 17920       | T       | 1536     | w_gateup   | -               |
| down     | 1536        | T       | 8960     | w_down     | -               |
| lm_head  | 151936      | R       | 1536     | embed      | -               |

`T` = batch rows (decode) or chunk rows (prefill). `R` = decode batch, or 1 in the final prefill
chunk (logits are only computed for the last prompt token - computing them for all prompt
positions would waste a [2048 x 151936] GEMM per chunk).

### 6.2 Custom kernels

Shared conventions: grid dimension one block per token row unless noted; FP32 accumulation;
padded rows (`seq_len == 0`) exit immediately.

1. **embed_gather** - `x[r] = embed[input_ids[r]]`; 256 threads copy 1536 halves vectorized
   (`half2`).
2. **rmsnorm / fused_add_rmsnorm** - one block (256 threads) per row, each thread owns 6
   elements. `fused_add_rmsnorm(inp, residual, w)` reproduces HF's operation order exactly.
   HF materializes `hidden_states = residual + x` **in FP16** and then upcasts that *rounded*
   tensor for the norm, so the kernel must not normalize the unrounded FP32 sum:
   `h = fp16(fp32(residual) + fp32(inp))` (identical to a native half add; written back as the
   new residual), `h32 = fp32(h)`, accumulate `mean_sq` over `h32`, then
   `y = fp16(h32 * rsqrt(mean_sq + eps)) * w`. Both FP16 roundings are load-bearing: the
   residual round *before* the mean-square (normalizing the unrounded sum is a systematic
   drift across all 57 norm sites that kernel tolerances never catch), and the norm round
   *before* the FP16 weight multiply (Qwen2RMSNorm order). A unit case with inputs near
   half-ULP rounding boundaries pins this against an HF-operation-order reference
   (Section 12a). Plain `rmsnorm` (no add) is used only for layer 0 input.
3. **rope** - in-place on `qkv_out`, NeoX-style half rotation: pair `(d, d+64)`, `d < 64`;
   `inv_freq[d] = theta^(-2d/128)` precomputed FP32 in constant memory;
   `angle = positions[r] * inv_freq[d]`, `sincosf` (precise, matches HF's FP32 rotary path);
   applied to 12 Q heads + 2 K heads (14 heads x 64 pairs per row). HF's Qwen2 uses
   `rotate_half` (first half / second half), not interleaved even/odd - verified against a
   captured HF fixture in unit tests (Section 12).
4. **kv_scatter** - per layer: writes the row's K (256 halves) and V (256 halves) straight from
   the strided `qkv_out` views (`k` at +1536, `v` at +1792, row stride 2048) into the paged
   pool. The slot is derived **in-kernel**: `p = positions[r]`, physical block =
   `block_table_row[p >> 4]`, slot `p & 15`, then the Section 5 index formula. No host-computed
   slot-index array exists, keeping the replay uploads at the four arrays of Section 5 at the
   cost of one extra global load per token. Decode passes the device block tables with their
   normal row stride (row `r` = batch slot `r`); prefill passes the one sequence's row with
   row-stride 0 (all chunk rows share it).
5. **kv_gather** (prefill only) - inverse of scatter: copies `[0, ctx_end)` K and V for one
   sequence from paged blocks into contiguous `khat/vhat [2, ctx, 128]` per layer. Rationale:
   prefill attention then runs on dense cuBLAS GEMMs, the paged pool remains the single source
   of truth (no shadow linear cache to keep coherent), and gather bandwidth (~1 MiB per layer
   per chunk at 4K ctx) is noise next to the GEMMs. This deviates from a dual-write design on
   purpose.
6. **paged GQA decode attention** - grid `(num_seqs, 2 KV heads)`, 128 threads (4 warps),
   online softmax, FP32 accumulation. The **KV-head-centric mapping is the primary v1 kernel**,
   not a later optimization: each CTA serves the 6 Q heads of its GQA group, so every K/V tile
   streams from DRAM once instead of six times. (A per-Q-head grid `(num_seqs, 12)` carries 6x
   duplicated KV traffic - at the bench workload of 64 seqs, ctx ~1150, that is ~12.7 GB/step
   vs ~2.1 GB ideal, turning a ~2.5 ms kernel into ~15 ms and dominating the step. The dev GPU
   masks the defect entirely: at batch 8 the step is weight-bound, so it would surface only on
   paid pod time. Hence KV-centric from day one; the per-Q-head variant is still written, but
   only as a slow, obviously-correct **test oracle** for Section 12a - never on the hot path.)
   - Load the CTA's 6 Q heads from the strided `qkv_out` views into shared memory
     (6x128 FP16 = 1.5 KiB), pre-scaled by 1/sqrt(128).
   - Iterate KV blocks `j = 0 .. ceil(seq_len/16)-1` via `block_table[r][j]`; per 16-token
     tile: (a) stage the K tile (16x128 FP16 = 4 KiB) in shared memory once; warps compute the
     6x16 head/position scores by lane-strided dot + shuffle reduction, with **select-style
     masking** - `score = (pos < seq_len) ? dot : -inf` - so garbage (even NaN) in untouched
     pool slots can never contaminate a result (Section 12a tests exactly this with NaN-filled
     pools; an additive `-inf` mask would propagate NaN); (b) per-head online-softmax update of
     running max `m[h]` and sum `l[h]`, rescale `alpha = exp(m_old - m_new)`; (c) accumulate
     `acc[h][d] = acc[h][d]*alpha + sum_t p_t * V[t][d]` into a 6x128 FP32 shared accumulator
     (3 KiB) - V tile reads are coalesced and shared across the 6 heads.
   - Epilogue: `attn_out[r, h*128 + d] = fp16(acc[h][d] / l[h])` (out row stride 1536).
   - Shared-memory budget: Q 1.5 + K tile 4 + V tile 4 + acc 3 + softmax state < 16 KiB -
     comfortably inside Turing's 48 KiB/block limit on sm_75.
   - *(Shipped configuration note: the 128-thread/one-stage shape above is the pre-sm_80
     fallback as built. The measured tuning pass moved sm_80+ to 256 threads with a
     two-stage `cp.async` double-buffered pipeline and a register output accumulator -
     ~19.6 KiB of shared memory - per-arch constants in `src/kernels/paged_attention.cu`,
     measurements in `docs/PROFILING.md` sections 2.2 and 4.)*
7. **prefill softmax** - causal row softmax on `scores fp32 [12, T, ctx]`: query global position
   `q = chunk_start + i` masks keys `> q`; FP32 max/sum; writes FP16 `probs` (values in [0,1],
   safe in FP16).
8. **silu_mul** - `gateup[r,c] = silu_f32(gateup[r,c]) * gateup[r, 8960+c]`, elementwise over
   `[rows, 8960]`, written in place over the gate half (safe: each thread reads and writes only
   column `c` / reads `8960+c`); the down GEMM then reads the `[rows, 8960]` slice with
   `lda = 17920`.
9. **argmax** - grid = rows, 256 threads grid-striding over 151,936 logits (~594/thread),
   tracking `(value, index)` with **lowest-index tiebreak** to match `torch.argmax`; block
   reduction; writes `int32` token.

### 6.3 Step composition

Decode step (this exact sequence is what gets graph-captured):

```
embed_gather
for l in 0..27:
    (l==0 ? rmsnorm : fused_add_rmsnorm(mlp_out))   # produces normed, updates residual
    qkv GEMM -> rope -> kv_scatter(l) -> decode_attn(l) -> o GEMM
    fused_add_rmsnorm(attn_out) -> gateup GEMM -> silu_mul -> down GEMM
fused_add_rmsnorm(mlp_out, final_norm) -> lm_head GEMM -> argmax
```

That is ~285 launches per decode step (28 x (6 kernels + 4 GEMMs) + 5). At 3-8 us CPU launch
cost each, launch overhead alone approaches or exceeds the GPU math time of a small-batch step -
this is the quantitative motivation for CUDA graphs (Section 9).

Prefill chunk (eager only) replaces `decode_attn` with the composite:
`kv_gather -> scores GEMMs -> prefill softmax -> PV GEMMs`. The scores/PV GEMMs run as **two
strided-batched `cublasLtMatmul` calls each** (one per KV head, batch = 6 Q heads), expressed in
the same column-major dual as Section 6.1 so the lda/stride tricks are exact rather than
hand-waved:

- scores, KV head `g` - logical `S_h[T, ctx] = Q_h[T, 128] @ K_g[ctx, 128]^T`, batched over
  `h = 6g..6g+5`. Dual: `D[ctx, T] = op(A) op(B)` with `A` = `khat[g]` read as col-major
  `[128, ctx]`, `opA = T`, `lda = 128`, **batch stride 0** (all 6 heads share the K tile);
  `B` = Q head `h` straight out of `qkv_out` as col-major `[128, T]`, `opB = N`, `ldb = 2048`,
  batch stride 128; `D` = `scores + h*T*ctx` FP32, `ldd = ctx`, batch stride `T*ctx`;
  `alpha = 1/sqrt(128)`.
- PV, KV head `g` - logical `O_h[T, 128] = P_h[T, ctx] @ V_g[ctx, 128]`. Dual:
  `D[128, T] = op(A) op(B)` with `A` = `vhat[g]` as col-major `[128, ctx]`, `opA = N`,
  `lda = 128`, batch stride 0; `B` = `probs + h*T*ctx` as col-major `[ctx, T]`, `opB = N`,
  `ldb = ctx`, batch stride `T*ctx`; `D` written directly into `attn_out` at column offset
  `h*128` as col-major `[128, T]`, `ldd = 1536`, batch stride 128 - no head-to-row repack
  kernel exists in the design.

If an Lt heuristic rejects these layouts on either arch (batch-stride-0 broadcast and the
strided `D` are the risky parts), the first fallback is 6 non-batched calls per KV head (12
small GEMMs, identical descriptors minus batching); the second is explicit repack kernels.
The decision is probed and recorded per arch at init (Section 16).

Chunking: prompts are processed in `prefill_chunk` token chunks (dev preset 1024, bench preset
2048; API default is the dev value, Section 10); chunk `n`
attends to all previously scattered positions plus itself (causal). The final chunk computes
logits for the last prompt token and emits the first generated token via argmax.

Cost of the composite, quantified up front so M5 TTFT is not a surprise: the FP32 `scores` +
FP16 `probs` round-trips move ~144 MiB x 28 layers ≈ 4 GB per 1024-token chunk (~5 ms at
~850 GB/s effective), and a 2048-token chunk at 4K context moves ~32 GB ≈ 35 ms of pure
softmax traffic. A 64-prompt closed wave therefore serializes roughly 350-400 ms of
decode-stalling prefill before its last request starts decoding. Decode throughput is
insulated by the metric definitions of Section 13, but **TTFT p50/p99 is expected to lose to
fused-prefill engines** - pre-stated here and in `bench/FAIRNESS.md`, published anyway; a
fused prefill kernel is the designated post-v1 upgrade (Section 16).

## 7. KV block allocator (host)

This API - reservation-based, matched verbatim by `src/core/block_allocator.hpp` so scaffold
and design can never diverge on the core contract again - is what the scheduler is written
against. A plain `Allocate/Free` interface with watermark-only admission was rejected at
review: admission would bound only `prompt_len` while decode growth is bounded by `max_new`,
so any config with `max_batch * ceil(max_seq_len/16) > usable blocks` exhausts the pool
mid-decode with no recovery path.

```cpp
class BlockAllocator {
  std::vector<BlockId> free_;  // stack of free physical ids, init [num_blocks-1 .. 1]
                               // (block 0 = reserved dummy block, Section 9)
  std::int64_t reserved_ = 0;  // blocks promised to admitted sequences, not yet allocated
public:
  bool    CanReserve(std::int64_t n) const;  // free_.size() - reserved_ >= n
  void    Reserve(std::int64_t n);           // admission time
  void    Unreserve(std::int64_t n);         // unused remainder, on finish/abort
  BlockId AllocReserved();                   // pop free_, --reserved_   (lazy, at first use)
  void    FreeBlock(BlockId b);              // push back
};
```

Fixed-size blocks mean zero external fragmentation; alloc/free are O(1). The invariant
`free_.size() >= reserved_` guarantees that a lazily requested block always exists. Blocks are
assigned lazily: prompt blocks materialize during prefill scatter; decode allocates one new
block whenever a sequence's next position crosses a 16-token boundary. On finish, all of a
sequence's blocks return to the free list and its unused reservation is released.

## 8. Scheduler - iteration-level continuous batching

Sequence: `{id, tokens, prompt_len, n_prefilled, n_generated, max_new, ignore_eos, block_ids,
slot, state}` with states `WAITING -> PREFILL -> RUNNING -> FINISHED(eos|length|aborted)`.

**Admission** (policy flag, default `reserve_full`):

- `reserve_full`: admit the FIFO head only if `CanReserve(ceil((prompt_len + max_new)/16))`.
  Worst-case KV is reserved up front, so decode can *never* run out of blocks mid-flight -
  this is what makes "no preemption in v1" sound rather than hopeful. Cost: admission is
  conservative when callers pass large `max_new` they will not use. This deviates from a
  watermark-only seed design deliberately.
- `watermark`: admit if `(free - reserved) - ceil(prompt_len/16) >= 0.10 * usable_blocks`
  (uncommitted free against a floor over *usable* blocks = num_blocks - 1 after the reserved
  dummy block; `max_new` is deliberately irrelevant under watermark). Higher occupancy,
  but a decode step may find the pool empty; recovery is to abort the **youngest** running
  sequence - free its blocks and emit its finish with `finish_reason = kAborted`, a
  first-class reason visible through both APIs so an abort is never mistaken for a natural
  finish. Simpler than preemption and bounded. Init logs a warning when
  `max_batch * ceil(max_seq_len/16)` exceeds the usable pool - the configurations in which
  the abort path is load-bearing rather than theoretical. Documented tradeoff; benchmarks use
  `reserve_full`, whose reservation invariant makes exhaustion impossible by construction.

**Step algorithm** (one engine step = one prefill chunk OR one decode batch):

```
step():
  # admit
  while !waiting.empty() and running+prefill < max_batch and admits(waiting.front()):
      s = pop(waiting); reserve(s); assign scheduler-side slot id; state=PREFILL; push(prefill_q)
      # `slot` is scheduler-side identity only, NOT a device-buffer row - per-step device
      # rows are batch-compaction indices owned by the executor (section 6.3).

  if !prefill_q.empty():                       # prefill steps are dedicated (v1 tradeoff:
      s = prefill_q.front()                    #  running decodes stall for one chunk's
      run_prefill_chunk(s, prefill_chunk)      #  duration; bounded by prefill_chunk knob)
      if s.n_prefilled == s.prompt_len:
          emit first token (from final-chunk argmax); s.n_generated=1; state=RUNNING
          pop(prefill_q); check finish (eos/max_new==1)
      return emissions

  batch = running sequences                    # decode
  for s in batch: if next position crosses block boundary: s.block_ids += AllocReserved()
                  # watermark mode: on allocation failure, abort youngest (emit kAborted)
  fill pinned mirrors; 4 batched H2D copies (ids, positions, seq_lens, block-table slab), Sec. 9
  bucket = smallest b in buckets with b >= |batch|
  graphs enabled and bucket exists ? replay_graph(bucket) : run_decode_eager(|batch|)
  download argmax_out; for each s: append token; emit (id, token, finished?)
  finished: free blocks, unreserve remainder, release slot
  return emissions
```

Stop conditions: token in the EOS set (from `generation_config.json`) - unless the request set
`ignore_eos`, which skips the membership check so benchmark runs can force exact output lengths
(Section 13; generated EOS tokens still appear in the output stream) - or
`n_generated == max_new`. No reordering (FIFO admission), no preemption. A request with an
empty prompt, a non-positive `max_new`, or `prompt_len + max_new > max_seq_len` (or that can
never fit the pool) is rejected at `add_request` with an exception, not queued forever.

## 9. CUDA graphs (decode only)

Buckets: `{1,2,4,8,16,32,48,64}` capped at `max_batch` (dev preset uses `{1,2,4,8}`). Prefill
and any unexpected shape run eager; `enable_cuda_graphs=false` disables replay entirely (the
eager path is permanently maintained and tested equal, Section 12).

**Capture (at init, after model load):**

1. Warm up: run one eager decode step per bucket with dummy-but-valid inputs (each slot:
   `seq_len=1`, block table pointing at one sacrificial "dummy block"). This forces cuBLASLt
   algo selection, workspace binding, and module loading *before* capture - Lt heuristic
   queries and any lazy allocation inside a capture region are bugs.
2. For each bucket `b`: write dummy inputs, `cudaStreamBeginCapture(s0, ThreadLocal)`, enqueue
   the exact decode sequence of Section 6.3 with `T=b` reading only persistent buffers,
   `cudaStreamEndCapture`, `cudaGraphInstantiate`. Store `exec[b]`.

**Replay (per decode step):**

1. Write this step's values into the **pinned host mirrors** (Section 5), then issue exactly
   four H2D `cudaMemcpyAsync` calls on `s0` - one contiguous copy per array: `input_ids`,
   `positions`, `seq_lens`, and the block-table slab as a **single**
   `[bucket, max_blocks_per_seq]` copy (64 KiB at bench max, ~6 us) rather than up to 64
   per-changed-row copies. This rule is load-bearing: async copies from pageable memory
   silently degrade to staged, partially synchronous transfers, and dozens of tiny copies each
   cost ~5-10 us of API/queue time - precisely the host-gap metric Section 14 says graphs must
   drive to ~0 (worse still under WSL2). Padded slots get `seq_len=0`, `position=0`,
   `token=0`, block-table row filled with the dummy block.
2. `cudaGraphLaunch(exec[bucket], s0)`.
3. `cudaMemcpyAsync` D2H of `argmax_out[0..bucket)` into its pinned mirror +
   `cudaStreamSynchronize(s0)` - the scheduler needs tokens on host every iteration; this sync
   is the step boundary.

**Sentinel policy (one policy everywhere):** physical block 0 is reserved at init as the dummy
block - it never enters the allocator's free list, so every id that can reach a device block
table resolves to an in-bounds address, even for padded rows/slots. `kInvalidBlockId` (-1) is
a host-side-only debug sentinel and must never be written into a device table
(debug-asserted at mirror-fill time). Kernels still skip padded rows on `seq_len == 0`; the
dummy block is belt-and-suspenders against stray address math, not a correctness dependency.

Because every kernel argument is a *pointer to a persistent buffer* and only buffer *contents*
change between steps, graphs never need re-instantiation or `cudaGraphExecUpdate`; shape
variation is entirely absorbed by bucket padding. Rules that make this safe: no allocation
after init; single stream; no host callbacks or NVTX inside the captured region (NVTX wraps the
replay call instead: `graph_replay[b=8]`).

Wasted-work bound from padding: the worst ratio is 17 live rows padded to bucket 32 (1.88x
GEMM rows); attention cost scales with live tokens only (padded rows exit immediately) -
accepted, and observable via the `stats()` bucket histogram.

## 10. API surface

Python (`src/api/pybind.cpp`, module `redline`). Defaults are the **dev preset** (Section 11):
the smallest supported GPU must be safe by default, and the bench harness always passes its
full configuration explicitly (FAIRNESS.md lists it):

```python
Engine(model_dir: str,
       kv_pool_gb: float = 1.0,
       max_batch: int = 8,
       enable_cuda_graphs: bool = True,
       max_seq_len: int = 2048,
       prefill_chunk: int = 1024,
       admission_policy: str = "reserve_full")   # or "watermark"

Engine.add_request(token_ids: list[int], max_new_tokens: int,
                   ignore_eos: bool = False) -> int   # req_id; raises on empty prompt,
                                                      # max_new <= 0, or oversize prompt
Engine.step() -> list[tuple[int, int, bool, str]]  # (req_id, token_id, finished, finish_reason
                                                   #  in {"", "eos", "length", "aborted"});
                                                   #  may be empty; GIL released for the call
Engine.has_capacity() -> bool                      # waiting queue below cap (1024): pure
                                                   # backpressure; KV admission happens inside
                                                   # step()
Engine.stats() -> dict   # total_blocks, free_blocks, reserved_blocks, waiting, running,
                         # steps, prefill_tokens, decode_tokens, graph_replays,
                         # eager_decodes, bucket_histogram, aborts
```

`ignore_eos` exists so the benchmark harness can force exact output lengths (Section 13) -
without it, the FAIRNESS.md forced-length contract would be unimplementable for Redline itself.
`step()` uses `py::call_guard<py::gil_scoped_release>()`. The typical driver is a pump loop:
submit while `has_capacity()`, call `step()` until all requests finish.

CLI (profiling driver - published cross-engine numbers come from the Python harness only,
Section 13):

```
redline_bench --model DIR [--kv-gb 1.0] [--max-batch 8] [--requests 64]
              [--prompt-len 1024] [--gen-len 256] [--graphs 1] [--ignore-eos 1]
              [--seed 0] [--json out.json]
```

Generates the same splitmix64 fixed-seed synthetic token IDs as `bench/workload.py` (algorithm
normative in `bench/FAIRNESS.md`; numpy and C++ `<random>` streams would never match), pumps
the engine, prints and optionally writes JSON: output token throughput, TTFT, ITL p50/p99,
step-time stats, `stats()` dump - metric definitions identical to Section 13.

## 11. Memory budgets

Formulas: weights 2944 MiB (Section 1); KV pool = `num_blocks * 448 KiB`; scratch from Section 5
sized by `(chunk, max_seq_len, max_bucket)`; scores is the dominant scratch term:
`12 * chunk * max_seq_len * 4 B` (FP32) + half that again for FP16 probs.

**Dev preset (RTX 2060 6 GiB)** - `kv_pool_gb=1.0, max_batch=8, max_seq_len=2048,
prefill_chunk=1024, buckets {1,2,4,8}`:

| Item                                | MiB  | Arithmetic                                    |
|-------------------------------------|------|-----------------------------------------------|
| CUDA context + cuBLASLt images      | ~512 | measured at M2; budgeted conservatively       |
| Weights FP16                        | 2944 | Section 1                                     |
| KV pool (2340 blocks = 37,440 tok)  | 1024 | floor(1 GiB / 448 KiB) = 2340                 |
| scores FP32 + probs FP16            | 144  | 12*1024*2048*(4+2) B                          |
| gateup + qkv + norm/residual bufs   | ~54  | 1024*17920*2 = 35; 1024*2048*2 = 4; 5x3       |
| kv gather + logits + inputs         | ~5   | 2 + 8*151936*2 = 2.3 + <1                     |
| cuBLASLt workspace                  | 32   | fixed                                         |
| CUDA graphs (4 buckets)             | 64   | budget; measured at M3                        |
| **Total**                           | **~4780** | vs ~5300 usable (display steals up to ~900) |

Headroom ~0.5 GiB. `kv_pool_gb` up to 1.5 still fits but erases margin; init performs a
`cudaMemGetInfo` check and fails fast with the computed budget if the config cannot fit.

**Bench preset (RTX 4090 24 GiB)** - `kv_pool_gb=8, max_batch=64, max_seq_len=4096,
prefill_chunk=2048, buckets {1..64}`:

| Item                                  | MiB   | Arithmetic                                  |
|----------------------------------------|-------|---------------------------------------------|
| CUDA context + libs                   | ~512  |                                             |
| Weights FP16                          | 2944  |                                             |
| KV pool (18,724 blocks = 299,584 tok) | 8192  | floor(8 GiB / 448 KiB) = 18,724             |
| scores + probs                        | 576   | 12*2048*4096*(4+2) B                        |
| gateup/qkv/hidden + logits + gather   | ~131  | 70 + 8 + 30 + 64*151936*2 = 18.6 + 4        |
| workspace + graphs (8 buckets)        | 160   | 32 + 128 budget                             |
| **Total**                             | **~12,500** | vs ~24,000 - >11 GiB headroom          |

Primary benchmark workload check: 64 seqs x 1280 tokens (1024 in + 256 out) =
64 x ceil(1280/16) = 5120 blocks = 2240 MiB - 27% of the 8 GiB pool, so admission never gates
the primary workload even under `reserve_full`.

## 12. Testing

**(a) Kernel units (gtest, run on GPU, compared to double-precision CPU references).** The
reference computes the same math in FP64 on FP16-quantized inputs. Shapes sweep block-boundary
edges: seq_len in {1,15,16,17,63,64,65,257}, batch in {1,3,8} with padded rows present. Initial
tolerances (tightened empirically once measured): elementwise kernels (rmsnorm, rope, silu_mul,
scatter/gather round-trip) max-abs 1e-2 / mean-abs 1e-3; attention (decode kernel and prefill
composite) max-abs 2e-2; argmax exact including crafted-tie inputs (lowest index wins). RoPE is
additionally validated against a fixture of Q/K tensors captured from HF's rotary implementation
for fixed inputs - this pins the NeoX-vs-interleaved decision to evidence. Further mandatory
units: (1) **BF16->FP16 conversion, exhaustive** - all 65,536 BF16 bit patterns against a
torch-generated golden table (round-to-nearest-even, subnormals preserved, overflow hard-fails;
Section 4); (2) **fused_add_rmsnorm rounding order** - inputs near half-ULP boundaries against
an HF-operation-order reference (asserts the FP16-rounded-residual order of Section 6.2);
(3) **NaN-poisoned pool attention** - every pool slot the test did not write is pre-filled with
NaN bit patterns; the select-style masking of Section 6.2 passes, while an additive-mask
implementation converts uninitialized pool memory into NaN outputs and fails loudly here
instead of generating silently wrong tokens in production; (4) **oracle cross-check** - the
KV-head-centric decode kernel against the naive per-Q-head oracle kernel on randomized shapes
(the oracle exists only for tests). Every launcher takes explicit row strides, so units cover
both packed (`stride == width`) and `qkv_out`-strided (`stride == 2048`) inputs.

**(b) Allocator + scheduler units (CPU-only, mock executor).** The mock returns scripted tokens
(including scripted EOS positions). Asserted: reservation arithmetic (`CanReserve/Reserve/
Unreserve/AllocReserved`) for both admission policies; lazy block allocation at 16-token
boundaries; block/reservation release on finish; free-list conservation after full drain
(`free == usable total`); bucket selection sequence; emission ordering; the `watermark` abort
path emitting `kAborted` for the youngest sequence; rejection of oversize, empty-prompt, and
non-positive-`max_new` requests; EOS on the *first* generated token and `max_new == 1`
finishing straight out of the PREFILL state (the release path must free prompt blocks and
unreserve the remainder); `ignore_eos` suppressing the EOS stop but never the length stop; and
a randomized admit/finish/abort churn property test asserting
`free + outstanding == usable total` plus reservation consistency, with request sizes placed at
`prompt_len + max_new ≡ 0, 1 (mod 16)` boundaries.

**(c) End-to-end HF parity (Python, dev GPU).** `tests/e2e/gen_reference.py` runs HF
`transformers` FP16 on 20 prompts - 5 short (50-100 prompt tokens), 10 medium (100-300), 5 long
(300-500); plain-text continuations tokenized without chat template so both sides consume
identical IDs; 128 new tokens each. **The generation config is neutralized and asserted**
(MODEL_SPEC F6): `do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None,
top_k=None` - the shipped `generation_config.json` sets `repetition_penalty=1.1`, which HF
applies *even when* `do_sample=False`, so a reference generated with `do_sample=False` alone is
not greedy and can never match. `attn_implementation` is pinned explicitly and recorded. The
script caches generated tokens, per-step top-k logits and top1-top2 margins to JSON keyed by
(model revision, GPU name, torch/transformers versions, attn_implementation), and runs **as a
separate OS process before** the Redline process so both fit 6 GiB sequentially (never one
interpreter). Pass criteria, in order of authority:

1. **Teacher-forced agreement (primary).** Feed HF's generated token at every step; compare the
   engine's argmax at all 128 positions per prompt. Bar: >= 99% position-wise agreement over
   all 2,560 positions, and every disagreement must sit on a documented FP16 coin-flip - HF
   top1-top2 logit margin < 0.1 at that position. A disagreement with a larger margin fails the
   suite regardless of the aggregate rate. Teacher forcing removes divergence compounding, so
   one early coin-flip cannot flake an entire prompt (the failure mode of a free-running
   prefix-match bar), while still checking behavior at every position after any divergence
   (which a prefix metric never sees).
2. **Free-running greedy prefix match (secondary, reported, not gating).** Per-prompt matched
   prefix length under self-fed decoding; every divergence reported with position, both token
   IDs, and the HF margin.
3. **First-token check.** Top-5 token sets of the first step must match, and where the argmax
   differs the HF margin must be < 0.1. (Replaces a scale-free logits max-abs-diff <= 0.5
   bound, which exceeds typical top-2 margins and could pass while tokens differ.)

Expected mismatch sources are FP16 reduction-order differences (different GEMM tiling than
cuBLAS-via-torch); the margin gate keeps them honest.

Every run of the suite writes an **auditable receipt JSON** on teardown (`$REDLINE_PARITY_OUT`,
default `./hf_parity_report.json` - the same report convention as suites (d)/(e)): agreement
counts and rate, coin-flip vs fatal disagreement tallies, first-token and echo-tripwire
outcomes, thresholds, reference identity (path, neutralization record, position count), and
the engine module version/kwargs. A run that backs a published parity claim commits its
receipt under `bench/results/parity/` - the suite's headline number must never exist only in
a terminal summary or a commit message. *(Provenance note: the committed receipt backing the
parity claim is `bench/results/parity/hf_parity_rtx2060_dev.json` - a fresh, complete run of
this suite (teacher-forced agreement 2547/2560 = 99.49%, 13 coin-flip disagreements all below
the 0.1 margin, zero fatal; first-token 20/20; echo tripwire passed) on the sm_75 dev GPU
(RTX 2060, WSL2) under the dev-preset engine defaults, against a reference regenerated in the
same session. That receipt is dev-GPU, dev-preset evidence: the bench-preset re-runs of this
suite ran on the sm_89 pod GPU in the benchmark sessions, which predate the receipt mechanism,
so the committed cross-engine correctness evidence for the published benchmark rows remains
the equivalence-gate transcripts under `bench/results/equivalence/`. The earlier sm_75
bring-up agreement result also predates the mechanism and has no artifact; the committed
receipt supersedes it.)*

**(d) Paged/batching invariance.** Byte-identical token equality across configurations is only
sound when both sides run **identical GEMM shapes**: a different `m` lets cuBLASLt pick a
different algo/split-K, changing FP32 reduction order and legitimately flipping near-tie tokens
- which would send the suite hunting paging bugs that are actually numerics. The tests
therefore exploit the engine's own bucket padding: the same prompt greedy-decoded alone but
**padded to bucket 8** must produce byte-identical tokens to the run where it is one of 8
concurrent sequences (7 filler prompts). Identical shape ⇒ same cached algo ⇒ each row's result
is independent of other rows' contents ⇒ any mismatch is definitively a
block-table/scatter/attention bug; no triage protocol, no fixture replacement. Variants:
fragmented pool (allocator pre-churned so physical blocks are non-contiguous) and boundary
lengths {15,16,17}. The selected Lt algo per (shape, arch) is recorded in the test output JSON,
keeping the shape-equality premise auditable. A true batch-1 (unpadded) vs batch-8 comparison
is additionally *reported* with logit margins as an informational cross-shape check; it does
not gate.

**(e) Graph equivalence.** With identical requests, `enable_cuda_graphs` on vs off must emit
identical tokens. The eager side of each gating comparison runs padded to the same bucket the
graph used (same-shape rule of (d)), exercised across bucket sizes; the unpadded eager run is
reported informationally.

**(f) Layer-0 parity fixture (GPU, before any e2e debugging).** For one fixed prompt, HF
activations are captured after layer 0's `input_layernorm`, after its attention block output,
and after its MLP output, then compared stage-by-stage against the engine's buffers. This
localizes packing mistakes - K rows swapped with V inside `w_qkv`, a bias applied on the wrong
axis, a `w_gateup` concat off-by-one - that otherwise produce fluent-looking garbage
diagnosable only by bisecting 28 layers on a 6 GiB card. The fused-weight layouts and the BIAS
epilogue are validated *directly against HF numbers* here, not only against unit references.

## 13. Benchmark methodology (bench/)

`bench/FAIRNESS.md` is the complete, reviewer-facing contract; it restates every rule below
normatively and ships with zero open policy TODOs. This section is the engineering summary.

Engines, all serving the same model in FP16 with greedy decoding on the same pod GPU:

- **redline** - driven by the same Python harness as the baselines (client-path rule below).
- **vLLM** - **latest stable release at the benchmark lock date** (exact version pinned in
  `bench/requirements.lock` at M5 and re-recorded into every results JSON). A design-time pin
  would be stale by measurement time - "you benchmarked a year-old vLLM" is the standard fatal
  rebuttal, and the legacy pre-V1 `AsyncLLMEngine` API this document once named has been
  removed upstream. *(Measured outcome: the original comparison rows deviated from this
  policy - `vllm==0.9.2` was measured while 0.24.0 was the latest stable at lock; a second
  formal `vllm==0.24.0` arm was then measured under the full per-arm gate-first protocol
  and is published alongside the 0.9.2 rows; the two-arm disclosure is recorded in
  `bench/FAIRNESS.md` "Engine configuration".)* The adapter targets the V1 engine's public APIs: offline
  `LLM.generate` for the client-overhead cross-check, the V1 async engine's streaming API for
  per-token timestamps. FP16, CUDA-graph execution on (its default mode), `ignore_eos=True`,
  prompts passed as token IDs; full config listed, including the compilation/cudagraph mode
  chosen by the tuning pass below.
- **llama.cpp** - pinned commit; FP16 GGUF produced by `convert_hf_to_gguf.py --outtype f16`
  (conversion command + file hash documented in FAIRNESS.md); `llama-server` with `-ngl 99`,
  flash attention on, `--parallel 64 --cont-batching`, and - mandatory context arithmetic -
  **`-c >= 64 * (1024 + 256) = 81920`** for the primary case (`llama-server` splits `-c`
  across parallel slots; the default would truncate every prompt and the workload simply would
  not run). `-b`/`-ub` batch sizes set by the documented tuning sweep. Driven over its HTTP
  streaming API by the same async client; `n_predict` fixed per request; prompts passed as
  token arrays (verified at M5 - any fallback must preserve exact token counts, FAIRNESS.md).
  A `llama-batched-bench` run cross-checks that HTTP overhead is not distorting throughput.

**Client-path rule.** Every published cross-engine number comes from the same Python harness
(`bench/run_bench.py`), same asyncio client structure, so measured-path overhead is shared
rather than engine-specific. The pure-C++ `redline_bench` CLI is a profiling tool only; its
numbers never appear in a cross-engine table. Client overhead is bounded and reported per
baseline: vLLM streaming adapter vs an offline `LLM.generate` run; llama.cpp HTTP vs
`llama-batched-bench`.

**Baseline best-configuration duty.** "Each engine on its best supported execution path" is
demonstrated, not asserted: at M5 each baseline gets a bounded, documented tuning pass - vLLM:
default vs full-CUDA-graph compilation mode; llama.cpp: `-b`/`-ub` sweep - with the best
result published and the losing configs committed in the raw results.

**Workload** (fixed seed **0** - the single seed used by workload.py, run_bench.py, and
redline_bench): synthetic prompts with exact token counts. The token generator is specified as
an explicit algorithm so C++ and Python produce bit-identical streams (numpy vs `<random>`
never would): `token_id(seed, req, pos) = 1000 + splitmix64((seed << 40) ^ (req << 20) ^ pos)
mod 99000` - uniform over [1000, 100000), far below the special-token range (>= 151643).
splitmix64 constants and reference code are normative in FAIRNESS.md; implementations live in
`bench/workload.py` and `src/bench_main.cpp`.

Cases (all greedy, all with EOS ignored via each engine's mechanism - Redline `ignore_eos`,
vLLM `ignore_eos=True`, llama.cpp fixed `n_predict`; the harness asserts every request
produced exactly its forced output length on every engine):

| case      | requests | in/out    | arrival                         | purpose                        |
|-----------|----------|-----------|---------------------------------|--------------------------------|
| `primary` | 64       | 1024/256  | all at t=0 (closed single wave) | headline throughput            |
| `arrival` | 64       | 1024/256  | request i at t = i * 100 ms     | prefill/decode interference    |
| `longout` | 64       | 1024/1024 | all at t=0                      | ~4x wider decode window        |
| `batch1`  | 1        | 1024/256  | t=0                             | single-stream latency          |

The closed single wave is the *easiest* regime for a dedicated-prefill scheduler (all prefill
front-loaded, no mid-decode admissions, uniform finishes) - which is exactly why `arrival`
exists: it drives arrivals into running decode so the Section 16 row-1 stall shows up in ITL
p99 instead of being hidden by workload shape. Every published claim is scoped to its case.
One unmeasured warmup trial per engine/case, then 3 measured trials.

Metrics (client-side wall-clock timestamps per token; formulas normative in FAIRNESS.md):

- `TTFT_i = t_i,0 - t_i,submit`; report p50/p99 across requests. Redline's composite prefill
  is expected to lose TTFT to fused-prefill baselines (quantified in Section 6.3); the number
  is published anyway.
- `ITL` = successive token deltas within a request, first token excluded; p50/p99 pooled.
- **Output token throughput** (headline) = `sum_i n_i / (max_i t_i,last - min_i t_i,submit)`
  - total generated tokens over the full wall window, **including all prefill work**, and
  labeled as such. (An earlier draft claimed a window opening at the first generated token
  "excludes prefill by construction" - false: with 64 concurrent requests, ~63/64 of all
  prefill executes inside that window. No published metric carries a "prefill excluded" label
  unless it is computed purely from within-request windows.)
- **Per-request decode rate** (secondary; genuinely prefill-free by construction) =
  `(n_i - 1) / (t_i,last - t_i,0)` per request - the reciprocal of its mean ITL - reported as
  a distribution across requests. Other requests' prefills stalling a decode lower this
  number; that is measured interference, not noise.
- Report the median trial **with min-max**; dispersion rule: if (max − min) > 3% of the median
  on a headline metric, re-run the case once and flag the number if the spread persists.
  Commit all raw per-token JSON under `bench/results/`.

**Cross-engine output-equivalence gate.** Before any measured trial, the same 5 short greedy
prompts run through all three engines; token streams must match up to documented FP16
coin-flips (logit-margin protocol of Section 12c). This is the evidence that all three engines
compute the same function on the gate workload (5 prompts, 64 in / 32 out, batch 5 - not the
measured 1024-token/batch-64 shape; the binding scope note in `bench/FAIRNESS.md` "Model and
inputs" records what covers that gap and why a same-shape gate would need a pod session) - a
silently broken GGUF conversion or a misconfigured baseline is either a strawman or an
inflated target, and "all engines served the same model" must not be an unverified assertion.
The transcript is committed next to the results JSON.

Environment capture per run: full `nvidia-smi -q` (driver, clocks, power limits, temperature),
`lscpu` + total RAM (client-side timestamps are CPU-sensitive on rented pods), kernel version,
`nvcc`/`gcc` versions, `pip freeze`, engine commit/version, exact command lines - embedded in
the results JSON. GPU clocks are locked via `nvidia-smi -lgc` when the pod permits; if not,
that is recorded. `bench/report.py` renders results JSON into the README markdown table; no
number appears in the README that is not generated from a committed results file, and the
table caption scopes results to this model size and these workloads (a 1.5B FP16 model at high
batch is the regime where host-side per-step overhead matters most - favorable to a lean C++
engine - so no claim of general superiority is made).

## 14. Profiling

`scripts/profile_decode.sh`: `nsys profile --trace=cuda,nvtx` around a 500-step decode run of
`redline_bench` after warmup. NVTX taxonomy: `step`, `prefill[len,chunk]`, `decode[b]`,
`graph_replay[b]`, `sample`, `h2d_inputs`, `d2h_tokens`. `docs/PROFILING.md` skeleton: (1) env
and method, (2) kernel-time breakdown table per decode step, (3) timeline gap analysis (host
gaps between GPU work - the metric CUDA graphs must drive to ~0; the pinned-mirror batched-copy
rule of Section 9 is what makes ~0 reachable), (4) decode-attention kernel time vs its
DRAM-traffic bound, (5) prefill cost accounting against the Section 6.3 expectation, (6) top-3
actions with before/after measurements. Kernel-level tuning decisions must land with an
nsys-backed number in this file.

## 15. Milestones

- **M0 - Scaffold.** CMake+Ninja builds `redline_core`, module, CLI, and tests for sm_75+sm_89
  from a fresh WSL2 clone; FetchContent deps pinned; `python -c "import redline"` works;
  a trivial gtest passes via ctest; `scripts/build.sh` documented in README.
- **M1 - Core authored.** Safetensors parser (unit-tested against a small synthetic fixture
  file), weight prep (BF16->FP16 conversion passing the exhaustive golden-table unit),
  allocator (reservation API), scheduler, all kernels and both execution paths written;
  suite (b) fully green on CPU; kernel gtests compile and run (may be red pending device
  debugging). No pod time used.
- **M2 - Correctness on RTX 2060.** Suites (a), (c), (d), (f) green on sm_75 under the dev
  preset; peak VRAM during (c)/(d) logged and < 5.2 GiB; loader validates config and rejects
  mismatches; single-request generation produces coherent text end-to-end.
- **M3 - Batching + graphs bring-up (still 2060).** Multi-request continuous batching runs;
  suite (e) green (graphs == eager tokens across buckets {1,2,4,8}); `redline_bench` completes
  the dev workload (8 x 512/128) without allocation after init (verified by absence of
  `cudaMalloc` in an nsys trace); NVTX ranges visible in nsys.
- **M4 - Performance on RTX 4090.** All correctness suites re-run green on sm_89 under the
  bench preset; buckets through 64 captured; decode-attention kernel time validated against
  its DRAM-traffic bound with an nsys capture recorded in PROFILING.md (the KV-head-centric
  kernel is primary from M1 - Section 6.2 - so M4 verifies the bound rather than rescuing a
  6x-traffic kernel on paid pod time); profiling loop iterated until the decode timeline shows
  no dominant unexplained host gap; step-time histogram stable across 3 runs.
- **M5 - Benchmark + publish.** vLLM version locked (latest stable at lock date; shipped
  outcome deviated - `vllm==0.9.2`, disclosed in `bench/FAIRNESS.md`) and
  `requirements.lock` written; cross-engine output-equivalence gate passed with the transcript
  committed; per-baseline tuning passes documented; harness runs all three engines on the same
  pod session; FAIRNESS.md final with zero open TODOs; 3-trial medians (with min-max) for all
  four cases committed as JSON; README table generated from results; PROFILING.md gap analysis
  comparing Redline's decode timeline against its own eager mode and discussing baseline
  deltas; repository made public.

## 16. Known risks & fallbacks

| Risk | Consequence | Mitigation / fallback |
|---|---|---|
| Composite (cuBLAS) prefill attention is slow vs fused-kernel engines | Worse TTFT (~350-400 ms serialized prefill for the 64-prompt wave, Section 6.3); decode stalls during dedicated prefill steps | Accepted for v1 (correctness-first); cost pre-quantified in Section 6.3 and pre-stated in FAIRNESS.md; `prefill_chunk` bounds the stall; the `arrival` case measures the interference; a fused prefill kernel is the designated post-M5 upgrade, interface already isolated |
| CUDA graph capture fails (capture-illegal call, Lt edge case, WSL2 driver quirk) | No graph path | Eager fallback behind `enable_cuda_graphs=false` is a permanent, tested code path; capture failure degrades to eager with a warning, never aborts |
| cuBLASLt heuristic returns zero algorithms for a required config - BIAS epilogue on the qkv shape, or the strided-batch tricks of Section 6.3 - on one arch | First-written GEMM wrapper or prefill attention unavailable | Every (shape, epilogue) combination probed at init, before capture, decision recorded per arch. qkv fallback: plain GEMM + bias add folded into the RoPE kernel. Prefill fallback chain: 6 non-batched per-head GEMMs, then explicit repack kernels (small, unit-testable) |
| sm_75 feature gap: no `cp.async`, smaller shared-mem/SM, different Lt algo space | Kernel configs that work on sm_89 fail or underperform on sm_75 | Tuning constants templated per `__CUDA_ARCH__`; v1 kernels stay within Turing limits (<=48 KiB smem/block, no async copies); Lt heuristics cached per arch |
| BF16 -> FP16 overflow on conversion | Corrupted weights, or silent divergence from the HF reference if clamped | No clamping: hard fail at load (expected overflow count: zero - Section 4); conversion exhaustively unit-tested against a torch golden table (RNE, subnormals preserved) |
| FP16 nondeterminism (reduction order) flips greedy tokens | e2e/batch-invariance flakiness | Gating equality tests run under the same-GEMM-shape (bucket-padding) protocol of Section 12(d/e), where bit-equality is sound; cross-shape comparisons are informational with logit margins; e2e uses the teacher-forced margin-gated metric of Section 12(c) |
| Naive per-Q-head decode attention looks healthy on the dev GPU | Batch-8 dev steps are weight-bound, so 6x duplicated KV traffic would surface only at bench batch on paid pod time (~15 ms vs ~2.5 ms attention) | KV-head-centric kernel is the primary v1 implementation (Section 6.2); the per-Q-head variant exists only as a test oracle; M4 verifies the DRAM-traffic bound |
| `watermark` admission exhausts pool mid-decode | A running sequence cannot grow | Default policy is `reserve_full` (impossible by invariant); watermark mode aborts the youngest sequence deterministically, emitting `kAborted`; init warns when the config makes the abort path load-bearing |
| 2060 VRAM shared with Windows display via WSL2 | Init OOM or noisy dev timings | `cudaMemGetInfo` preflight with computed budget and actionable error; `kv_pool_gb` down-tunable; dev timings treated as non-authoritative - only pod numbers are published |
| Batch-1 long-context decode underutilizes SMs (2 blocks for 2 KV heads per seq) | Poor batch-1 ITL at long ctx | Out of primary-workload scope; flash-decode-style split-K reduction is the documented follow-up if secondary-workload profiling shows it matters |
| Graph memory growth across 8 buckets | VRAM budget overrun | Measured at M3 against the 128 MiB budget; bucket set is config-trimmable (e.g. drop 48) |
| Baseline version drift (vLLM, llama.cpp commit) | Non-reproducible or strawman comparison | Policy: vLLM pinned to the latest stable release at benchmark lock date (M5), never a design-time version (*shipped outcome: `vllm==0.9.2` measured first, 0.24.0 latest at lock - a formal `vllm==0.24.0` arm was then measured gate-first and published alongside; two-arm disclosure in FAIRNESS.md*); llama.cpp pinned by commit; exact pins in `bench/requirements.lock` + FAIRNESS.md; versions re-recorded into every results JSON at runtime |
| RoPE convention wrong (interleaved vs half-rotation) | Garbage generations, subtle | Pinned by HF-captured fixture test before any e2e debugging; single switch point in the kernel |

## 17. References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, SOSP 2023 - source of the paged-KV/block-table concept used in Sections 5-8.
- Qwen2.5 model card and `config.json` / `generation_config.json` (Hugging Face,
  `Qwen/Qwen2.5-1.5B-Instruct`) - constants mirrored in `docs/MODEL_SPEC.md`.
- NVIDIA CUDA 12.6 documentation: cuBLASLt, CUDA Graphs, Nsight Systems, NVTX v3.
- safetensors format specification (header layout, dtype encodings).

## Appendix A - pre-implementation review notes (2026-07)

Three independent design reviews ran against this document, `docs/MODEL_SPEC.md`, and the
scaffold before M1. All blocking and major findings were folded directly into the sections
above (GEMM formulation, kernel stride contracts, allocator/scheduler reservation contract,
pinned-mirror uploads, KV-head-centric attention, rmsnorm rounding order, BF16 conversion
policy, HF-reference neutralization, teacher-forced parity metric, same-shape determinism
protocol, `ignore_eos`, benchmark metric/pinning/client-path/workload rules). Minor findings
and their dispositions:

| # | Finding | Disposition |
|---|---|---|
| A1 | Scaffold/doc drift: engine header said "per-layer KV pools" vs the single layer-outer allocation; `EngineOptions` defaults (1.5 GiB pool, 256 blocks/seq) contradicted the dev preset; device block-table padding said `kInvalidBlockId` vs Section 9's dummy block | Fixed. Scaffold matches Sections 5/9/11: single pool allocation, dev-preset defaults (1.0 GiB, `max_seq_len` 2048 -> 128 blocks/seq, chunk 1024), dummy-block device padding with `kInvalidBlockId` demoted to a host-only debug sentinel (Section 9 sentinel policy) |
| A2 | API drift: pybind ctor missing `max_seq_len`/`prefill_chunk`/`admission_policy`; `step()` dropped `finish_reason`; no `kAborted`; `EngineStats` missing graph/bucket/abort counters; `has_capacity` semantics diverged (watermark vs queue cap) | Fixed in scaffold + Section 10: `step()` returns `finish_reason`, aborts are first-class, stats carry `graph_replays`/`eager_decodes`/`bucket_histogram`/`aborts`, `has_capacity` = queue-depth backpressure only |
| A3 | CI built only `redline_core` (bench/tests/pybind stubs could rot uncompiled) and `gtest_discover_tests` executed the test binary at build time - breaks GPU-less builders if any CUDA init leaks into static scope | Fixed: CI builds all targets (PYTHON=OFF) and runs `ctest`; discovery uses `DISCOVERY_MODE PRE_TEST` |
| A4 | Accepted composite-prefill cost carried no number, so M5 TTFT would read as a surprise | Fixed: cost model in Section 6.3, expectation pre-stated in Section 13 and FAIRNESS.md, PROFILING.md gains a prefill-accounting section |
| A5 | 3-trial median had no dispersion rule | Fixed: median reported with min-max plus the 3%-spread re-run/flag rule (Section 13, FAIRNESS.md, report.py) |
| A6 | No scope statement for the 1.5B regime (host-overhead-dominated - the friendliest regime for a lean C++ engine) | Fixed: scope statement required in FAIRNESS.md and the README table caption; no generalization claimed |
| A7 | llama.cpp token-ID fallback underspecified - a detokenize/retokenize fallback can silently change prompt lengths | Fixed: FAIRNESS.md prohibits any fallback that does not hard-assert exact token-ID-sequence equality |

Where reviews proposed different fixes for the same defect, the chosen resolution and
rationale:

- **kv_scatter slot input** - add a fifth per-step `slot_indices` upload vs derive slots
  in-kernel from `positions` + block table. Chosen: in-kernel derivation (Sections 5, 6.2). It
  keeps the replay upload set at four arrays, matches the index math this document already
  specifies, and costs one extra global load per token.
- **Benchmark seed** - 42 (this document, previously) vs 0 (workload.py, run_bench.py,
  bench_main.cpp). Chosen: 0. The value is arbitrary; three of four sources already agreed.
- **batch1 case shape** - 128/128 (this document, previously) vs 1024/256 (FAIRNESS.md,
  workload.py). Chosen: 1024/256, so the latency case isolates concurrency effects against
  `primary` rather than changing two variables at once.
- **Python API defaults** - this document previously documented bench-preset defaults (8 GiB
  pool, batch 64) while the scaffold used ad-hoc values. Chosen: dev-preset defaults
  (Section 10); defaults must be safe on the smallest supported GPU, and the harness always
  passes explicit config.
- **Headline throughput metric** - whole-window output throughput (labeled as including
  prefill) vs an ITL-derived decode-only rate. Chosen: both, with the whole-window number as
  the headline (standard and honest) and the per-request decode rate as the explicitly
  prefill-free secondary; the former "prefill excluded by construction" claim was removed as
  mathematically false (Section 13).
- **Watermark admission soundness** - one review suggested an init-time hard invariant
  (`max_batch * ceil(max_seq_len/16) <= usable blocks`) as an alternative to an abort path.
  Chosen: keep the abort-youngest path (a hard invariant would forbid legitimate
  high-occupancy configs) and add an init-time *warning* when the invariant does not hold,
  plus first-class `kAborted` reporting so aborts are observable (Section 8).
