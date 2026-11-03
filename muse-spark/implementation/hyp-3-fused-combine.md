# Hypothesis 3: fused combine kernel

- **Owner:** lead
- **Status:** building
- **Version:** 2
- **Updated:** 2026-11-03T21:55:00Z
- **Next:** builders 3-a (single kernel, thread-per-output grid) and 3-b (single kernel, threadgroup-per-row with staged weights) competing

## Inefficiency

Slot `combine`. One entry:

- `ineff-model-2` (rank 5, mean 4.3% of step): each layer's weighting launches 5 kernels (bf16→fp32 cast, multiply, sum, fp32→bf16 cast, add shared) costing 0.61 ms at 1 row and 0.33 ms at 32 rows for ~6 MB of activations. Fusion to 1 kernel/layer saves 0.47 / 0.30 / 0.19 ms at 1/8/32 rows.

Design: one Metal kernel per layer that weights the 4 experts' bf16 outputs by the fp32 weights, sums in fp32, rounds once to bf16, and adds the shared expert. G2: every element within 2 bf16 ULP (floor at 0.001). Note the piece-harness lesson from hyp-2: the fused-kernel floor is ~24 µs/layer, not 6.81 µs — predict against the harness, and G3 decides the real saving.

## Boundary

*written by boundary-3, 2026-11-03T22:05Z*

### Inputs and outputs

Source: the brief's combine table (`claude/muse-agent-brief.md:49-62`) and `bench/grader/slots.py:19-32,51-52`. All arrays row-contiguous (brief:24). The harness flattens the port's lead dims, so the slot sees flat rows (slots.py:107,116).

| | Name | Shape at 1 / 8 / 32 rows | Dtype | Layout |
|---|---|---|---|---|
| input | `y` | [1,4,768] / [8,4,768] / [32,4,768] | bfloat16 | the 4 experts' outputs, in `inds` order (brief:53) |
| input | `weights` | [1,4] / [8,4] / [32,4] | float32 | `weights[r,j]` belongs to `y[r,j,:]` |
| input | `shared` | [1,768] / [8,768] / [32,768] | bfloat16 | the shared expert's output |
| output | `out` | [1,768] / [8,768] / [32,768] | bfloat16 | `(y * weights[..., None]).sum(axis=-2).astype(bfloat16) + shared` |

### Before and after in `MoE.__call__`

Source: `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232` and the split harness `bench/grader/slots.py:105-119`.

- **Before (1), the experts call:** `y = self.experts(x, inds)` (`mobilemoe.py:230`) produces `y` in `inds` order (brief:53). From rows ≥ 16 `SwitchGLU` sorts pairs by expert internally and unscatters the result (`switch_layers.py:181`), but that is invisible here: `y[..., j, :]` always corresponds to `weights[..., j]` positionally, whatever per-row order routing emitted.
- **Before (2), routing weights:** `scores` normalized (`+1e-20` guard) and scaled × 2.5 in float32 (`mobilemoe.py:226-228`, `slots.py:48`). Reference values are non-negative and sum to 2.5 per row, but a fused end-to-end model may feed anything finite — the kernel must treat `weights` as arbitrary finite float32.
- **Before (3), the shared expert:** `self.shared_expert(x)` (`mobilemoe.py:232`), a dense int4 `MLP(dim=768, hidden=1536)` (`mobilemoe.py:215,131-139`), computed on the same `x` as the routed path, outside all slots — a slot kernel may not change it.
- **After:** the residual add `h + feed_forward(...)` in `DecoderLayer.__call__` (`mobilemoe.py:258`), a bf16 add on `[rows,1,768]`, then the next layer. It expects `out` exactly as the reference value below, within G2.

Checklist (per `tracking.md`) for each neighbour:

- **Producer `SwitchGLU(x, inds)`.** Callers: `MoE.__call__` (`mobilemoe.py:230`). Callees: `gather_qmm` per product plus internal sort/unsort from rows ≥ 16 (`switch_layers.py:176-199`) — all upstream of the slot. Data: `y` [rows,4,768] bf16. Neighbours: routing before it; this slot after. Cost: the experts slot (hyp-1), untouched. Invariants: positional pairing — column `j` of `y` pairs with `weights[j]` as given; the slot must not re-sort or re-pick experts.
- **Producer routing-weights chain.** Callers: `MoE.__call__` (`mobilemoe.py:226-228`). Callees: none into the slot — it only supplies `weights`. Data: `weights` [rows,4] f32. Neighbours: top-4 selection before (untouched). Cost: the routing slot (hyp-2, ready), untouched. Invariants: no range assumption beyond finite float32.
- **Producer shared-expert MLP.** Callers: `MoE.__call__` (`mobilemoe.py:232`). Callees: 3 quantized matmuls + one fused silu-mul (`tables.md:141-144`: 4 kernels/layer, 80/step). Data: `shared` [rows,768] bf16. Neighbours: inside the slot boundary as input, outside all slots as computation. Cost: knock-out 0.51 ms @1, 2.90 ms @32 (`step-breakdown.md:172,213`), untouched. Invariants: bf16 values; the slot reads them as-is.
- **Consumer residual add.** Callers: `DecoderLayer.__call__` (`mobilemoe.py:258`). Callees: none (plain bf16 add). Data: `out` [rows,768] bf16. Neighbours: next layer's input_norm after. Cost: outside the slot. Invariants: `out` bit-compatible with the reference within G2 — the residual add amplifies nothing, but end-to-end greedy tokens (brief:185) feel any drift.

### What must stay identical

- **Rounding points:** the reference (`slots.py:51-52`, brief:61) casts `y` bf16→fp32 (exact), multiplies by the fp32 weights and sums the 4 terms in fp32, rounds the sum once to bf16 (the `.astype`), then adds `shared` in bf16. Today that is 5 kernels/layer: `copy_v`, broadcast `binary_g` multiply, strided-small `col_reduce_small` sum, `copy_v`, `binary_vv` add (`territory-other-kernels.md` checklist G, `tables.md:140`). The two bf16 roundings — the astype and the add — are the contract: a kernel that merges them (e.g. adding `shared` in fp32 and rounding once) changes the reference arithmetic and is acceptable only if every element still lands within 2 bf16 ULP at every row count and every input. Boundary checks will probe any such move beyond the captured inputs.
- **Tie rule:** no selection happens in this slot, so the 4th/5th within-1e-6 tie freedom (brief:183) belongs to routing only. Combine must pair `weights[r,j]` with `y[r,j,:]` strictly positionally — no re-sorting, re-picking, or re-normalizing experts.
- **Allowed reorderings:** the summation order of the 4 (row, j) products per output element is free within the 2-ULP tolerance, as is any thread/layout decomposition. Nothing else may move: shapes, dtypes, the fp32 accumulation, the single astype before `+ shared`, and the positional `weights[j]` ↔ `y[j]` pairing all stay.

### What the grader checks, and what it can't see

- **G1** compiles the Metal and runs it on the captured inputs at 1, 8 and 32 rows (brief:181, `check_candidate.py:206`).
- **G2 combine** checks exact shape [rows,768] and bf16 dtype, and every element within 2 bf16 ULP of the reference with the ULP floored at |r| = 0.001 (`check_candidate.py:44-47,96-109`); `pass` iff all 20 layers pass (`check_candidate.py:133-138,192-197`). Inputs are captured from all 20 layers of one real decode step per row count (`slots.py:137-171`). In a `routing+combine` candidate each part is checked on its own and combine receives the captured reference weights, never the candidate's (brief:171, `check_candidate.py:96-98,181-187`). Piece timing (median of 30 × all-20-layers eval ÷ 20) must beat `mx.compile(ref_combine)` at all three row counts to stop the slot (brief:193,195, `check_candidate.py:209-226`).
- **What it can't see:** row counts other than 1, 8 and 32 — a decode step can have any row count from 1 to 32 (brief:24); inputs other than the captured ones (other prompts, later decode steps, non-reference weights from a fused routing kernel); threadgroup-memory use near the 32 KB limit (`slots.build` enforces only the 1024-thread cap, `slots.py:275`); results that depend on thread timing; the end-to-end greedy-token check (24 tokens at 1 and 32 requests, plain MLX and vLLM) and G3 full-step timing, which the team runs after merging (brief:185-186,195). Boundary checks will probe exactly these.

### The GPU limits

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory (brief:98); `grid` counts threads, not threadgroups (brief:175). Hardware: Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` available in Metal (brief:99).

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
