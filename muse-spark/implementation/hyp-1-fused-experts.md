# Hypothesis 1: fused grouped experts kernel

- **Owner:** lead
- **Status:** building
- **Version:** 2
- **Updated:** 2026-11-03T21:35:00Z
- **Next:** builders 1-a (full fused grouped kernel), 1-b (fusion keeping per-pair reads), 1-c (two-kernel split) competing

## Inefficiency

Slot `experts`. One combined fix for five confirmed entries (savings do not add; joint ceiling 443 MB at streaming rate ≈ 11.5 ms at 32 rows, per the verifier):

- `ineff-quantized-kernels-1` (rank 1, mean 19.9% of step): per-pair qvm grids at ~1/3 peak; ideal 0.5 / 2.5 / 8.6 ms at 1/8/32 rows.
- `ineff-findings-2` (rank 3, mean 10.4%): measured 25% grouped-kernel win; 0.40 / 1.30 / 3.67 ms.
- `ineff-switch-layers-2` (rank 4, mean 4.3%): unfused products+activation; ~0.4 ms at all row counts.
- `ineff-runs-1` (rank 6, mean 4.2%): 465 MB excess re-reads; floor 0.0 / 0.6 / 2.3 ms.
- `ineff-switch-layers-1` (rank 7, mean 0.8%): sort glue 0.7 ms at 32 rows, removable only inside this fix.

Design: one Metal kernel that batches same-expert pairs with coalesced reads, runs gate/up/down plus the activation with intermediates on-chip, and needs no sort/unsort. Must keep the G2 boundary (2 bf16 ULP, fp32 accumulation, bf16 rounding points).

## Boundary

(written by boundary-1; every line carries its source)

### Inputs and outputs

Sources: the brief's `experts` slot section (`claude/muse-agent-brief.md:70-92`) and `bench/grader/slots.py:19-40,271`. Constants `D,E,K,H = 768,60,4,384` (slots.py:19; brief:26). All arrays row-contiguous (brief:24). `rows` is read from the first slot input's `shape[0]` (slots.py:271); any 1–32 in decode (brief:24).

| Name | Shape at 1 / 8 / 32 rows | Dtype | Layout / notes |
|---|---|---|---|
| `x` | [1,768] / [8,768] / [32,768] | bfloat16 | brief:74 |
| `inds` | [1,4] / [8,4] / [32,4] | uint32 | brief:75; cast to u32 on load, slots.py:219-220 |
| `gate_w`, `up_w` | [60,768,48], rows-independent | uint32 | [expert,in,out] transposed layout (brief:76; mobilemoe.py:42,202-214); 8 4-bit values per word along out; output column `o` sits in word `o/8` at bits `4*(o%8)` (brief:82) |
| `gate_scales`, `gate_biases`, `up_scales`, `up_biases` | [60,768,12] each, rows-independent | bfloat16 | one scale and bias per 32 outputs (brief:77); `w = scale*q + bias`, affine, group_size 32, bits 4 (brief:82; slots.py:67-77) |
| `down_w` | [60,384,96], rows-independent | uint32 | same packing as gate/up (brief:78) |
| `down_scales`, `down_biases` | [60,384,24] each, rows-independent | bfloat16 | one scale and bias per 32 outputs (brief:79) |
| `y` (output) | [1,4,768] / [8,4,768] / [32,4,768] | bfloat16 | brief:80; `y[r,j]` is expert `inds[r,j]`'s output, positional (see neighbour B) |

Weight bytes per layer (arithmetic shown): gate/up/down `_w` are each 60·768·48 = 60·384·96 = 2,211,840 words = 8,847,360 B; gate/up/down scales and biases are each 60·768·12 = 60·384·24 = 552,960 values = 1,105,920 B. Total ≈ 3·(8,847,360 + 2·1,105,920) ≈ 33.2 MB per layer, shared by all rows.
The spec's `slot_outputs` must provide `y` (slots.py:266-268). Grid counts threads, not threadgroups (slots.py:283).

### Before and after

Source: `MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`) and the split MoE the grader installs (`bench/grader/slots.py:97-119`, bit-identical to the port per `check_identity`, slots.py:122-134). Order per layer: post_attention_layernorm → `x`; router → `inds`, `weights`; experts(`x`, `inds`) → `y`; shared_expert(`x`) → `shared` (independent of this slot, runs alongside); combine(`y`, `weights`, `shared`) → `out` → residual / next layer (mobilemoe.py:230-232,257-258; slots.py:105-118). The grader flattens model `[B,L,D]` to `[rows,D]` and restores shapes afterwards (brief:24; slots.py:106-117).

Neighbour A — what produces the slot's inputs (layernorm + routing), per tracking.md checklist:
- **Callers:** `MoE.__call__` in all 20 layers of every decode step (mobilemoe.py:217-230; brief:24).
- **Callees:** router Linear in float32, sigmoid, `argpartition(-biased, kth=3)`, `take_along_axis`, normalize `+1e-20` then `×2.5` (slots.py:44-48; mobilemoe.py:219-228).
- **Data:** `x` is bf16 RMSNorm output, flattened [rows,768]; `inds` is u32 [rows,4], 4 distinct experts per row in arbitrary order, values in [0,60) (slots.py:44-48; routing G2 rejects inds ≥ 60, check_candidate.py:65).
- **Neighbours:** this slot must accept `inds` in ANY order, including either side of a routing near-tie: within-row order is free (brief:47) and routing G2 matches expert sets, skipping rows whose 4th/5th biased scores are within 1e-6 (check_candidate.py:71-80).
- **Cost:** routing's own ~12 kernels/layer (brief:47) — not this slot's cost.
- **Invariants:** `inds` valid [0,60); `x` finite bf16; row count any 1–32 (brief:24), not just 1/8/32.

Neighbour B — what consumes `y` (combine + residual), per tracking.md checklist:
- **Callers:** combine is called as `combine(y.reshape(-1,K,D), weights, shared.reshape(-1,D))` (slots.py:116).
- **Callees:** `(y * weights[...,None]).sum(axis=-2).astype(bf16) + shared` (slots.py:51-52; mobilemoe.py:231-232).
- **Data:** `y` [rows,4,768] bf16 "in `inds` order" (brief:53); `weights` [rows,4] f32; `shared` [rows,768] bf16; `out` [rows,768] bf16 → reshaped to lead dims, added to the residual, fed to the next layer (slots.py:116-117).
- **Neighbours:** combine pairs `y[...,j,:]` with `weights[...,j]` POSITIONALLY (slots.py:52), so `y[r,j]` MUST be the output of expert `inds[r,j]` — no reordering of the K axis is visible downstream. Slot-edge traffic floor: `y` (rows·6144 B per layer) is written here and re-read by combine; ≈0.04 ms/step at 32 rows, unremovable within the interface (`territory-grader.md`, Boundary floor).
- **Cost:** combine's own 5 kernels/layer (brief:64) — not this slot's cost.
- **Invariants:** `y` must have the exact shape and bf16 dtype or G2 fails (check_candidate.py:117-118).

### What must stay identical

- **Reference math:** for each row `r` and `j`, with `e = inds[r,j]`: `y[r,j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]` (brief:84-88); port code `ref_experts` (slots.py:55-82) with `swiglu(gate,x) = silu(gate)*x` (activations.py:9-11), `gather_qmm(..., transpose=False, group_size=32, bits=4, mode="affine")` (slots.py:67-77).
- **Rounding points:** each gate/up/down product accumulates in float32 and rounds its output to bfloat16; `silu(gate)*up` is computed on those bfloat16 values and rounded to bfloat16 before the down product (brief:90). Keep fp32 accumulation (nothing narrower) and all four bf16 rounding points (carry no wider precision across them) — this hypothesis's stated G2 boundary.
- **Tie rule:** there is NONE inside this slot — instead the slot inherits routing's freedom: it must handle any 4 valid expert ids per row in any order, never assuming sorted `inds` or the reference `argpartition` order (brief:47; check_candidate.py:71-80).
- **Allowed reorderings:** any summation order inside fp32 accumulation; any grouping of (row, expert) pairs and any thread/block decomposition; sort or no sort at ANY row count including ≥16 (the reference sorts iff `inds.size >= 64`, slots.py:62, switch_layers.py:181, but G2 compares outputs only, so the choice is free — `territory-grader.md` note on ineff-runs-3); any dequantization order. **Forbidden:** moving any `y` element off its `[r,j,d]` position, changing shapes/dtypes, narrowing accumulation below fp32, skipping a bf16 rounding, or fusing across the slot edges (`x`/`inds` in, `y` out stay in DRAM between slots).

### What the grader checks, and what it can't see

Checks (`bench/grader/check_candidate.py`):
- **G1:** compiles and runs on real inputs at 1, 8 and 32 rows (`g1=True` after the loops, 206; any exception → error + g1 false, 227-228, Metal errors trimmed by `short_error`, 153-163). Note: shape/dtype mismatches are per-layer stats, not exceptions — they fail G2 with g1 true (188-191).
- **G2 experts** (`check_experts`, 112-130): exact shape [rows,4,768] and bf16 dtype (117-118); EVERY element within 2 bf16 ULP of the reference, with the ULP floored at |r| = 0.001 so reorderings near zero don't fail (`ulp_bf16`, 44-47; 119-123). All 20 layers' captured inputs at each of 1/8/32 rows must pass (179-205). Inputs are seeded (`mx.random.seed(rows)`, slots.py:158) — deterministic every run.
- **Piece timing** runs only if G2 passes (209-226): median of 30 repeats of all-20-layer evals, per layer; `beats_compiled` needs candidate < compiled reference at ALL three row counts. "It is a guide; G3 decides" (brief:193).
What it can't see (checked by hand later — boundary.md *check an attempt*):
- Row counts other than 1, 8 and 32: any 1–32 occur in decode (brief:24); grids and indexing must be valid for all of them.
- Inputs beyond the captured set: one seeded run of 128-token random prompts, decode step 8 (slots.py:158-168). Real prompts, steps and activation distributions differ — no assumption fitted to the captured values survives.
- Reads of threadgroup memory never written, results that depend on thread timing, out-of-bounds reads that luckily pass, threadgroup memory over 32 KB (`build` enforces only the thread COUNT, slots.py:274-276, not the 32 KB).
- End-to-end and G3, run after merge (`time_candidates.py`): the same greedy tokens over 24 generated tokens at 1 and 32 requests in plain MLX and vLLM (brief:185), and a lower decode-step period than both the original and the compiled reference with the gap exceeding the repeat spread (brief:186).

### The GPU limits

- At most 1,024 threads per threadgroup (brief:98; count enforced in slots.py:275) and 32 KB of threadgroup memory (brief:98; NOT enforced by `build` — verify by hand from the Metal source).
- Grid counts threads (slots.py:283); sizes are ints or products of ints and rows/D/E/K/H only (slots.py:230-240); dtypes float32/bfloat16/float16/uint32/int32 (slots.py:22-28).
- Only `mx.fast.metal_kernel`, MLX 0.32.4; the grader runs no Python of the candidate's — the spec is data, only the Metal compiles (brief:96-97). A grader run should take under 3 minutes (brief:98).
- Hardware: Apple M1 Pro GPU, 16 cores, SIMD width 32, bfloat16 as `bfloat` in Metal (brief:99).

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
