# Hypothesis 2: fused routing kernel

- **Owner:** lead
- **Status:** ready
- **Version:** 4
- **Updated:** 2026-11-03T21:52:00Z
- **Next:** none — attempt-1 goes to G3

## Inefficiency

Slot `routing`. One entry:

- `ineff-runs-2` (rank 2, mean 11.0% of step): routing costs 1.02 ms at 1 row and 1.68 ms at 32 rows while reading ~4 MB — 12 tiny kernels/layer of pure overhead (argpartition runs a full 60-element sort per row; the weights-norm sum runs on 1 thread at 1 row). Fusion to ~1 kernel/layer saves 0.9 / 1.2 / 1.4 ms at 1/8/32 rows.

Design: one Metal kernel per layer that computes router scores (matmul + sigmoid + bias), selects the top-4 experts, and normalizes/scales the weights, with the 4 experts in any order per row. G2 gives freedom only on within-1e-6 tie rows and 1e-4 relative weights by expert.

## Boundary

*written by boundary-2, 2026-11-03T21:45Z*

### Inputs and outputs

Source: the brief's routing table (`claude/muse-agent-brief.md:28-36`) and `bench/grader/slots.py:19-48`. All arrays row-contiguous (brief:24).

| | Name | Shape at 1 / 8 / 32 rows | Dtype | Layout |
|---|---|---|---|---|
| input | `x` | [1,768] / [8,768] / [32,768] | bfloat16 | flat rows of the MoE input, row-contiguous |
| input | `w_router` | [60,768] at all row counts | float32 | router weight, used transposed (`x @ w_router.T`) |
| input | `bias` | [60] at all row counts | float32 (the model's bfloat16 `expert_bias`, converted exactly) | one value per expert |
| output | `inds` | [1,4] / [8,4] / [32,4] | uint32 | 4 expert indices per row, each < 60 |
| output | `weights` | [1,4] / [8,4] / [32,4] | float32 | `weights[r,j]` belongs to `inds[r,j]` |

### Before and after in `MoE.__call__`

Source: `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232` and the split harness `bench/grader/slots.py:105-119`.

- **Before:** the slot input `x` is the MoE block's input, i.e. `post_attention_layernorm(h)` reshaped to flat `[rows,768]` (`mobilemoe.py:258-260`, `slots.py:106-107`). `w_router` (`self.router.weight`) and `bias` (`self.expert_bias.astype(float32)`, `slots.py:108`) are parameters. The producer expects nothing back; the slot must read `x` as bfloat16 — the reference casts to float32 itself (`mobilemoe.py:219`).
- **After (1), the experts call:** `SwitchGLU(x, inds)` gathers expert weights per (row, j) by index (`switch_layers.py:176-199`). From `inds.size >= 64` (rows ≥ 16) it sorts pairs by expert internally and unscatters the result back, so its output `y[..., j, :]` always corresponds to `inds[..., j]` whatever per-row order the slot emits (brief:53: `y` is "in `inds` order"). It expects 4 valid indices per row, each < 60; order is free.
- **After (2), combine:** `(y * weights[..., None]).sum(axis=-2).astype(bfloat16) + shared` (`mobilemoe.py:231`, `slots.py:51-52`). It expects `weights[r,j]` to be the normalized weight of `inds[r,j]` (brief:47). The shared expert runs independently of the slot.

Checklist (per `tracking.md`) for each neighbour:

- **Producer (layernorm output / MoE entry).** Callers: `DecoderLayer.__call__` (`mobilemoe.py:258-260`). Callees: none into the slot — it only supplies `x`. Data: `x` [rows,768] bf16. Neighbours: attention block before (untouched). Cost: outside the slot. Invariants: `x` values are whatever the model produces; no assumption about their range beyond bf16.
- **Consumer `SwitchGLU(x, inds)`.** Callers: `MoE.__call__` (`mobilemoe.py:230`). Callees: `gather_qmm` per product, with internal sort/unsort from rows ≥ 16 (`switch_layers.py:176-199`). Data: `inds` [rows,4] uint32. Neighbours: combine after it. Cost: the experts slot (hyp-1), untouched by this hypothesis. Invariants: one output row per (row, j) pair, in `inds` order; any per-row permutation of `inds` is safe as long as `weights` permutes with it.
- **Consumer combine.** Callers: `MoE.__call__` (`mobilemoe.py:231`). Callees: none (reference arithmetic). Data: `weights` [rows,4] f32, `y` [rows,4,768] bf16, `shared` [rows,768] bf16. Neighbours: residual add `h + ...` in `DecoderLayer` after. Cost: 5 kernels/layer, untouched. Invariants: the bf16 cast happens here, outside the slot — the slot outputs pure float32.

### What must stay identical

- **Rounding points:** everything inside the slot is float32 — `x` cast bf16→f32 (exact), matmul + sigmoid in f32, gather of the *unbiased* scores, f32 sum/divide (epsilon 1e-20) and × 2.5 (`slots.py:44-48`, `mobilemoe.py:219-228`). There is no bf16 rounding inside the slot; the only cast is combine's `.astype(bfloat16)`, outside the slot. So reduced precision anywhere in the fused kernel is allowed only if weights still land within 1e-4 relative per expert; a different summation order is fine within that tolerance.
- **The bias rule:** the bias changes *which* experts are selected, not their weights (`mobilemoe.py:221-225`): selection uses `scores + bias`, weights use the unbiased `scores` gathered at the selected indices.
- **Tie rule:** each row must select the same 4-expert *set* as the reference top-4 of biased scores, except rows where the reference's 4th and 5th biased scores are within 1e-6 of each other (brief:183, `check_candidate.py:67-80`). The 4 indices per row must be distinct and < 60 (`check_candidate.py:65`).
- **Allowed reorderings:** any permutation of the 4 (row, j) pairs within a row; the `inds[r,j]` ↔ `weights[r,j]` pairing must be preserved (brief:47). Nothing else may move: the slot may not change shapes, dtypes, the epsilon, the 2.5 scale, or the unbiased-score rule.

### What the grader checks, and what it can't see

- **G1** compiles the Metal and runs it on the captured inputs at 1, 8 and 32 rows (brief:181, `check_candidate.py:206`).
- **G2 routing** checks exact shapes/dtypes, `inds < 60`, expert-set equality per row (near-tie rows counted and skipping the weight check), and weights by expert within 1e-4 relative (`check_candidate.py:50-94,133-138`). Inputs are captured from all 20 layers of one real decode step per row count (`slots.py:137-171`, brief:182). Piece timing (median of 30 × all-20-layers eval ÷ 20) must beat `mx.compile(ref_routing)` at all three row counts to stop the slot (brief:195, `check_candidate.py:209-226`).
- **What it can't see:** row counts other than 1, 8 and 32 — a decode step can have any row count from 1 to 32 (brief:24); inputs other than the captured ones (other prompts, later decode steps, other layers' drift); threadgroup-memory use near the 32 KB limit (`slots.build` enforces only the 1024-thread cap, `slots.py:275`); results that depend on thread timing; the end-to-end greedy-token check (24 tokens, plain MLX and vLLM) and G3 full-step timing, which the team runs after merging (brief:185-186,195). Boundary checks will probe exactly these.

### The GPU limits

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory (brief:98); `grid` counts threads, not threadgroups (brief:175). Hardware: Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` available in Metal (brief:99).

## Builders

### builder-2-a — direction (a): single fused kernel per layer

Router scores (matmul + sigmoid + bias), top-4 selection, and weight
normalize/scale in one launch, one threadgroup (32 threads) per row; x staged
in threadgroup memory, thread 0 does the serial top-4 scan.

| Attempt | Predicted µs/layer 1/8/32 | Measured µs/layer 1/8/32 (ref / compiled) | G1 | G2 |
|---|---|---|---|---|
| `muse-spark/kernels/routing/attempt-1` | 8 / 14 / 36 | 28.5 / 23.6 / 41.8 (36.1,48.0,65.4 / 30.5,42.0,49.1) | pass | pass, worst max_rel_err 2.6e-6, 0 wrong/near-tie rows |

Attempt 1 beats compiled at all three row counts, so per the brief's stop
rule it stops the slot (lead to confirm vs 2-b's in-flight attempt). Taught:
the piece-harness fixed floor is ~24 µs, not one 6.81 µs launch — the
reference's 12 kernels cost only ~3 µs each here, so fusion saves 7.6 µs/layer
at 1 row, and 8 rows runs faster than 1 row (better fixed-cost amortization).
Full arithmetic and miss explanation in the attempt's `notes.md`. One change
I'd make next: nothing in the kernel body (compute is negligible, G2 margin
is 40x) — the remaining cost is dispatch/eval floor outside the slot's reach.

### builder-2-b — direction (b): scores kernel + select+norm kernel

Kernel 1 fuses matmul + sigmoid + bias over a dense rows x 60 grid (k tiled
in 64s through 17.4 KB of threadgroup memory shared across 8 rows, 480
threads/group); kernel 2 fuses the top-4 scan plus normalize/scale, one
thread per row. Outputs two [rows,60] f32 intermediates.

| Attempt | Predicted µs/layer 1/8/32 | Measured µs/layer 1/8/32 (ref / compiled) | G1 | G2 |
|---|---|---|---|---|
| `muse-spark/kernels/routing/attempt-2` | 15 / 15 / 18 | 36.3 / 34.9 / 49.0 (36.3,56.9,51.7 / 29.3,42.9,46.6) | pass | pass, worst max_rel_err 2.6e-6, 0 wrong/near-tie rows |

Beats compiled only at 8 rows. Taught: the piece harness has a ~24-28 µs
floor per fused kernel (attempt-1 measures 28.5 at 1 row), so my 2 x 6.81 µs
fixed-cost model was wrong by 6x; the second launch plus 24 barriers plus
strided w reads cost more than the tiling saves. Direction (a) beats this
split at all three row counts. Full arithmetic and miss explanation in the
attempt's `notes.md`.

## Boundary checks

*checked by boundary-2, 2026-11-03T21:47Z — both attempts against the five grader-blind spots*

### attempt-1 (single fused kernel) — pass

- **Other row counts (1..32):** grid `rows*32`, threadgroup 32 → exactly one full threadgroup per row at every count; no partial groups, no bounds assumptions. Pass.
- **Unwritten threadgroup reads:** `xsh` fully written (32 threads × 24 = 768, `block-0.metal:18-21`); `scores[0..59]` fully written (tids 0..31 write `e=tid`, tids 0..27 write `e=tid+32`, `:25-36`) and barrier-separated (`:37`) before thread 0 reads. Pass.
- **Threadgroup memory:** 768×4 + 60×4 = 3,312 B (~3.2 KB of 32 KB); 32 threads ≤ 1,024. Pass.
- **Thread timing:** one writer per `scores[e]`, barrier-separated phases, thread 0 alone writes outputs (`:39-78`). Deterministic. Pass.
- **Captured-input assumptions:** none. Sigmoid overflow collapses to 0/1 exactly as the reference `mx.sigmoid`; no range assumption on `x` beyond bf16; strictly-greater insertion (earliest index wins ties) with exact ties G2-exempt; all biased scores finite so all 4 slots fill with distinct indices; denom sums in `inds` order like the reference (`slots.py:44-48`); `inds`↔`weights` pairing preserved. Pass.

### attempt-2 (two-kernel split) — pass

- **Other row counts (1..32):** kernel 1 grid `rows*60`, threadgroup 480. `rows*60 mod 480 = (rows mod 8)*60`, always whole rows, so partial groups (including grid < group at rows < 8) always cover whole local rows; group bases are multiples of 480 = multiples of 60, so the `(lr, e)` ↔ `(r, e)` mapping in `block-0.metal:10-14` stays consistent in full and partial groups. Kernel 2 is one independent thread per row (`block-1.metal:5`). Pass.
- **Unwritten threadgroup reads:** partial groups always hold ≥ 60 threads, so `lr==0` covers all 60 experts staging `w_tile`; every occupied local row has its `e==0` stager for `x_tile`; both barrier-separated (`block-0.metal:32,36`). Pass.
- **Threadgroup memory:** 60·64·4 + 8·64·4 = 17,408 B (~17.0 KB of 32 KB); 480 threads ≤ 1,024. Over half the limit but clear of it; `slots.build` checks only the thread cap (`slots.py:275`), hardware accepts the rest. Pass.
- **Thread timing:** disjoint tile writes per `(lr, e)`, barrier-separated phases; kernel-2 threads share nothing. Deterministic. Pass.
- **Captured-input assumptions:** none beyond attempt-1's (same sigmoid form, same strictly-greater tie rule, denom in `inds` order, pairing preserved); the two `[rows,60]` intermediates scale trivially with rows. Pass.

## Decision

Best: `muse-spark/kernels/routing/attempt-1` (builder-2-a, single fused kernel). Why: the only attempt beating `mx.compile(ref_routing)` at all three row counts (28.5 / 23.6 / 41.8 vs 30.5 / 42.0 / 49.1 µs/layer; attempt-2 wins only at 8 rows and is slower everywhere); G2 margin 40× (worst rel err 2.6e-6 vs 1e-4 limit, 0 wrong-expert rows); boundary pass with 3.3 KB of 32 KB threadgroup memory. No hardening round: the kernel is already one minimal launch per layer, and both builders' miss analyses agree the residual ~24 µs/layer is dispatch/eval floor that no kernel-body change removes — no further change worth making. G3 should show: a faster step at all three concurrencies. Knock-out basis predicts −0.9 / −1.2 / −1.4 ms/step; the piece timing shows −0.15 / −0.49 / −0.47 ms/step ((36.1−28.5), (48.0−23.6), (65.4−41.8) µs/layer × 20 layers); the truth is between and G3 decides. The 1-row margin is thinnest, so watch the repeat spread there.

## From other agents
