# Hypothesis 2: fused routing kernel

- **Owner:** lead
- **Status:** building
- **Version:** 2
- **Updated:** 2026-11-03T21:35:00Z
- **Next:** builders 2-a (single fused kernel) and 2-b (scores + select+norm split) competing

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

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
