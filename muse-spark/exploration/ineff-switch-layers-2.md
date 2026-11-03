# Inefficiency: gate/up/down products and activation unfused

- **Owner:** explorer-switch-layers
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:27:42Z
- **Next:** verifier to confirm against the kernel inventory and brief rounding points

## Slot

`experts`

## What happens

Each layer launches three separate `mx.gather_qmm` kernels (up, gate, down;
`switch_layers.py:188-194` via `mobilemoe.py:172-184`) plus one compiled
silu-mul activation kernel (`activations.py:9-11`, `tables.md:136,141`): 4
on-chain kernels/layer, 80/step at every row count, plus 3 off-chain default
`Arange` kernels/layer (60/step, measured ~free — `step-breakdown.md:438-463`).
Between them the gate, up and silu-mul bf16 outputs are written to memory and
re-read: at 32 rows 3 × [128,1,384] bf16 = 0.29 MB/layer = 5.9 MB/step; at 1
row 3 × [4,1,384] = 9 KB/layer. One fused kernel could do all three products
and the activation in registers/shared memory.

## Cost

Per decode step (plain MLX, 128-token prompts):

- 1 row: experts knock-out 0.92 ms for 140 kernels = 6.6 µs each
  (`step-breakdown.md:170,182`). Fixed-cost share of the 4 fusable kernels:
  4 × 6.81 µs × 20 = 0.54 ms; intermediates 0.2 MB ≈ 0.00 ms.
- 8 rows: same structure as 1 row (no sort; 867 ops, `tables.md:153`). No
  knock-out at 8 rows; fixed-cost share again ≈ 0.54 ms (estimate, same kernels).
- 32 rows: experts knock-out 14.67 ms (`step-breakdown.md:211`). Fixed-cost
  share of the 4 ≈ 0.54 ms; intermediates 5.9 MB ≈ 0.03 ms at peak. The rest of
  the 14.67 ms is per-pair weight reads (`ineff-runs-1`) and small-kernel
  efficiency, not this entry.

## Kind

Overhead-bound: 80 on-chain kernels/step doing 4 launches where 1 suffices.

## Spans

`switch-layers`, `quantized-kernels`.

## Predicted saving

Fusing the 3 products + activation into one kernel (keeping per-pair reads —
grouping is `ineff-runs-1`'s prize; one combined fix, savings do not add):

- 1 row: 3 launches/layer saved ≈ 3 × 6.81 µs × 20 = 0.4 ms.
- 8 rows: ≈ 0.4 ms (same kernel structure; estimate).
- 32 rows: ≈ 0.4 ms fixed costs + 5.9 MB intermediate traffic (0.03 ms at
  peak); larger threadgroups should also lift efficiency — unquantified.

A fused kernel must keep the boundary: fp32 accumulation + bf16 rounding per
product, silu-mul on bf16 values rounded to bf16 before the down product
(brief §experts).

## Verdict

verifier, 2026-11-03T21:27:42Z — **confirmed**. switch_layers.py:188-194 (up/gate/down `gather_qmm` calls) ✓ via mobilemoe.py:172-184 (the `gather_qmm` wrapper) ✓; compiled silu-mul activation (activations.py:9-11 `swiglu`) ✓. 80 on-chain + 60 off-chain Arange per step (tables.md:136,152-154 ✓); Aranges ~free (step-breakdown §5 repeat-fixes :438-463: −0.02/−0.04/+0.04 noise) ✓. Intermediates: 3×128×384×2 B = 0.29 MB/layer = 5.9 MB/step at 32 rows; 3×4×384×2 = 9 KB/layer at 1 row ✓. Cost: 0.92/140 = 6.6 µs (:170,182 ✓); 4×6.81×20 = 0.54 ms ✓; 0.2 MB ≈ 0 ms at 1 row ✓; 867 ops at 8 rows (:153 ✓); 5.9/200 = 0.03 ms at 32 rows ✓. Saving 3×6.81×20 = 0.41 ≈ 0.4 ms at all rows ✓. Boundary (fp32 accumulation + bf16 rounding per product, silu-mul rounded before down — brief:90) ✓. Absorbs ineff-quantized-kernels-2 (lead settlement — same fusion; qk-2's launch/intermediate figures agree within rounding).

## From other agents

explorer-quantized-kernels, 2026-11-03T22:05Z — kernel-side detail for this
entry (my `ineff-quantized-kernels-2` superseded into it): each product is an
`affine_gather_qvm` grid (`quantized.h:2355-2416`) of 64-thread threadgroups —
gate/up 6 per pair, down 12 per pair — so fusion also merges 24 grids/pair into
1. Gate and up share `x` and `rhs_indices`, so a fused kernel reads both once.
Intermediate traffic beyond the 5.9 MB of writes: the activation re-reads gate
+up outputs (~3.9 MB @32), and down re-reads the activation output strided 24x
per pair (~46 MB/step @32, counted in my `ineff-quantized-kernels-1` x-traffic);
all of it stays on-chip when fused. Agree the prize is one combined fix with
`ineff-runs-1`.
