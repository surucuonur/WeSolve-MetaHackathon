# Inefficiency: gate/up/down are three qvm launches per pair with DRAM round-trips between

- **Owner:** explorer-quantized-kernels
- **Status:** superseded by ineff-switch-layers-2
- **Version:** 2
- **Updated:** 2026-11-03T22:05Z
- **Next:** none (superseded; kernel-side numbers moved to a note on ineff-switch-layers-2)

Superseded: `ineff-switch-layers-2` (posted first, broader: 3 products +
activation, ~0.4 ms) covers this entry's launch-fusion cause. Kept for the
record; the intermediate-traffic detail lives on as my note there.

## Slot

`experts`

## What happens

Each layer launches 3 `GatherQMM` kernels (gate, up, down: 60/step at every row
count, `results/runs/2026-11-01-breakdown/tables.md:152-154`) because each
product is its own `affine_gather_qvm` grid (`quantized.h:2355-2416`). Gate/up
outputs (`[B,1,384]` bfloat16 each) are written to DRAM, re-read by the
activation kernel, whose output is written and re-read by down: ~3.8 KB
intermediate traffic per pair per layer (weights-independent). A fused kernel
keeps all of it on-chip and launches once (gate+up share `x` and `rhs_indices`).

## Cost

Per decode step, plain MLX:

- Launches x fixed cost: 60 GatherQMM x ~6 us (`step-breakdown.md:179-185`)
  ~ 0.36 ms at 1, 8 and 32 rows; fusion to 1 launch/layer saves 40 x 6 us
  ~ 0.24 ms.
- Intermediate bytes: 3.8 KB x pairs x 20 layers = 0.3 / 2.4 / 9.8 MB at
  1/8/32 rows (counter corroboration: experts saves 26 MB written at 32 rows,
  `results/runs/2026-11-02-gpu-counters/tables.md:18`).
- Measured experts totals: 0.92 ms @1, ~5.1 ms @8 (micro, `moe_parts.txt:6`),
  14.67 ms @32.

## Kind

Overhead-bound: launch count plus small round-trips (~10 MB at 32 rows vs 908 MB
the piece reads).

## Spans

`quantized-kernels` -> slot `experts`. Neighbour `other-kernels` (activation
kernel between the launches).

## Predicted saving

40 fewer launches (~0.24 ms) + intermediates on-chip (~10 MB @150 GB/s ~ 0.07 ms
@32), per decode step: **0.2 ms @1 row, 0.3 ms @8 rows, 0.3 ms @32 rows**.
Overlap: the launch saving is shared with `ineff-quantized-kernels-1`'s fused
ideal (its 150 GB/s assumes fusion); count the ~0.2 ms once. No overlap with
`ineff-runs-1`/`ineff-runs-3` (weight bytes, not launches).

## Verdict

(verifier to write)
