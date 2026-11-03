- **Owner:** explorer-model
- **Status:** superseded by ineff-runs-2
- **Version:** 2
- **Updated:** 2026-11-03T21:24Z
- **Next:** none — code-side detail appended as a note on ineff-runs-2

## Slot

`routing`.

## What happens

Each layer's router/top-4/weights (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:219-228`)
launches 12 dependent kernels: AsType (x→fp32), Matmul, Sigmoid, AsType (bias→fp32), Add,
Negative, ArgPartition, GatherAxis, Sum, Add, Divide, Multiply — 240 per step at every row
count (`results/runs/2026-11-01-breakdown/tables.md:133-135`). Two casts and the negate exist
only to feed single-use elementwise ops; the whole chain reads 3.7 MB of router weight per
step and writes [rows, 4] indices + weights.

## Cost

- Kernels × fixed cost: 240 × ~4.3 µs @1 row (1.02 ms ÷ 240, measured) ≈ 1.02 ms;
  @32 rows 240 kernels for 1.68 ms ≈ 7.0 µs each — the chain cost is the fixed cost per kernel
  (`claude/findings/step-breakdown.md:179-185`: most pieces cost 4–7 µs per kernel).
- Bytes ÷ peak: 3.7 MB ÷ 200 GB/s = 0.02 ms (negligible).
- Measured (knock-out, plain MLX, `tables.md:333`, `step-breakdown.md:169,210`):
  1.02 ms @1 row (22% of the 4.75 ms step), 1.68 ms @32 rows (6% of 28.19 ms).
  @8 rows knock-outs were not run (`step-breakdown.md:91`); kernel inventory is identical
  to 1 row (867 total, `tables.md:153`), so cost estimated ~1.3 ms by interpolation.
- CPU record: 173 µs @1, 338 µs @32 per step (`tables.md:96-98` summed).

## Kind

Overhead-bound: 4 GB/s achieved, 2% of peak bandwidth (`step-breakdown.md:250,265`).

## Spans

`model` (the chain) → slot `routing`. Neighbour: `other-kernels` (post_norm feeds x).

## Predicted saving

One fused kernel per layer (20 kernels/step ≈ 20 × 7 µs ≈ 0.14 ms residual + small encode)
removes the chain. Saving per decode step: 1.02 − 0.14 ≈ **880 µs @1 row**;
~1.3 − 0.14 ≈ **1150 µs @8 rows** (estimate); 1.68 − 0.14 ≈ **1540 µs @32 rows**.
Fusion also removes the 173–338 µs CPU record time and most of the chain's encode time
(serialized encode 0.80–1.04 ms for router alone, `tables.md:207,256`).

## Verdict

(verifier writes)
