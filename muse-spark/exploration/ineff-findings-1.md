# Inefficiency: routing+combine bookkeeping dominates the 1-row step

- **Owner:** explorer-findings
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:35Z
- **Next:** verifier to check the arithmetic against tables.md

## Slot

`routing` + `combine` (the brief's `routing+combine` slot).

## What happens

Routing (cast, router product, sigmoid, bias add, negate, argpartition, gather,
weight normalisation) launches 240 kernels per step and combine (weight, sum, cast
back, add shared) 100 more — 340 of the step's 867 kernels at 1 row
(`results/runs/2026-11-01-breakdown/tables.md`, "Each piece at 1 row", lines
358-375; op mix in "Operations per step by type", lines 149-159). The router
product is the only real math; the rest is selection and weighting bookkeeping
(`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`, as split in
step-breakdown "How the step was cut into pieces").

## Cost

Plain MLX, knock-out savings (tables.md "What removing a piece saves", lines
320-356) over bare step periods (4.75 / 13.37 / 28.19 ms):

- **1 row:** 340 kernels. Routing 1.02 ms / 240 = 4.3 µs per kernel; combine
  0.61 ms / 100 = 6.1 µs per kernel. Bytes: routing must read 3.7 MB/step, so
  3.7 MB / 200 GB/s = 19 µs; combine reads activations only. Measured total:
  **1.63 ms of the 4.75 ms step (34%)**.
- **8 rows:** knock-outs did not run at 8 rows (step-breakdown: knock-outs at 1
  and 32 rows only), and the kernel count is identical to 1 row (867, same op
  mix). Estimate by linear interpolation in rows: routing
  1.02 + (1.68 − 1.02) × 7/31 = 1.17 ms; combine
  0.61 + (0.33 − 0.61) × 7/31 = 0.55 ms; total **≈1.72 ms of 13.37 ms (13%)**.
- **32 rows:** routing 1.68 ms / 240 = 7.0 µs per kernel; combine 0.33 ms / 100 =
  3.3 µs per kernel. Measured total: **2.01 ms of the 28.19 ms step (7%)**.

## Kind

Overhead-bound (glossary: far below both limits). Routing moves 2-4 GB/s; combine
reads no weights. The cost is 340 fixed kernel costs plus their CPU encode.

## Spans

`findings` (measured costs), `model` (MoE block code), `quantized-kernels` (the
router product kernel).

## Predicted saving

If each slot fuses to 1 kernel per layer (20 + 20 = 40 kernels vs 340), saving
300 fixed costs per step:

- **1 row:** routing (240 − 20) × 4.3 = 946 µs; combine (100 − 20) × 6.1 =
  488 µs; total **≈1420 µs per step**.
- **8 rows (estimate):** routing 220 × (1.17 ms / 240 = 4.9 µs) = 1078 µs;
  combine 80 × (0.55 ms / 100 = 5.5 µs) = 440 µs; total **≈1500 µs per step**.
- **32 rows:** routing 220 × 7.0 = 1540 µs; combine 80 × 3.3 = 264 µs; total
  **≈1800 µs per step**.

Supporting evidence: compiling the whole 20-layer MoE stack as one function cut
the block 6.02 → 4.27 ms (−29%) at 1 row in the chained test
(decode-bottlenecks §3) — a different measurement (synchronous, includes
build+encode), so an existence proof, not a prediction. Caveats: knock-out
savings include CPU encode, which fusion also cuts (favourable); but a faster
version gains less than the full saving (step-breakdown Limits).

## Verdict

(not yet written — the verifier writes this)
