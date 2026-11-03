# Attempt 2 — builder-3-b, single fused kernel, threadgroup-per-row (direction b)

First attempt in this direction: one Metal kernel per layer fusing the whole
combine chain, organized as one threadgroup (192 threads) per row instead of
direction (a)'s flat thread-per-output grid. The threadgroup cooperatively
stages weights (4 f32) and shared (768 bf16, bit-exact to fp32) into
threadgroup memory (~3.1 KB of the 32 KB limit) with vectorized 8-byte
loads, hits one barrier, then each thread weights its 4 consecutive outputs
(y also streamed with vectorized u64 loads, unpacked bit-exactly), sums in
fp32, rounds once with `bfloat(acc)`, and bf16-adds the staged shared.

## Inefficiency removed

`ineff-model-2`: each layer's weighting launches 5 tiny kernels (AsType,
Multiply, Sum, AsType, Add) costing 0.61 ms/step at 1 row for ~6 MB of
activations — pure launch overhead on an overhead-bound chain. This attempt
replaces all 5 with 1 launch/layer, keeping both bf16 rounding points of the
reference (`slots.py:51-52`): the astype before `+ shared`, then the bf16
add with the same single-rounding op as attempt-1. Per-element fp32 sum runs
j = 0..3, the reference's natural order (order is free per the boundary).

## Prediction (µs per layer)

Per the hyp-2 lesson, predict against a fused-kernel harness floor, refined
by attempt-1's verdict: ~17 µs/layer for a barrier-free streaming kernel,
~24 µs for one with barriers/serial tails (routing attempt-1). This kernel
sits between: one barrier plus staging, no serial section, but only
1 / 8 / 32 threadgroups (vs 3 / 24 / 96 for attempt-1), so single-row
amortization is worse. Take a 21 µs floor at 1 row, 19 µs at 8/32 rows.
Traffic per layer is rows×9232 B ≈ 9.0 KB/row (attempt-1 notes); at
200 GB/s (brief:9) that is 0.05 / 0.4 / 1.4 µs at 1/8/32 rows:

- 1 row: 21 + 0.05 ≈ **21 µs**
- 8 rows: 19 + 0.4 ≈ **19 µs**
- 32 rows: 19 + 1.4 ≈ **20 µs**

Attempt-1 measured 17.2 / 17.0 / 18.5 vs compiled 22.9 / 26.2 / 28.0, so
this direction needs ≤ ~22/26/28 µs to also beat compiled; the prediction
says it does, but slower than attempt-1 (staging buys nothing on a
streaming op — the bet is only that it costs little).

## Verdict

G1 pass, G2 pass, beats `mx.compile(ref_combine)` at all three row counts.
Grader output (µs per layer, reference / compiled / candidate):

- 1 row: 24.8 / 20.9 / **17.9** (predicted 21)
- 8 rows: 24.0 / 23.0 / **16.1** (predicted 19)
- 32 rows: 43.3 / 27.9 / **18.1** (predicted 20)

Correctness: 0 bad elements at all row counts (limit 2 bf16 ULP, floor at
0.001); max ULP 0.0 / 2.0 / 1.0 at 1/8/32 rows — identical margins to
attempt-1, as expected from identical arithmetic (same j = 0..3 sum order,
same `bfloat(acc)` + single-rounding bf16 add; staging is bit-exact).

No >20% miss: predictions came in 15% / 15% / 10% high. The staging +
barrier cost ~nothing measurable — this direction lands on the same ~17 µs
dispatch floor as attempt-1's barrier-free kernel (attempt-1: 17.2 / 17.0 /
18.5). The 19–21 µs floor guess was conservative; attempt-1's refined
lesson holds and extends: one barrier plus ~3 KB of threadgroup staging
does not lift the elementwise floor.

First grading caught a bug before the verdict above: G2 failed with ~97%
bad elements because the y-stream stride used 96 u64/expert instead of 192
(768 bf16 = 192 u64, not 96 — halved twice). Fixed `yp[j * 192u + tid]`;
shared staging (192 u64/row) was already right.

Per the brief's stop rule the slot already stopped at attempt-1 (graded
first, beats compiled everywhere); this attempt is the competing direction
(b) datapoint — same speed, more machinery, so attempt-1 stays the pick.
