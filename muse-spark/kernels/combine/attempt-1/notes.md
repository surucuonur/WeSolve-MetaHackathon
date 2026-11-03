# Attempt 1 — builder-3-a, single fused kernel, thread-per-output (direction a)

First attempt in this direction: one Metal kernel per layer fusing the whole
combine chain (bf16→fp32 cast, multiply, sum, fp32→bf16 cast, + shared). One
thread per output element on a rows×768 grid, threadgroup 256 (768 = 3×256,
so the grid is always whole threadgroups). Each thread loads its 4 bf16
values, converts exactly to fp32, weights, sums in fp32, rounds once with
`bfloat(acc)`, and adds `shared` with the same single-rounding bf16 add as
MLX's `binary_vv`.

## Inefficiency removed

`ineff-model-2`: each layer's weighting launches 5 tiny kernels (AsType,
Multiply, Sum, AsType, Add) costing 0.61 ms/step at 1 row for ~6 MB of
activations — pure launch overhead on an overhead-bound chain. This attempt
replaces all 5 with 1 launch/layer, keeping both bf16 rounding points of the
reference (`slots.py:51-52`): the astype before `+ shared`, then the bf16
add. The 4-term fp32 sum order is free per the hyp-3 boundary.

## Prediction (µs per layer)

Per the hyp-2 lesson (routing attempt-1 notes), predict against the ~24
µs/layer fused-kernel harness floor, not 6.81 µs/launch. Traffic per layer
is y rows×4×768 bf16 + weights rows×4 f32 + shared rows×768 bf16 + out
rows×768 bf16 = rows×9232 B ≈ 9.0 KB/row; at 200 GB/s (brief:9) that is
0.05 / 0.4 / 1.4 µs at 1/8/32 rows — negligible next to the floor. The grid
holds 3 / 24 / 96 threadgroups, so amortization is at least as good as
routing's 8-row case (23.6 µs):

- 1 row: 24 + 0.05 ≈ **24 µs**
- 8 rows: 24 + 0.4 ≈ **24 µs**
- 32 rows: 24 + 1.4 ≈ **25 µs**

Reference (brief:64, ÷20 layers): ≈30.5 @1, ≈22.5 @8 (interpolated in
ineff-model-2), ≈16.5 @32 — whole-step knock-out figures, so the piece
harness will differ and G3 decides the real saving.

## Verdict

G1 pass, G2 pass, beats `mx.compile(ref_combine)` at all three row counts.
Grader output (µs per layer, reference / compiled / candidate):

- 1 row: 22.4 / 22.9 / **17.2** (predicted 24)
- 8 rows: 22.1 / 26.2 / **17.0** (predicted 24)
- 32 rows: 30.7 / 28.0 / **18.5** (predicted 25)

Correctness: 0 bad elements at all row counts (limit 2 bf16 ULP, floor at
0.001); max ULP 0.0 / 2.0 / 1.0 at 1/8/32 rows — the 8-row case touches the
2-ULP boundary on at least one element, so the sum-order margin is thinnest
there, but everything passes on all 20 layers × all captured rows.

Per the brief's stop rule (first attempt beating compiled at all three row
counts), this attempt stops the slot; G3/end-to-end are the team's call.

## Why the prediction missed (>20%, on the fast side)

Predicted ~24 µs from the hyp-2 "~24 µs fused-kernel floor"; measured ~17 µs
(28% under). The 24 µs figure was routing-specific, not a universal harness
floor: routing attempt-1 runs one threadgroup/row with two threadgroup
barriers, a 60-element serial top-4 scan on thread 0, and ~92k FLOP/row of
dot products — at 1 row that is a single wavefront plus a serial tail on a
16-core GPU. This kernel is a trivially parallel streaming op (~12 FLOP per
element, no barriers, no serial section, no threadgroup memory) over 3 / 24 /
96 threadgroups, so its fixed cost is the bare dispatch floor of ~17 µs in
this harness — consistent with the reference's own ~4.5 µs/kernel here
(22.4 µs ÷ 5 kernels at 1 row). Lesson refined: predict a fused elementwise
kernel against ~17 µs/layer, a fused kernel with barriers/serial tails
against ~24 µs.

Piece-timing saving vs the reference is (22.4−17.2)×20 ≈ 0.10 ms/step at 1
row and (30.7−18.5)×20 ≈ 0.24 ms/step at 32 rows — well below the 0.47/0.19
ms the knock-out scaling predicted, because the knock-out and the piece
harness disagree on the slot's standalone cost (same caveat as routing
attempt-1). G3 decides which one the fused kernel actually cashes.
