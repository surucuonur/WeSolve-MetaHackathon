# Attempt 1 — builder-2-a, single fused kernel (direction a)

First attempt in this direction: one Metal kernel per layer fusing router
scores (matmul + sigmoid + bias), top-4 selection, and weight normalize/scale.
One threadgroup (32 threads) per row; thread t computes experts t and t+32;
thread 0 does the serial top-4 scan and writes the outputs.

## Inefficiency removed

`ineff-runs-2`: routing costs 1.02 ms/step at 1 row for ~4 MB of traffic —
12 tiny kernels/layer of pure launch overhead (argpartition full-sorts 60
elements/row; the weights-norm sum runs on 1 thread at 1 row). This attempt
replaces all 12 with 1 launch/layer, keeping the same traffic floor
(w_router 60*768 f32 = 184 KB/layer/row, read once per row).

## Prediction (µs per layer)

Fixed cost per tiny kernel 6.81 µs
(`results/runs/2026-10-30-micro/step_parts.txt:2`); peak bandwidth 200 GB/s
(brief:9). One launch ≈ 6.8 µs plus traffic rows*184 KB at 200 GB/s:

- 1 row: 6.8 + 184 KB / 200 GB/s (0.9) ≈ **8 µs**
- 8 rows: 6.8 + 1.47 MB / 200 GB/s (7.4) ≈ **14 µs**
- 32 rows: 6.8 + 5.9 MB / 200 GB/s (29.5) ≈ **36 µs**

Compute (2*60*768 = 92k FLOP/row) and the 60-element serial scan are
negligible. Original reference per layer ≈ 1.02 ms/20 = 51 µs at 1 row
(ineff-runs-2 ÷ 20 layers).

## Verdict

G1 pass, G2 pass, beats `mx.compile(ref_routing)` at all three row counts.
Grader output (µs per layer, reference / compiled / candidate):

- 1 row: 36.1 / 30.5 / **28.5** (predicted 8)
- 8 rows: 48.0 / 42.0 / **23.6** (predicted 14)
- 32 rows: 65.4 / 49.1 / **41.8** (predicted 36)

Correctness: 0 wrong-expert rows, 0 near-tie rows, max rel err 2.6e-6 at all
row counts (limit 1e-4) — the f32 dot + sigmoid + exact-op-order weights match
the reference with ~40x margin, and no selection flip occurred on any of the
41 captured rows × 20 layers.

Per the brief's stop rule (first attempt beating compiled at all three row
counts), this attempt stops the slot; G3/end-to-end are the team's call.

## Why the 1/8-row predictions missed (>20%)

The traffic model held (32 rows: 41.8 vs 36 predicted, +16%), but the fixed
cost is ~24 µs, not one 6.81 µs launch. The verdict itself proves it: the
reference's 12 kernels measure 36.1 µs/layer at 1 row, i.e. ~3 µs each in
this harness — far below the 6.81 µs whole-step figure from
`step_parts.txt:2` that I used. So per-launch cost in the piece-timing
harness (20 calls enqueued, one GPU wait, ÷20) is small, and my single
kernel's 28.5 µs is dominated by fixed per-harness cost: one wavefront on a
16-core GPU (single threadgroup at 1 row), MLX `metal_kernel` dispatch +
command encoding per call, and the amortized eval/wait. Consistent with this,
8 rows (23.6 µs) is *faster* than 1 row (28.5 µs): more threadgroups amortize
the fixed cost better, the opposite of a traffic model.

Consequence for the saving: piece-timing saving vs the reference at 1 row is
(36.1−28.5)×20 = 0.15 ms/step, not the 0.9 ms the knock-out scaling
predicted — the knock-out (1.02 ms) and the piece harness (0.72 ms) disagree
on the slot's standalone cost, and G3 decides which one the fused kernel
actually cashes. The compute itself (92k FLOP/row, 60-element scan) is
negligible either way; further work on this slot should attack the ~24 µs
fixed floor (e.g. fewer, larger dispatches across layers is out of scope for
the slot — so nothing left here), not the kernel body.
