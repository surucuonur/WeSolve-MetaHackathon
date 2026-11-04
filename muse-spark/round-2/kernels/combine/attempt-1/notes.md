# Attempt 1 — builder-3-a, direction (a): thread-per-output + precise stepwise sum

First attempt in this direction (round 2, `hyp-3-combine-exact`, slot
`combine`, baseline `none`). Builds on the starting point
`muse-spark/kernels/combine/attempt-1` (`block-0.metal:12-25`,
`spec.json`: grid `rows*768`, threadgroup [256,1,1], no threadgroup
memory). The starting point is not a baseline: under round 2's bit-exact
G2 it fails at five of six sizes on FMA contraction alone (decode
0 / 2 / 2 differ, prefill 6 / 50 / 130; `ineff-baseline-combine-1`,
`territory-baseline-combine.md` checklist D), so it counts as the attempt
this one changes.

## What changed

Only the arithmetic. Indexing, grid, threadgroup, inputs/outputs and the
final line `bfloat(acc) + shared[sbase]` are verbatim from the starting
point. The single-expression sum

```metal
float acc = float(y0)*w0 + float(y1)*w1 + float(y2)*w2 + float(y3)*w3;
```

which the compiler FP-contracts into FMAs (each product kept unrounded
into its add), becomes seven fma-free stepwise ops with explicit
rounding barriers:

```metal
volatile float p0 = float(y[ybase]) * weights[wbase];        // rounded to f32
volatile float p1 = float(y[ybase+768u]) * weights[wbase+1u];
volatile float p2 = float(y[ybase+1536u]) * weights[wbase+2u];
volatile float p3 = float(y[ybase+2304u]) * weights[wbase+3u];
volatile float s01 = p0 + p1;                                // rounded to f32
volatile float s012 = s01 + p2;                              // rounded to f32
volatile float acc = s012 + p3;                              // rounded to f32
out[sbase] = bfloat(acc) + shared[sbase];                    // verbatim
```

Each `volatile` store materialises its f32 value, so no product can stay
unrounded into the next add and no add can be fused with a later mul;
the fold is exactly `((p0+p1)+p2)+p3` with a rounding after every add,
then one bfloat16 round and the single-rounding bfloat16 `+ shared`,
matching the reference's separate kernels (`binary_g` multiply,
`col_reduce_small` sum, `copy_v`, `binary_vv`; `slots.py:51-52`,
Boundary steps 1–5). No reassociation, no `fma`, no threadgroup memory.

Note on spelling: the literal `precise float` qualifier does not compile
under MLX 0.32.4 (`metal::precise` is a function namespace in the MLX
header, e.g. `metal::precise::exp`; a leading `precise` is parsed as
that namespace and rejected). Volatile stepwise barriers are the
"explicit rounding barriers" the hypothesis design names as the
alternative to `metal::precise` operations, with the same semantics:
every mul rounds once to f32, every add rounds once to f32. Probes
outside the grader (random [512,4,768] inputs vs the eager MLX
reference, 3 × 393,216 elements) gave differ 3–4 for the starting
point's contracted expression and differ 0 for this kernel.

## Inefficiency removed

`ineff-baseline-combine-1` (confirmed, gate): FMA contraction breaks
bit-exactness and forfeits the slot's whole saving while G2 fails.
`ineff-baseline-combine-2` (confirmed, keep) is retained, not removed:
the kernel stays on the ~17 µs/layer dispatch floor (one launch/layer,
traffic + compute < 1.6 µs even at 32 rows).

## Prediction (µs per layer, decode 1 / 8 / 32 rows)

Floor + traffic + barrier cost, on the old independent piece basis (the
only same-run basis available; round 2 ranks by chained timing, which
has never run on this slot and must replace these numbers):

- Dispatch floor (starting point, both round-1 organisations):
  ~17 µs/layer, flat in rows (`ineff-baseline-combine-2`).
- Traffic: 9,232 B/row ÷ 200 GB/s = 0.05 / 0.37 / 1.48 µs at 1 / 8 / 32
  rows; compute ≈ 7 f32 ops × 768 outputs ≤ 0.03 µs.
- Barrier cost: 7 volatile thread-local f32 spills per output element,
  still under the floor; `ineff-baseline-combine-1` predicts < 0.5
  µs/layer for losing FMA fusion. Allow 0.3 µs.

Predicted candidate: 17 + 0.05/0.37/1.48 + 0.3 ≈ **17.4 / 17.7 / 18.8
µs/layer**. Compiled reference on the same basis: 23.8 / 26.1 / 32.0
µs/layer (Claude re-grade, `ineff-baseline-combine-1` *Cost*), so the
attempt is predicted to pass G2 with differ 0 at all six sizes and to
beat the compiled reference at all three decode sizes. Starting point
on the same basis: 18.6 / 17.0 / 18.9 µs/layer (untimed under round 2 —
G2-gated); the prediction is that this attempt matches it within the
0.5 µs barrier allowance.

## Verdict

G1 pass, G2 pass at all six sizes, `beats_compiled` true. Baseline is
`none`, so there is no `beats_baseline` / `wins_vs_baseline` bar.

Grader output (single run, `check_candidate.py ... combine`):

- Correctness (differ / max ULP, summed over 20 layers): decode 1 row
  0 / 0.0, 8 rows 0 / 0.0, 32 rows 0 / 0.0; prefill 127 rows 0 / 0.0,
  1,016 rows 0 / 0.0, 2,048 rows 0 / 0.0 — every element bit-equal to
  the eager reference. First difference: none. The starting point's
  0 / 2 / 2 and 6 / 50 / 130 differs are closed.
- Chained timing (µs per layer, reference / compiled / candidate):
  1 row 59.0 / 35.4 / **31.8**, 8 rows 45.0 / 38.2 / **33.6**, 32 rows
  52.1 / 43.4 / **39.5**. Candidate beats the compiled reference at all
  three decode sizes by 3.6 / 4.6 / 3.9 µs/layer (0.072 / 0.092 /
  0.078 ms/step over 20 layers).

### Why the timing missed the prediction (>20%)

Predicted 17.4 / 17.7 / 18.8 µs/layer; measured (chained) 31.8 / 33.6 /
39.5 — +83% / +90% / +110%. The miss is the timing basis, not the
barrier. The prediction used the old independent piece timing, where
the 20 layer calls overlap on the GPU and hide per-call latency; round
2 times chained through the grader's `link` kernel so each layer waits
for the previous one (`check_candidate.py:178-213`), pricing the serial
latency the independent basis hid. The compiled reference moved the
same way between bases (23.8 → 35.4, 26.1 → 38.2, 32.0 → 43.4
µs/layer, +36–49%), and the candidate still beats it at every size, so
the dispatch floor is retained in relative terms. The volatile-barrier
cost (< 0.5 µs/layer predicted) cannot be isolated from the basis
change — the starting point was never chained-timed because G2 gated
it — but no barrier regression is visible in the candidate-vs-compiled
gap, which matches the banked-saving direction of
`ineff-baseline-combine-1` (5.2 / 9.1 / 13.1 µs/layer on the independent
basis; 3.6 / 4.6 / 3.9 chained). Lesson: predict combine against the
chained basis (~32–40 µs/layer here), not the ~17 µs independent floor.

Next change if hardening is wanted: none for correctness — the kernel
is exact at all six sizes and already beats compiled. Any further work
would be shaving the chained floor itself, which
`ineff-baseline-combine-2` predicts no kernel-body change can do.
