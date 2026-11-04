# attempt-4 (round 3): grouped/tiled prefill replica, reserved-word fix (builder-2-d)

Hypothesis `hyp-2-experts-prefill`, slot `experts`, baseline `none`,
starting point `muse-spark/round-2/kernels/experts/attempt-7`
(G1+G2 pass, slower at 1 row and at every prefill size).
Direction (c): grouped/tiled replica of `gather_qmm_rhs` issue.

## The one main change

Copy of attempt-3 with the single fix its notes.md prescribes: `half`
-> `part` in `dequant_tile_4bit` (`block-3.metal`: 1 declaration + 4
uses, lines 29/37/38/42/44). `half` is a reserved Metal type name, so
attempt-3 failed G1 with six parse errors; nothing else was reached.
Verified: all other files byte-identical to attempt-3
(`block-0/1/2/4/5.metal`, `spec.json`); `grep -w half` on the new
`block-3.metal` returns 0 matches. No `max_rows`: kernels run at every
row count (decode kernel below 60, prefill kernel at/above, selected by
the same threshold MLX uses). For the four-kernel design, limits and
exactness argument see attempt-3's notes.md; they are unchanged.

## Which inefficiency it removes, and how

Same as attempt-3: `ineff-baseline-experts-1` (scalar prefill 15-28x
slower). The scalar `qmm_accum` per-pair branch never runs (decode
kernel exits at rows >= 60, grouped prefill kernel replaces it).
Cross-row weight sharing removes the distinct-traffic gap (one dequant
per tile serves <= 16 rows); hardware MMA removes the scalar-issue gap;
64-thread groups remove the 512-thread-per-pair overprovisioning. Takes
the tile-waste and offsets prizes for free. `ineff-baseline-experts-2`
(1-row decode gap) deliberately kept for hyp-4.

## Predicted us per layer at the six sizes (chained)

Identical to attempt-3's (same kernels, same arithmetic; the rename
changes no instruction). Basis (measured, piece,
`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`):
starting point vs compiled 92.1 vs 78.6 (1 row), 162.8 vs 235.5 (8),
515.3 vs 797.8 (32), 15,220.0 vs 960.8 (127), 119,154.4 vs 4,306.0
(1,016), 240,681.1 vs 11,998.9 (2,048). Baseline `none`.

- 1 row: 92.1 + 3 = **95** (vs 78.6 compiled, `slower`, keeps the
  starting-point loss for hyp-4).
- 8 rows: 162.8 + 3 = **166** (vs 235.5, `faster`, keeps the win).
- 32 rows: 515.3 + 4 = **519** (vs 797.8, `faster`, keeps the win).
- 127 rows: **900** (vs 960.8, `faster`).
- 1,016 rows: **4,100** (vs 4,306.0, `faster`).
- 2,048 rows: **11,500** (vs 11,998.9, `faster`).

G1 prediction: pass (the only defect reached was the reserved word).
G2 prediction: pass with `differ == 0` (decode byte-identical to
attempt-7; prefill replicates dequant values, BK=32 frag order and the
four bfloat16 points; select exact). `beats_compiled` prediction: false
(prefill `faster` everywhere, decode `slower` at 1 row) -- the prefill
phase win is the claim.

## What this attempt leaves unmeasured

Same as attempt-3: kernel speed at 33-126 rows; sizes above 2,048;
prefill beyond the first seeded chunk, other lengths/distributions,
mixed batches; the 32 KB limit (hand-checked: prefill 18,752 B);
unwritten threadgroup reads and thread-timing dependence (none by
construction); NaN-vs-NaN equality; whole-request composition (ready
check). The 2,048-row compiled basis carries run-to-run variance
(round-3 `territory-quantized-kernels.md` S5 caveat).

## Verdict

G1 **pass**, G2 **pass** (`differ == 0`, `max_ulp == 0.0` at all six
sizes) — the rename was the only defect, and attempt-3's "first
suspect" (the MMA frag mapping) is refuted: the grouped prefill path is
bit-exact. (`grade.json`)

Timing (µs/layer, candidate vs compiled, `vs_compiled`):

- 1 row: 122.2 vs 77.9 (reference 78.6) — `slower` (0 wins).
  Predicted 95.
- 8 rows: 206.4 vs 237.6 (reference 237.3) — `faster` (29 wins).
  Predicted 166.
- 32 rows: 584.7 vs 807.4 (reference 803.4) — `faster` (29 wins).
  Predicted 519.
- 127 rows: 2,684.3 vs 1,005.7 (reference 1,046.2) — `slower`
  (0 wins). Predicted 900.
- 1,016 rows: 9,311.2 vs 4,073.3 (reference 4,071.2) — `slower`
  (0 wins). Predicted 4,100.
- 2,048 rows: 17,229.0 vs 7,417.4 (reference 7,549.1) — `slower`
  (0 wins). Predicted 11,500.

`phases_vs_compiled`: decode `slower`, prefill `slower`;
`beats_compiled` false. Prefill ratios vs compiled: 2.67x / 2.29x /
2.32x slower. The scalar branch is gone — cross-run absolutes vs the
starting point's 15,220/119,154/240,681 are 5.7x/12.8x/14.0x faster —
but the "compiled rate minus banked waste" prediction missed by ~3x.

## Why prefill missed (miss analysis)

The prediction assumed grouped MMA runs at the compiled rate. Three
structural gaps in the code explain most of the ~2.3-2.7x (unprofiled —
ranked by arithmetic, not measurement):

1. **`x` is gathered 12x per tile.** The gate/up nest is `g` outer,
   `k` inner (`block-2.metal:54-78`): every (N-tile, K-tile) pair
   reloads its `x` tile from device memory. Per tile group that is 12
   x 24 x 1 KB = 288 KB of gathers instead of 24 KB (attempt-3's notes
   claimed "x staged once per K-tile" — the nest orders it per N-tile
   too; L2 absorbs an unknown share). At 2,048 rows, 512 tiles x 288 KB
   = 147 MB/layer of `x` traffic, ~7.4 ms/layer at 20 GB/s if
   uncached — same order as the whole 17.2 ms measurement.
2. **The MMA is scalar-fed at low occupancy.** Each `mma_step`
   (`block-3.metal:95-130`) issues ~34 scalar threadgroup float loads
   per lane plus 2 `simdgroup_barrier`s to feed 4 MMA ops, and the
   group holds only 64 threads behind ~19 KB of threadgroup memory
   (1-3 groups resident per core), so barrier and load latency has
   little to hide behind. Count: 12x24x4x2 + 24x12x4 = 3,456
   `mma_step`s and 1,152 full-group barriers per tile group.
3. **Partial tiles at small sizes.** At 127 rows the mean expert owns
   ~8.5 of 16 pairs, so roughly half the MMA work covers zero-padded
   rows (127 rows is also the worst ratio, 2.67x). MLX pads too, so
   this is at most a fraction of the gap.

Decode overhead also missed: +30/+44/+69 µs vs the round-2 basis
(92.1/162.8/515.3) against +3/+3/+4 predicted. Part is the select
pass (decode writes `y_decode`, select reads + rewrites it: +12/98/393
KB traffic ≈ +0.6/5/20 µs at 20 GB/s) plus 3 extra launches x 20
layers, but ~24 µs/layer at 1 row is unexplained by that model —
recorded as unexplained, not as free.

## Next change

Interchange the gate/up nest to `k` outer / `g` inner, loading each
`x` tile once and reusing `Xs` across the 12 N-tiles (accumulators
`Cg`/`Cu` become per-`g` arrays). One change, aimed at cause 1; if
prefill then stays >1.5x slower, the scalar MMA feed (cause 2) is the
bigger term and direction (c) needs a cooperative frag-load redesign.
