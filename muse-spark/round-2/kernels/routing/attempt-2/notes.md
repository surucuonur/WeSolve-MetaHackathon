# Attempt 2 (round 2) — builder-2-b, direction (b): minimal-change scalar exact fix

Starting point: `muse-spark/kernels/routing/attempt-1` (single kernel, one
32-thread threadgroup per row). That starting point fails G2 at every size
under round 2's bit-exact grader (product accumulation order +
sigmoid formula; see `muse-spark/round-2/exploration/territory-baseline-routing.md`
§3.1–3.2). Baseline for this slot: `none`, so there is no `beats_baseline`
bar; the bar is the compiled reference.

## What changed (only the scores)

Kept exactly as attempt-1: one kernel, grid `rows*32`, threadgroup 32, one
threadgroup per row, `x` staged in threadgroup memory, thread `t` computes
experts `t` and `t+32`, thread 0's serial strictly-greater top-4 insertion
scan (same tie rule), and the weights sum `s0+s1+s2+s3 + 1e-20f`, divide,
×2.5 in `inds` order. Changed only how each expert's score is computed:

1. **Sigmoid, MLX verbatim** (`unary_ops.h:308-314`):
   `y = 1/(1 + metal::precise::exp(metal::abs(acc)))`,
   `scores[e] = (acc < 0) ? y : 1 - y`. Removes `ineff-baseline-routing-2`.
2. **Product accumulation reordered scalar-wise, no MMA intrinsics:**
   - `rows == 1` (GEMV regime): 32 lane partials, lane `l` serially sums
     `d = i*128 + l*4 + 0..3` for `i = 0..5` (`gemv.h:162-182`), then the
     shuffle tree `sn = 16,8,4,2,1` (`gemv.h:202-208`) emulated in place on
     the partial array (ascending `l` reads `lane_part[l+sn]` before it is
     overwritten in that round).
   - `rows > 1` (split-K regime): 32 partitions of BK=16 chunks —
     partitions 0..30 cover 16 elements each, partition 31 covers the
     remaining 272 (steel split-K at 8/32 rows: `gemm_k_iterations =
     (768/16)/32 = 1`, `split_k_partition_size = 16`, last partition absorbs
     the rest, `steel_gemm_splitk.h:140-155`). Each partial is summed
     scalar-wise in increasing `d`; partials are summed sequentially in
     partition order from 0, as in `gemm_splitk_accum`
     (`steel_gemm_splitk.h:186-189`).

This is the controlled test of whether scalar reordering can be bit-exact:
at 1 row the GEMV grouping is fully scalar (lane serial sums + shuffle
tree), so it should reproduce the reference if FMA contraction matches;
at 8/32 rows the partition boundaries and sequential partial sum are exact,
but each partition's internal accumulation in MLX is an 8×8 simdgroup MMA
whose 8-product hardware accumulation a scalar loop may not reproduce
(`ineff-baseline-routing-1`). Note this attempt deliberately does **not**
reproduce MLX's partition count at 127 rows (8 partitions) or the regular
steel GEMM at 1,016/2,048 rows — direction (b) fixes 32 partitions at all
larger rows, so those sizes test the same scalar question less cleanly.

## Inefficiency removed

`ineff-baseline-routing-2` (sigmoid, fully) and, if scalar reordering
rounds identically, `ineff-baseline-routing-1` (product order). It does not
touch `ineff-baseline-routing-3` (one simdgroup per row + serial scan) or
`-4` (`w_router` re-read per row) — minimal change by design.

## Prediction (µs per layer at 1 / 8 / 32 rows)

Same flop count as attempt-1 (768 multiply-adds per expert) and same
traffic (`rows × 184 KB` of `w_router`). Extra work per expert: 32 partial
stores + 31 tree/partition adds (≈8% more adds) and one `precise::exp`
instead of fast `exp` (60 per row, negligible). Attempt-1 measured
28.5 / 23.6 / 41.8 µs per layer in its own (round-1, unchained) grader run
(`muse-spark/kernels/routing/attempt-1/notes.md:35-37`), so ×1.08:

- 1 row: **≈31 µs** (compiled reference 30.5 in that run)
- 8 rows: **≈25 µs** (compiled 42.0)
- 32 rows: **≈45 µs** (compiled 49.1)

Round 2's chained timing may shift all figures; the prediction is against
attempt-1's harness numbers. G2 prediction: pass at 1 row; fail at 8/32 if
(and only if) the MMA internal accumulation differs from a scalar serial
sum within each partition.

## Verdict

Graded 2026-11-04 (one grader run, `routing` slot; baseline `none`, so no
`beats_baseline` / `wins_vs_baseline`):

- **G1: pass.**
- **G2: fail overall — but pass at all three decode sizes.**
  - decode 1 row: `rows_differ` 0, `rows_other_experts` 0, max rel err 0.0 — **pass**
  - decode 8 rows: 0, 0, 0.0 — **pass**
  - decode 32 rows: 0, 0, 0.0 — **pass**
  - prefill 127 rows: `rows_differ` 2,453 (of 2,540), experts always match,
    max rel err 3.3e-06 — fail. First difference (layer 0, row 0): experts
    [59, 55, 13, 5] in both; weights [1.142134189605713, 0.663673996925354,
    0.3643006682395935, 0.3298913240432739] vs reference
    [1.142134189605713, 0.6636739373207092, 0.3643006384372711,
    0.32989129424095154].
  - prefill 1,016 rows: `rows_differ` 20,128 (of 20,320), max rel err
    1.41e-05 — fail (first difference layer 0, row 0, same experts).
  - prefill 2,048 rows: `rows_differ` 40,453 (of 40,960), max rel err
    1.39e-05 — fail (first difference layer 0, row 0, same experts).
- **Timing:** none. The grader prints `timing_us_per_layer` only for a
  correct attempt, and `beats_compiled` is false; so the predicted
  ≈31 / 25 / 45 µs per layer could not be checked (neither confirmed nor
  missed by >20% — unmeasured, not a prediction miss). For context from the
  same round: builder-2-a's MMA replica (attempt-1) measured chained
  54.9 / 91.4 / 114.8 vs compiled 56.2 / 91.2 / 98.9 µs per layer.

## What this proves and disproves

**Scalar reordering CAN be bit-exact at decode sizes.** At 1 row the scalar
GEMV emulation (32 lane partials in `gemv.h` lane order + shuffle-tree
association) plus the verbatim sigmoid reproduces every score bit for bit
on all 20 layers. At 8 and 32 rows the scalar split-K emulation — exact MLX
partition boundaries (31 × BK=16 + last partition absorbing the remaining
272) with each partial summed serially in increasing `d` and partials
summed sequentially — is also bit-exact on all 160 / 640 rows. So the
8-product accumulation inside `simdgroup_float8x8` MMA, at least as it
affects these outputs, rounds identically to a serial scalar sum over the
same K order; the claim in `ineff-baseline-routing-1` that a scalar
reordering cannot reproduce the MMA regimes' rounding is **disproved at
8/32 rows** (and the GEMV half at 1 row).

The prefill failures do not test that claim: by design (direction (b) fixed
32 partitions at all larger rows) this kernel uses the wrong accumulation
grouping there — MLX uses 8 partitions of 96 at 127 rows and the regular
steel GEMM (BK=16 chunks accumulated chunk after chunk into one
accumulator) at 1,016/2,048 rows. The failure pattern confirms grouping is
the whole story: experts and order match everywhere (`rows_other_experts`
0 at all six sizes), only last-bit weight differences remain.

**Next change (one):** branch the partition geometry on `rows` as MLX does
— 8 partitions of 96 at the 127-row split-K sub-regime and a single
chunk-after-chunk scalar accumulation (BK=16 chunks, one accumulator) in
the regular-GEMM regime above 768 rows — keeping everything else in this
kernel identical. If the decode result generalises (MMA ≡ serial scalar
over the same K order), that attempt passes G2 at all six sizes with a
far simpler and likely faster kernel than the MMA replica; the grader's
chained timing would then be the first timing measurement of the scalar
design.
