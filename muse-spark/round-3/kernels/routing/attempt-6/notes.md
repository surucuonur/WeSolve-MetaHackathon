# attempt-6 (round 3): byte-identical re-grade of attempt-1 — one-run noise test (builder-3-f)

Hypothesis `hyp-3-routing-prefill` (slot `routing`, baseline `none`),
hardening of direction (a): the `max_rows` hand-off. Starting point
for this attempt is attempt-1 itself
(`muse-spark/round-3/kernels/routing/attempt-1/`: max_rows 32;
decode `faster` x3, prefill `same`/`same`/`slower` with the 2,048-row
`slower` at 8/30 wins on literally identical code).

## The one main change

None — deliberately. This is a one-run noise test, not a kernel
change. `block-0.metal`, `block-1.metal` and `spec.json` are byte
copies of attempt-1's, with no edits at all:

```
diff attempt-1/block-0.metal attempt-6/block-0.metal  # empty
diff attempt-1/block-1.metal attempt-6/block-1.metal  # empty
diff attempt-1/spec.json attempt-6/spec.json          # empty
```

(`grade.json` and `notes.md` are not copied: the former is this
attempt's own verdict, the latter this file.) Purpose: test whether
attempt-1's 2,048-row `slower` (exactly the 22-loss threshold, medians
within 2.7% of compiled on identical code) reproduces on a fresh
paired run. A non-reproduction confirms the identical-code-noise call
in attempt-1's notes and the failure ledger; a reproduction would
force re-opening it as a possible real hand-off cost.

## Which inefficiency it removes, and how

No new inefficiency. It defends direction (a)'s claim that the full
prefill gap (`ineff-baseline-routing-1` traffic +
`ineff-baseline-routing-2` shape, 139.8/1,077.7/2,222.6 µs/layer) is
already banked by the hand-off: above `max_rows` the candidate IS the
compiled reference (separate `mx.compile` instance), so any `slower`
there can only be paired noise, and this run tests that.

## max_rows: 32, and why

Inherited byte-identically from attempt-1: bottom of the boundary's
32–126 window (see attempt-1's notes.md). Every decode step (1–32
rows) stays on the staged scalar kernels; all three graded prefill
sizes hand to compiled. No analysis is repeated here.

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Baseline `none`: predictions are against attempt-1's own measured
times from its grading run (`grade.json`: decode candidate
103.6/129.9/126.5 vs compiled 139.4/157.2/149.7; prefill candidate
91.3/229.3/299.7 vs compiled 95.1/228.4/291.7).

- **Decode 1/8/32: `faster` x3 kept.** Byte-identical kernels,
  identical grids, identical dispatch at rows ≤ 32. Chained absolutes
  drift with machine state (attempt-1's own 1-row absolute was +108%
  vs its basis while the relative win held), so no absolute is
  predicted — but the 35.8/27.3/23.2 µs/layer margins at 25/28/25
  wins are far above the ~2–3 µs coin-flip band that flipped
  attempt-3/attempt-5 to `same`, and identical code cannot be
  systematically slower than itself.
- **Prefill 127/1,016/2,048: `same`-or-better at every size, 2,048
  included.** Rows above `max_rows` run `mx.compile(ref_routing)`,
  the same function the bar times — identical code, so medians
  within ~3% (attempt-1 saw −4.0%/+0.4%/+2.7%) and win counts a fair
  coin. P(≤8/30 | fair) ≈ 0.008, so a second `slower` at 2,048 is
  not expected; `same` (8–22 wins either way) is the prediction,
  `faster` (≥22 wins) also consistent with noise.

Predicted: G1 pass; G2 pass at all six sizes (nothing changed);
`phases_vs_compiled` decode `faster`, prefill `not slower`;
`beats_compiled` true.

## What this attempt leaves unmeasured (for the hypothesis entry's Unverified list)

Same list as attempt-1 (see its notes.md) — this run adds nothing
except a second paired sample at the six graded sizes. In particular
it does not probe ungraded decode counts, the (by-construction empty)
33–126 band, other chunk sizes, sizes above 2,048, later chunks, or
any boundary can't-see item beyond the six graded sizes.

## Verdict

Graded 2026-11-04T18:27Z, one grader run
(`check_candidate.py .../routing/attempt-6 routing --out .../grade.json`,
no `--baseline`: baseline `none`). Verdict kept in `grade.json`.

G1 pass. **G2 pass at all six sizes** (`rows_differ` 0,
`rows_other_experts` 0, `max_rel_err` 0.0 everywhere — no differing
row anywhere, so no first difference to cite).

Chained µs/layer, reference / compiled / candidate (`grade.json`,
medians of 30 paired repeats):

| size | reference | compiled | candidate | wins | verdict |
|---|---|---|---|---|---|
| rows 1 | 71.9 | 66.3 | 51.7 | 28 | `faster` |
| rows 8 | 113.0 | 104.0 | 90.9 | 27 | `faster` |
| rows 32 | 150.8 | 152.0 | 132.2 | 25 | `faster` |
| prefill 127 | 103.9 | 84.0 | 83.6 | 13 | `same` |
| prefill 1,016 | 234.5 | 222.3 | 220.8 | 16 | `same` |
| prefill 2,048 | 319.6 | 288.8 | 289.5 | 19 | `same` |

`phases_vs_compiled`: decode `faster`, prefill `not slower`.
**`beats_compiled` true.**

## What the run says

The 2,048-row `slower` did **not** reproduce: 19/30 candidate wins
(`same`) on byte-identical code, median +0.2% vs compiled. All
three prefill medians sit within 0.7% of the bar (−0.5%/−0.7%/+0.2%),
tighter than attempt-1's −4.0%/+0.4%/+2.7% and far inside the brief's
5% same-kernel band — exactly what identical code timed twice should
show. The identical-code-noise call in attempt-1's notes and the
failure ledger is confirmed: attempt-1's 8/30 was a ~1-in-125
verdict-level event, not a hand-off cost. Decode kept `faster` x3
(28/27/25 wins, ahead 14.6/13.1/19.8 µs/layer), matching the
prediction.

No prediction missed by >20%: decode verdicts were predicted without
absolutes (chained absolutes drift — this run's machine was quieter
than attempt-1's: compiled 66.3/104.0/152.0 here vs
139.4/157.2/149.7 there) and all three held; prefill medians were
predicted within ~3% and landed within 0.7%. Nothing to explain
beyond the already-documented machine-state drift of chained
absolutes.
