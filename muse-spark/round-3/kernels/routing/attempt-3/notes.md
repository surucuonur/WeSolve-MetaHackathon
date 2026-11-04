# Round-3 attempt 3 — builder-3-c, hyp-3-routing-prefill direction (c)

Builds on `muse-spark/round-2/kernels/routing/attempt-4` (the slot's
starting point: G1+G2 pass, decode `faster` x3, prefill `slower` x3 at
2.6/5.9/8.7x compiled). Baseline: none, so no `beats_baseline`; the bar
is the compiled reference.

## The one main change

Split the staged path at the regime boundary `rows = 768` into two
kernels so the single-partition regime (partitions == 1, i.e. exactly
rows > 768) drops the 2,048-float `parts` array:

- Kernel-A (`block-0.metal`): attempt-4 verbatim plus a uniform
  `if (rows > 768) return;` first and outputs renamed to
  `inds_a`/`weights_a`. Serves rows <= 768 (GEMV + split-K).
- Kernel-B (`block-2.metal`): at rows <= 768 copies kernel-A's outputs
  through (bit-exact, 4 threads/row); at rows > 768 runs the
  single-partition path with each expert's whole 768-product chain in
  one thread-private accumulator across the 12 staged groups — no
  `parts`, so threadgroup is (816 + 60 + 4,096) x 4 B = 19,888 B
  (60.7% of 32 KB) instead of 28,080 B (85.7%).

Two kernels are required because threadgroup arrays are statically
sized: one kernel would still declare `parts` for its general path.
Each kernel's outputs are fresh buffers (`bench/grader/slots.py:343-356`,
`env.update` overwrites per kernel), so same-name sharing is impossible
and the copy-through is the wiring.

Kept verbatim: the `rows`-branched regimes (the split is at the
regime boundary, never at graded values), staging layout/stride,
product order within every (expert, partition) chain (kernel-B's
register chain visits chunks/groups in the same increasing order the
`parts` path resumes them in), the 24 in-loop barriers, sigmoid,
strictly-greater top-4, and the weights fold. G2 prediction: pass at
all six sizes — storage changes only (register vs threadgroup vs copy),
no order changes.

Design alternatives ruled out while writing (for the failure ledger /
inefficiency notes): register-held partials in ONE kernel cannot shed
`parts`, because the final ordered partition-sum needs all
partitions x 60 partials simultaneously visible (a gather scratch of up
to 1,920 floats ~= `parts` itself); 8-chunks/group needs 8,192 floats
of `wsh` alone (> 32 KB) and 6-chunks/group needs 36,272 B with `parts`
(34,224 B even at stride 60) — both infeasible, so attempt-4's notes'
"stage 8, halve the barriers" suggestion does not fit; sharding one
expert's K across threads would reorder its serial chain and fail G2.

## Inefficiency removed

`ineff-baseline-routing-2` (shape), partially: the 28 KB residency cap
at the two single-partition prefill sizes (1,016/2,048 rows), where the
running kernel's footprint falls 29% (28,080 -> 19,888 B). Barriers
(24), single-partition duty (60/256) and all of `ineff-baseline-routing-1`
(traffic) are untouched — so prefill stays `slower` (the traffic floor
alone exceeds compiled total 1.35x/4.3x/6.6x); this attempt buys only a
slice of the 142.7/336.8 us/layer shape residual at 1,016/2,048 rows.

## Predicted us per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows

Anchors (different runs, drift caveat per builder-1-b): decode
47.5/69.1/72.9 vs compiled 60.7/84.9/89.5 (round-2 grade run);
prefill measured 227.1/1298.8/2513.0 vs compiled 87.3/221.1/290.4
(`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`).
Extra cost: one more kernel launch per layer. Launch cost estimated
from (reference - attempt-4)/11 kernels in the round-2 run:
(71.1-47.5)/11 = 2.1, (102.9-69.1)/11 = 3.1, (104.9-72.9)/11 = 2.9
us/layer at 1/8/32 rows -> ~2-3 us/layer for the empty/copy kernel
(copy traffic is rows x 64 B, negligible).

- 1 row: 47.5 + ~2.5 = **50** (compiled ~60.7, faster, margin ~10 us).
- 8 rows: 69.1 + ~3 = **72** (compiled ~84.9, faster).
- 32 rows: 72.9 + ~3 = **76** (compiled ~89.5, faster).
- 127 rows: 227.1 + ~3 (copy kernel) = **230** (compiled ~87.3, slower).
- 1,016 rows: 1298.8 + ~2.5 (empty kernel-A) - residency saving.
  Residency is an unmeasured mechanism (entry-2 verdict), so the saving
  is a guess: ~10% of the 354.7 residual ~= **-35 -> ~1266**
  (compiled ~221, slower).
- 2,048 rows: 2513.0 + ~2.5 - ~10% of 609.8 ~= **-70 -> ~2446**
  (compiled ~290, slower).

Prediction: decode faster x3 (wins kept), prefill slower x3,
`beats_compiled` false. G1 pass, G2 pass (0 rows differing, all sizes).

No `max_rows`: direction (c) is a kernel fix; the hand-off is
direction (a)'s. Every size runs the kernels.

## What this leaves unmeasured (for the hypothesis entry's Unverified)

Ungraded row counts on the new dispatch: 769-1,015 (kernel-B compute
path in kind, untested), 33-767 (kernel-A + copy-through in kind),
2-7/9-31 (same), above 2,048; the residency mechanism itself (still
inferred, not measured); the empty/copy launch cost on other machine
states; exact 4th/5th ties (no graded row is one); prefill beyond the
first seeded chunk.

## Verdict (2026-11-04T18:18:49Z, one grader run)

G1 pass, **G2 pass at all six sizes** (rows_differ 0,
rows_other_experts 0, max_rel_err 0.0). Chained us/layer
(ref / compiled / candidate, `grade.json`): 1 row 77.9 / 64.7 /
**66.1** (14 wins, `same`); 8 rows 109.8 / 95.4 / **96.1** (15,
`same`); 32 rows 117.4 / 103.9 / **109.5** (9, `same`); 127 rows
108.7 / 96.9 / **266.0** (0, `slower`); 1,016 rows 242.2 / 235.3 /
**1098.3** (2, `slower`); 2,048 rows 324.8 / 299.7 / **2097.3** (0,
`slower`). `phases_vs_compiled`: decode `not slower`, prefill
`slower`; **`beats_compiled` false**. Baseline none, no
`beats_baseline`.

## Why the decode timing missed the prediction by >20%

Predicted 50/72/76/230/1266/2446, measured 66.1/96.1/109.5/266.0/
1098.3/2097.3: decode +32/+33/+44%, 127 rows +16%, prefill-big
-13%/-14% (better than predicted). Two causes, in order:

1. **Machine-state drift (dominant).** The anchor mixed two runs
   already; this run's compiled bar (64.7/95.4/103.9) sits above the
   round-2 anchor (60.7/84.9/89.5), while sibling attempt-1, graded
   minutes earlier, saw compiled at 139.4/157.2/149.7 — a ~2x swing
   of the bar between adjacent runs. Attempt-4's own decode margin
   compressed from -13/-16/-17 us to ~+/-few us of coin-flip races
   (14/15/9 wins). Same lesson as builder-1-b's and builder-3-a's:
   chained absolutes drift; only paired verdicts in one run count.
2. **The real wiring cost (~2-3 us, as predicted).** Kernel-A on the
   decode path is attempt-4 plus a uniform branch (free) plus the
   copy-through launch. In a compressed-margin state that small real
   cost flips `faster` to `same` — attempt-1's identical decode
   kernel without the copy won `faster` x3 in its (higher-margin)
   run. The cost exists; the verdict flip is drift meeting it.

Prefill-big reads the other way and is the encouraging half:
candidate 1098/2097 vs the 1299/2513 anchor (-15%/-17%) against a
compiled headwind (+6%/+3% slower bar) — consistent with removing
most of the 355/610 us shape residual via the 29% footprint cut,
far above the guessed 10%. But it is cross-run and
drift-confounded (see cause 1), so it stays a suggestion until a
same-run baseline comparison confirms it. At 127 rows no change was
expected (same kernel-A work + copy) and the +17% against a +11%
bar move is noise/drift on a small absolute.

Next change (one): copy this attempt byte-identically to a new
attempt number and re-grade WITH `--baseline
muse-spark/round-2/kernels/routing/attempt-4`, so one run isolates
the copy-kernel cost at decode and the residency saving at
1,016/2,048 rows from machine-state drift. (The hypothesis names no
baseline, so the lead must bless the flag; it changes no verdict,
only adds the paired column.)
