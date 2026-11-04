# attempt-1 (round 4): first-touch partial init — dead `parts` zeroing removed (builder-2-b)

Hypothesis `hyp-2-routing-margin` (slot `routing`, baseline `none`),
direction (b): trim the staged scalar path's shape cost without
touching the per-row `w_router` re-read. Starting point is
`muse-spark/round-3/kernels/routing/attempt-6` (byte-identical
round-2 attempt-4 staged scalar kernels plus `"max_rows": 32`;
G2 exact, decode `faster` ×3; round-4 re-grade margins
17.7 / 15.8 / 19.7 µs/layer over compiled at 1 / 8 / 32 rows).

## The one main change

Delete the 2,048-write `parts[]` zeroing loop and initialise each
(partition, expert) partial on first touch instead:

- Removed (was starting-point lines 30–32): `for (i = tid; i < 2048;
  i += 256) parts[i] = 0.0f;` before the first barrier.
- In the scalar compute, `acc = parts[p*64+e]` becomes
  `acc = (c_lo == pc0) ? 0.0f : parts[p*64+e]`: the first staged group
  covering a partition starts from the same `0.0f` the zero loop
  stored; later groups resume the stored partial exactly as before.

Nothing else changes: `block-1.metal` and `spec.json` are byte copies
of the starting point; grid `rows*256`, threadgroup 256 (unchanged,
inside Metal's real per-pipeline cap — no thread-count change to
re-prove), one kernel, no added launch (round 3 attempt-3's lesson),
threadgroup footprint unchanged at (816 + 60 + 2048 + 4096) × 4 B =
28,080 B, staging, partition formula, barrier count (27/row),
sigmoid, strictly-greater top-4 and weights fold all verbatim.

## Which inefficiency it removes, and how

`ineff-baseline-routing-2` item 4 (dead work at `rows == 1`) and the
used-but-avoidable half of the same zeroing at 2–32 rows. At
`rows == 1` the GEMV branch reads neither `parts` nor `wsh`, so the
2,048 threadgroup writes (8 per thread) in front of the first barrier
were pure waste. At 2–32 rows the zeroing was load-bearing only as
the partials' initial value; first-touch initialisation supplies the
identical `0.0f` at the identical point in each accumulator's life,
so the writes are dead there too. Items 1–2 (27 barriers, 23%-duty
tail groups) and item 5 (serial tails) are deliberately untouched:
this is the one change that is certain in direction, per the entry's
own bound (dead zeroing: sub-µs/layer at 1 row, certain).

## Why this is bit-exact

Every (partition, expert) accumulator still starts at `0.0f` and adds
the same chunks in the same increasing-`d` order (group iteration,
`c_lo..c_hi` clamp and partition formula unchanged); a threadgroup
store/load of a float is exact, so resuming from `parts` vs starting
from the literal `0.0f` cannot change any rounding. The GEMV branch,
the final partition-order sum, the verbatim sigmoid, the
strictly-greater earliest-index-wins top-4 and the
`(((s0+s1)+s2)+s3)+1e-20`, divide, ×2.5 fold are untouched. No
accumulation is reordered, sharded or regrouped; the per-row
`w_router` re-read (`ineff-baseline-routing-1`) is not touched.

## max_rows: 32, and why

Kept from the starting point. Decode steps have 1–32 rows, so every
decode step stays on these kernels; 127 / 1,016 / 2,048 rows run the
compiled reference and are reported `handed off` (not timed). No
`min_rows`: handing off 1 row would run MLX's own code for a whole
1-request workload. Sizes above `max_rows` and (no `min_rows`) below
it: none below; all above run the reference.

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Anchor: the starting point's round-4 re-grade candidate times
45.1 / 144.2 / 151.3 µs/layer at 1 / 8 / 32 rows (compiled
62.8 / 160.0 / 171.0 by the margins 17.7 / 15.8 / 19.7;
`ineff-baseline-routing-2` cost table). Saving arithmetic: 2,048
threadgroup writes removed per row = 8 stores per thread; even at
8 cycles per store that is ~64 cycles ≈ 0.05 µs at ~1.3 GHz, plus
the loop's index/branch overhead — bound the saving at ≤0.5 µs/layer
at 1 row and ≤0.3 µs/layer at 8/32 rows (more threadgroups resident,
more overlap). Prediction is therefore the anchor minus ≤ the bound:

- 1 row: **≈45 µs/layer** (45.1 − ≤0.5), verdict `faster` (margin 17.7).
- 8 rows: **≈144 µs/layer** (144.2 − ≤0.3), verdict `faster` (margin 15.8).
- 32 rows: **≈151 µs/layer** (151.3 − ≤0.3), verdict `faster` (margin 19.7).
- 127 / 1,016 / 2,048 rows: `handed off` — not timed; the attempt's
  time there is the compiled reference's by construction.

Chained absolutes drift with machine state (identical starting-point
bytes measured 51.7 / 90.9 / 132.2 in round 3's quieter run), so the
load-bearing prediction is the verdicts: G1 pass, G2 pass at all six
sizes, decode `faster` ×3, prefill `handed off` ×3,
`beats_compiled` true. The deleted writes are far too small to flip
any verdict either way; any measured delta beyond ~1 µs/layer vs the
same-run starting point would be machine drift, not this change.

## What this attempt leaves unmeasured (for the hypothesis entry's Unverified list)

Everything the starting point left: ungraded decode counts 2–7 and
9–31 (same `tm = 1, partitions = 32, base_iters = 1` geometry, now
also exercising first-touch init — reasoned bit-exact, not graded);
the 33–126 band (empty by construction under `max_rows: 32`);
hand-off exactness above `max_rows` at untested sizes; sizes above
2,048; later prefill chunks; other prompts and later decode steps;
unseen exact 4th/5th ties; the barrier/tail-duty costs (items 1–2,
5) themselves, still undecomposed; whether the sub-µs saving is
resolvable at all inside paired-repeat noise; whole-request (G3)
effect, which only the ready check measures as a set.

## Verdict

Graded 2026-11-04T21:54Z, one grader run
(`.venv/bin/python bench/grader/check_candidate.py
muse-spark/round-4/kernels/routing/attempt-1 routing --out
muse-spark/round-4/kernels/routing/attempt-1/grade.json`, no
`--baseline`: baseline `none`). Verdict kept in `grade.json`.

G1 pass. **G2 pass at all six sizes** (`rows_differ` 0,
`rows_other_experts` 0, `max_rel_err` 0.0 at 1 / 8 / 32 and
127 / 1,016 / 2,048 — no differing row anywhere, so no first
difference to cite). First-touch init is bit-exact as reasoned.

Chained µs/layer, reference / compiled / candidate (medians of 30
paired repeats):

| size | reference | compiled | candidate | wins | verdict |
|---|---|---|---|---|---|
| rows 1 | 84.2 | 86.1 | 58.5 | 30 | `faster` |
| rows 8 | 164.8 | 157.3 | 136.9 | 26 | `faster` |
| rows 32 | 112.4 | 92.1 | 74.7 | 30 | `faster` |
| prefill 127 | 101.2 | 104.6 | 104.6 (= compiled) | — | `handed off` |
| prefill 1,016 | 259.8 | 232.9 | 232.9 (= compiled) | — | `handed off` |
| prefill 2,048 | 340.3 | 276.7 | 276.7 (= compiled) | — | `handed off` |

`phases_vs_compiled`: decode `faster`, prefill `handed off`.
**`beats_compiled` true.** Baseline `none`, so no `beats_baseline`.
Decode margins over compiled this run: 27.6 / 20.4 / 17.4 µs/layer
(re-grade starting point: 17.7 / 15.8 / 19.7).

## Why the absolutes missed the prediction by >20% (1 and 32 rows)

Predicted ≈45 / 144 / 151 µs/layer (starting point's re-grade
anchor minus ≤0.5); measured 58.5 / 136.9 / 74.7 — +30% at 1 row,
−5% at 8 rows, −51% at 32 rows. The miss is machine state, not this
change, and the same run proves it: the compiled bar itself moved
+37% at 1 row (62.8 → 86.1) and −46% at 32 rows (171.0 → 92.1)
against the identical re-grade anchor, while the candidate/compiled
ratios stayed consistent (0.68 / 0.87 / 0.81 here vs 0.72 / 0.90 /
0.88 at re-grade) and the paired wins were decisive (30 / 26 / 30).
A deleted 2,048-write loop worth ≤0.5 µs/layer cannot move a chained
absolute by 13–76 µs; identical starting-point bytes had already
shown the same drift across runs (51.7 / 90.9 / 132.2 in round 3's
grade vs 45.1 / 144.2 / 151.3 at re-grade). The change's own effect
is inside paired noise, as predicted; what is measured is that the
margin survived intact — decode `faster` ×3 with 17.4–27.6 µs/layer
to spare. This note is also appended to the inefficiency entry.
