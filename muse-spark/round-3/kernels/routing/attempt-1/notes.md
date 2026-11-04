# attempt-1 (round 3): max_rows hand-off on round-2 attempt-4 (builder-3-a)

Hypothesis `hyp-3-routing-prefill` (slot `routing`, baseline `none`),
direction (a): the `max_rows` hand-off. Starting point
`muse-spark/round-2/kernels/routing/attempt-4` (staged scalar path;
G1+G2 pass at all six sizes, wins every decode size, `slower` at
every prefill size: 2.6/5.9/8.7x compiled).

## The one main change

`block-0.metal` and `block-1.metal` are byte copies of the starting
point's (verified: `diff` against
`muse-spark/round-2/kernels/routing/attempt-4/` is empty).
`spec.json` is the starting point's plus one key: `"max_rows": 32`.

Rows 1-32 run the candidate kernels; rows above 32 run the slot's
reference under `mx.compile` through the attempt's own dispatch
(`bench/grader/slots.py:327-337`). The regime branch on `rows`
(GEMV at `rows == 1`, split-K while `rows <= 768`), the
`rows`-only split-K partition formula, the verbatim sigmoid, the
strictly-greater top-4 scan, the weights fold, grids, and the
28,080 B threadgroup layout are untouched. The split-K branch for
33-768 rows and the single-accumulator branch above 768 are now
dead code at every graded size (all graded sizes above 32 hand
off), kept verbatim so the kernels stay valid if `max_rows` is
ever raised.

## Which inefficiency it removes, and how

`ineff-baseline-routing-1` (confirmed, traffic: per-row `w_router`
re-read, excess 116.1/935.4/1886.5 us/layer at 127/1,016/2,048
rows) and `ineff-baseline-routing-2` (confirmed, shape: 24
barriers, 28 KB residency cap, 23% single-partition duty, residual
24.1/142.7/336.8 us/layer) together. This attempt removes the
whole staged path from the graded prefill sizes: every prefill
size hands to the compiled reference, so prefill times `same` as
compiled by construction and the full prefill gap (measured -
compiled = 139.8/1077.7/2222.6 us/layer) is banked without
touching a line of kernel code. Decode is untouched, so the
starting point's decode wins are kept.

## max_rows: 32, and why

The boundary's hand-off window is 32 <= `max_rows` <= 126: every
decode step (1-32 rows, `claude/muse-agent-brief.md:26`) must stay
on the staged scalar path that already wins there (`max_rows >=
32`), while any `max_rows <= 126` hands all three graded prefill
sizes (127/1,016/2,048) to compiled. Routing has no MLX kernel
switch at 60 rows (that is the experts slot's); its reference
regimes change at `rows = 1` and `rows = 768` only.

32 is the bottom of the window, picked because there is no
measured win to preserve above it: the staged path's traffic
floor alone exceeds the compiled reference's whole measured time
from 127 rows on (1.35x/4.3x/6.6x), so unlike the experts hand-off
(where 33-59 share the `qvm` decode path the kernels win), keeping
33-126 on the candidate could only slow real serving. 32 also
empties the boundary's can't-see band 33-126 (kernel-visible
counts in chunked-prefill serving with no piece verdict) and
dead-ends the regular-GEMM branch wherever the kernels run: below
the hand-off the kernels face only the GEMV (`rows == 1`) and
split-K (1 < rows <= 32) roundings, both already G2-passing and
winning. Sizes above `max_rows` exercise the compiled reference
through this attempt's own dispatch, and the compiled reference
was bit-exact with the reference at all six sizes on 2026-11-04
(`bench/micro/compiled_exact.py`).

## Predicted us per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Baseline `none`: predictions are against the starting point's
measured times and the compiled bar from the same basis run
(`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`,
as tabulated in `ineff-baseline-routing-1`: starting point
49.8/113.5/121.3 decode and 227.1/1298.8/2513.0 prefill vs
compiled 65.1/135.7/145.2 decode and 87.3/221.1/290.4 prefill).

- **Decode 1/8/32: 49.8 / 113.5 / 121.3.** Byte-identical kernels,
  identical grids, identical dispatch at rows <= 32 -- the chained
  time is the starting point's up to run-to-run machine-state
  drift of chained absolutes (round 2's grading run measured
  47.5/69.1/72.9 vs compiled 60.7/84.9/89.5 on a quieter machine;
  both runs agree the candidate beats compiled by
  13-24 us/layer). Predicted verdicts: `faster` x 3.
- **Prefill 127/1,016/2,048: = compiled (87.3 / 221.1 / 290.4 in
  the basis run).** Rows above `max_rows` run `mx.compile(ref_routing)`,
  the same function the bar times, through the same dispatch --
  identical code, so identical time up to pair noise. Predicted
  verdicts: `same` x 3. Saving vs the starting point:
  227.1-87.3 = 139.8, 1298.8-221.1 = 1077.7, 2513.0-290.4 =
  2222.6 us/layer -- the full prefill gap (traffic 116.1/935.4/1886.5
  + shape 24.1/142.7/336.8 to rounding), i.e. 2.80/21.6/44.5 ms
  per chunk over 20 layers.

Predicted: G1 pass; G2 pass at all six sizes (kernels unchanged
where they run, compiled reference where they don't);
`phases_vs_compiled` decode `faster`, prefill `not slower`;
`beats_compiled` true.

## What this attempt leaves unmeasured (for the hypothesis entry's Unverified list)

- Ungraded decode 2-7 and 9-31 on the graded split-K path
  (boundary can't-see item 1): kernels unchanged, but no piece
  verdict exists there.
- The 33-126 band (item 2) is empty by construction (`max_rows`
  32): no kernel-visible count there in any serving shape, graded
  or chunked-prefill.
- Prefill chunks of other sizes on both sides of the split-K /
  regular boundary at `rows = 768` (item 3): all hand off, so they
  exercise the compiled reference, whose exactness above
  `max_rows` at untested sizes is itself untested (item 5).
- Sizes above 2,048 (item 3), prefill beyond the first
  seeded-random chunk (item 4), whole-request noise at 1 request
  (item 6), other prompts / later decode steps / layers' drift /
  unseen exact 4th/5th ties (no graded row is one), score
  magnitudes saturating the sigmoid differently, threadgroup
  memory (item 8, unchecked by `slots.build`), unwritten
  threadgroup reads and thread-timing dependence.
- `check_sizes.py` (25 sizes incl. 33/65/129/257/768/769) not run
  by this builder -- boundary tooling, not a gate.

## Verdict

Graded 2026-11-04T18:14Z, one grader run
(`check_candidate.py .../routing/attempt-1 routing --out .../grade.json`,
no `--baseline`: baseline `none`). Verdict kept in `grade.json`.

G1 pass. **G2 pass at all six sizes** (`rows_differ` 0,
`rows_other_experts` 0, `max_rel_err` 0.0 at 1/8/32 and
127/1,016/2,048 -- no differing row anywhere, so no first
difference to cite).

Chained us/layer, reference / compiled / candidate (`grade.json`,
medians of 30 paired repeats):

| size | reference | compiled | candidate | wins | verdict |
|---|---|---|---|---|---|
| rows 1 | 251.3 | 139.4 | 103.6 | 25 | `faster` |
| rows 8 | 154.7 | 157.2 | 129.9 | 28 | `faster` |
| rows 32 | 173.7 | 149.7 | 126.5 | 25 | `faster` |
| prefill 127 | 172.2 | 95.1 | 91.3 | 18 | `same` |
| prefill 1,016 | 260.1 | 228.4 | 229.3 | 18 | `same` |
| prefill 2,048 | 322.2 | 291.7 | 299.7 | 8 | `slower` |

`phases_vs_compiled`: decode `faster`, prefill `slower`.
**`beats_compiled` false** -- on the 2,048-row prefill size alone,
where the candidate *is* the compiled reference.

## Why the attempt failed, and why 1-row decode missed by >20%

Two findings, both machine/verdict-level: the kernels are
byte-identical to the starting point, so no kernel change can
explain either.

1. **The 2,048-row `slower` is paired noise on identical code,
   not a slowdown.** Above `max_rows` the candidate runs
   `mx.compile(REFERENCE[slot])` built in `slots.build`
   (`bench/grader/slots.py:331-337`); the bar is a second
   `mx.compile` of the same reference built at timing time
   (`bench/grader/check_candidate.py:361`). Same function, same
   shapes, same `chained` wrapper, 3 warmup calls each, order
   balanced across the 30 repeats (2 functions alternate,
   `check_candidate.py:231-232`). The three prefill outcomes are
   18/18/8 candidate wins with medians 4.0% under, 0.4% over and
   2.7% over compiled -- all inside the brief's own "the same
   kernel timed twice differs by up to 5%" band
   (`claude/muse-agent-brief.md:211`). A systematic handicap in
   the candidate-side compile instance would lose at all three
   sizes; instead 127/1,016 split for the candidate and only the
   last-timed size (hottest machine) hit exactly the 22-loss
   `slower` threshold. P(<=8/30 | fair coin) ~= 0.008 -- a rare
   verdict-level event over 30 correlated repeats, and the prize
   itself (prefill runs compiled) is genuinely banked: the worst
   median gap anywhere above the hand-off is 2.7%.
2. **The 1-row decode absolute missed because this run's machine
   was ~2x slower at small sizes.** Predicted 49.8, measured
   103.6 (+108%); but compiled moved 65.1 -> 139.4 (+114%) and the
   plain reference measured 251.3 in the same run. 8 rows:
   predicted 113.5, measured 129.9 (+14%); 32 rows: 121.3 ->
   126.5 (+4%). The relative wins hold and match the prediction
   (`faster` x 3, ahead 35.8/27.3/23.2 us/layer, 25/28/25 wins):
   chained absolutes drift with machine state (same observation
   as builder-1-b's and builder-2-b's notes this round), most
   where fixed overhead dominates. Nothing in the kernels changed,
   so the drift is the whole story.

Next change (one): re-grade a byte-identical copy under a new
attempt number to confirm the 2,048-row `slower` does not
reproduce -- no kernel edit can move identical-code timing, and a
second paired run is the only probe that distinguishes a 1-in-125
noise event from a real hand-off cost. The lead decides whether
that re-run happens; this attempt's verdict stands as graded.
