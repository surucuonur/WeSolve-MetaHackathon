# attempt-2 (round 4): same 1-row hand-off with narrower kernel window (builder-1-b)

Hypothesis `hyp-1-experts-minrows` (slot `experts`, baseline `none`),
direction (b): same 1-row hand-off with a narrower kernel window.
Starting point `muse-spark/round-3/kernels/experts/attempt-1`
(byte-identical round-2 attempt-7 kernels, `max_rows: 59`; G1 pass,
G2 differ 0 at all six sizes, 1 row `slower` 96.9 vs 86.6 µs/layer,
8/32 rows `faster` by 69.3 / 328.7 µs/layer).

## The one main change

`block-0.metal` and `block-1.metal` are byte copies of the starting
point's (MD5 `a1e95fc49844f714afd1a1e377967dae` /
`631e710f5dd343427667a2270a166e6d`; `diff` clean against
`muse-spark/round-3/kernels/experts/attempt-1/`). `spec.json` is the
starting point's plus `"min_rows": 2` with `"max_rows"` narrowed
from 59 to 32. No kernel byte, grid, threadgroup (512 threads),
accumulation order or bfloat16 rounding point changes.

Rows 2–32 run the candidate kernels; rows < 2 (exactly `rows == 1`)
and rows > 32 run the slot's compiled reference through the
attempt's own dispatch (`bench/grader/slots.py:330-345`). At the six
graded sizes: 1 row handed off, 8/32 rows on the kernels,
127/1,016/2,048 rows handed off (all > 32, as > 59 before). The
ungraded 33–59 band, which direction (a) (`max_rows: 59`) keeps on
the kernels, now runs the compiled reference too — the only
difference from direction (a), and one no graded size can see.

## Which inefficiency it removes, and how

The starting point loses `beats_compiled` on the 1-row decode size
alone: 96.9 vs 86.6 µs/layer `slower` (5 wins), while winning at 8/32
rows by 69.3 / 328.7 µs/layer (round-3 `grade.json`). Round 3's
kernel fixes for 1 row bought nothing (vectorised scalar) or 2×
worse (unfused), both refuted. This attempt removes the 1-row loss
from the verdict, not from the machine: with `min_rows: 2` the
1-row size runs the compiled reference itself, so it reports
`handed off` and decode is judged on 8 and 32 rows alone — both
already `faster`. Narrowing `max_rows` to 32 does not change any
graded size: decode never exceeds 32 rows and the smallest graded
prefill is 127 rows, so the 33–59 band is ungraded either way; in
serving it only affects hypothetical mid-size calls, which now take
the compiled `gather_qvm` path instead of the candidate's identical
`qvm` decode path. Prefill is unchanged by construction. The claim
is "faster at 8/32-row decode, unchanged everywhere else"
(boundary, *What must stay identical*).

## min_rows: 2 and max_rows: 32, and why

- **`min_rows: 2` (required exactly).** `rows < 2` is exactly
  `rows == 1` (`rows` ≥ 1). Among graded decode sizes only 1 row
  hands off; 8/32 rows run the kernels; in serving every 1-row
  decode step hands off and every 2–32-row step runs the kernels.
  `min_rows: 1` would hand off nothing and leave the 1-row loss;
  `min_rows ≥ 3` would also hand off serving counts
  2…`min_rows−1`, which neither direction proposes. With 1 request
  every decode step has 1 row, so this slot then runs MLX's own
  code for the whole 1-request workload and can pass G3 only in a
  set whose other kernels are faster there (boundary; ADR 0015) —
  a whole-request fact the piece timing shows only as `handed off`
  at 1 row.
- **`max_rows: 32` (narrow window).** MLX switches its experts
  kernel at 60 rows (`gather_qvm` below, `gather_qmm_rhs` at/above).
  32 is the bottom of the boundary's valid window
  32 ≤ `max_rows` ≤ 59 with `min_rows: 2`: it keeps every decode
  step from 2 to 32 rows on the fused path (1 row is the sole,
  intended exception) and keeps the kernels off the `qmm_rhs` side
  entirely. The first handed-off count above, 33, is still on the
  `qvm` side of the switch, taken by the compiled reference. The
  in-kernel `x_shape[0] >= 60` dispatch and scalar `qmm_accum`
  prefill branch stay as unreachable fallback wherever the kernels
  run (2–32 rows).

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Baseline `none`: predictions are against the starting point's
measured times in its own grade run
(`muse-spark/round-3/kernels/experts/attempt-1/grade.json`:
candidate / compiled 96.9/86.6, 168.9/238.2, 554.7/883.4,
1,053.5/1,131.1, 4,106.9/4,133.5, 7,557.0/7,557.8 µs/layer).

- **Decode 1 row: 86.6, `handed off`.** Handed off by construction:
  the attempt runs literally the compiled reference, so its time
  is that run's compiled time; 86.6 is the starting run's compiled
  value. The starting point's 96.9 candidate time and its `slower`
  verdict (5 wins) disappear as `handed off`, not as a win. Saving
  vs the starting candidate: 96.9 − 86.6 = 10.3 µs/layer of loss
  removed from the verdict (the machine still pays the compiled
  86.6; with 1 request this slot contributes no speed change).
- **Decode 8 rows: 168.9, `faster`.** Kernels, grid and dispatch at
  8 rows are byte-identical to the starting point, so the chained
  time should reproduce its 168.9 vs compiled 238.2 (29 wins):
  saving 238.2 − 168.9 = 69.3 µs/layer × 20 layers ≈ 1.39 ms/step.
- **Decode 32 rows: 554.7, `faster`.** Same arithmetic: reproduce
  554.7 vs compiled 883.4 (29 wins): saving
  883.4 − 554.7 = 328.7 µs/layer × 20 ≈ 6.57 ms/step.
- **Prefill 127 / 1,016 / 2,048: 1,131.1 / 4,133.5 / 7,557.8, all
  `handed off`.** All > 32, so each runs literally the compiled
  reference and takes that run's compiled times; the numbers are
  the starting run's compiled values. Direction (b)'s narrowing
  changes none of them: every prefill size handed off under
  `max_rows: 59` already, and still does under 32.
- **G1: pass** — the launch is the starting point's (512 threads,
  3,840 B threadgroup memory: `tg_x[768] + tg_gate/up/h[384]`
  bfloat16, `block-0.metal:24-27`); `min_rows`/`max_rows` add no
  kernel.
- **G2: pass, differ 0 at all six sizes** — rows 2–32 run the
  starting point's G2-exact kernels; handed-off sizes run the
  compiled reference, bit-exact with the reference at all six
  sizes on 2026-11-04 (`results/runs/2026-11-04-handoff-check/`).
- **`beats_compiled`: true (predicted)** — decode `faster` (every
  kernel-run size, 8 and 32, `faster`; at least one such size),
  prefill `handed off`, neither phase `slower`
  (`bench/grader/check_candidate.py:253-271`). Baseline `none`,
  so no `beats_baseline`. Direction (b) predicts the same graded
  verdicts as direction (a): the 33–59 difference is ungraded.

## What this attempt leaves unmeasured

- Kernel speed at decode counts 2–7 and 9–31 (kernel-run in
  serving; the grader times/tests only 1/8/32 there), and — in
  this direction — the 33–59 band's hand-off cost: it now runs the
  compiled reference, whose speed there is unmeasured by the piece
  timing (no graded size in 33–59). Direction (a) leaves the
  band's kernel speed unmeasured instead; neither measures it.
  Kernel correctness at 2–7/9–31 rests on the per-pair,
  rows-independent construction; 33–59 correctness here is the
  compiled reference's.
- The exact hand-off bounds: at the six graded sizes any
  `min_rows` in 2–8 hands off only the 1-row size and any
  `max_rows` in 32–59 runs 8/32 and hands off all prefill — the
  grader cannot tell `min_rows: 2` (right) from 3–8, nor
  `max_rows: 32` (this attempt) from 59, nor from a value above
  59 (wrong side: kernels would meet `gather_qmm_rhs` rows).
  Only reading `spec.json` (`min_rows == 2`,
  `32 ≤ max_rows ≤ 59`, ints, not bools) distinguishes them
  (boundary, *can't see*).
- The 1-request whole-request consequence (this slot runs MLX's
  own code for the whole 1-request workload; passable only inside
  a faster set) and the three-slot interaction at 8 requests —
  visible only to the ready check / G3.
- Sizes above 2,048 rows; prefill beyond the first chunk of seeded
  128-token prompts; other lengths/distributions; mixed batches.
- The boundary's full *can't see* list (unwritten threadgroup
  reads, thread-timing dependence, 32 KB limit by hand — 3,840 B,
  unchanged — NaN-vs-NaN equality, 1-request G3 noise).

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** — no first difference. Chained timing µs/layer
(`grade.json`; candidate vs reference vs compiled):
**1 row 76.1 / 79.2 / 76.1 (`handed off`)**; 8 rows 170.5 /
237.2 / 236.8 (`faster`, 30 wins); 32 rows 538.1 / 809.1 / 823.2
(`faster`, 30 wins); prefill 127: 991.6 / 1,043.8 / 991.6
(`handed off`); prefill 1,016: 3,998.9 / 4,090.0 / 3,998.9
(`handed off`); prefill 2,048: 7,441.5 / 7,440.4 / 7,441.5
(`handed off`). `phases_vs_compiled`: decode `faster`, prefill
`handed off`. **`beats_compiled` true.** Baseline `none`, so no
`beats_baseline`.

Margins this run (compiled − candidate, kernel-run sizes only):
8 rows 236.8 − 170.5 = 66.3 µs/layer × 20 ≈ 1.33 ms/step; 32 rows
823.2 − 538.1 = 285.1 µs/layer × 20 ≈ 5.70 ms/step. The starting
point's 1-row `slower` (96.9 vs 86.6, 5 wins) is now `handed off`
by construction — the slot runs the compiled reference there, so
at 1 request it contributes no speed change and G3 can pass only
inside a faster set. Prefill is `handed off` at all three sizes
under `max_rows: 32` exactly as under 59; direction (b)'s graded
verdicts match direction (a)'s (attempt-1) in kind, as predicted:
the 33–59 band difference is ungraded.

## Miss analysis (prediction 86.6 / 168.9 / 554.7 decode, 1,131.1 / 4,133.5 / 7,557.8 prefill)

All six sizes inside the 20% bar — no miss to explain. Measured
vs predicted: 1 row (76.1 − 86.6)/86.6 = −12.1%; 8 rows
(170.5 − 168.9)/168.9 = +0.9%; 32 rows (538.1 − 554.7)/554.7 =
−3.0%; prefill 127 (991.6 − 1,131.1)/1,131.1 = −12.3%; 1,016
(3,998.9 − 4,133.5)/4,133.5 = −3.3%; 2,048
(7,441.5 − 7,557.8)/7,557.8 = −1.5%. The shift is run-to-run
machine level, not the change: handed-off sizes equal the
compiled bar exactly by construction (candidate == compiled at
all four), and kernel-run sizes run byte-identical kernels —
the compiled bar itself sits at 76.1 / 236.8 / 823.2 this run vs
86.6 / 238.2 / 883.4 in the starting point's run.

Lesson for the 1-row inefficiency: the spec-only hand-off
removes the 1-row `slower` from the verdict at zero kernel cost
and flips `beats_compiled` false → true; narrowing the window
to 2–32 rows costs nothing at any graded size. What remains in
this slot is the disjoint 8/32-row serial-chain headroom
(`ineff-baseline-experts-2`, confirmed: 80.4 / 200.8 µs/layer
above the byte floor, 0 banked) — untouched here, and only
worth a kernel if the `min_rows` set fails.

## Next change

No kernel change in this slot: ready-check the three-slot set
(`muse-all`) — the experts slot's 1-request contribution is
`handed off` by construction, so the set verdict, not a new
kernel, is the next measurement. If the set needs more 8/32-row
margin, attack `ineff-baseline-experts-2`'s serial K-chain at
8/32 rows in a new attempt on this spec.
