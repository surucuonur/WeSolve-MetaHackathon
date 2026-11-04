# attempt-1 (round 4): pure spec change — `min_rows: 2` on the starting point (builder-1-a)

Hypothesis `hyp-1-experts-minrows` (slot `experts`, baseline `none`),
direction (a): pure spec change. Starting point
`muse-spark/round-3/kernels/experts/attempt-1` (byte-identical round-2
attempt-7 kernels, `max_rows: 59`; G1 pass, G2 differ 0 at all six sizes,
1 row `slower` 96.9 vs 86.6 µs/layer, 8/32 rows `faster`).

## The one main change

`block-0.metal` and `block-1.metal` are byte copies of the starting
point's (MD5 `a1e95fc49844f714afd1a1e377967dae` /
`631e710f5dd343427667a2270a166e6d`; `diff` clean against
`muse-spark/round-3/kernels/experts/attempt-1/`). `spec.json` is the
starting point's plus one key: `"min_rows": 2` (`"max_rows": 59` kept).
No kernel byte, grid, threadgroup (512 threads), accumulation order or
bfloat16 rounding point changes.

Rows 2–59 run the candidate kernels; rows < 2 (exactly `rows == 1`)
and rows > 59 run the slot's compiled reference through the attempt's
own dispatch (`bench/grader/slots.py:330-345`). At the six graded
sizes: 1 row handed off, 8/32 rows on the kernels, 127/1,016/2,048
rows handed off (all > 59, as before).

## Which inefficiency it removes, and how

The starting point loses `beats_compiled` on the 1-row decode size
alone: 96.9 vs 86.6 µs/layer `slower` (5 wins), while winning at 8/32
rows by 69.3 / 328.7 µs/layer (round-3 `grade.json`). Round 3's kernel
fixes for 1 row bought nothing (vectorised scalar) or 2× worse
(unfused), both refuted. This attempt removes the 1-row loss from the
verdict, not from the machine: with `min_rows: 2` the 1-row size runs
the compiled reference itself, so it reports `handed off` and decode
is judged on 8 and 32 rows alone — both already `faster`. Prefill is
unchanged by construction (handed off above 59 rows, as in the
starting point). The claim is "faster at 8/32-row decode, unchanged
everywhere else" (boundary, *What must stay identical*).

## min_rows: 2 and max_rows: 59, and why

- **`min_rows: 2` (required exactly).** `rows < 2` is exactly
  `rows == 1` (`rows` ≥ 1). Among graded decode sizes only 1 row
  hands off; 8/32 rows run the kernels; in serving every 1-row decode
  step hands off and every 2–32-row step runs the kernels.
  `min_rows: 1` would hand off nothing and leave the 1-row loss;
  `min_rows ≥ 3` would also hand off serving counts 2…`min_rows−1`,
  which neither direction proposes. With 1 request every decode step
  has 1 row, so this slot then runs MLX's own code for the whole
  1-request workload and can pass G3 only in a set whose other
  kernels are faster there (boundary; ADR 0015) — a whole-request
  fact the piece timing shows only as `handed off` at 1 row.
- **`max_rows: 59` (kept).** MLX switches its experts kernel at 60
  rows (`gather_qvm` below, `gather_qmm_rhs` at/above). 59 is the
  last `gather_qvm` count; the first handed-off count above, 60, is
  exactly MLX's first `gather_qmm_rhs` count. The kernels therefore
  run only on 2–59 rows and never need `qmm_rhs` arithmetic; the
  in-kernel `x_shape[0] >= 60` dispatch and scalar `qmm_accum`
  prefill branch stay as unreachable fallback. Valid window per the
  boundary is 32 ≤ `max_rows` ≤ 59 with `min_rows: 2`; direction (a)
  keeps the top of it (direction (b) takes 32).

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Baseline `none`: predictions are against the starting point's
measured times in its own grade run
(`muse-spark/round-3/kernels/experts/attempt-1/grade.json`:
candidate / compiled 96.9/86.6, 168.9/238.2, 554.7/883.4,
1,053.5/1,131.1, 4,106.9/4,133.5, 7,557.0/7,557.8 µs/layer).

- **Decode 1 row: 86.6, `handed off`.** Handed off by construction:
  the attempt runs literally the compiled reference, so its time is
  that run's compiled time; 86.6 is the starting run's compiled
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
  `handed off`.** All > 59, so each runs literally the compiled
  reference and takes that run's compiled times; the numbers are
  the starting run's compiled values. Unchanged vs the starting
  point's prefill (`same` 1,053.5/4,106.9/7,557.0 — same code, run
  noise) by construction.
- **G1: pass** — the launch is the starting point's (512 threads,
  3,840 B threadgroup memory: `tg_x[768] + tg_gate/up/h[384]`
  bfloat16, `block-0.metal:24-27`); `min_rows` adds no kernel.
- **G2: pass, differ 0 at all six sizes** — rows 2–59 run the
  starting point's G2-exact kernels; handed-off sizes run the
  compiled reference, bit-exact with the reference at all six sizes
  on 2026-11-04 (`results/runs/2026-11-04-handoff-check/`).
- **`beats_compiled`: true (predicted)** — decode `faster` (every
  kernel-run size, 8 and 32, `faster`; at least one such size),
  prefill `handed off`, neither phase `slower`
  (`bench/grader/check_candidate.py:253-271`). Baseline `none`, so
  no `beats_baseline`.

## What this attempt leaves unmeasured

- Kernel speed and correctness at decode counts 2–7 and 9–31 and at
  33–59 rows (kernel-run in serving; the grader times/tests only
  1/8/32 there). Correctness there rests on the per-pair,
  rows-independent construction.
- The exact hand-off bounds: at the six graded sizes any
  `min_rows` in 2–8 and any `max_rows` in 32–59 grade identically;
  only reading `spec.json` (`min_rows == 2`, `32 ≤ max_rows ≤ 59`)
  distinguishes them (boundary, *can't see*).
- The 1-request whole-request consequence (this slot runs MLX's own
  code for the whole 1-request workload; passable only inside a
  faster set) and the three-slot interaction at 8 requests —
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
**1 row 73.9 / 76.2 / 73.9 (`handed off`)**; 8 rows 157.6 /
233.0 / 232.1 (`faster`, 30 wins); 32 rows 571.1 / 842.5 / 832.6
(`faster`, 30 wins); prefill 127: 1,026.6 / 1,016.0 / 1,026.6
(`handed off`); prefill 1,016: 4,013.4 / 4,070.0 / 4,013.4
(`handed off`); prefill 2,048: 7,412.5 / 7,452.5 / 7,412.5
(`handed off`). `phases_vs_compiled`: decode `faster`,
prefill `handed off`. **`beats_compiled` true** — as predicted.
Baseline `none`, so no `beats_baseline`.

Next to the starting point's own run (96.9 / 168.9 / 554.7
candidate; 86.6 / 238.2 / 883.4 compiled): the 1-row `slower`
(5 wins) is gone as `handed off`, and the 8/32-row wins are kept
and slightly wider this run (ahead 74.5 / 261.5 µs/layer of
compiled: 232.1 − 157.6, 832.6 − 571.1; ≈ 1.49 / 5.23 ms/step
× 20 layers, prediction from this run's margins).

## Miss analysis (prediction vs measured)

All six sizes inside the 20% bar — no miss to explain:
1 row 73.9 vs 86.6 predicted (−14.7%), 8 rows 157.6 vs 168.9
(−6.7%), 32 rows 571.1 vs 554.7 (+3.0%), prefill 127 1,026.6 vs
1,131.1 (−9.2%), 1,016 4,013.4 vs 4,133.5 (−2.9%), 2,048
7,412.5 vs 7,557.8 (−1.9%). Handed-off sizes equal the compiled
time exactly in this run (73.9 = 73.9, 1,026.6 = 1,026.6, etc.)
by construction; the level shift vs the prediction is run-to-run
machine noise — the compiled bar itself moved between runs
(86.6 → 73.9 at 1 row, 238.2 → 232.1 at 8, 883.4 → 832.6 at 32),
and the kernel-run sizes use byte-identical kernels.

Lesson: the pure spec change does exactly what the boundary
claims — it removes the 1-row loss from the verdict at zero
kernel cost, and `beats_compiled` flips false → true on the
strength of the already-measured 8/32-row wins. What remains
unproven is whole-request: with 1 request this slot runs MLX's
own code in every layer and step, so G3 can pass only inside a
set whose other kernels win there.

## Next change

None in this slot's kernels — direction (a) is the ceiling of the
spec surface (`min_rows: 2`, `max_rows: 59`). Next: grade
direction (b) (`max_rows: 32`) for comparison, then take this
attempt into the ready check as part of the three-slot set,
where the 1-request workload and the three-slot interaction are
actually measured.
