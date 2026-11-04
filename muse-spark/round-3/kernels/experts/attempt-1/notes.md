# attempt-1 (round 3): max_rows hand-off on round-2 attempt-7 (builder-2-a)

Hypothesis `hyp-2-experts-prefill` (slot `experts`, baseline `none`),
direction (a): the `max_rows` hand-off. Starting point
`muse-spark/round-2/kernels/experts/attempt-7` (fused decode path +
G2-exact scalar prefill branch; G1+G2 pass, `slower` at 1 row and at
every prefill size).

## The one main change

`block-0.metal` and `block-1.metal` are byte copies of the starting
point's (verify: `diff` against
`muse-spark/round-2/kernels/experts/attempt-7/`). `spec.json` is the
starting point's plus one key: `"max_rows": 59`.

Rows 1–59 run the candidate kernels; rows above 59 run the slot's
reference under `mx.compile` through the attempt's own dispatch
(`bench/grader/slots.py:327-337`). The fused decode path, the
`x_shape[0] >= 60` dispatch threshold (`block-0.metal:22`), the scalar
`qmm_accum` prefill branch, grids, and all four bfloat16 rounding
points are untouched — the scalar branch is now dead code at every
graded size (it can only trigger at rows ≥ 60, which all hand off),
kept as fallback so the kernels stay valid if `max_rows` is ever
raised.

## Which inefficiency it removes, and how

`ineff-baseline-experts-1` (confirmed): the `rows >= 60` scalar
`qmm_accum` branch runs 15.8/27.7/20.1× slower than compiled
(15,220/119,154/240,681 vs 961/4,306/11,999 µs/layer) — ~29.5
µs/pair serial issue at 18.8 GB/s. This attempt removes the whole
branch from the graded path: every prefill size (127/1,016/2,048
rows) runs the compiled reference, so prefill times `same` as
compiled by construction. Decode is untouched, so the 8/32-row wins
(ahead 72.7/282.5 µs/layer per the hypothesis entry) are kept.

## max_rows: 59, and why

The boundary's hand-off window is 32 ≤ `max_rows` ≤ 59:
decode steps run 1–32 rows and must stay on the fused path
(`max_rows >= 32`), while MLX switches its own experts kernel at 60
rows (`gather_qvm` below, `gather_qmm_rhs` at/above), so `qvm`-only
kernels must never see rows ≥ 60 (`max_rows <= 59`). This attempt's
kernels replicate both arithmetics (the scalar branch is G2-exact),
but decode rows 33–59 take the `qvm` decode path in both the kernels
(`block-0.metal:22`) and the reference, so keeping them on the
candidate is exact and keeps the fast path where it wins. 59 is the
top of the window: every decode-size input (≤ 32, plus 33–59) stays
on the candidate, every prefill-size input hands off. Round 2's
checked hand-off copy also used 59
(`results/runs/2026-11-04-handoff-check/`), and the compiled
reference was bit-exact with the reference at all six sizes there.

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Baseline `none`: predictions are against the starting point's
measured times and the compiled bar from its run.

- **Decode 1/8/32: 93.8 / 160.1 / 510.1.** Identical kernels,
  identical grids, identical dispatch at rows ≤ 32 — the chained
  time should reproduce attempt-7's 93.8 / 160.1 / 510.1 vs
  compiled 74.8 / 236.5 / 792.7 (attempt-7 `notes.md`, Verdict),
  modulo run-to-run noise (compiled shifted 75.9 → 74.8 at 1 row
  between attempt-5/7 runs). Verdicts: `slower` at 1 row, `faster`
  at 8 and 32.
- **Prefill 127/1,016/2,048: 961 / 4,306 / 11,999 (= compiled).**
  The hand-off runs literally the compiled reference, so each
  prefill size times `same` as the compiled bar measured in the
  same run; the numbers are attempt-7's run (hypothesis entry,
  *Inefficiency*). Saving vs the starting point: 14,259 / 114,848
  / 228,682 µs/layer = 285 ms / 2.30 s / 4.57 s per chunk
  (hypothesis entry arithmetic).
- **G1: pass** — the launch is attempt-7's (512 threads, 3,840 B
  threadgroup memory); `max_rows` adds no kernel.
- **G2: pass, differ 0 at all six sizes** — rows ≤ 59 run
  attempt-7's G2-exact kernels; rows above run the compiled
  reference, bit-exact with the reference at all six sizes on
  2026-11-04 (`results/runs/2026-11-04-handoff-check/`).
- **`beats_compiled`: false** — prefill `same` at all three sizes
  by construction, decode `faster` at 8/32 but `slower` at 1 row;
  the bar needs one phase `faster` at every size with neither
  `slower` anywhere. The claim is "faster decode, unchanged
  prefill" (ADR 0011): this attempt banks the prefill prize
  (removes `slower` at all three prefill sizes) and leaves the
  1-row decode gap to hyp-4.

## What this attempt leaves unmeasured

- Kernel speed at 33–59 rows (runs the candidate's `qvm` decode
  path in real serving; the grader times only 1/8/32 there).
- Sizes above 2,048 rows; prefill beyond the first chunk of seeded
  128-token prompts; other lengths/distributions; mixed batches.
- The boundary's full *can't see* list (unwritten threadgroup
  reads, thread-timing dependence, 32 KB limit by hand — 3,840 B,
  unchanged — NaN-vs-NaN equality, 1-request G3 noise).

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** — no first difference. Chained timing µs/layer
(`grade.json`; candidate vs reference vs compiled):
**1 row 96.9 / 82.2 / 86.6 (`slower`, 5 wins)**; 8 rows 168.9 /
245.0 / 238.2 (`faster`, 29 wins); 32 rows 554.7 / 812.4 / 883.4
(`faster`, 29 wins); prefill 127: 1053.5 / 1007.7 / 1131.1
(`same`, 19 wins); prefill 1,016: 4106.9 / 4140.7 / 4133.5
(`same`, 15 wins); prefill 2,048: 7557.0 / 7668.8 / 7557.8
(`same`, 16 wins). `phases_vs_compiled`: decode `slower`,
prefill `not slower`. **`beats_compiled` false** — only the 1-row
decode size misses. Baseline `none`, so no `beats_baseline`.

The prefill prize is banked: `slower` at all three prefill sizes
(starting point: 15,220/119,154/240,681 µs/layer) is now `same`
(1,053.5/4,106.9/7,557.0), saving ≈ 14,167/115,047/233,124
µs/layer ≈ 283 ms / 2.30 s / 4.66 s per chunk against the starting
point's run. Decode keeps the 8/32-row wins (ahead 69.3/328.7
µs/layer of compiled this run).

## Miss analysis (prediction 93.8 / 160.1 / 510.1 decode, = compiled prefill)

Decode misses are +3.3% / +5.5% / +8.7% — all inside the 20% bar.
The whole shift is run-to-run machine noise, not the change: this
attempt's kernels are byte-identical to attempt-7's at rows ≤ 32,
and the compiled bar itself moved between runs (74.8 → 86.6 at 1
row, 236.5 → 238.2 at 8, 792.7 → 883.4 at 32). Prefill matched
the prediction exactly in kind (`same` at all three sizes); the
127-row candidate mean (1,053.5) sitting below the compiled mean
(1,131.1, 19 wins) is the same noise — both sides run literally
the same compiled code, and 19 wins is well short of the 22-win
`faster` bar.

Lesson for `ineff-baseline-experts-1`: the hand-off removes the
confirmed prefill cause in full at zero decode cost (decode
verdicts identical to the starting point's: slower/faster/faster).
What remains in this slot is the disjoint 1-row decode gap
(`ineff-baseline-experts-2`, hyp-4).

## Next change

Do not touch prefill — `same` is the ceiling by construction.
Build hyp-4 on this attempt: attack the 1-row decode remainder
(`tg_x` load + silu + barrier chain, chained fixed cost) while
keeping `max_rows: 59` and every accumulation order.
