# Attempt 2 — builder-3-b, threadgroup-per-row staged, contraction suppressed by register barriers (direction b)

Builds on the starting point `muse-spark/kernels/combine/attempt-1` (which
fails round-2 G2 on FMA contraction only: 0 / 2 / 2 differ at decode
1 / 8 / 32 rows, 6 / 50 / 130 at prefill 127 / 1,016 / 2,048) and on
round 1's `muse-spark/kernels/combine/attempt-2` organization (one
192-thread threadgroup per row, weights + shared staged in threadgroup
memory, vectorized u64 loads, one barrier; same ~17 µs floor, same
contracted fingerprint). First attempt in this direction under the
exact-G2 rules. Claim note: attempt-1 in this folder was also claimed by
builder-3-a (log 2026-11-04T03:57:57Z); builder-3-b took attempt-2 to
avoid overwriting it.

## What changed

Organization (from round-1 attempt-2, unchanged): grid `rows*192`,
threadgroup 192, one threadgroup per row; weights (16 B) and shared
(768 f32, the exact expansion of the bf16 values) staged in threadgroup
memory (3,088 B of 32 KB); each thread then owns 4 consecutive outputs,
streaming `y` as one u64 per expert.

Arithmetic (the fix): per output, the 4 products are computed in
separate statements into separate f32 variables, each passed through a
volatile register round-trip (`{ volatile float b = p; p = b; }`), which
forces the product's f32 rounding to materialize so no mul+add pair can
contract into an FMA across it. The fold is the reference's
`((p₀+p₁)+p₂)+p₃`, each partial sum likewise volatile-rounded. Both bf16
roundings are kept: `bfloat(acc)` (the astype), then the bf16 add of the
staged shared value. This reproduces the eager reference's evaluation
sequence (products rounded once, rounding after every add) by
construction, at every row count; there is no `rows`-dependent path.

Probes (scratch, in `probes/`, never graded): the probe harness first
reproduced the starting point's known 0 / 2 / 2 differs verbatim on the
captured inputs. The asm-barrier variant
(`asm volatile("" : "+r"(p) :: "memory")`) compiled but corrupted values
(~100% differ) — Metal's inline asm does not give the Clang register
barrier semantics, so it was dropped. The volatile variant above gives
0 differs on all 20 layers × decode 1 / 8 / 32 captured inputs and on
5 × 2,048-row random stress inputs (7.8 M elements). A no-barrier staged
control also gave 0 on those inputs, but that rests on the compiler not
contracting across statements in this code shape; the volatile
round-trips make exactness independent of that choice.

## Inefficiency removed

`ineff-baseline-combine-1` (gate): FMA contraction of the 4-term sum
breaks bit-exactness (0 / 2 / 2 decode, 6 / 50 / 130 prefill) and
forfeits the slot's ≈0.10 / 0.18 / 0.26 ms/step (independent basis).
This attempt suppresses contraction while keeping the fold and both
bf16 roundings. `ineff-baseline-combine-2` (the ~17 µs/layer dispatch
floor) is kept, not attacked: one launch per layer, traffic + compute
<1.6 µs/layer even at 32 rows.

## Prediction (µs per layer, decode 1 / 8 / 32 rows)

Traffic is 9,232 B/row → 0.05 / 0.37 / 1.48 µs at 200 GB/s; compute
≤0.03 µs. The same organization without barriers measured
17.9 / 16.1 / 18.1 µs/layer in round 1 (independent timing;
attempt-1 shape: 17.2 / 17.0 / 18.5) — the dispatch floor. The 24
volatile round-trips per thread are thread-local register/stack traffic
with no extra launch, barrier or sync; expected cost <0.5 µs/layer
(`ineff-baseline-combine-1`'s bound for contraction suppression).
Round-2 timing is chained and has never run on this slot (it requires
G2 first), so predict the candidate lands on the same floor:

- 1 row: **≈18 µs/layer**
- 8 rows: **≈17 µs/layer**
- 32 rows: **≈19 µs/layer**

Against the compiled reference (round-1 independent: 22.9 / 26.2 / 28.0
µs/layer; round-2 chained figure comes from the same grader run), this
beats compiled at all three sizes if the chained compiled time stays in
that range; the 1-row margin is the thinnest. Baseline is `none`, so
there is no `beats_baseline` bar; the bar is G2 exact at all six sizes
plus `beats_compiled` on chained decode timing.

## Verdict

G1 pass, G2 pass at all six sizes, `beats_compiled` true. Baseline is
`none`, so there is no `beats_baseline` / `wins_vs_baseline` bar.

Grader output (single run,
`.venv/bin/python bench/grader/check_candidate.py muse-spark/round-2/kernels/combine/attempt-2 combine`):

- Correctness (differ / max ULP, summed over 20 layers): decode 1 row
  0 / 0.0, 8 rows 0 / 0.0, 32 rows 0 / 0.0; prefill 127 rows 0 / 0.0,
  1,016 rows 0 / 0.0, 2,048 rows 0 / 0.0 — every element bit-equal to
  the eager reference. First difference: none. The starting point's
  0 / 2 / 2 and 6 / 50 / 130 differs are closed.
- Chained timing (µs per layer, reference / compiled / candidate):
  1 row 67.5 / 47.3 / **45.2**, 8 rows 66.5 / 55.6 / **37.6**, 32 rows
  87.8 / 43.0 / **33.8**. Candidate beats the compiled reference at all
  three decode sizes by 2.1 / 18.0 / 9.2 µs/layer
  (0.042 / 0.360 / 0.184 ms/step over 20 layers). The 1-row margin is
  thin: the compiled chained figure varied between runs today
  (35.4 in builder-3-a's run, 47.3 here); `beats_compiled` is judged
  within the run.

## Why the timing missed the prediction (>20%)

Predicted ≈18 / 17 / 19 µs/layer; measured (chained) 45.2 / 37.6 / 33.8
— +151% / +121% / +78%. Two causes, both in the timing basis and the
organization, not the barrier arithmetic:

1. The prediction used round 1's independent piece timing, where the
   20 layer calls overlap and hide per-call latency. Round 2 times
   chained through the grader's `link` kernel
   (`check_candidate.py:178-213`), pricing serial latency instead.
   Every function moved the same way: builder-3-a's direction-(a)
   candidate missed +83–110% on the same basis change (predicted
   17.4 / 17.7 / 18.8, measured 31.8 / 33.6 / 39.5), and the compiled
   reference itself measured 35.4–47.3 µs/layer chained at 1 row
   against 23.8 independent. The kernel still beats compiled at every
   size, so the dispatch floor is retained in relative terms.
2. Direction (b)'s own 1-row tax, visible only once calls serialize:
   one threadgroup per row is one threadgroup total at 1 row, plus the
   staging barrier, against direction (a)'s 3 threadgroups — 45.2 vs
   31.8 µs/layer at 1 row. The ranking flips by 32 rows (33.8 vs
   39.5), where 32 threadgroups amortize the staging. The volatile
   round-trips (24 per thread) show no isolable cost: the 32-row
   candidate is the fastest combine kernel measured chained today.

Lesson: predict combine on the chained basis (~32–45 µs/layer at
1 row), and at 1 row count threadgroups, not just launches.

Next change, if the lead wants one: raise 1-row parallelism without
touching the arithmetic — e.g. two threadgroups per row (384 outputs
each) — keeping the volatile-barrier fold verbatim. Both combine
attempts now pass G1 + G2 + `beats_compiled`; the pick between them
is the lead's.
