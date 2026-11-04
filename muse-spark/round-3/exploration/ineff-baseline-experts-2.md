# Inefficiency: 1-row decode chain still 13.5 µs/layer over compiled — serial K-chain behind 4 groups at the register ceiling

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:46:33Z
- **Next:** done — confirmed by verifier (see Verdict)

## Slot

`experts`

## What happens

At 1 row the slot kernel (`muse-spark/round-2/kernels/experts/attempt-7/block-0.metal`,
`!prefill` path) launches 4 threadgroups of 512 threads —
one per pair — each running a barrier-serialized chain:
`tg_x` load + barrier (`:29-32`); gate/up interleaved, 24
strided steps (`K=768`) over 12 of 16 simdgroups
(`:51-58`); barrier (`:73`); full-bfloat16 silu-mul, 384
of 512 threads (`:77-85`); barrier (`:86`); down
interleaved, 12 merged steps (`K=384`) over 12 of 16
simdgroups (`:95-102`). That is 36 heavy K steps per
thread — each ~2×(2 scale/bias + 4 weight words) plus 64
dependent accumulates — with 4 of 16 simdgroups idle in
every phase, 4 `simd_sum` trees per thread, and an
exp/divide silu between two of the three barriers.

Why the gap survives everything round 2 tried (six fused
exact packings plateaued at 92–110 µs/layer chained:
104.3 / 103.6 / 109.5 / 94.0 / 92.1 / 93.8 — attempts
1–5, 7 own runs, per the builder notes on round-2
`ineff-baseline-experts-2.md:120-251`, frozen):

1. **The chain is at the register ceiling.** Widening
   threads/pair bought ~29 µs (132.9 → 104.3) and the
   gate/up interleave ~10 µs (104.3 → 94.0, cross-run),
   but attempt-6 proved a third 32-float accumulator
   fails G1 at 512 threads (pipeline max 448); exactly
   two fit (`attempt-7/notes.md:10-15, 227-240`). The
   per-thread chain cannot be widened further in this
   packing.
2. **Merging spent the available parallelism.** The down
   interleave is latency-neutral at 1 row (93.8 vs 92.1
   cross-run, +1.8%): down's blocks already overlapped
   across the 16-simdgroup wave, so merging trades
   cross-simdgroup parallelism for in-step parallelism
   while each step doubles its loads
   (`attempt-7/notes.md:237-255`). Group count past 4
   is not the lever either — 4, 48+96 and 12+24 groups
   all plateaued at 103.6–109.5 before the interleave
   (attempt-3 builder note). What remains is chain
   depth: 24 merged gate/up steps + 12 merged down
   steps that no packing shortens.
3. **The structural remainder is fixed.** `tg_x` load,
   silu (serial exp + divide per element), three
   barriers, and the chained launch/link cost sit
   outside every K-loop change; attempt-1 priced that
   block at ~20 µs of the 1-row layer time
   (`attempt-1/notes.md:46`, cross-harness model,
   indicative). Parallel stores saved only ~2 µs at 1
   row (94.0 → 92.1): stores were never the 1-row
   bottleneck.

Meanwhile the compiled reference runs 7 short kernels
per layer at 1 row (3×(Arange+GatherQMM) + silu-mul,
round-2 territory §E, frozen) with 24/24/48
threadgroups per product (`(M,(N+63)/64,B)` 64-thread
groups, round-2 territory §F) and no cross-phase
barriers inside any kernel — each kernel short, each
latency hidden across dozens of groups — totalling
78.6 µs/layer including the `link`. The candidate's one
kernel does less launching but one long chain: 92.1.

## Cost

Measured, piece (chained median µs/layer,
`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`):

- **1 row:** candidate 92.1 vs compiled 78.6 (gap
  **+13.5**, +17.2%; 1/30 wins, `slower`) and vs
  reference 84.0 (+8.1). Own run
  (`attempt-7/notes.md:203-209`): 93.8 vs 74.8 (+19.0)
  — the task's "~19 µs" is the own-run gap; +13.5 is
  the operative re-grade bar. Per decode step: +13.5 ×
  20 = **270 µs ≈ 0.27 ms** (own-run basis 0.38 ms),
  reproducing the G3 effect of +0.29 ms with 1 request
  (`console-vllm.txt`, via round-3 README) to within
  run shift — no second in-step cause needed.
- Bytes: 4 pairs × 552,960 B = 2.21 MB/layer → floor
  11.1 µs at 200 GB/s peak (`claude/GLOSSARY.md:10`).
  Effective: candidate 24.0 GB/s chained (12% of peak)
  vs compiled 28.1 GB/s on the same pair model.
  Kernels: candidate 1 + `link` vs compiled 7 + `link`
  per layer per call.
- **8 rows:** 162.8 vs 235.5 (ahead 72.7, −30.9%,
  30/30 wins). 32 pairs, 17.69 MB → 108.7 GB/s. No
  loss; the win must be kept.
- **32 rows:** 515.3 vs 797.8 (ahead 282.5, −35.4%,
  30/30). 128 pairs, 70.78 MB → 137.3 GB/s. No loss.
- **Prefill 127/1,016/2,048 rows:** this entry costs
  0 there — the loss at those sizes is the scalar
  branch (`ineff-baseline-experts-1`), a disjoint
  cause. A decode-path change must leave the prefill
  branch (and its G2) intact.

Decomposition of the 92.1 (prediction from static
structure + cross-run attempt deltas, not measured):
~20 µs structural (launch/link, `tg_x`, silu,
barriers — attempt-1 model) + ~60 µs exposed K-chain
latency (36 merged steps, serial accumulates,
`simd_sum` trees) + ~11 µs byte floor, overlapped.
The ordering evidence: interleave moved 1 row by ~10,
stores by ~2, down-merge by ~0 — the serial gate/up
chain and the structural block are what is left.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): 24.0 GB/s = 12%
of peak bandwidth and a small fraction of peak compute
at 1 row; time goes to waiting on a serial chain
behind 4 threadgroups, not to bytes or arithmetic.

## Spans

`baseline-experts` (`block-0.metal` decode path,
`block-1.metal` pair helpers, `spec.json` grid),
`grader` (chained-timing definition — the bar is the
chained median, `check_candidate.py:192-221`).

## Predicted saving

If the 1-row chained layer time fell to the compiled
reference's with 8/32-row wins and prefill G2 kept:

- 1 row: (92.1 − 78.6) × 20 = **270 µs ≈ 0.27 ms per
  decode step** (own-run basis: (93.8 − 74.8) × 20 =
  0.38 ms; G3 measured +0.29 ms with 1 request — the
  saving should reproduce there, since the chain
  reproduces the step to 0.02–0.11 ms).
- 8 rows: **0** (already ahead 72.7 µs/layer, 1.45
  ms/step; a 1-row fix must not spend it).
- 32 rows: **0** (already ahead 282.5 µs/layer, 5.65
  ms/step).
- Prefill 127 / 1,016 / 2,048 rows: **0** from this
  entry (disjoint cause; see `ineff-baseline-experts-1`
  for the 14.3 / 114.8 / 228.7 ms/layer prefill
  headroom). Do not add the two entries' savings at any
  size without a combined design: a decode fix that
  slows prefill, or vice versa, is not an improvement
  (ADR 0011).

This is a prediction: the headroom, not a promise — it
requires shortening the serial chain or the structural
block without breaking per-block `simd_sum` order
(decode exactness) and without adding fixed cost on
the chain. Round 2's evidence says the next lever is
not another fused packing (plateau 92–110 across six
attempts) but a reference-shaped unfused structure for
small row counts (`muse-spark/round-2/summary.md:9`) —
separate gate/up/down kernels with per-block groups,
which also fits rows 33–59 (sorted from 16 rows yet
still `qvm` below 60, round-2 territory §E, frozen).

## Verdict

verifier, 2026-11-04T17:46:33Z — **confirmed**. Every claim the cost and the saving rest on holds; one correction (a range endpoint) changes nothing.

- **confirmed:** 1-row chain — 4 threadgroups × 512 threads, one per pair (block-0:15-16); `tg_x` + barrier (:29-32); gate/up interleaved 24 strided steps over K=768 on 12 of 16 simdgroups (:51-58, helper `k=lane;k<K;k+=32`); barrier (:73); silu-mul 384 of 512 (:77-85); barrier (:86); down interleaved 12 merged steps over K=384 on 12 of 16 (:95-102); 4 of 16 simdgroups idle in every phase ✓.
- **confirmed:** per-step shape — each K step 2×(2 scale/bias + 4 weight words) + 64 dependent accumulates (both pair helpers) ✓; 4 `simd_sum` trees per thread (gate, up, down×2; block-1:121-127,188-194) ✓; 36 heavy K steps = 24 + 12 ✓.
- **confirmed:** plateau 104.3/103.6/109.5/94.0/92.1/93.8 for attempts 1/2/3/4/5/7 own runs (round-2 ineff-baseline-experts-2:120-251, all six builder notes in range) ✓; widening 132.9→104.3 ≈ 29 µs ✓; interleave 104.3→94.0 ≈ 10 cross-run ✓; attempt-6 G1 fail on the third 32-float array at 512 threads (pipeline max 448), exactly two fit (attempt-7 notes:10-15,227-240 + block-0:7-13 header) ✓.
- **confirmed:** down-merge latency-neutral at 1 row — 93.8 vs 92.1 = +1.85% → +1.8% cross-run, mechanism in notes:237-255 ✓; group counts 4 / 48+96 / 12+24 plateaued 103.6–109.5 (attempt-3 builder note) ✓; structural block ~20 µs (attempt-1 notes:46) ✓; parallel stores 94.0→92.1 ≈ 2 µs ✓.
- **confirmed:** compiled reference 7 short kernels/layer at 1 row (3×(Arange+GatherQMM) + silu-mul, territory §E) with 24/24/48 threadgroups per product (`(M,(N+63)/64,B)` 64-thread groups, §F: gate/up (1,6,B), down (1,12,B)) ✓, totalling 78.6 incl. link (JSON) ✓.
- **confirmed:** cost — 1 row 92.1 vs 78.6 (+13.5, +17.2%; 1/30 wins, slower) and vs 84.0 (+8.1) (JSON) ✓; own run 93.8 vs 74.8 (+19.0, notes:203-209) ✓; +13.5×20 = 270 µs = 0.27 ms, own-run 0.38, G3 +0.29 (round-3 README) ✓; bytes 4×552,960 = 2.21 MB → floor 11.1, effective 24.0 vs 28.1 GB/s ✓; 8 rows ahead 72.7 (−30.9%, 17.69 MB → 108.7 GB/s) ✓; 32 rows ahead 282.5 (−35.4%, 70.78 MB → 137.3 GB/s) ✓; prefill 0, disjoint from entry 1 ✓; decomposition honestly labelled prediction (20+60+11 ≈ 92.1) ✓.
- **confirmed:** saving — 0.27 ms/step at 1 row; keep-outs 1.45/5.65 ms/step at 8/32 rows ✓; prefill headroom 14.3/114.8/228.7 ms/layer matches entry 1 ✓; reference-shaped-unfused next lever (round-2 summary:9) ✓; rows 33–59 sorted-from-16 yet `qvm`-below-60 (§E) ✓; chained-median bar (check_candidate.py:192-221) ✓.
- **wrong (range endpoint):** "reproduces the step to 0.02–0.11 ms" — vs G3's 0.29 the errors are 0.02 (regrade 0.27) and 0.09 (own-run 0.38); 0.11 is own-run minus regrade-basis (0.38−0.27), not vs-G3. Correct range: 0.02–0.09.

## From other agents

- builder-2-e, 2026-11-04T18:16:56Z: attempt-5
  (`muse-spark/round-3/kernels/experts/attempt-5/`) tested this
  entry's suggested next lever — a reference-shaped unfused
  structure (gate/up+silu and down as separate per-block kernels,
  48/96 groups, all accumulation orders kept) scoped to rows == 1
  on top of attempt-1. G1+G2 pass (differ 0 all six), 8/32-row wins
  and prefill `same` x3 kept, but 1-row scores 234.3 vs compiled
  118.4 (`slower`, 0/30; within-run 1.98x vs fused 1.12x). Analysis
  in attempt-5 `notes.md`: ~+18 us from 5 dispatches vs 2 at this
  run's ~6 us/dispatch price (reference-vs-compiled), the remaining
  ~98 us from per-group setup x144 groups vs the fused kernel's
  amortized x4 (prime suspect). The gap survives unfused regrouping
  as well as the six fused packings; the reference's speed comes
  from one mx.compile dispatch, not just its shape.
