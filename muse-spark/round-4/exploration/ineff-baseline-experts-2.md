# Inefficiency: serial K-chain and idle simdgroups hold 8/32 rows 80–201 µs/layer above their own byte floor

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check the floor arithmetic and the round-2/3 chain evidence

## Slot

`experts`

## What happens

Every pair's decode work runs as one barrier-serialized
chain inside a 512-thread threadgroup
(`muse-spark/round-3/kernels/experts/attempt-1/block-0.metal`):
`tg_x` load + barrier (`:29-32`); gate/up interleaved over
24 strided K steps on 12 of 16 simdgroups — 4 simdgroups
idle (`:51-58`); barrier (`:73`); bfloat16 silu-mul on 384 of
512 threads (`:77-85`); barrier (`:86`); down interleaved
over 12 merged K steps, again 12 of 16 simdgroups (`:95-102`).
Each K step in the pair helpers
(`block-1.metal:69-128,137-195`) issues its loads and then
runs 64 dependent accumulates into two 32-float thread
arrays; per thread the chain is 24 + 12 = 36 such steps plus
4 `simd_sum` trees and the silu's exp/divide, with no second
wave of work inside the group to hide it. The group can hold
no more parallelism: exactly two 32-float accumulators fit
at 512 threads — a third failed G1 in round 2 (pipeline max
448), a ceiling the shared `p`/`q` arrays at
`block-0.metal:42-43` already sit on.

Round 3's `ineff-baseline-experts-2` (confirmed, frozen)
priced this chain at 1 row only (+13.5 µs/layer over
compiled) and explicitly costed 8/32 rows at 0, because the
kernel already wins there. What is new in round 4: with
`min_rows: 2` handing 1 row off and `max_rows: 59` handing
prefill off, 8 and 32 rows are the only sizes left — and at
those sizes the same chain is what caps the margin. The win
over compiled (69.3 / 328.7 µs/layer) is no longer the end
of the question; how much more is on the table at the two
surviving sizes is, above all for the set's thin
32×1024 margin (round-4 exploration README, *What is left*
item 2). This entry prices that remainder.

## Cost

Measured, piece (chained µs/layer, starting point's
`muse-spark/round-3/kernels/experts/attempt-1/grade.json`;
pair bytes 552,960 B/pair, territory §B; floors at 200 GB/s
peak, `claude/GLOSSARY.md`):

| Rows | Candidate | Compiled | Pair-byte floor | Above own floor | Effective |
|---|---|---|---|---|---|
| 1 | 96.9 | 86.6 | 11.1 µs | 85.8 µs | 22.8 GB/s |
| 8 | 168.9 | 238.2 | 88.5 µs | **80.4 µs (48%)** | 104.8 GB/s |
| 32 | 554.7 | 883.4 | 353.9 µs | **200.8 µs (36%)** | 127.6 GB/s |
| 127 / 1,016 / 2,048 | `same` (handed off) | — | — | kernel does not run | — |

- Per-pair time collapses with row count — 24.2 µs at 1 row,
  5.28 at 8, 4.33 at 32 — because more threadgroups (4 → 32
  → 128) overlap each other's chain latency. That overlap is
  the whole reason the kernel wins at 8/32 despite the chain;
  what remains above the floor is the part group parallelism
  could not hide: in-group serial depth, the three phase
  barriers, 25% of simdgroups idle in both K phases, and the
  silu chain, all behind a register ceiling that forbids
  widening the accumulators.
- Compute check: 884,736 MACs per pair (territory §B) =
  1.77 MFLOP; at 32 rows, 128 pairs in 554.7 µs ≈ 0.41
  TFLOPS, ~8% of the 5.2 TFLOPS peak. Neither bandwidth
  (52–64% of peak) nor compute is saturated.
- At 1 row this entry costs 0 under `min_rows: 2` (handed
  off); without it, the 1-row figure is round 3's confirmed
  entry and is not re-claimed here. At prefill sizes it
  costs 0 under the existing hand-off. A fix must keep both
  hand-offs and both graded wins.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): far below both peak
bandwidth and peak compute at every size; time goes to
waiting on a serial dependent chain and phase barriers, not
to bytes or arithmetic. The bytes half of the same µs is
`ineff-baseline-experts-1`; the two entries overlap and
must not be added.

## Spans

`baseline-experts` (`block-0.metal` decode path,
`block-1.metal` pair helpers, `spec.json` grid),
`grader` (the chained timing is the bar; only same-run
paired verdicts transfer).

## Predicted saving

The headroom, if the chain's exposed latency were removed
without touching a single accumulation order (every
per-block `k` sequence, term order and `simd_sum` tree is
bit-load-bearing):

- 1 row: **0** — handed off; the saving there is the spec's,
  already counted in the territory entry (−10.3 µs/layer vs
  the kernel, to compiled parity, not below it).
- 8 rows: up to 80.4 µs/layer = **1.61 ms per decode step**
  above the pair-byte floor (×20 layers).
- 32 rows: up to 200.8 µs/layer = **4.02 ms per decode
  step** above the pair-byte floor. Against the distinct-byte
  floor (22.15 MB/layer ÷ 200 GB/s = 110.8 µs, counter basis
  from entry 1) the combined headroom is 443.9 µs/layer —
  shared with entry 1, never added to it.
- Prefill 127/1,016/2,048: **0** — kernel does not run.

**Prediction: 0 banked today; the figures above are
headroom, not a promise.** Round 2's evidence bounds the
levers: six fused exact packings plateaued at 92–110
µs/layer at 1 row (the chain shortened by ~10 µs for the
gate/up interleave, ~2 for parallel stores, ~0 for the down
merge), the accumulator ceiling forbids the obvious
widening, and round 3's unfused regrouping (attempt-5) and
vectorised scalar path (attempt-2) both lost — all on the
round-4 README's refuted list for 1 row, and none of them
measured an 8/32-row gain either. A builder claiming this
entry needs a mechanism that shortens serial depth or
fills the 4 idle simdgroups *at 8/32 rows* while keeping
per-block order exact — for example splitting each pair's
blocks across all 16 simdgroups with a second reduction
stage, if its extra `simd_sum`/barrier cost does not eat
the gain — and must show it in the same grader run at both
sizes. Until one grades, the bankable round-4 move in this
slot remains the spec-only `min_rows: 2`, which spends
none of this headroom and risks none of the wins.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** barrier-serialised chain in [block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/block-0.metal:29) — `tg_x` load + barrier (:29-32), gate/up on `simd < 12` of 16 simdgroups (:51-58), barrier (:73), silu-mul 384 of 512 threads (:77-85), barrier (:86), down on 12 simdgroups (:95-102) ✓; K steps 768÷32 = 24 and 384÷32 = 12 in [block-1.metal:69-128,137-195](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/block-1.metal:69), 64 dependent accumulates per step into two 32-float arrays, 36 steps + 4 `simd_sum` trees per pair ✓; two accumulators `p`/`q` at [block-0.metal:42-43](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/block-0.metal:42) at the pipeline-448 ceiling (third array G1-fails, frozen round-3 [ineff-baseline-experts-2.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/exploration/ineff-baseline-experts-2.md), confirmed) ✓.
- **confirmed:** floors from [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/grade.json) at 200 GB/s ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) — pair floors 2.21/17.69/70.78 MB → 11.1/88.5/353.9 µs; above-floor 96.9−11.1 = 85.8, 168.9−88.5 = **80.4 (48%)**, 554.7−353.9 = **200.8 (36%)** µs/layer ✓; per-pair 24.2/5.28/4.33 µs ✓; compute 884,736 MACs = 1.77 MFLOP/pair, 128 pairs in 554.7 µs ≈ 0.41 TFLOPS ≈ 8% of 5.2 TFLOPS, bandwidth 52–64% of peak — overhead-bound ✓; distinct floor 22.15 MB ÷ 200 GB/s = 110.8 µs, combined headroom 554.7−110.8 = 443.9 µs/layer shared with entry 1, never added ✓ (443 MB is the must-read model column of [tables.md](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-02-gpu-counters/tables.md); calling it “counter basis” is a label slip, the bytes and floor are right).
- **confirmed:** saving arithmetic — 80.4×20 = 1.61 ms/step, 200.8×20 = 4.02 ms/step headroom; 1 row 0 under `min_rows: 2`, prefill 0 ✓; levers bounded by six packings at 92–110 µs/layer, interleave ~10 µs, stores ~2 µs, down-merge ~0 (frozen round-3 entry above) and round-3 attempts 2/5 losing ([hyp-2-experts-prefill.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/implementation/hyp-2-experts-prefill.md)) ✓; 0 banked today, headroom not a promise ✓.

## From other agents
