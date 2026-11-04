# Inefficiency: baseline-combine-2 — ~17 µs/layer dispatch floor

- **Owner:** explorer-baseline-combine
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:22:29Z
- **Next:** verifier to confirm; no builder action expected beyond keeping the exact fix (ineff-baseline-combine-1) on this kernel

## Slot

`combine`.

## What happens

The starting point already removed round 1's `ineff-model-2` (5 kernels
per layer → 1: `muse-spark/kernels/combine/attempt-1/spec.json`, body
`block-0.metal:12-25`; 20 launches per step instead of the reference's
100, `results/runs/2026-11-01-breakdown/tables.md:140`). What remains is
a flat ~17 µs per layer that does not move with rows: candidate
18.6 / 17.0 / 18.9 µs/layer at 1 / 8 / 32 rows (Claude re-grade) and
17.2 / 17.0 / 18.5 in round 1's own run (`attempt-1/notes.md`). The
competing organization — threadgroup-per-row, staged weights/shared,
vectorized loads, one barrier — measured 17.9 / 16.1 / 18.1
(`muse-spark/kernels/combine/attempt-2/notes.md`): same floor. The
time is the fixed cost of one `mx.fast.metal_kernel` dispatch + eval
per layer call (in round 2's timing, chained through the grader's
`link` kernel, `bench/grader/check_candidate.py:178-213`, the same for
every function timed), plus kernel launch latency at tiny grids:
3 / 24 / 96 threadgroups of 256 threads — at 1 row, 768 threads on a
16-core GPU leave most of the chip idle while the latency runs.

## Cost

Per layer at 1 / 8 / 32 rows (arithmetic shown; peak 200 GB/s and
5.2 TFLOPS from `claude/GLOSSARY.md`):

- Kernels × fixed cost: 1 kernel × ~17 µs ≈ **17 µs**, i.e. ~0.34–0.38
  ms/step over 20 layers at every row count.
- Bytes ÷ peak: (4·768·2 + 4·4 + 768·2 + 768·2) B/row = 9,232 B/row →
  9,232 / 73,856 / 295,424 B = **0.05 / 0.37 / 1.48 µs** — 0.3–8% of
  the measured time.
- Compute: ~7 f32 ops × 768 outputs ≈ 5k / 43k / 172k FLOP =
  **≤0.03 µs** at 32 rows.
- Measured: candidate 18.6 / 17.0 / 18.9 µs/layer vs compiled reference
  23.8 / 26.1 / 32.0 and eager reference 23.1 / 22.5 / 39.5 (old
  independent piece timing, `claude/findings/muse-round-1.md` §1,
  `results/runs/2026-11-04-muse-round-1/grade-combine.json`). At least
  ~15 µs/layer (≥85%) is therefore neither bytes nor arithmetic at any
  row count. The knock-out basis disagrees on the slot's standalone
  cost (0.61 / 0.33 ms/step at 1 / 32 rows, `tables.md:335`), as round
  1 already recorded; the piece basis is the one comparable to the
  compiled reference here.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): far below both the bandwidth
and compute limits; the time is launching and dispatching the kernel.

## Spans

`baseline-combine` (the fused kernel and its launch spec), `grader`
(the chained timing that prices each layer's call, link kernel
included for every function alike).

## Predicted saving

Already banked against the compiled reference by the starting point,
and retained if `ineff-baseline-combine-1` is fixed without leaving
the floor: (23.8−18.6) / (26.1−17.0) / (32.0−18.9) µs/layer × 20 =
**0.104 / 0.182 / 0.262 ms per decode step at 1 / 8 / 32 rows**
(independent piece basis; round 1's own run gives 0.114 / 0.184 /
0.190 ms — same conclusion). No further saving is predicted from
kernel-body changes: two organizations, with and without barriers,
staging and vectorized loads, land on the same floor, and traffic +
compute total <1.6 µs/layer even at 32 rows, so the remaining
~15 µs/layer is a floor this slot cannot go below — it can only be
lost (a slower exact fix) or kept. The chained-timing figure, which
round 2 ranks by, is still unmeasured for this slot (timing runs only
after G2 passes, `check_candidate.py:308-337`); treat the numbers
above as the prediction it must confirm, with G3 as the end-to-end
check (round 1 vLLM: attempt-1 alone −0.51 / −0.71 / −0.39 ms vs the
compiled reference's −0.29 / −0.26 / −0.27 ms, `muse-round-1.md` §3).

## Verdict

verifier, 2026-11-04T03:22:29Z — **confirmed**. Checked: 5→1 kernels/layer (20 vs 100/step) from [spec.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/combine/attempt-1/spec.json) and [tables.md:140](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-01-breakdown/tables.md:140) (`combine` 100 = AsType 40 + Multiply 20 + Sum 20 + Add 20) ✓. Flat ~17 µs/layer: Claude re-grade 18.6/17.0/18.9 vs compiled 23.8/26.1/32.0 and eager 23.1/22.5/39.5 ([grade-combine.json](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/grade-combine.json)) ✓; round-1 own 17.2/17.0/18.5 and attempt-2 17.9/16.1/18.1 ([attempt-1/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/combine/attempt-1/notes.md), [attempt-2/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/combine/attempt-2/notes.md)) — same floor, with/without barrier/staging ✓. Chained `link` pricing at [check_candidate.py:178-213](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:178) applies to every timed function ✓. Grids 3/24/96 threadgroups (rows·768/256) ✓. Bytes (4·768·2+4·4+768·2+768·2)=9,232 B/row → 9,232/73,856/295,424 B = 0.05/0.37/1.48 µs at 200 GB/s ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) ✓; compute ~7·768≈5k/43k/172k FLOP ≤0.03 µs at 5.2 TFLOPS ✓; remainder ≥15 µs/layer (≥85%) is neither bytes nor arithmetic ✓. Knock-out disagreement 0.61/0.33 ms/step ([tables.md:335](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-01-breakdown/tables.md:335)) ✓. Banked saving (23.8−18.6)/(26.1−17.0)/(32.0−18.9)×20 = 0.104/0.182/0.262 ms/step, own-run 0.114/0.184/0.190 ✓; no-body-change-below-floor prediction follows (traffic+compute <1.6 µs at 32 rows). Chained figure still requires G2 ([check_candidate.py:309](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:309)) as caveated.
