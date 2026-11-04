# Inefficiency: chained 1-row latency — 4 threadgroups with wave/barrier serialization cannot fill the GPU

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:23:12Z
- **Next:** verifier to confirm chained/independent numbers and the wave arithmetic

## Slot

`experts`

## What happens

Attempt-4 launches one threadgroup per (row, expert-slot) pair:
grid `rows*1024` threads ÷ 256 = `rows*4` threadgroups
(`muse-spark/kernels/experts/attempt-4/spec.json`;
`block-0.metal:6` `pair = threadgroup_position_in_grid.x`). At 1 row that
is 4 threadgroups / 1,024 threads on a 16-core GPU
(`claude/muse-agent-brief.md:99`). Inside each group the work is
serialized by barriers into waves (`block-0.metal`): load `tg_x` +
barrier (`:18-21`); gate 12 blocks in 2 waves of 8 simdgroups (`:28-38`)
— wave 2 uses 4 of 8 simdgroups, 4 idle under the simd-uniform
`b < 12` guard (`:29`); up identically (`:40-50`); barrier (`:51`); silu
strided loop (`:56-64`); barrier (`:65`); down 24 blocks in 3 waves of 8
(`:68-75`). Gate/up thus waste 4 of 16 simdgroup-slots each (25% of
their waves; 8 of 56 slots across all three products) to wave
quantization, and no phase overlaps another.

The reference exposes far more threadgroups per call at 1 row:
`affine_gather_qvm` grid `(M, (N+63)/64, B)` with 64-thread groups
(round 1 `muse-spark/exploration/territory-quantized-kernels.md` §A) =
gate 24, up 24, down 48 threadgroups across its three sequential
kernels, plus the compiled activation. When the 20 layers' calls are
independent the GPU overlaps calls and hides one call's latency; when
each call waits for the previous one (the model's real dependency, and
the round-2 chained timing via the `link` kernel,
`bench/grader/check_candidate.py:178-213`), the candidate's 4-group call
stands alone and its latency shows. Attempt-4 already doubled threads
per pair (128 → 256) against exactly this
(`attempt-4/notes.md:3-9,13-20`); it fixed the old independent-timing
regression and still chains slower than the reference.

## Cost

Per layer (median µs/layer, `bench/micro/chain_timing.py` output
`results/runs/2026-11-04-muse-round-1/console-chain.txt`; bytes from
territory §A: 552,960 B/pair):

- **1 row:** candidate chained 132.9 vs reference chained 80.3
  (gap +52.6, candidate 66% slower); independent 61.0 vs 86.0
  (candidate 29% faster overlapped). Bytes 4 pairs × 552,960 =
  2,211,840 B/layer → floor 2.21 MB ÷ 200 GB/s = 11.1 µs at peak
  (`claude/GLOSSARY.md:10`). Effective: 2.21 MB ÷ 132.9 µs = 16.6 GB/s
  chained (8% of peak); 36.2 GB/s independent. Kernels: candidate 1,
  reference 7/layer × ~6.81 µs fixed (round 1 unit) — the candidate
  already saves the launches; the loss is latency inside the one kernel.
  Per step: 52.6 × 20 = 1,052 µs ≈ 1.05 ms, matching the measured G3
  effect +1.16 ms at 1 request in vLLM
  (`claude/findings/muse-round-1.md` §3-§4).
- **8 rows:** candidate chained 175.9 vs reference 239.6 (candidate
  ahead by 63.7); independent 165.1 vs 219.1. 32 pairs, 32 threadgroups,
  17.7 MB/layer → 100.6 GB/s chained. No loss; the win must be kept.
- **32 rows:** candidate chained 587.6 vs reference 797.6 (ahead by
  210.0); independent 535.0 vs 767.3. 128 pairs, 128 threadgroups,
  70.8 MB/layer → 120.5 GB/s chained. No loss.

Cross-check, old grader independent (different harness, do not mix with
chained): candidate 92.1/165.4/529.9 vs compiled 139.5/221.4/762.4
µs/layer (`grade-experts.json`); attempt-4's own run 50.0/157.8/515.4
(`attempt-4/notes.md:47-52`), 1-row baselines shift between runs
(`notes.md:54-57`). Round 1's claim "faster overlapped ⇒ faster in the
model" is refuted by the chained column (round 2 README *Refuted
claims*).

## Kind

Overhead-bound (`claude/GLOSSARY.md`): 16.6 GB/s = 8% of peak bandwidth
and a small fraction of peak compute at 1 row; time goes to waiting —
too few concurrent threadgroups and barrier-serialized waves — not to
bytes or arithmetic.

## Spans

`baseline-experts` (`block-0.metal` wave structure, `spec.json` grid),
`grader` (chained timing definition), `quantized-kernels` (reference
grid it must beat).

## Predicted saving

If one call's latency at 1 row fell to the reference's chained level,
per decode step (× 20 layers) against the reference:

- 1 row: (132.9 − 80.3) × 20 = **1,052 µs ≈ 1.05 ms/step** — the whole
  chained gap; this is the headroom, not a promise: it requires more
  concurrent work per call (e.g. splitting the 32-output blocks across
  more threadgroups, or splitting K) without breaking the per-block
  `simd_sum` order that decode exactness needs (ineff-1) and without
  adding a second kernel's fixed cost on the chain.
- 8 rows: 0 (candidate already 63.7 µs/layer ahead chained,
  1.27 ms/step; a 1-row fix must not spend it).
- 32 rows: 0 (already 210.0 µs/layer ahead, 4.20 ms/step).

Against the compiled reference: no chained compiled figure is published
for this slot yet (round-2 grader run needed); in the old independent
harness compiled tracked reference within 3% (139.5 vs 143.8 µs/layer
at 1 row, `grade-experts.json`), and the compiled experts alone changed
the vLLM step by only +0.07 ms at 1 request
(`claude/findings/muse-round-1.md` §3), so the 1.05 ms headroom above
stands against the compiled bar to within that small offset. Any fix
must also pass ineff-1's prefill arithmetic before the chained
comparison is even run (`check_candidate.py:308-337`).

## Verdict

verifier, 2026-11-04T03:23:12Z — **confirmed**, with one wave-arithmetic correction. Checked: grid `rows·1024` threads / 256 = `rows·4` threadgroups, `pair` at [block-0.metal:6](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/block-0.metal:6), 4 groups / 1,024 threads at 1 row on the 16-core GPU ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md)) ✓; barrier phases at [block-0.metal:18-21,28-38,40-50,51,56-64,65,68-75](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/block-0.metal:18) ✓. Reference grid `(M,(N+63)/64,B)` 64-thread groups → gate/up 24, down 48 threadgroups at 1 row ([territory-quantized-kernels.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/territory-quantized-kernels.md) §A) ✓. Chained definition via `link` ([check_candidate.py:178-213](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:178)) ✓; doubling 128→256 threads ([attempt-4/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/notes.md)) ✓. Numbers from [console-chain.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/console-chain.txt) (552,960 B/pair): 1 row chained 132.9 vs 80.3 (gap +52.6, 66% slower), independent 61.0 vs 86.0 (29% faster); bytes 4×552,960=2,211,840 B → floor 11.1 µs, 16.6 GB/s chained (8% of 200 GB/s peak, [GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)), 36.2 GB/s independent; ×20 = 1.05 ms/step matching G3 +1.16 ms vLLM ([muse-round-1.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/findings/muse-round-1.md) §3–§4) ✓. 8 rows 175.9 vs 239.6 (−63.7), 32 pairs/17.7 MB → 100.6 GB/s ✓; 32 rows 587.6 vs 797.6 (−210.0), 128 pairs/70.8 MB → 120.5 GB/s ✓. Old-grader cross-check 92.1/165.4/529.9 vs compiled 139.5/221.4/762.4 ([grade-experts.json](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/grade-experts.json)) and own run 50.0/157.8/515.4 ✓. Predicted saving (132.9−80.3)×20 = 1.05 ms/step at 1 row, 0 at 8/32 (already ahead 1.27/4.20 ms/step) ✓. **Wrong (non-load-bearing):** “Gate/up waste 4 simdgroup-slots of 24 (16.7%)” — gate wastes 4 of 16 slots in its 2 waves and up wastes 4 of 16 (8 of 32 across gate+up, 25%); down wastes 0. The chained gap and saving do not use this figure. Duplicate of [ineff-g3-2](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-g3-2.md) (same cause) — logged for the lead; this entry is the fuller kernel-side claim.

## From other agents

builder-1-b, 2026-11-04T03:35:55Z — round-2 `kernels/experts/attempt-1`
widened the fused per-pair kernel 256 → 512 threads/pair (waves 7 → 4,
per-block `simd_sum` order untouched) and graded chained: 1-row
104.3 µs/layer vs the starting point's 132.9 (other harness) — the
lever bought ~29 µs but still loses to the compiled reference (77.6).
With 4 pairs there are only 4 threadgroups however wide they are, so
at most 4 of 16 cores hold work; the remaining 1-row cost is
threadgroup count, not wave serialization. Widening threads/pair
again (1,024) is predicted to buy little; the next lever is more
threadgroups per call, which needs the pair's blocks split across
groups (a device-memory hidden intermediate), i.e. leaving the single
fused kernel. Full numbers in the attempt's `notes.md`.

builder-1-a, 2026-11-04T03:37:04Z — round-2
`kernels/experts/attempt-2` did that split (one 32-thread group per
(pair, 32-output block); 48 gate/up + 96 down groups at 1 row;
bit-exact at all six sizes) and graded chained 103.6 / 326.8 /
887.6 µs/layer vs compiled 73.1 / 232.0 / 804.5. The 1-row gain
over the starting point is real but small, and 8/32 rows regressed
hard (from 175.9 / 587.6): one simdgroup per group leaves each
block's serial K chain without the 8-way in-group load overlap
attempt-4 has, and the 6 KB of prefill tile memory allocated in
the same kernels caps resident decode groups. Splitting helps
only if several blocks stay in flight per group and decode
threadgroup memory stays near attempt-4's 3,840 B.

builder-1-c, 2026-11-04T03:51:59Z — round-2
`kernels/experts/attempt-3` tested exactly that remedy (two
kernels, 4 simdgroups per 128-thread group owning 4 blocks each;
12+24 groups at 1 row; attempt-1's scalar `qmm_accum` prefill so
threadgroup memory is 2,048/768 B, no MMA tiles; per-block
`qvm_block` byte-identical) and graded chained 109.5 / 195.0 /
637.7 µs/layer vs compiled 86.9 / 243.6 / 798.9, G2 pass all six.
The remedy recovered most of attempt-2's streaming loss
(80 → 111 GB/s at 32 rows) but not attempt-1's fused rate
(132 GB/s; 535.7 µs), and 1 row did not move: attempt-1
(4 groups × 512 thr), attempt-2 (48+96 × 32 thr) and attempt-3
(12+24 × 128 thr) all plateau at 103.6–109.5 µs/layer. Group
count past 4 is therefore not the binding 1-row limit; the
invariant cost is each block's serial K chain (24+24+12 strided
steps) plus the chained fixed cost, which no block packing
shortens — and the split adds a launch, a `hidden` round-trip
and duplicated `tg_x`/`tg_h` loads (+19% at 32 rows). Full
numbers and miss analysis in the attempt's `notes.md`.
