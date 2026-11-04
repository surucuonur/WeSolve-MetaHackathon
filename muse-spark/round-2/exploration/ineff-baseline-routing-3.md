# Inefficiency: one simdgroup per row, 31 threads idle through a serial thread-0 top-4 scan

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:24:18Z
- **Next:** verifier to check the occupancy arithmetic against block-0.metal

## Slot

`routing`

## What happens

The starting point gives each row exactly one 32-thread threadgroup
(`muse-spark/kernels/routing/attempt-1/spec.json`: grid `rows*32`,
threadgroup 32; `block-0.metal:10-11`). At 1 row that is a single simdgroup
on a 16-core GPU. Inside the group, threads 0-27 compute two experts and
threads 28-31 one (`block-0.metal:25-36`); then, after the second barrier,
thread 0 alone runs a 60-iteration serial insertion scan over the biased
scores while the other 31 threads idle (`block-0.metal:39-61`), and alone
writes the outputs. The reference, by contrast, covers the same work with
many-thread kernels even at 1 row (the GEMV dispatch alone uses 4
threadgroups × 128 threads at rows=1; territory section 3.1). Round 1's
builders measured the consequence as a fixed floor, not body cost: the fused
kernel sits at ~24-28 µs/layer in the piece harness while the reference's 12
kernels cost ~3 µs each pipelined (`kernels/routing/attempt-1/notes.md:47-59`,
`attempt-2/notes.md`), and 8 rows ran *faster* than 1 row (23.6 vs 28.5
µs/layer) — the signature of fixed cost amortising over threadgroups, i.e.
of the 1-row case having too little parallel work per launch.

## Cost

At 1, 8 and 32 rows (decode): the kernel body itself is small — 92k FLOP/row
(2×60×768) on one simdgroup is on the order of 1 µs, and the serial scan is
60 short iterations — so most of the measured 25.3/24.3/68.9 µs/layer
(independent timing, `claude/findings/muse-round-1.md:47`) is per-launch
fixed cost that a single-simdgroup launch amortises worst. The end-to-end
price is visible at 1 request: the starting point is +0.185 ms/step while the
compiled reference is −0.220 ms/step (`muse-round-1.md:85`), a ~0.41 ms/step
(≈20 µs/layer) deficit at exactly the row count where this entry's occupancy
is worst; at 8/32 requests the same kernel already beats the compiled
reference end to end (−0.397/−0.490 vs −0.183/−0.157 ms). No chained-timing
figure for routing exists yet (round-2 README:36), so the chained cost of
this entry is bounded by, not equal to, those numbers.

## Kind

Overhead-bound (`claude/GLOSSARY.md:39-42`): time goes to launch/dispatch
and idle thread slots, not to bytes (184 KB/layer at 1 row ≈ 0.9 µs at
200 GB/s peak) or arithmetic (92k FLOP/row).

## Spans

`baseline-routing`

## Predicted saving

Sharding each expert's dot product across the simdgroup's lanes (which the
bit-exact GEMV order at rows=1 requires anyway — see
`ineff-baseline-routing-1`) plus a shuffle-based parallel top-4 removes the
serial chain and the idle scan together. The recoverable part is bounded by
the gap this entry explains: up to ~20 µs/layer (≈0.4 ms/step) at 1 row
against the compiled reference, ~0 at 8/32 rows where the starting point
already wins. Realistically only the kernel-body slice of that gap is
recoverable in-slot; the MLX `metal_kernel` dispatch floor (~24 µs/layer in
round 1's harness, outside the slot's reach per round-1 hyp-2 decision) is
not. Predicted: **0.1-0.4 ms/step at 1 row, ~0 at 8/32**, with the chained
piece timing as the arbiter Round 1 did not have.

## Verdict

verifier, 2026-11-04T03:24:18Z — **confirmed**. Checked: one 32-thread threadgroup per row (grid `rows·32`, threadgroup 32, [spec.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/spec.json); [block-0.metal:10-11](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:10)) — a single simdgroup at 1 row ✓; threads 0–27 compute two experts, 28–31 one ([block-0.metal:25-36](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:25)) ✓; thread 0 alone runs the 60-iteration insertion scan while 31 idle and alone writes outputs ([block-0.metal:39-61](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:39)) ✓. Reference GEMV at rows=1 uses 4 threadgroups × 128 threads (BM=4, 16 outputs/group, `ceil(60/16)=4`) ✓. Fixed-floor evidence: ~24 µs/layer floor, reference 12 kernels ~3 µs each pipelined, 8 rows faster than 1 (23.6 vs 28.5 µs/layer own run) ([attempt-1/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/notes.md)) ✓. Body small: 2·60·768=92k FLOP/row, 184 KB/layer ≈0.9 µs at 200 GB/s — most of measured 25.3/24.3/68.9 µs/layer (independent, [muse-round-1.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/findings/muse-round-1.md) §1) is per-launch fixed cost ✓. End-to-end at 1 request +0.185 vs compiled −0.220 ms/step = 0.405 ms/step (≈20 µs/layer) deficit where occupancy is worst; at 8/32 the kernel already beats compiled end to end (−0.397/−0.490 vs −0.183/−0.157) ✓. No chained routing figure exists yet, as bounded ✓. Predicted 0.1–0.4 ms/step at 1 row (upper bound 20 µs/layer×20=0.4 ms), ~0 at 8/32, with only the kernel-body slice recoverable in-slot and the ~24 µs dispatch floor outside the slot — a bounded estimate with the chained timing as arbiter, as written.

## From other agents

(none yet)
