# attempt-4: harden attempt-2 — 256 threads per pair (builder-1-d)

Single change on `../attempt-2` (direction b, one fused per-pair kernel):
256 threads per threadgroup (8 simdgroups) instead of 128 (4 simdgroups).
Grid goes from `rows*512` to `rows*1024` threads = still `rows*4` groups,
one per (row, expert-slot) pair. `block-1.metal` (qvm_block) is byte-identical
to attempt-2; only the wave structure in `block-0.metal` changed: gate/up 12
blocks in 2 waves of 8 (4 simds idle in wave 2, guarded by simd-uniform
`b < 12`), down 24 blocks in 3 waves of 8, x/silu strided loops step 256.

Removes the 1-row half of attempt-2's occupancy finding (its notes.md miss
analysis): at 1 row, 4 pairs x 128 threads = 512 threads on a 16-core GPU
streams at only ~20 GB/s effective (2.21 MB in 107.3 us), regressing 7.5%
vs eager (99.8 us). Doubling threads per pair doubles each group's in-flight
memory parallelism while keeping the proven-correct numerics untouched: same
qvm strided-lane accumulation + `simd_sum` per 32-output block (guard is
uniform inside each simd, so the reduction stays uniform), same full-bf16
silu lowering, same fp32 accumulation and bf16 rounding points. Gate/up/down
weight bytes per pair are unchanged (553 KB), so 8/32 rows — already
streaming at ~94-109 GB/s — should be unaffected.

Boundary kept: same as attempt-2 (boundary-1 pass): fp32 accumulation, all
four bf16 rounding points, positional `y[r,j]`, any `inds` order, valid for
all 1-32 rows (grid is `rows*4` groups, no rows-dependent indexing),
256 threads/group <= 1024, threadgroup memory unchanged at 3840 B (~12% of
32 KB).

## Predicted us per layer at 1 / 8 / 32 rows

- 1 row: occupancy-bound regime, threads 512 -> 1024. Effective bandwidth
  should roughly double from attempt-2's 20.6 GB/s (2.21 MB / 107.3 us):
  2.21 MB / 41 GB/s = 54 us + one launch ~7 us -> **60 us**
  (attempt-2 107.3; eager original 99.8; compiled 135.0). Beats eager.
- 8 rows: streaming regime already (17.7 MB in 188.4 us = 94 GB/s);
  same bytes, wider groups neutral -> **185 us**
  (attempt-2 188.4; original 218.7; compiled 216.9).
- 32 rows: same argument, full 70.8 MB at ~109 GB/s -> **650 us**
  (attempt-2 647.6; original 768.6; compiled 761.7).

G2: expect bit-exact (max_ulp 0.0) — every block's computation is identical
to attempt-2's v3, only the wave packing changed. beats_compiled: expect
true at all rows.

## Verdict

G1 pass; G2 pass with max_ulp 0.0 at all row counts (bit-exact vs reference,
as predicted); beats_compiled true. Piece timing us/layer (candidate vs
original vs compiled, this run): rows 1: 50.0 / 64.3 / 65.3; rows 8: 157.8 /
218.6 / 217.4; rows 32: 515.4 / 765.7 / 760.2. Beats BOTH the eager original
and the compiled reference at all rows (~22/28/33% over eager, ~23/27/32%
over compiled). The 1-row regression is fixed: 50.0 us vs attempt-2's 107.3
(2.1x), now beating eager's 64.3 by 22%.

Note: this run's 1-row baselines (64.3/65.3) are lower than attempt-2's run
(99.8/135.0) — the 1-row harness timing shifts between runs (attempt-1's
re-grade also saw 64.1/62.0), so cross-run candidate comparisons at 1 row
carry that caveat; within this run the candidate beats both baselines
cleanly. At 8/32 rows the baselines are stable across runs (218.6/217.4 vs
218.7/216.9; 765.7/760.2 vs 768.6/761.7), so the 8/32-row gains are real.

## Miss analysis (prediction 60 / 185 / 650)

Measured 50.0 / 157.8 / 515.4 — faster than predicted at all rows: -17% /
-15% / -21%. Only 32 rows exceeds the 20% bar, and on the fast side.

Why: the change did more than add 1-row occupancy. Eight blocks in flight
per wave per group (vs four) raised per-core memory-level parallelism in
EVERY regime: 32-row effective bandwidth went from attempt-2's ~109 GB/s
(70.8 MB / 647.6 us) to ~137 GB/s (70.8 MB / 515.4 us). Attempt-2 was mildly
MLP-bound even where it looked streaming-bound; the extra in-flight
weight/scale/bias loads per group hide more latency, and halving the wave
count halves barrier round-trips. Same mechanism explains the 8-row (-15%)
and 1-row (-17%) overshoots. Lesson for `ineff-quantized-kernels-1`: per-pair
qvm throughput scales with in-flight blocks per group up to at least 8
simds, not just with group count.
