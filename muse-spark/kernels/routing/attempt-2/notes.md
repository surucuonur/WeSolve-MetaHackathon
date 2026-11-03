# Attempt 2: two-kernel split (direction b)

First attempt in direction (b): kernel 1 fuses router scores (dense matmul +
sigmoid + bias add), kernel 2 fuses top-4 selection plus weight
normalize/scale. The dense matmul keeps a regular GEMM shape (one thread per
(row, expert), k tiled through threadgroup memory); the 11 bookkeeping kernels
around it collapse into the two fused kernels.

Removes `ineff-runs-2`: 12 kernels/layer of pure launch overhead become 2.
Kernel 1 additionally avoids re-reading the 184 KB router weight once per row
by staging k-chunks in threadgroup memory shared across 8 rows (17.4 KB <
32 KB limit; 480 threads/group < 1024).

## Predicted µs per layer (piece timing)

Fixed cost: 2 kernels x 6.81 µs (step_parts.txt:2) = 13.6 µs. Traffic at
200 GB/s peak:

- 1 row: w 184 KB (1 group) + x 1.5 KB + intermediates ~1 KB = ~187 KB ~= 0.9 µs.
  Total ~= 14.5 -> **15 µs/layer** (original 51).
- 8 rows: w 184 KB (1 group) + x 12 KB + intermediates ~8 KB = ~204 KB ~= 1.0 µs.
  Total ~= 14.6 -> **15 µs/layer** (original ~59, interpolated).
- 32 rows: w 737 KB (4 groups) + x 49 KB + intermediates ~32 KB = ~818 KB ~= 4.1 µs.
  Total ~= 17.7 -> **18 µs/layer** (original 84).

Predicted saving per step (x20 layers): 0.72 / 0.88 / 1.32 ms at 1/8/32 rows.

## Verdict

G1 pass. G2 pass: 0 wrong-expert rows, 0 near-tie rows at all row counts,
worst max_rel_err 2.6e-6 (40x inside the 1e-4 tolerance).

| rows | candidate | reference | compiled | beats compiled |
|---|---|---|---|---|
| 1 | 36.3 | 36.3 | 29.3 | no |
| 8 | 34.9 | 56.9 | 42.9 | yes |
| 32 | 49.0 | 51.7 | 46.6 | no |

`beats_compiled: false`. Attempt-1 (direction a, single kernel) measured
28.5 / 23.6 / 41.8 against compiled 30.5 / 42.0 / 49.1 on its run and stops
the slot; this attempt is slower than it at every row count.

## Why the prediction missed (predicted 15 / 15 / 18, off by 142 / 133 / 172%)

1. Wrong fixed-cost model. I charged 2 x 6.81 µs = 13.6 µs from
   step_parts.txt. The piece harness shows a ~24-28 µs floor even for one
   fused kernel (attempt-1: 28.5 at 1 row), while the reference's 12 kernels
   cost only ~3 µs each when pipelined 20 layers deep. A heavy fused kernel
   costs a dispatch floor plus low-occupancy ramp, not one microbench launch.
2. Two launches, each under-occupied. Kernel 1 runs 60 threads (2 simdgroups,
   1 of 16 cores) at 1 row with 24 threadgroup barriers; kernel 2 runs a
   single thread at 1 row -- a whole launch for ~100 ns of ALU.
3. Sub-peak bandwidth. Kernel 1's w reads are 3 KB apart across SIMD lanes,
   far from the assumed 200 GB/s, and the two [rows,60] intermediates add a
   write-plus-reread round trip the single-kernel design avoids.
4. Attempt-1 is the control: re-reading w once per row (no tiling) still wins
   at 32 rows, so my barrier-heavy tiling bought traffic savings that never
   mattered and paid overhead that did.

One change I would make next (staying in direction b): delete the
threadgroup tiling in kernel 1 -- stream each expert's w column from device
memory with zero barriers -- since traffic is not the binding constraint.
Honestly, direction (a) dominates this split: one launch and no intermediates
beats two launches at all three row counts.
