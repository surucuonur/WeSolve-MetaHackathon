# Round-3 attempt 2 — builder-3-b, hyp-3-routing-prefill direction (b)

Builds on `muse-spark/round-2/kernels/routing/attempt-4` (the slot's
starting point: G1+G2 pass, decode `faster` x3, prefill `slower` x3).
Baseline: none, so no `beats_baseline`; the bar is the compiled reference
in both phases.

## The one main change

A second kernel with a GEMM shape for prefill sizes, sharing each
`w_router` K-chunk across 16 rows, behind a size branch that keeps decode
on the winning staged path:

- Kernel 0 (`block-0.metal` + `block-1.metal` header) is attempt-4
  verbatim — same GEMV branch at `rows == 1`, same split-K staging,
  same sigmoid, same top-4 — plus `if (rows > 32) return;` at the top.
  Its outputs are renamed `inds0`/`weights0`. Grid `rows*256`,
  threadgroup 256, threadgroup memory unchanged at 28,080 B.
- Kernel 1 (`block-2.metal`) is new. Grid `rows*60`, threadgroup 960:
  one threadgroup spans 16 rows with one thread per (row, expert); 60
  divides 960 so every group covers whole rows and no bounds check is
  needed (round-1 attempt-2's grid precedent). Each of the 12 staged
  groups stages 4 K-chunks of `w_router` once per 16 rows with
  attempt-4's exact coalesced `float4` mapping into `wsh[4096]`, plus
  the 64 current d-values per row into `xst[1024]`; each thread
  accumulates its (row, expert) score into per-partition partials in
  registers, each partial a serial scalar sum in increasing d, folded
  in partition order — the exact rows-branched per-row orders, so G2
  follows attempt-4's proof (feeding changes without order changes).
  Sigmoid, strictly-greater top-4 and the left-fold normalise are
  verbatim copies, run by one thread per row. At `rows <= 32` it only
  copies kernel 0's outputs. Threadgroup memory
  (4096+1024+1024)*4 = 24,576 B of 32 KB; 25 barriers per group of 16
  rows (12x2 + 1), i.e. ~1.6 per row against attempt-4's 24.

Against round-1 attempt-2's failure (slower everywhere): that kernel
staged with serial per-thread loops (60 threads x 64 serial loads),
2 launches with two [rows,60] intermediates, and strided device reads.
This one stages cooperatively (960 threads share 960 vector loads),
keeps one [rows,4] hand-off, and reuses attempt-4's proven coalesced
mapping — but the proof is the chained timing below, not the contrast.

## Inefficiency removed

`ineff-baseline-routing-1` (per-row `w_router` re-read) and
`ineff-baseline-routing-2` (per-row shape cost) together: `w_router`
device traffic falls from rows x 184,320 B to ceil(rows/16) x 184,320 B
(16x at full groups), barriers from 24/row to ~1.6/row amortised, duty
from 23% to 100% in the single-partition regime (every thread owns one
expert of its row through all 12 groups), footprint from 28,080 B to
24,576 B.

## Predicted us per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows

Anchor: the round-3 re-grade of attempt-4
(`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`):
ref / compiled / candidate = 68.4/65.1/49.8 at 1 row,
169.2/135.7/113.5 at 8, 159.3/145.2/121.3 at 32,
102.3/87.3/227.1 at 127, 231.3/221.1/1298.8 at 1,016,
310.0/290.4/2513.0 at 2,048. Decode margins: 15.3/22.2/23.9.

- Decode (1/8/32): kernel 0 is attempt-4 plus a never-taken branch
  (~0); kernel 1 adds one copy launch (grid rows*60 threads, rows*4
  elements x 8 B copied). Allowance 3 us: **53 / 117 / 124** vs
  compiled 65.1/135.7/145.2 — still `faster`, spending ~3 of 15-24 us
  of margin.
- Prefill traffic floors at 200 GB/s (w staged once per 16 rows):
  - 127 rows: 8 groups x 184,320 B (w) + 127 x 1,536 B (x) + 127 x 32 B
    (out) = 1,673,696 B -> 8.4 us. + 25 barriers x 0.5 us = 12.5 +
    idle kernel-0 launch 4 + chained fixed cost 25 -> **~50** vs
    compiled 87.3.
  - 1,016 rows: 64 x 184,320 + 1,016 x 1,536 + 1,016 x 32 = 13.4 MB ->
    67 us. 64 groups / 16 cores = 4 waves x 25 x 0.5 = 50 + idle
    launch 6 + fixed 25 -> **~148** vs compiled 221.1.
  - 2,048 rows: 128 x 184,320 + 2,048 x 1,536 + 2,048 x 32 = 26.7 MB ->
    134 us. 8 waves x 25 x 0.5 = 100 + idle launch 8 + fixed 25 ->
    **~267** vs compiled 290.4 — thin: if barriers cost 1 us not 0.5,
    this is ~367 and `slower`. The barrier cost is a guess; the
    chained timing decides.

Prediction: **53 / 117 / 124 / 50 / 148 / 267 us**. G2 prediction: pass
at all six sizes — every accumulation order is copied op-for-op from
attempt-4, which passes G2 at all six (it fails on speed, not bits).

No `max_rows`: the hand-off is direction (a)'s fix; this attempt must
prove the GEMM kernel same-or-better at prefill on its own merits.

## What this attempt leaves unmeasured

For the hypothesis entry's Unverified list: ungraded row counts 2-7 /
9-31 (kernel-0 staged path, same code as graded 8/32 but other
partition groupings); 33-126 (kernel-1 split-K path with 16/8
partitions, exercised in kind but not in verdict); 129-767 and 769+
(kernel-1 paths past the 129/257/768 transitions); above 2,048 rows;
prefill past the first chunk; whole-request composition (ready check);
the 24,576 B threadgroup sum (unchecked by slots.build); unwritten
threadgroup reads and thread-timing dependence (one equality run per
layer cannot expose them).

## Verdict

G1 fail, no G2, no timing (`grade.json`): `ValueError: Thread group size
(960) is greater than the maximum allowed threads per threadgroup (640).`
One grader run, one attempt — recorded as graded; the fix is a copied new
attempt (see below).

## Why G1 failed

The brief's 1,024-thread cap (brief:100) is the device max; Metal derives
each pipeline's actual `maxTotalThreadsPerThreadgroup` from the kernel's
own resource use — here 640, against my 960-thread (16 rows x 60 experts)
threadgroup. The likely binder is registers (the `part[16]` partial array
plus staging temporaries per thread), with 24,576 B of threadgroup memory
as the other candidate; either way the number to design under is 640, not
1,024. Round-1 attempt-2's 480-thread groups passed G1, consistent with a
cap in the 480-960 band. Nothing about the GEMM direction is refuted: no
correctness or timing evidence exists (the kernel never ran), and the
predicted 53/117/124/50/148/267 us stands uncompared — a >20% analysis is
not applicable with no measurement.

Next change (one, in a copied new attempt): 10 rows per threadgroup
(600 threads, `xst`/`sc` to 10x64, `Rg` cap and row stride 16->10, grid
stays `rows*60`). 600 < 640 under either binding constraint, and 60 still
divides 600 so groups stay row-aligned with no bounds checks. Re-derived
prefill traffic floors at 200 GB/s: 127 rows: 13 x 184,320 + 127 x 1,536
= 2.6 MB -> ~13 us; 1,016 rows: 102 x 184,320 + 1,016 x 1,536 = 20.4 MB
-> ~102 us; 2,048 rows: 205 x 184,320 + 2,048 x 1,536 = 40.9 MB ->
~205 us (thin vs compiled 290.4 — the chained timing decides). If that
attempt passes G1, the open question is unchanged: whether barriers and
waves fit the remaining headroom.
