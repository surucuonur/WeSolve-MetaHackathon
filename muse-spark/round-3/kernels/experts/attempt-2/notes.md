# attempt-2 (round 3): vectorized per-pair scalar prefill — one thread per weight word (builder-2-b)

First attempt in direction (b) for `hyp-2-experts-prefill` (slot
`experts`, baseline `none`). Starting point
`muse-spark/round-2/kernels/experts/attempt-7` (fused single kernel,
512 threads/pair, G2 pass all six, prefill 15,220 / 119,154 /
240,681 µs/layer at 127 / 1,016 / 2,048 rows).

## The one main change

Only the `rows >= 60` prefill branch is rewritten; the fused decode
path is byte-identical (verified by diff: `block-0.metal` differs
from attempt-7 only in the header comment and the three `} else {`
prefill branches; `block-1.metal` decode helpers `qvm_block`,
`qvm_gate_up_pair`, `qvm_down_pair` byte-identical; `spec.json`
byte-identical — same kernel, same `rows*2048` grid, no `max_rows`).

- **Before:** one thread per output called scalar `qmm_accum`: 32
  threads per 32-output block, each walking all of K with one
  accumulator, each issuing its own W + S + B + x loads per k.
- **After:** one thread per weight word calls `qmm_accum8`
  (`block-1.metal`): 4 threads per block, each accumulating the
  word's 8 outputs into 8 independent accumulators (`p[0..7]` —
  the kernel-scope `p` reused in this disjoint phase, re-zeroed on
  entry, so no third thread array and the same 512-thread pipeline
  budget as attempt-7). One W/S/B/x load per k feeds all 8
  outputs: 8× fewer issues, 8 accumulators overlapping in flight.
- **Packing:** 8 blocks per simdgroup (`block = simd*8 +
  (lane>>2)`, `word = lane&3`): gate/up 12 blocks in one wave of
  2 simdgroups (48 active threads, 16 lanes idle in the second),
  down 24 blocks in one wave of 3 simdgroups (96 active, none
  idle) — replacing gate/up's 384-thread phases and down's 2
  waves.
- **Bits:** each output keeps the scalar branch's exact per-output
  sequence — same contiguous `BK=32` tile order, same
  `q = (w >> shift) & 0xf` extraction, same
  `wq = s*q + b` / `acc += x*(bfloat)wq` terms (S/B/x loaded once
  per k hold the same values every output would load itself), same
  `(bfloat)` store. Bit-exact by the same G2-proven coincidence
  the scalar branch relies on.

## Which inefficiency it removes, and how

`ineff-baseline-experts-1` (round 3, confirmed): the serial scalar
prefill chain at ~29.5 µs/pair and 18.8 GB/s (9.4% of peak). This
attempt removes its first two causes inside each pair — the
single-accumulator serial chain (now 8 overlapping accumulators)
and the 12.8× redundant scalar loads (now 8× fewer W/S/B/x issues:
per k per block 32 threads × 4 loads → 4 threads × 4 loads) —
while keeping every output's accumulation order. It deliberately
keeps the third cause (no cross-row reuse: per-pair grid, 553 KB
unique per pair) — that is direction (c)'s prize.

## Limits (by hand; the grader does not check the 32 KB cap)

512 threads/group ≤ 1,024. Threadgroup memory unchanged: 3,840 B.
Distinct thread arrays unchanged: two 32-float accumulators (`p`,
shared `q`) — prefill reuses `p[0..7]`, adding none, so the
pipeline's `maxTotalThreadsPerThreadgroup` should still admit 512
(attempt-6's G1 lesson). Grid `rows*4` groups, valid at every
`rows ≥ 1` on both sides of the unchanged `x_shape[0] >= 60`
dispatch. Coverage: gate/up 48 threads × 8 outputs = 384 each,
down 96 × 8 = 768, each output written exactly once.

## Predicted µs per layer at the six sizes (chained, grader timing)

Starting point (grade-phases run): decode 93.8 / 160.1 / 510.1,
prefill 15,220.0 / 119,154.4 / 240,681.1; compiled prefill 960.8 /
4,306.0 / 11,998.9. Baseline `none`.

- **Decode 1 / 8 / 32 rows: 94 / 160 / 510.** The decode path is
  byte-identical code on the same grid, so timing should repeat
  attempt-7's within run noise (±5%): still losing at 1 row,
  winning at 8/32.
- **Prefill 127 / 1,016 / 2,048 rows: 3,000 / 24,000 / 49,000**
  (≈6 µs/pair vs 29.5, ~5×). Arithmetic: 508 / 4,064 / 8,192
  pairs × 6 µs = 3,048 / 24,384 / 49,152. Basis: the branch is
  issue-bound (18.8 GB/s flat across sizes), and this change
  cuts issues 8× while overlapping the serial FMA chain 8-wide;
  capped below 8× because the FMA count is unchanged (884,736
  MACs/pair now share each thread's pipeline), gate/up stay
  sequential phases, silu + barriers + x-load + launch/link cost
  are untouched, and only 48–96 of 512 lanes are active per
  group. Still 3–11× above compiled (no cross-row reuse), so
  prefill stays `slower` and `beats_compiled` stays false —
  this attempt prices direction (b), it does not close the gap.

G1 prediction: pass — same launch, same array count as attempt-7.
G2 prediction: pass at all six sizes, `differ == 0` — each
prefill output's terms and order are the scalar branch's by
construction, decode is untouched.

With `max_rows`: none — kernels run at every row count (both
arithmetics kept), so no size runs the compiled reference.

## What this attempt leaves unmeasured

For the hypothesis entry's **Unverified** list: kernel speed at
33–59 rows (decode path, unchanged code, untimed bracket); sizes
above 2,048 rows; prefill beyond the first chunk of seeded
128-token prompts and other lengths/distributions; whole-request
effect (no ready check on this attempt); the 32 KB cap (3,840 B
by hand).

## Verdict

G1 pass — reusing kernel-scope `p[0..7]` added no thread array and
the 512-thread pipeline still compiles, as predicted. **G2 pass at
all six sizes, differ 0 / max_ulp 0.0 everywhere** (no first
difference) — the 8-output widening preserves the scalar branch's
bits exactly, as designed. But prefill is not faster.

Chained timing µs/layer, this run (`grade.json`: candidate vs
reference vs compiled): decode **1 row 152.8 / 153.6 / 113.0
(`slower`, 5 wins); 8 rows 173.3 / 249.9 / 240.4 (`faster`, 28);
32 rows 634.1 / 896.0 / 896.3 (`faster`, 24)**; prefill **127 rows
20,443.5 / 1,095.3 / 1,113.0 (`slower`, 0 wins); 1,016 rows
156,415.6 / 4,862.1 / 6,774.6 (`slower`, 0); 2,048 rows 303,542.8 /
8,126.1 / 16,333.0 (`slower`, 0)**. `phases_vs_compiled`: decode
`slower`, prefill `slower`; `beats_compiled` false. Baseline
`none`. Decode verdicts (`slower`/`faster`/`faster`) repeat the
starting point's; prefill stays `slower` at all three sizes.

Caveat — this run's machine was much slower than the basis run's:
the byte-identical decode path timed 152.8/173.3/634.1 vs
attempt-7's 93.8/160.1/510.1 while reference and compiled slowed
too (compiled prefill 1,113/6,775/16,333 vs basis 961/4,306/11,999;
sibling attempt-1's run, minutes apart, saw compiled prefill
1,131/4,134/7,558). Absolute µs therefore overstate any
regression. The within-run signal is candidate-vs-compiled:
18.4×/23.1×/18.6× here vs 15.8×/27.7×/20.1× for the starting
point — worse at 127 rows, marginally narrower at 1,016/2,048,
no speedup demonstrated anywhere. Per-pair absolute 40.2/38.5/37.1
µs vs the starting point's 29.96/29.32/29.38.

## Miss analysis (predicted prefill 3,000 / 24,000 / 49,000, measured 20,444 / 156,416 / 303,543)

Missed by ~6.5× at all three sizes — the prediction's issue model
was wrong in two reinforcing ways:

1. **The loads stayed on the critical path.** The 8 accumulators
   overlap the 8 FMAs *within* one k iteration, but every
   iteration's FMAs depend on that same iteration's W/S/B loads,
   and without k-loop unrolling the GPU issues the 768
   load→FMA steps in order. Each per-thread step got longer (8
   FMAs behind one load instead of 1), never shorter; the serial
   depth per thread is unchanged at 768 (+768+768 across
   phases). Widening across outputs without unrolling across k
   cannot hide load latency — it stacks more ALU behind the
   same stall.
2. **It removed the cross-thread latency hiding the scalar
   branch relied on.** Gate/up went from 384 active threads per
   group (12 full simdgroups) to 48, down from up to 512 to 96 —
   8× fewer outstanding load streams per group. The branch was
   already latency-bound at 18.8 GB/s; fewer warps covering the
   same per-thread stall is strictly worse. Total FMA count never
   changed (884,736 MACs/pair), so nothing compensated.

Lesson for `ineff-baseline-experts-1` (note left there): per-pair
widening across outputs is not the lever — the scalar branch's
bottleneck is per-k load latency with in-order issue, and its
only hiding came from thread count. A scalar-branch speedup must
unroll or pipeline across k (several k-steps' loads outstanding
per thread), not across outputs; cross-row reuse stays the
structural fix.

## Next change

Keep the word-packing (8 outputs, one W/S/B load per k) and unroll
across k: 4 k-steps × 8 outputs = 32 accumulators fits `p[0..31]`
exactly, putting 4 loads outstanding per thread with 32 FMAs to
hide them behind — no new array, same G1 budget. If that still
shows no gain, direction (b) is exhausted and only grouped reuse
(direction (c)) or the hand-off (attempt-1) moves prefill.
