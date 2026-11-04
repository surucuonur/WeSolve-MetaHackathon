# Round-3 attempt 4 — builder-3-d, hyp-3-routing-prefill direction (b)

Builds on `muse-spark/round-3/kernels/routing/attempt-2` (direction (b),
G1 fail: kernel 1 asks 960 threads/group against the pipeline max 640;
no G2, no timing). Baseline: none, so no `beats_baseline`; the bar is the
compiled reference in both phases.

## The one main change

10 rows per threadgroup instead of 16 in kernel 1 (the GEMM-shaped
prefill kernel), exactly the fix attempt-2's notes prescribe — nothing else:

- `spec.json`: kernel 1 `threadgroup` 960 → 600; grid stays `rows*60`.
  60 still divides 600, so every threadgroup covers whole rows (the last
  possibly fewer than 10) and no bounds checks are needed — edge groups
  clip to whole rows exactly as in attempt-2's 16-row design.
- `block-2.metal`: `xst`/`sc` 1024 → 640 floats (10 rows × 64 stride);
  `wsh[4096]` unchanged (4 K-chunks × 16 k × 64-expert stride is
  row-count independent); row stride `g * 16u` → `g * 10u` in `r`, `Rg`
  and `rr`; `Rg` cap 16 → 10. The `w`-staging loop bound stays 960
  (4 slots × 240 elements of total work, independent of rows/group);
  every remaining `16u` is K-chunk geometry (16 d-values per chunk,
  `part[16]`), correctly untouched.
- `block-0.metal`, `block-1.metal`: byte-identical to attempt-2
  (`diff` empty) — decode path untouched.

Threadgroup: (4096 + 640 + 640) × 4 B = 21,504 B of 32 KB (was 24,576 B).
600 threads < 640 pipeline cap under either binding constraint
(registers or threadgroup memory — both shrank), so G1 should pass.

## Inefficiency removed

Same as attempt-2: `ineff-baseline-routing-1` (per-row `w_router`
re-read) and `ineff-baseline-routing-2` (per-row shape cost) together.
`w_router` device traffic falls from rows × 184,320 B to
ceil(rows/10) × 184,320 B (~10× at full groups, was 16×), barriers from
24/row to 25 per group of 10 rows (~2.5/row amortised, was ~1.6),
duty from 23% to 100% in the single-partition regime, footprint from
28,080 B to 21,504 B.

## Predicted us per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows

Anchor: the round-3 re-grade of attempt-4
(`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`):
ref / compiled / candidate = 68.4/65.1/49.8 at 1 row,
169.2/135.7/113.5 at 8, 159.3/145.2/121.3 at 32,
102.3/87.3/227.1 at 127, 231.3/221.1/1298.8 at 1,016,
310.0/290.4/2513.0 at 2,048. Decode margins: 15.3/22.2/23.9.

- Decode (1/8/32): kernel 0 is attempt-4 plus a never-taken branch
  (~0); kernel 1 adds one copy launch (grid rows*60 threads, rows*4
  elements × 8 B copied). Allowance 3 us: **53 / 117 / 124** vs
  compiled 65.1/135.7/145.2 — still `faster`, spending ~3 of 15–24 us
  of margin. Unchanged from attempt-2 (decode code is byte-identical).
- Prefill traffic floors at 200 GB/s (`w` staged once per 10 rows):
  - 127 rows: 13 groups × 184,320 B (w) + 127 × 1,536 B (x) + 127 × 32 B
    (out) = 2,595,296 B → 13.0 us. 13 groups fit 16 cores → 1 wave ×
    25 barriers × 0.5 us = 12.5 + idle kernel-0 launch 4 + chained
    fixed cost 25 → **~55** vs compiled 87.3 (`faster`).
  - 1,016 rows: 102 × 184,320 + 1,016 × 1,536 + 1,016 × 32 =
    20,393,728 B → 102 us. ceil(102/16) = 7 waves × 12.5 = 87.5 +
    idle launch 6 + fixed 25 → **~220** vs compiled 221.1 — tied;
    the chained timing decides.
  - 2,048 rows: 205 × 184,320 + 2,048 × 1,536 + 2,048 × 32 =
    40,996,864 B → 205 us. ceil(205/16) = 13 waves × 12.5 = 162.5 +
    idle launch 8 + fixed 25 → **~400** vs compiled 290.4 —
    `slower` under this barrier model.

Prediction: **53 / 117 / 124 / 55 / 220 / 400 us**. G1 prediction: pass
(600 < 640; 21,504 B < 32 KB; grid divisible along whole rows). G2
prediction: pass at all six sizes — every accumulation order is copied
op-for-op from attempt-4 (via attempt-2, which this copies), and only
the grouping changed, never any row's order. `beats_compiled`
prediction: false under the 0.5-us/barrier guess (2,048 rows slower,
1,016 tied) — but the barrier cost is a guess, and barriers may
overlap across resident groups; the chained timing decides whether
prefill-big is `same`-or-better. The decode `faster` × 3 is the firm
half of the prediction.

No `max_rows`: the hand-off is direction (a)'s fix; this attempt must
prove the GEMM kernel same-or-better at prefill on its own merits.

## What this attempt leaves unmeasured

For the hypothesis entry's Unverified list: ungraded row counts 2–7 /
9–31 (kernel-0 staged path, same code as graded 8/32 but other
partition groupings); 33–126 (kernel-1 split-K path with 16/8
partitions, exercised in kind but not in verdict); 129–767 and 769+
(kernel-1 paths past the 129/257/768 transitions); above 2,048 rows;
prefill past the first chunk; whole-request composition (ready check);
the 21,504 B threadgroup sum (unchecked by slots.build); unwritten
threadgroup reads and thread-timing dependence (one equality run per
layer cannot expose them).

## Verdict

G1 fail, no G2, no timing (`grade.json`): `ValueError: Thread group size
(600) is greater than  the maximum allowed threads per threadgroup (576).`
One grader run, one attempt — recorded as graded; the fix is a copied new
attempt (see below).

## Why G1 failed

The pipeline cap moved against the fix: attempt-2's kernel 1 reported a
cap of 640, this kernel — the same code with 10 rows/group (600 threads)
and 21,504 B threadgroup (was 24,576 B) — reports 576. Both numbers are
Metal's per-pipeline `maxTotalThreadsPerThreadgroup` derived from the
kernel's own resource use, not the brief's 1,024 device max; "design
under 640" was wrong advice because the cap is not stable across
near-identical kernels.

Likely mechanism (inferred, not measured): register allocation. Both
caps are whole-SIMD multiples (640 = 20 × 32, 576 = 18 × 32), as a
registers-per-thread budget would produce, and the recompile with
different grouping constants can schedule ~11% more registers per
thread (640/576 = 1.11) even though the per-thread logic is unchanged.
Threadgroup memory shrank, so it cannot explain a lower cap. Nothing
about the GEMM direction is refuted: no correctness or timing evidence
exists (the kernel never ran), and the predicted 53/117/124/55/220/400
us stands uncompared — a >20% analysis is not applicable with no
measurement.

Next change (one, in a copied new attempt): 8 rows per threadgroup
(480 threads, `xst`/`sc` to 8×64, `Rg` cap and row stride 10→8, grid
stays `rows*60`, 60 still divides 480 so groups stay row-aligned with
no bounds checks). 480 sits inside round-1 attempt-2's proven-passing
band — but this attempt proves the cap moves with the kernel, so if the
cap drops below 480 again the following fix must attack per-thread
registers (e.g. shrink `part[16]` toward the reachable partition max),
not just the group size.
