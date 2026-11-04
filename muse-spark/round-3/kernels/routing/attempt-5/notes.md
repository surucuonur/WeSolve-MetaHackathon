# Round-3 attempt 5 — builder-3-e, hyp-3-routing-prefill direction (b)

Builds on `muse-spark/round-3/kernels/routing/attempt-4` (direction (b),
G1 fail: kernel 1 asks 600 threads/group against the pipeline max 576;
no G2, no timing). Baseline: none, so no `beats_baseline`; the bar is the
compiled reference in both phases.

## The one main change

8 rows per threadgroup instead of 10 in kernel 1 (the GEMM-shaped
prefill kernel), exactly the fix attempt-4's notes prescribe — nothing else:

- `spec.json`: kernel 1 `threadgroup` 600 → 480; grid stays `rows*60`.
  60 still divides 480, so every threadgroup covers whole rows (the last
  possibly fewer than 8) and no bounds checks are needed — edge groups
  clip to whole rows exactly as in attempt-4's 10-row design.
- `block-2.metal`: `xst`/`sc` 640 → 512 floats (8 rows × 64 stride);
  `wsh[4096]` unchanged (4 K-chunks × 16 k × 64-expert stride is
  row-count independent); row stride `g * 10u` → `g * 8u` in `r`, `Rg`
  and `rr`; `Rg` cap 10 → 8. The `w`-staging loop bound stays 960
  (4 slots × 240 elements of total work, independent of rows/group);
  every remaining `16u` is K-chunk geometry (16 d-values per chunk,
  `part[16]`), correctly untouched.
- `block-0.metal`, `block-1.metal`: byte-identical to attempt-4
  (`diff` empty) — decode path untouched.

Threadgroup: (4096 + 512 + 512) × 4 B = 20,480 B of 32 KB (was 21,504 B).
480 threads sits inside round-1 attempt-2's proven-passing band and 96
threads under attempt-4's reported 576 cap — but attempt-4 proved the cap
moves with the kernel, so G1 passing is a measurement, not a certainty.

## Inefficiency removed

Same as attempt-2/attempt-4: `ineff-baseline-routing-1` (per-row `w_router`
re-read) and `ineff-baseline-routing-2` (per-row shape cost) together.
`w_router` device traffic falls from rows × 184,320 B to
ceil(rows/8) × 184,320 B (~8× at full groups, was 10×), barriers from
24/row to 25 per group of 8 rows (~3.1/row amortised, was ~2.5),
duty from 23% to 100% in the single-partition regime, footprint from
28,080 B to 20,480 B.

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
  of margin. Unchanged from attempt-4 (decode code is byte-identical).
- Prefill traffic floors at 200 GB/s (`w` staged once per 8 rows):
  - 127 rows: 16 groups × 184,320 B (w) + 127 × 1,536 B (x) + 127 × 32 B
    (out) = 3,148,256 B → 15.7 us. 16 groups fit 16 cores → 1 wave ×
    25 barriers × 0.5 us = 12.5 + idle kernel-0 launch 4 + chained
    fixed cost 25 → **~57** vs compiled 87.3 (`faster`).
  - 1,016 rows: 127 × 184,320 + 1,016 × 1,536 + 1,016 × 32 =
    25,001,728 B → 125 us. ceil(127/16) = 8 waves × 12.5 = 100 +
    idle launch 6 + fixed 25 → **~256** vs compiled 221.1 —
    `slower` under this barrier model.
  - 2,048 rows: 256 × 184,320 + 2,048 × 1,536 + 2,048 × 32 =
    50,397,184 B → 252 us. 256/16 = 16 waves × 12.5 = 200 +
    idle launch 8 + fixed 25 → **~485** vs compiled 290.4 —
    `slower` under this barrier model.

Prediction: **53 / 117 / 124 / 57 / 256 / 485 us**. G1 prediction: pass
(480 < 576 with 96 threads of headroom, inside the proven-passing band;
20,480 B < 32 KB; grid divisible along whole rows). G2
prediction: pass at all six sizes — every accumulation order is copied
op-for-op from attempt-4 (via attempt-2/attempt-4, which this copies), and
only the grouping changed, never any row's order. `beats_compiled`
prediction: false under the 0.5-us/barrier guess (1,016/2,048 rows slower)
— but the barrier cost is a guess, and barriers may
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
the 20,480 B threadgroup sum (unchecked by slots.build); unwritten
threadgroup reads and thread-timing dependence (one equality run per
layer cannot expose them).

## Verdict

G1 pass; G2 pass at all six sizes, 0 rows differing (`grade.json`) —
the first direction-(b) kernel to compile AND prove exact. Timing
(candidate vs compiled, wins/30): decode 1 row 72.6 vs 76.2
(`faster`, 24); 8 rows 123.3 vs 120.4 (`same`, 10); 32 rows 111.9 vs
107.7 (`same`, 16). Prefill 127 rows 147.3 vs 102.1 (`slower`, 3);
1,016 rows 561.6 vs 222.6 (`slower`, 0); 2,048 rows 1,052.7 vs 300.8
(`slower`, 0). Original reference in the same run: 78.0 / 118.6 /
127.1 / 109.8 / 244.9 / 325.7. `phases_vs_compiled`: decode
`not slower`, prefill `slower`; `beats_compiled` false.

## Why the prefill missed the prediction (and what decode says)

Predicted 53 / 117 / 124 / 57 / 256 / 485 us; measured 72.6 / 123.3 /
111.9 / 147.3 / 561.6 / 1,052.7 — decode within drift (+37/+5/−10%),
prefill 2.2–2.6× over at every size (+158/+119/+117%).

- Decode: the 1-row +37% is machine-state drift, not a code cost —
  the compiled bar moved 65.1 → 76.2 (+17%) and the original 68.4 →
  78.0 between the anchor run and this one, and the relative win
  holds (`faster`, 24/30). 8/32 rows landed within ±10% of the
  copy-launch allowance, but drift compressed attempt-4's margin to
  coin-flip races (`same` 10/16 wins) — the same verdict compression
  attempt-3's notes describe.
- Prefill: the traffic sharing works — vs the staged-path anchor
  (227.1/1,298.8/2,513.0) this kernel does 147.3/561.6/1,052.7
  (0.65×/0.43×/0.42×), removing over half the gap — but a per-group
  overhead ~4× the model dominates. Excess over the traffic floors
  (15.7/125/252): 132/437/801 us over 16/127/256 groups ≈
  8.2/3.4/3.1 us/group (127 rows carries small-size fixed cost; big
  sizes settle at ~3.2) vs ~0.8 modeled. In wave terms at 2,048 rows:
  801 us over 16 waves = 50 us/wave vs 12.5 modeled — either barriers
  cost ~2 us each (4× the 0.5 guess) or occupancy serializes ~4× worse
  than one group per core (register pressure from `part[16]` × 480
  threads is the suspect; inferred, not measured). Net: 1.44×/2.52×/
  3.50× slower than compiled, which shares `w_router` across all rows
  while this kernel re-stages it every 8.

Next change (one, in a copied new attempt): amortize the staging and
the 25 barriers over multiple sequential row-blocks inside one
threadgroup — stage each `w` chunk once, stream 2–4 × 8-row blocks of
`x` through it. Traffic/row and barriers/row fall 2–4× with no more
threads and no reordering of any row's accumulation (G2-safe by the
same argument as this attempt). If that still trails, shrink the live
`part[16]` register set toward the reachable partition max to raise
occupancy.
