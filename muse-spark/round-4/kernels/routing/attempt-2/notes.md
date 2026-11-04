# attempt-2 (round 4): share w_router across 4 rows at rows > 16 (builder-2-a, direction a)

Hypothesis `hyp-2-routing-margin` (slot `routing`, baseline `none`).
Starting point `muse-spark/round-3/kernels/routing/attempt-6`
(byte-identical round-2 attempt-4 staged scalar kernels,
`max_rows: 32`; G2 exact, decode `faster` x3, prefill handed off;
round-4 re-grade margins 17.7 / 15.8 / 19.7 µs/layer over compiled at
1 / 8 / 32 rows). Attempt-1 in this round's folder is builder-2-b's
direction (b); this is the first attempt in direction (a).

## The one main change

At `rows > 16` only, share each `w_router` byte across 4 rows; every
other size runs the starting point's path verbatim. One kernel,
threadgroup 256, grid `rows*256` as before:

- A threadgroup covers 4 rows: 240 `(row, expert)` compute threads
  (`lr = tid/60`, `e = tid%60`) plus 16 staging helpers. Threadgroups
  with `gpos*4 >= rows` return immediately, so at 32 rows 8 groups
  compute and 24 exit; at 17 rows 5 compute.
- `w_router` chunks are staged once per 4-row group with the starting
  point's exact coalesced `float4` loads and transposed
  `wsh[slot][k][expert]` layout (expert stride 64), then reused for
  all 4 rows. Device `w_router` traffic at 32 rows falls from
  32 x 184,320 B = 5.90 MB/layer to 8 x 184,320 B = 1.47 MB/layer.
- Each compute thread holds its `(row, expert)` partition partial in
  a register (`pacc`), flushing it into `tot` (`tot += pacc`) when
  the partition index advances and once at the end. A partition's
  products are added in increasing `d` (chunk, then `k`), partitions
  fold in increasing order — the same addition sequence as the staged
  path's `parts[]` store/resume plus sequential fold, so the scalar
  accumulation order is bit-identical. No `parts[]` array is kept on
  this path.
- The 4 rows' `x` are preloaded once into the same padded layout
  (`lr*816 + c*17 + k`), one initial barrier; then 12 staged groups x
  2 barriers + 1 sigmoid barrier = 26 barriers per 4-row group
  (vs 27 per row on the staged path).
- Sigmoid (verbatim `1/(1+precise::exp(abs(s)))`, `(s<0)?y:1-y`),
  the strictly-greater insertion top-4 in ascending expert order
  (earliest index wins ties; per-row scan by the `e == 0` thread),
  and the `(((s0+s1)+s2)+s3)+1e-20`, divide, `x2.5` fold in emitted
  order are copied op-for-op.

At `rows <= 16` the kernel is the starting point remapped onto one
threadgroup buffer `tg[7616]` (staged: `xpad` at 0, `scores` at 816,
`parts` at 876, `wsh` at 2924; shared: 4-row `xpad` at 0, scores at
3264, `wsh` at 3520 — the two layouts never live at once). The GEMV
branch (`rows == 1`), split-K partition formula, staged loops,
sigmoid, top-4 and fold are otherwise verbatim, so 1- and 8-row
behaviour (code and timing, one extra `rows > 16` compare aside) is
inherited, not re-proven. `block-1.metal` and `spec.json` are byte
copies of the starting point (`diff` empty, checked).

Threadgroup memory: `7616 x 4 B = 30,464 B` of 32 KB (starting point
28,080 B); 256 threads, inside the 512 thread cap the task sets and
the proven 480-thread passing band — no pipeline-cap risk of the
round-3 600/960-thread kind.

## Which inefficiency it removes, and how

`ineff-baseline-routing-1` (confirmed): every row's threadgroup
streams its own full copy of `w_router` — excess over read-once
0 / 6.5 / 28.6 µs/layer (floor at 200 GB/s) at 1 / 8 / 32 rows. The
shared path reads each weight byte once per 4 rows at rows > 16, so
at 32 rows the excess falls to (8-1) x 184,320 B / 200 GB/s =
6.4 µs/layer floor. The staged path below the threshold keeps
re-reading per row, deliberately: at 8 rows the excess (6.5 µs
floor, ~1.1 µs at the measured 17% overlap rate) does not pay for
dropping from 8 parallel row-groups to 2 four-row groups.

## max_rows: 32, and why

Inherited from the starting point's `spec.json`, unchanged (byte
copy): every decode step (1-32 rows) stays on the kernels, all three
graded prefill sizes (127 / 1,016 / 2,048) hand off to the compiled
reference and report `handed off`. No `min_rows`: the 1-row GEMV
path, where the relative win is strongest, stays on the kernel.

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Anchor: the round-4 re-grade of the identical starting-point bytes
(`results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`):
candidate / compiled 45.1 / 62.8 at 1 row, 144.2 / 160.0 at 8,
151.3 / 171.0 at 32; compiled 85.7 / 213.1 / 282.0 at the prefill
sizes. Chained absolutes drift between runs (the round-3 grade of
the same bytes read 51.7 / 90.9 / 132.2), so the firm predictions
are the verdicts and margins; absolutes are the anchor's.

- **1 row: ~45 µs, `faster`.** GEMV path verbatim; no sharing is
  possible or needed (excess 0). One extra compare.
- **8 rows: ~144 µs, `faster`.** Staged path verbatim (threshold
  keeps it); excess 6.5 µs floor left in place by design.
- **32 rows: ~147 µs, `faster`, margin widening 19.7 -> ~24 µs.**
  Arithmetic: traffic floor saved (32-8) x 184,320 B / 200 GB/s =
  22.1 µs; at the measured overlap rate (marginal 0.16 µs/row vs
  0.92 µs/row floor, ~17%, round-2 attempt-4 notes) the realised
  slice is 24 rows x 0.16 µs = 3.8 µs, so 151.3 - 3.8 = 147.5.
  Barrier cost is already in that marginal rate's staging term;
  the risk is occupancy (8 groups on 16 cores vs 32), which the
  chained timing decides.
- **127 / 1,016 / 2,048: `handed off`** (= compiled,
  ~86 / 213 / 282 on the anchor run); the kernels never run there.

Prediction: G1 pass (256 threads, 30,464 B); G2 pass at all six
sizes (no accumulation order changes anywhere — sharing changes
where operands are read from, never the order); decode `faster` x3,
prefill `handed off` x3, `beats_compiled` true (baseline `none`, so
no `beats_baseline`).

## What this attempt leaves unmeasured (for the hypothesis entry's Unverified list)

Ungraded decode counts 17-31 on the new shared path (graded 32 runs
the same `Rg = 4` full groups; 17-31 add a partial last group) and
2-7 / 9-16 on the staged path in-kind; the 33-126 band (empty by
construction under `max_rows: 32`); sizes above 2,048; prefill past
the first chunk; hand-off exactness above `max_rows` at untested
sizes; unseen exact 4th/5th ties; the 30,464 B threadgroup sum
(unchecked by `slots.build`); unwritten threadgroup reads and
thread-timing dependence (one equality run per layer cannot expose
them); whole-request composition (ready check only).

## Verdict (2026-11-04T21:56:01Z, one grader run)

Graded once with
`.venv/bin/python bench/grader/check_candidate.py muse-spark/round-4/kernels/routing/attempt-2 routing --out muse-spark/round-4/kernels/routing/attempt-2/grade.json`
(no `--baseline`: baseline `none`). Verdict kept in `grade.json`.

G1 pass. **G2 pass at all six sizes** (`rows_differ` 0,
`rows_other_experts` 0, `max_rel_err` 0.0 at 1 / 8 / 32 and
127 / 1,016 / 2,048 — no differing row anywhere, so no first
difference to cite). Chained µs/layer, reference / compiled /
candidate (medians of 30 paired repeats):

| size | reference | compiled | candidate | wins | verdict |
|---|---|---|---|---|---|
| rows 1 | 68.6 | 59.2 | 44.4 | 30 | `faster` |
| rows 8 | 112.9 | 87.6 | 71.6 | 30 | `faster` |
| rows 32 | 105.9 | 92.1 | 83.8 | 29 | `faster` |
| prefill 127 | 98.5 | 80.9 | 80.9 | — | `handed off` |
| prefill 1,016 | 223.4 | 203.7 | 203.7 | — | `handed off` |
| prefill 2,048 | 349.6 | 299.9 | 299.9 | — | `handed off` |

`phases_vs_compiled`: decode `faster`, prefill `handed off`.
**`beats_compiled` true.**

Margins over compiled: 14.8 / 16.0 / 8.3 µs/layer at 1 / 8 / 32
rows, against the starting point's 17.7 / 15.8 / 19.7 (round-4
re-grade) and 14.6 / 13.1 / 19.8 (round-3 grade). The 1- and 8-row
margins hold (those paths are verbatim); the 32-row margin — the
one this attempt exists to widen, predicted ~24 — came in at 8.3.
Sharing is bit-exact and still beats compiled at 32 rows, but on
this run it did not widen the 32-row margin; it narrowed it.

## Why the prediction missed by >20%

Predicted ~45 / 144 / 147 µs; measured 44.4 / 71.6 / 83.8
(-2% / -50% / -43%). Two separate misses:

1. **Absolutes (all sizes): chained drift, as documented.** The
   anchor re-grade ran on a slow machine state: in this run the
   compiled bar itself moved 62.8 -> 59.2, 160.0 -> 87.6 and
   171.0 -> 92.1 µs, and the candidate moved with it. The 8-row
   absolute halved while its margin (16.0 vs 15.8 predicted-held)
   and its 30/30 wins held exactly — the code at 8 rows is
   byte-equivalent to the anchor's, so this is the drift the
   starting point's notes already document, not a code effect.
2. **The 32-row margin (the real miss): sharing traded away
   per-row parallelism the staged path was using.** The staged
   path spreads one row's 768 products over 32 partition threads
   (256 threads/row, 32 groups at 32 rows); the shared path gives
   each `(row, expert)` one thread holding a serial 768-product
   chain (60 threads/row, 8 compute groups on 16 cores, 24 groups
   launched only to exit). The traffic saved is real (5.90 ->
   1.47 MB/layer staged loads) but, as the inefficiency entry's
   overlap account warned, only ~17% of that floor was ever time;
   the parallelism given up costs more than the ~3.8 µs/layer the
   saving was priced at. Net: at the one size where sharing runs,
   the margin is ~11 µs/layer below the starting point's
   cross-run-stable ~19.7 (inferred from the two prior grades of
   identical bytes, not a same-run measurement).

Next change (one): keep the staged path's partition parallelism
and share `w_router` across rows *inside* it — e.g. one threadgroup
staging `w` once for 2 rows with both rows' `parts` interleaved in
the existing 2,048-float layout budget — instead of collapsing
each row to 60 serial threads; if that cannot fit the 32 KB
budget, direction (a) should be dropped in favour of (b).
