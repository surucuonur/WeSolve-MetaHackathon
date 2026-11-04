# attempt-3 (round 4): tail-group feeding — idle tail threads pre-stage the next `w_router` group (builder-2-c)

Hypothesis `hyp-2-routing-margin` (slot `routing`, baseline `none`),
direction (b) hardening of `muse-spark/round-4/kernels/routing/attempt-1`
(first-touch `parts[]` init, dead zeroing removed; G1+G2 pass at all six
sizes, decode `faster` ×3, margins 27.6 / 20.4 / 17.4 µs/layer over
compiled at 1 / 8 / 32 rows in its own grade). This is the next single
change named by attempt-1's builder: tail-group feeding against
`ineff-baseline-routing-2` items 1–2.

## The one main change

In the staged scalar path only, and only at the decode geometry
(`partitions == 32`, `base_iters == 1`, which every scalar-path decode
size 2–32 takes), the four tail staged groups (`gbase` = 32 / 36 /
40 / 44) compute a single partition on 60 of 256 threads (23.4%
duty). In the first three of those groups the 196 otherwise-idle
threads (`tid >= 60`) now pre-load the *next* staged group's
`w_router` block during the compute phase:

- Each idle thread holds up to 5 `float4`s in registers (`pre[5]`),
  covering the next group's 960 `float4` staging items exactly once
  (`lv = (tid - 60) + j * 196`, `j` = 0..4; same `(s, e, kv)` mapping
  and same coalesced `float4` device loads as the normal staging
  loop, so the bytes landing in `wsh` are identical).
- The next group then fills `wsh` from those registers with
  threadgroup stores only (same transposed `(s * 16 + kv * 4) * 64
  + e` destinations), instead of issuing fresh device loads between
  its two barriers. Device-load latency for groups 36 / 40 / 44 is
  moved behind the preceding tail group's compute phase.

Nothing else changes. `block-1.metal` and `spec.json` are byte
copies of attempt-1 (`diff` empty, checked); grid `rows*256`,
threadgroup 256, one kernel, no added launch, no added threadgroup
memory (registers only; footprint still (816 + 60 + 2048 + 4096) ×
4 B = 28,080 B), barrier count still 27 per row (2 per staged group
+ 3 outside), partition formula, first-touch `parts[]` init,
per-row `w_router` re-read structure, sigmoid, strictly-greater
top-4 and weights fold all verbatim from attempt-1. The `rows == 1`
GEMV path is untouched (the feeding state is declared inside the
scalar branch). Groups outside the tail (`gbase` < 32), the last
group (`gbase == 44`, no next group), and any non-decode geometry
(`partitions != 32` or `base_iters != 1`, unreachable under
`max_rows: 32`) run attempt-1's staging loop verbatim.

## Which inefficiency it removes, and how

`ineff-baseline-routing-2` items 1–2: 27 barriers per row and the
tail third of K-chunks running at 23% thread duty — in each tail
group 196 threads wait at the compute barrier while 60 threads run
the partition-31 chain, and the next group's device loads are then
issued cold between barriers. Feeding puts those idle threads to
work issuing the next group's `w_router` loads concurrently with
the tail compute, so the next group's staging phase shrinks to
register-to-threadgroup stores. This is the entry's legal
"feeding" fix: it touches no partition boundary, no chunk's
accumulator, and no add order (see below), so it cannot reorder
any row's accumulation the way rebalancing would.

## Why this is bit-exact

`wsh` contents for every group are bit-identical to attempt-1:
the prefetch reads the same `float4`s from the same device
addresses and stores them to the same transposed `wsh`
destinations; a device load into a register and back out to
threadgroup memory is exact for float32. Every (partition,
expert) accumulator still starts at `0.0f` on first touch
(attempt-1's init, kept) and adds the same chunks in the same
increasing-`d` order; partials are stored and resumed exactly as
before, then summed in partition order. The verbatim sigmoid,
the strictly-greater earliest-index-wins top-4 and the
`(((s0+s1)+s2)+s3)+1e-20`, divide, ×2.5 fold are untouched. No
accumulation is reordered, sharded or regrouped; the per-row
`w_router` re-read (`ineff-baseline-routing-1`) is deliberately
not touched — pre-staged bytes are still loaded once per row.

## max_rows: 32, and why

Kept from attempt-1 (byte-identical `spec.json`). Decode steps
have 1–32 rows, so every decode step stays on these kernels;
127 / 1,016 / 2,048 rows run the compiled reference and are
reported `handed off` (not timed). No `min_rows`: handing off
1 row would run MLX's own code for a whole 1-request workload.
Sizes above `max_rows` run the reference; none below (no
`min_rows`).

## Predicted µs per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Anchor: attempt-1's own measured grade — candidate
58.5 / 136.9 / 74.7 µs/layer against compiled 86.1 / 157.3 /
92.1 at 1 / 8 / 32 rows (margins 27.6 / 20.4 / 17.4).

Saving arithmetic. Feeding pre-stages 3 of the 12 staged groups
(36 / 40 / 44) at every scalar-path row. Each such group
otherwise issues 960 `float4` device loads = 15,360 B per row
(0.077 µs at the 200 GB/s floor — bytes are not the prize) plus
their exposed latency between two barriers. No barrier is
removed, so the inefficiency entry's outer envelope (its
inferred ~2 µs/barrier datum) does not apply; the realised
saving is only the load-issue + latency slice that overlaps
the preceding tail group's compute phase, minus the cost of
holding 5 extra `float4`s (20 floats) in registers. The tail
compute phase itself is short (60 threads × 4 chunks × 16
FMAs), and cross-threadgroup overlap at 8–32 rows already hid
part of this latency in attempt-1, so the honest expectation
is 0–2 µs/layer at 8 / 32 rows and exactly 0 at 1 row (GEMV
path, code untouched — any 1-row delta is machine drift).
Prediction, attempt-1 anchor minus ~1 µs/layer at 8/32:

- 1 row: **≈58 µs/layer** (58.5 − 0), verdict `faster` (margin 27.6).
- 8 rows: **≈136 µs/layer** (136.9 − ~1), verdict `faster` (margin 20.4).
- 32 rows: **≈74 µs/layer** (74.7 − ~1), verdict `faster` (margin 17.4).
- 127 / 1,016 / 2,048 rows: `handed off` — not timed; the attempt's
  time there is the compiled reference's by construction.

Chained absolutes drift with machine state (attempt-1's own
notes: identical starting-point bytes measured 51.7 / 90.9 /
132.2 in round 3 vs 45.1 / 144.2 / 151.3 at re-grade, with the
same run's compiled bar moving +37% / −46% at 1 / 32 rows), so
the load-bearing prediction is the verdicts: G1 pass, G2 pass
at all six sizes, decode `faster` ×3, prefill `handed off` ×3,
`beats_compiled` true. A measured delta beyond ~2 µs/layer
against attempt-1's absolutes is expected to be drift, not
this change, and will be judged against the same run's
compiled bar and margins.

## What this attempt leaves unmeasured (for the hypothesis entry's Unverified list)

Everything attempt-1 left, plus: whether the tail-feeding
overlap is resolvable at all inside paired-repeat noise (the
tail compute phase may be too short to hide 960 loads, and the
register cost may eat the saving); ungraded decode counts 2–7
and 9–31 (same `partitions = 32, base_iters = 1` geometry, so
feeding is reasoned to run there in-kind, not graded); the
33–126 band (empty under `max_rows: 32`); hand-off exactness
above `max_rows` at untested sizes; sizes above 2,048; later
prefill chunks; other prompts and later decode steps; unseen
exact 4th/5th ties; the barrier cost itself (items 1–2's
barrier count is unchanged and still undecomposed); the
serial tails (item 5, untouched); whole-request (G3) effect,
which only the ready check measures as a set.

## Verdict

Graded once (`.venv/bin/python bench/grader/check_candidate.py
muse-spark/round-4/kernels/routing/attempt-3 routing --out
muse-spark/round-4/kernels/routing/attempt-3/grade.json`, no
`--baseline`: baseline `none`). Verdict kept in `grade.json`.

G1 pass. **G2 pass at all six sizes** (`rows_differ` 0,
`rows_other_experts` 0, `max_rel_err` 0.0 at 1 / 8 / 32 and
127 / 1,016 / 2,048 — no differing row anywhere, so no first
difference to cite). Tail-group feeding is bit-exact as reasoned.

Chained µs/layer, reference / compiled / candidate (medians of 30
paired repeats):

| size | reference | compiled | candidate | wins | verdict |
|---|---|---|---|---|---|
| rows 1 | 70.5 | 66.2 | 49.3 | 30 | `faster` |
| rows 8 | 100.4 | 86.1 | 74.5 | 30 | `faster` |
| rows 32 | 105.4 | 94.8 | 81.8 | 27 | `faster` |
| prefill 127 | 102.0 | 81.4 | 81.4 (= compiled) | — | `handed off` |
| prefill 1,016 | 231.6 | 213.4 | 213.4 (= compiled) | — | `handed off` |
| prefill 2,048 | 372.2 | 332.2 | 332.2 (= compiled) | — | `handed off` |

`phases_vs_compiled`: decode `faster`, prefill `handed off`.
**`beats_compiled` true.** Baseline `none`, so no `beats_baseline`.
Decode margins over compiled this run: 16.9 / 11.6 / 13.0 µs/layer
(attempt-1's own run: 27.6 / 20.4 / 17.4; starting-point re-grade:
17.7 / 15.8 / 19.7).

## Why the 8-row absolute missed the prediction by >20%

Predicted ≈58 / 136 / 74 µs/layer (attempt-1's measured anchor
minus ~1 µs/layer at 8/32); measured 49.3 / 74.5 / 81.8 — −16%
at 1 row and +11% at 32 rows (inside 20%), but −45% at 8 rows.
The miss is machine state, not this change, and this run proves
it: the compiled bar at 8 rows moved 157.3 → 86.1 µs/layer
(−45%) against attempt-1's run and the uncompiled reference
moved 164.8 → 100.4 (−39%) in the same paired repeats, while
the candidate/compiled ratios stayed in attempt-1's band
(0.74 / 0.87 / 0.86 here vs 0.68 / 0.87 / 0.81 in attempt-1) and
the paired wins were decisive (30 / 30 / 27). Three pre-staged
groups' load overlap, bounded above at ~2 µs/layer in the
prediction, cannot move a chained 8-row absolute by 62 µs.
What this run does show against attempt-1 is a *narrower*
decode margin at 8 / 32 rows (11.6 / 13.0 vs 20.4 / 17.4
µs/layer, ratios 0.87 / 0.86 vs 0.87 / 0.81): the feeding
saving is inside paired noise, as the notes above allowed
(0–2 µs/layer, possibly eaten by the 20-float register hold
and by the tail compute phase being too short to hide 960
loads), and the residual margin difference is cross-run drift
of the kind attempt-1 already documented (+37% / −46% swings
of the compiled bar between runs of identical bytes). The
change is therefore kept as bit-exact and verdict-identical
to attempt-1 (`faster` ×3, `beats_compiled` true), but no
µs/layer saving is banked for it. This note is also appended
to the inefficiency entry.
