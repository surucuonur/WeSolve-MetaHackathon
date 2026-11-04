# Hypothesis 2: experts prefill at compiled speed

- **Owner:** lead
- **Status:** dropped
- **Version:** 8
- **Updated:** 2026-11-04T18:18:10Z
- **Next:** none — slot keeps the reference; attempt-1 (max_rows 59) is the next round's starting point

## Inefficiency

[ineff-baseline-experts-1.md](../exploration/ineff-baseline-experts-1.md) (confirmed): the starting point's `rows >= 60` scalar `qmm_accum` prefill branch runs 15.8/27.7/20.1× slower than the compiled reference (15,220/119,154/240,681 vs 961/4,306/11,999 µs/layer) — constant ~29.5 µs/pair serial issue at 18.8 GB/s — making prompt processing 6.63–14.50× slower and costing −10.0% to −82.5% of whole-request throughput. Slot: `experts`. Baseline: `none` (starting point `muse-spark/round-2/kernels/experts/attempt-7`, which passes G1+G2 but is `slower` at 1 row and at every prefill size).

Predicted effect if the prefill path matches compiled speed at equal bits: save 14,259/114,848/228,682 µs/layer = 285 ms / 2.30 s / 4.57 s per chunk, recovering the throughput loss to whatever the decode phase then gives (decode wins at 8/32 rows must be kept: ahead 72.7/282.5 µs/layer). Fix shapes: (a) vectorized per-pair scalar (one thread per word's 8 outputs, 8 accumulators, each output keeping contiguous `BK=32` order); (b) grouped/tiled replica of `qmm_rhs` issue (takes the tile-waste and offsets prizes, ranked prefill items 7–8, for free); (c) `max_rows` hand-off (Claude checked 59 bit-exact; decode stays on the fused path). The 1-row decode gap ([ineff-baseline-experts-2.md](../exploration/ineff-baseline-experts-2.md)) is a disjoint cause and a later hypothesis (hyp-4) built on this one's winner; a prefill fix that slows decode, or vice versa, is not an improvement (ADR 0011).

## Boundary

(written by boundary-2; delta only — every line carries its source)

**Cited boundary, checked line by line.** Round 2's boundary for this slot is
[`muse-spark/round-2/implementation/hyp-1-experts-exact.md`](../../round-2/implementation/hyp-1-experts-exact.md)
*Boundary* (itself a delta on round 1's
[`muse-spark/implementation/hyp-1-fused-experts.md`](../../implementation/hyp-1-fused-experts.md)).
I checked each line against the starting point
`muse-spark/round-2/kernels/experts/attempt-7` (`block-0.metal`, `block-1.metal`,
`spec.json`) and against the current G2 plus the two-phase judgment
(`claude/muse-agent-brief.md:173,185-187,209-213`;
`claude/adr/0011-decode-and-prefill.md:21-35`;
`bench/grader/check_candidate.py:224-264,308-378`;
`bench/grader/slots.py:304-337`). Still holding, not rewritten below:
input/output names, dtypes, layouts and weight shapes/bytes at all six sizes;
`y[r,j]` positional in `inds` order; neighbour contracts; the four bfloat16
rounding points in kind; the two reference arithmetics and the kernel switch at
60 rows; launch-spec surface; GPU limits. Baseline is `none` — no slot has one
this round, so the bar is the compiled reference everywhere
(`claude/adr/0011-decode-and-prefill.md:46`). Only what changed follows.

### Inputs and outputs — unchanged

Shapes, dtypes and layouts are round 2's verbatim (`x` [rows,768] bfloat16,
`inds` [rows,4] uint32, `y` [rows,4,768] bfloat16 at 1/8/32 and 127/1,016/2,048
rows; `claude/muse-agent-brief.md:74-82`). Rechecked against the starting
point: its `spec.json` input list is the slot's full 11 inputs and its output
`y` is `["rows",4,768]` bfloat16; it reads `rows` from `x_shape[0]`
(`block-0.metal:22`), the same value the grader reads from the first slot
input's `shape[0]` (`bench/grader/slots.py:334`).

### Before and after — unchanged

Neighbour code and contracts are round 2's verbatim (post-attention layernorm
→ `x`; router → `inds`, `weights`; `experts(x, inds)` → `y`; combine pairs
`y[r,j]` with `weights[r,j]` positionally;
`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`;
`bench/grader/slots.py:97-119`). The starting point's changes vs its own
predecessor (interleaved down pair, one shared kernel-scope `q[32]`,
`block-0.metal:42-43,95-102`) are internal to the slot; nothing crosses a
neighbour boundary.

### What must stay identical — changed: the `max_rows` hand-off is allowed

- **Bit-exactness unchanged.** G2 still requires `differ == 0` in all 20 layers
  at all six sizes (`claude/muse-agent-brief.md:185-191`;
  `bench/grader/check_candidate.py:161-177,322-344`); `max_ulp` stays
  diagnostic only. The two reference arithmetics are unchanged: `gather_qvm`
  below 60 rows (float32 dequant, strided-lane accumulation + `simd_sum`),
  `gather_qmm_rhs` at/above (bfloat16-rounded dequant tiles, `BK=32` MMA-order
  accumulation). The starting point still branches on the dispatch threshold
  itself, `x_shape[0] >= 60` (`block-0.metal:22`), so 33–59 stay `qvm`.
- **New: rows above `max_rows` run the compiled reference, not the kernels.**
  With `"max_rows"` in `spec.json` (int ≥ 1, bools rejected),
  `slots.build` sends inputs with `rows > max_rows` to the slot's reference
  under `mx.compile` (`bench/grader/slots.py:327-337`;
  `claude/muse-agent-brief.md:173`; `claude/adr/0011-decode-and-prefill.md:33-35`).
  G2 checks `run()`, so sizes above `max_rows` exercise the compiled reference
  through the attempt's own dispatch; the compiled reference was bit-exact with
  the reference at all six sizes on 2026-11-04 (`bench/micro/compiled_exact.py`,
  `results/runs/2026-11-04-handoff-check/`). This replaces the ADR 0010 line
  that required the kernels to run in every call
  (`claude/adr/0011-decode-and-prefill.md:17`).
- **Decode stays on the candidate: `max_rows >= 32`.** A decode step has 1–32
  rows (`claude/muse-agent-brief.md:26`); every decode step up to 32 rows must
  run the fused path, so the hand-off threshold must be at least 32. Below it
  the kernels face only `gather_qvm` rounding.
- **The hand-off sits below MLX's switch: `max_rows <= 59`.** MLX changes its
  own experts kernel at 60 rows (`bench/grader/slots.py:219-220`;
  `claude/muse-agent-brief.md:187`), so kernels that only replicate `qvm`
  arithmetic must never see rows ≥ 60. The valid hand-off window is therefore
  **32 ≤ `max_rows` ≤ 59**; round 2's checked hand-off copy used 59
  (`results/runs/2026-11-04-handoff-check/`). Above `max_rows` the attempt
  times `same` as the compiled bar by construction, never better — its claim
  is "faster decode, unchanged prefill"
  (`claude/adr/0011-decode-and-prefill.md:35,48`).
- **Consequence for the three fix shapes** (this entry, *Inefficiency*): (a)
  and (b) keep kernels at every row count, so they must keep both arithmetics
  bit-exact (or add a hand-off); (c) hands prefill-sized inputs off with
  `max_rows` in 32–59 and deletes or dead-ends the scalar `qmm_accum` branch.
  Tie rule and pair order unchanged: this slot inherits routing's ids as given
  and keeps `y[r,j]` positional, because combine sums positionally
  (`bench/grader/slots.py:51-52,116`).

### What the grader checks, and what it can't see — changed: two-phase timing

Checks (`bench/grader/check_candidate.py`): G1/G2 mechanics unchanged — G1 is
true only after the whole six-size correctness loop completes without
exception (`:342`); per-layer shape/dtype faults fail G2 with G1 true
(`:324-326`). **Timing now runs at all six sizes** (was decode-only), still
chained through the `link` kernel (`:345-378`). Per size, 30 paired repeats in
rotating order, `faster`/`slower` at ≥22 wins/losses, else `same` (`:224-245`);
per phase, `faster` iff every size is, `slower` iff any size is, else
`not slower`; `beats_compiled` iff one phase is `faster` and neither is
`slower` (`:248-264`). **Round 2's "a slow-but-exact prefill branch is free"
line is superseded** — prefill speed is now timed, and this hypothesis exists
because the starting point is `slower` at every prefill size. No `--baseline`
bar exists this round (no baselines; `territory-grader.md` §1).
What it still can't see (checked by hand later — *check an attempt*; full list
`muse-spark/round-3/exploration/territory-grader.md` §*What the grader still
cannot see*): any `max_rows` in 32–126 behaves identically at all six G2/timing
sizes — the grader cannot tell 59 (right side of 60) from 100 (wrong side);
only `check_sizes.py` (59/60/61) distinguishes them, and it is boundary
tooling, not a gate. Kernel speed at 33–126 rows is unmeasured by the piece
timing (with `max_rows` > 32 the kernels run there in real serving); sizes
above 2,048 untested; prefill beyond the first chunk of seeded 128-token
prompts, other lengths/distributions, mixed batches; unwritten threadgroup
reads, thread-timing dependence, the 32 KB limit; NaN-vs-NaN counts as equal
(`:101`); 1-request whole-request noise in G3.

### The GPU limits — unchanged, plus the pipeline ceiling

At most 1,024 threads per threadgroup (count enforced,
`bench/grader/slots.py:341-342`; `claude/muse-agent-brief.md:100`) and 32 KB
of threadgroup memory (not enforced by `build` — verify by hand from the Metal
source). Starting point for scale: 512 threads/group (`spec.json`) and 3,840 B
(`tg_x[768] + tg_gate/up/h[384]` bfloat16, `block-0.metal:24-27`), valid at
every row count. Carried invariant from round 2 (attempt-6 G1 fail → attempt-7
pass): the binding ceiling for this 512-thread fused kernel is the compiled
pipeline's `maxTotalThreadsPerThreadgroup`, not the 1,024 count — headroom is
exactly two distinct 32-float thread arrays (`p`, shared `q`); a third drops
the compiled maximum below 512 and fails G1 loudly
(`hyp-1-experts-exact.md` *Boundary checks*, attempt-6/7 notes).

## Builders

### builder-2-a — direction (a): the `max_rows` hand-off

Copy the starting point's kernels verbatim (fused decode path +
G2-exact scalar prefill branch as dead-code fallback) and hand
prefill-sized inputs to the compiled reference with `max_rows: 59`
(top of the boundary's 32–59 window: decode rows ≤ 32, plus 33–59
which take the `qvm` decode path on both sides, stay on the
candidate; MLX switches at 60).

- **attempt-1** (`muse-spark/round-3/kernels/experts/attempt-1/`):
  predicted 93.8 / 160.1 / 510.1 decode, = compiled prefill.
  Measured (`grade.json`): G1 pass; G2 pass all six, differ 0.
  Decode 1/8/32 rows: 96.9 / 168.9 / 554.7 vs compiled 86.6 /
  238.2 / 883.4 — `slower` (5 wins) / `faster` (29) / `faster`
  (29). Prefill 127/1,016/2,048 rows: 1053.5 / 4106.9 / 7557.0
  vs compiled 1131.1 / 4133.5 / 7557.8 — `same` (19/15/16 wins)
  at all three. `phases_vs_compiled` decode `slower`, prefill
  `not slower`; `beats_compiled` false (only 1-row decode
  misses). Saved ≈ 14,167/115,047/233,124 µs/layer vs the
  starting point's prefill. Taught: the hand-off removes the
  confirmed prefill cause in full at zero decode cost; decode
  verdicts identical to the starting point's; prefill `same` is
  the ceiling by construction — the next change belongs to hyp-4
  (1-row decode), built on this attempt.

### builder-2-c — direction (c): grouped/tiled prefill replica

Four kernels: the starting point's fused body as a decode kernel
(plus `if (prefill) return;`, output renamed `y_decode`), a
counting-sort count kernel (pairs by expert, offsets once), a fused
grouped prefill kernel (one 64-thread group per (expert, 16-pair
tile): bfloat-rounded dequant shared across <= 16 rows, 8-wide frag
MMA in BK=32 order, on-chip swiglu, positional stores), and an exact
select kernel on `rows >= 60`. No `max_rows`.

- **attempt-3** (`muse-spark/round-3/kernels/experts/attempt-3/`):
  predicted 95 / 166 / 519 decode (`slower`/`faster`/`faster` vs
  78.6 / 235.5 / 797.8) and 900 / 4,100 / 11,500 prefill (`faster`
  everywhere vs 960.8 / 4,306.0 / 11,998.9); `beats_compiled`
  false (1-row decode still misses, for hyp-4). Measured
  (`grade.json`): G1 **fail** — `block-3.metal` declares
  `uint half`, but `half` is a reserved Metal type name, so the
  build fails with six parse errors. No G2, no timing; predictions
  untested, not refuted. Taught: the reserved word is the only
  defect reached; the next change is a rename (`half` -> `part`)
  in a copied new attempt, then grade.

### builder-2-b — direction (b): vectorized per-pair scalar

One thread computes a weight word's 8 outputs with 8 accumulators
(`qmm_accum8`), each output keeping the scalar branch's contiguous
`BK=32` bfloat16-rounded-dequant order; packed 8 blocks per
simdgroup (gate/up one wave of 2, down one wave of 3); fused decode
path, dispatch, grid and `spec.json` byte-identical to the starting
point, no `max_rows`.

- **attempt-2** (`muse-spark/round-3/kernels/experts/attempt-2/`):
  predicted decode 94 / 160 / 510, prefill 3,000 / 24,000 / 49,000
  (~5× on 8× fewer issues + 8× overlap). Measured (`grade.json`):
  G1 pass; G2 pass all six, differ 0. Decode 1/8/32 rows: 152.8 /
  173.3 / 634.1 vs compiled 113.0 / 240.4 / 896.3 — `slower` (5)
  / `faster` (28) / `faster` (24). Prefill 127/1,016/2,048 rows:
  20,443.5 / 156,415.6 / 303,542.8 vs compiled 1,113.0 / 6,774.6
  / 16,333.0 — `slower` (0 wins) at all three.
  `phases_vs_compiled` decode `slower`, prefill `slower`;
  `beats_compiled` false. Taught: widening across outputs without
  unrolling across k buys nothing — each k step's FMAs depend on
  that step's loads (in-order issue, serial depth still 768) and
  384→48 threads/group removed the cross-thread latency hiding;
  within-run vs-compiled 18.4/23.1/18.6× vs the starting point's
  15.8/27.7/20.1× (this run's machine was much slower than the
  basis run's, so absolute µs overstate the gap — see notes.md).
  Next: keep the word-packing, unroll 4 k-steps × 8 outputs into
  `p[0..31]`; if that fails, direction (b) is exhausted.

### builder-2-d — direction (c) fix: attempt-3's reserved word renamed

Copy of attempt-3 with the single prescribed fix: `half` -> `part`
in `dequant_tile_4bit` (`block-3.metal`, 1 declaration + 4 uses);
all other files byte-identical. No `max_rows`.

- **attempt-4** (`muse-spark/round-3/kernels/experts/attempt-4/`):
  predicted 95 / 166 / 519 decode, 900 / 4,100 / 11,500 prefill
  (attempt-3's arithmetic, unchanged). Measured (`grade.json`):
  G1 pass; G2 pass all six, differ 0. Decode 1/8/32 rows: 122.2 /
  206.4 / 584.7 vs compiled 77.9 / 237.6 / 807.4 — `slower` (0) /
  `faster` (29) / `faster` (29). Prefill 127/1,016/2,048 rows:
  2,684.3 / 9,311.2 / 17,229.0 vs compiled 1,005.7 / 4,073.3 /
  7,417.4 — `slower` (0 wins) at all three (2.67/2.29/2.32×).
  `phases_vs_compiled` decode `slower`, prefill `slower`;
  `beats_compiled` false. Taught: the grouped path is bit-exact
  (frag-mapping suspect refuted) and 5.7/12.8/14.0× faster than
  the starting point's prefill in cross-run absolutes, but ~3×
  off the "compiled rate" prediction — candidate causes:
  `x` gathered 12× per tile (`g`-outer nest), scalar-fed MMA at
  64 threads/group, partial-tile padding. Decode overhead also
  missed (+30/+44/+69 vs +3/+3/+4). Next: interchange to
  `k`-outer/`g`-inner so each `x` tile loads once.

### builder-2-e — direction (e): reference-shaped unfused 1-row path

Four kernels on attempt-1, `max_rows: 59` kept: K0 gate/up+silu (48
fixed groups x 128 threads, one per (pair, block), -> `h_small`),
K1 down (96 groups x 128, -> `y_small`), K2 attempt-1's fused kernel
plus a uniform early return at rows == 1 and `y` stores renamed to
`y_fused`, K3 exact select-copy to `y`. Rows == 1 runs K0/K1;
every other count runs K2. Every per-block accumulation order kept
(same k sequence, terms, simd_sum trees, four bfloat16 points).

- **attempt-5** (`muse-spark/round-3/kernels/experts/attempt-5/`):
  predicted 68 / 176 / 562 decode (`faster` x3), = compiled prefill
  (`same` x3), `beats_compiled` true. Measured (`grade.json`): G1
  pass; G2 pass all six, differ 0. Decode 1/8/32 rows: 234.3 /
  252.7 / 560.4 vs compiled 118.4 / 291.7 / 802.4 — `slower` (0) /
  `faster` (25) / `faster` (30). Prefill 127/1,016/2,048 rows:
  967.2 / 3952.1 / 7362.5 vs compiled 968.0 / 3950.5 / 7358.5 —
  `same` (16/14/14) at all three. `phases_vs_compiled` decode
  `slower`, prefill `not slower`; `beats_compiled` false. Taught:
  the regrouping is bit-exact and keeps the 8/32-row wins and
  prefill `same`, but 1-row scores 1.98x compiled (0/30) vs fused
  1.12x — unfused work + 5 dispatches cost ~+100 us over the fused
  chain (dispatch ~+18 us at this run's ~6 us/dispatch reference-vs-
  compiled price; the rest is per-group setup x144 vs amortized x4,
  prime suspect). Reference-shaped is not reference-priced outside
  one mx.compile dispatch. Next: do not build on attempt-5; bank
  attempt-1 — the 1-row gap looks structural (see notes.md).

## Boundary checks

### attempt-1 — pass (boundary-2, 2026-11-04T17:59:14Z)

G1+G2 pass (differ 0 all six sizes). Kernels verified byte-identical to
the starting point (`diff` of `block-0.metal` and `block-1.metal`
against `muse-spark/round-2/kernels/experts/attempt-7/`); `spec.json`
differs by one key only, `"max_rows": 59` (int ≥ 1, not a bool —
accepted by `bench/grader/slots.py:329`). Dispatch sends rows >
`max_rows` to the compiled reference (`slots.py:335`), so rows ≤ 59
run the kernels.

- **Every decode step stays on the fused path.** A decode step has
  1–32 rows; 32 ≤ 59, so all of them run the kernels. Pass.
- **Hand-off sits below MLX's switch at 60.** Rows ≥ 60 hand off to
  the compiled reference; the kernels only ever see rows ≤ 59, where
  the reference uses `gather_qvm` — and the kernels' own dispatch
  (`block-0.metal:22`, `x_shape[0] >= 60`) takes the `qvm` path there
  too, so both sides agree at every kernel-visible count including
  the ungraded 33–59. The scalar `qmm_accum` prefill branch is dead
  code wherever the kernels run, kept as G2-exact fallback. Pass.
- **Row counts G2 doesn't test.** Kernel work is per-pair
  (one 512-thread group per pair, grid `rows*2048` threads total):
  no cross-row interaction, so the 1/8/32-row G2 verdict extends to
  every decode count up to 32 by construction; 33–59 take the same
  `qvm` path graded at 1/8/32. Prefill chunks of any size ≥ 60 run
  the compiled reference exactly. Speed at 33–59 rows stays
  unmeasured (grader times only 1/8/32 there) — noted, not a defect.
- **Unwritten threadgroup reads: none.** `tg_x[768]` is fully covered
  by the strided loop (`block-0.metal:29-31`); `tg_gate/tg_up[384]`
  are fully covered in both branches (12 simdgroups × 32 lanes write
  distinct `b*32+lane`, lines 51-58 / 61-71); `tg_h[384]` is fully
  covered by the strided silu loop (line 77). Every read follows its
  phase's barrier (lines 32, 73, 86).
- **Threadgroup memory far from the limit.** 3,840 B
  (`tg_x[768] + tg_gate/up/h[384]` bfloat16, `block-0.metal:24-27`)
  vs the 32 KB ceiling; 512 threads/group vs the 1,024 count and the
  inherited pipeline ceiling (two 32-float thread arrays `p`, `q`
  unchanged — lines 42-43 — so the attempt-6 G1 failure mode does
  not recur). Pass.
- **No thread-timing dependence.** All cross-thread sharing passes
  through `threadgroup_barrier`s; `y` slices are disjoint per pair
  and per (simdgroup, lane); no atomics; `simd_sum` reductions are
  fixed trees. The shared kernel-scope `q[32]` phases are separated
  by barriers with re-zeroing on entry (round-2 checked).
- **No new input assumptions.** The attempt adds no input-dependent
  logic beyond `rows > max_rows` on `x_shape[0]` — the same value the
  grader reads (`slots.py:334`) — and `inds` flows through untouched
  (`y[r,j]` stays positional in `inds` order).

## Decision

Dropped: no experts attempt has `beats_compiled`, and no further change with a mechanism can be named. The prefill prize is fully banked by attempt-1 (`max_rows` 59: prefill `same` × 3, G1+G2 pass, boundary pass, decode wins at 8/32 rows) — but decode at 1 row misses (96.9 vs 86.6, 5/30 wins), so `beats_compiled` is false and the attempt cannot close as a candidate. The 1-row gap now survives two approaches: six fused exact packings plateaued at 92–110 µs/layer (round 2), and the reference-shaped unfused path (attempt-5) came out nearly 2× worse (234.3 vs 118.4, 0/30 wins) — the reference's speed comes from one `mx.compile` dispatch, not just its shape. Per the approach rule, the next attempt would need a third approach; none exists in the spec surface (there is no hand-off below a row count, only `max_rows` above it). The slot keeps the reference. Durable gains: the `max_rows`-59 design (bit-exact, prefill `same`) and the bit-exact grouped prefill path (attempt-4, 2.3× off) stand as the next round's starting material.

## Failure ledger

- attempt-3 (grouped/tiled replica): G1 fail — `uint half` in block-3.metal, `half` is a reserved Metal type name; no G2, no timing. Cause: compile error, not a speed cause; fix is a rename in a copied attempt.
- attempt-2 (vectorized per-pair scalar): G1+G2 pass, prefill `slower` x3 (20,444/156,416/303,543 vs compiled 1,113/6,775/16,333; within-run ratios 18.4/23.1/18.6× vs starting point's 15.8/27.7/20.1× — no speedup). Cause: serial depth stayed 768 (no k-unrolling) while active threads fell 384→48/group, removing cross-thread latency hiding without shortening any chain. Approach: per-pair scalar widening.
- attempt-4 (grouped/tiled replica, renamed): G1+G2 pass, prefill `slower` x3 (2,684/9,311/17,229 vs compiled 1,006/4,073/7,417; 2.67/2.29/2.32×) — bit-exact grouped path, 6–12× better than the scalar branch but still over 2× off. Cause: `x` gathered 12× per tile by the `g`-outer nest (prime suspect). Approach: grouped/tiled replica. Line ends: even a 2× nest fix leaves it behind attempt-1's `same`, and the 1-row decode gap blocks every experts attempt equally — hardening continues on attempt-1.
- attempt-5 (reference-shaped unfused 1-row path on attempt-1): G1+G2 pass, decode 1-row 234.3 vs 118.4 (0/30 wins, `slower`) — nearly 2× worse than the fused chain; 8/32-row wins and prefill `same` × 3 kept. Cause: ~+100 µs over fused — ~18 µs from 5 dispatches vs 2 plus per-group setup × 144 groups vs amortized × 4. Approach: unfused regrouping (genuine approach change from fused packing — and it lost worse). No third approach exists; hypothesis dropped.

## Unverified

(at least the boundary's *can't see* list, once written)

## Checks

(none yet)
