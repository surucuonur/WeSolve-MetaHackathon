# attempt-5 (round 3): reference-shaped unfused 1-row path on attempt-1 (builder-2-e)

Hypothesis `hyp-2-experts-prefill` (slot `experts`, baseline `none`),
direction (e): fix the 1-row decode gap with a reference-shaped unfused
structure for small row counts. Builds on attempt-1 (the round's best:
fused decode path + `max_rows: 59`; prefill `same` x3, decode wins at
8/32 rows, loses only at 1 row).

## The one main change

At exactly 1 row, run two short unfused kernels (gate/up+silu, then
down) with per-block threadgroups instead of the fused kernel; every
other row count keeps attempt-1's fused path byte-for-byte. Four
kernels, `max_rows: 59` kept:

- K0 `gate_up_small` (`block-0.metal`): fixed grid of 48 groups x 128
  threads, one group per (pair, 32-output block). Loads the x row to
  threadgroup memory, computes 8 gate + 8 up outputs per simdgroup with
  qvm_block's exact per-output arithmetic, rounds to bfloat16, runs the
  fused kernel's exact silu-mul op sequence, stores `h_small`. Returns
  immediately unless `x_shape[0] == 1`.
- K1 `down_small` (`block-1.metal`): fixed grid of 96 groups x 128
  threads, one group per (pair, 32-output block). Loads the h row from
  `h_small`, computes 8 down outputs per simdgroup with qvm_block's
  exact per-output arithmetic over K=384, stores `y_small`. Returns
  immediately unless `inds_shape[0] == 1`.
- K2 `fused_experts_branch` (`block-2.metal`, header `block-3.metal`
  byte-identical to attempt-1's `block-1.metal`): attempt-1's
  `block-0.metal` plus a uniform early `return` at `x_shape[0] == 1`
  and its three `y[` stores renamed to `y_fused[`; all accumulation
  orders, silu, prefill branch and rounding points untouched.
- K3 `select_small` (`block-4.metal`): exact copy of `y_small`
  (rows == 1) or `y_fused` (else) to `y`, one thread per element.

## Which inefficiency it removes, and how

`ineff-baseline-experts-2` (confirmed): at 1 row the fused kernel runs
4 threadgroups x 512 threads through a barrier-serialized chain (tg_x
load, 24 merged gate/up K steps, silu, 12 merged down K steps, three
barriers) at the register ceiling — 92.1 vs 78.6 chained, +13.5
(+17.2%). Round 2's evidence says the next lever is not another fused
packing (six packings plateaued at 92-110) but a reference-shaped
unfused structure: the compiled reference runs 7 short kernels with
24/24/48 threadgroups per product and no cross-phase barriers.

This attempt is that structure, scoped to the one failing size: K0/K1
split the chain into two short kernels with 48/96 groups (vs 4),
each thread carrying 16/8 accumulators (vs 64) and one in-kernel
barrier (vs three), with silu fused elementwise into K0 (no barrier
needed — it is per-element on bfloat16 values). Cross-phase latency
hides across dozens of groups instead of serializing behind 4. Every
per-block accumulation order is kept: each output accumulates
k = lane, lane+32, ... serially with the same x*(s*q+bias) terms and
the same simd_sum tree as the fused kernel's helpers (the 8 outputs
of simdgroup s are exactly fused inner indices i = 4s..4s+3, one
shared weight word per k step), and the silu sequence is the fused
kernel's op-for-op on the same bfloat16 values.

## max_rows: 59, and why

Unchanged from attempt-1: top of the boundary's 32-59 window, every
decode-size input stays on the candidate, every prefill-size input
hands to the compiled reference (`same` by construction). The unfused
path only runs at rows == 1, where both the kernels and the reference
use `gather_qvm` arithmetic.

## Predicted us per layer at 1 / 8 / 32 / 127 / 1,016 / 2,048 rows (chained)

Basis: attempt-1's `grade.json` (96.9 / 168.9 / 554.7 decode vs
compiled 86.6 / 238.2 / 883.4; prefill `same` x3).

- **1 row: 68 (`faster`).** The ineff entry decomposes fused 92.1 as
  ~20 structural + ~60 exposed K-chain + ~11 byte floor. K0+K1 move
  ~2.5 MB (same weights as fused plus h_small traffic) with ~1/4 the
  per-thread serial depth and no cross-phase barriers; two short
  kernels should land near the reference's rate for the same bytes,
  ~60 for K0+K1 plus ~5 for K2's empty launch and ~3 for K3's 6 KB
  copy. Needs 22/30 pair wins vs ~87; the margin is thin and this is
  the attempt's whole bet.
- **8 rows: 176 (`faster`).** Fused path 168.9 plus two fixed-grid
  empty launches (48+96 groups exiting) and K3's 49 KB copy, ~7
  total; the win margin is 69.3.
- **32 rows: 562 (`faster`).** Fused path 554.7 plus the same fixed
  empty launches and K3's 196 KB copy, ~7 total; the margin is 328.7.
- **Prefill 127/1,016/2,048: = compiled (`same` x3).** Hand-off
  untouched; the kernels never run there.
- **G1: pass** — K0/K1/K3 are tiny kernels (16/8/0 thread floats,
  1,536/768/0 B threadgroup); K2 keeps attempt-7's two-array
  footprint at 512 threads.
- **G2: pass, differ 0 at all six sizes** — the unfused path replays
  the fused kernel's per-output arithmetic op-for-op (same terms,
  same order, same simd_sum trees, same four bfloat16 rounding
  points); rows != 1 run the fused kernel and an exact copy.
- **`beats_compiled`: true** — decode `faster` at all three sizes,
  prefill `not slower` (same x3).

## What this attempt leaves unmeasured

- Attempt-1's list (33-59-row kernel speed, sizes above 2,048,
  beyond-first-chunk prefill, other lengths/distributions, mixed
  batches, the boundary's can't-see list).
- The unfused path runs only at exactly rows == 1; rows 2-7 run the
  fused path in real serving (graded `faster` only at 8/32 by proxy).
  The empty-launch cost of K0/K1/K3 at rows 2-59 is inside the 8/32
  measurements only.
- 1-request whole-request noise in G3 (the boundary's list).

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** — no first difference; the unfused regrouping is
bit-exact. Chained timing us/layer (`grade.json`; candidate vs
reference vs compiled): **1 row 234.3 / 154.3 / 118.4 (`slower`, 0
wins)**; 8 rows 252.7 / 248.4 / 291.7 (`faster`, 25 wins); 32 rows
560.4 / 815.4 / 802.4 (`faster`, 30 wins); prefill 127: 967.2 /
971.8 / 968.0 (`same`, 16 wins); prefill 1,016: 3952.1 / 3957.9 /
3950.5 (`same`, 14 wins); prefill 2,048: 7362.5 / 7373.8 / 7358.5
(`same`, 14 wins). `phases_vs_compiled`: decode `slower`, prefill
`not slower`. **`beats_compiled` false.** Baseline `none`, so no
`beats_baseline`.

Kept: 8/32-row wins, prefill `same` x3, `max_rows: 59`, every
per-block accumulation order. Not fixed: the 1-row gap got worse,
not better (within-run 1.98x compiled vs attempt-1's 1.12x).

## Miss analysis (prediction 68 / 176 / 562 decode, = compiled prefill)

The 1-row prediction missed by +245% (68 -> 234.3); 8/32-row and
prefill matched in verdict (`faster`/`faster`/`same` x3), with 8-row
absolute 252.7 vs predicted 176. Part of every absolute is machine
state, not the change: this run's compiled bar moved 86.6 -> 118.4
at 1 row (1.37x) and 238.2 -> 291.7 at 8 rows (1.22x), and the
reference moved 82.2 -> 154.3 at 1 row (1.88x); at 32 rows the
machine agreed (802.4 vs 883.4) and the candidate matched
attempt-1's absolute (560.4 vs 554.7), bounding the 32-row empty
launches + K3 copy overhead near zero. The real signal is the
within-run ratio at 1 row: 1.98x compiled (0/30 wins) where the
fused path scores 1.12x — the unfused structure is genuinely ~1.8x
worse than fused against the same bar, far beyond noise.

Why, from `grade.json` structure plus static counts (no other GPU
evidence is allowed, so this is analysis, not measurement):

1. **Dispatch count explains ~1/6 of the +116 us gap.** Within this
   run, the uncompiled reference (~7 dispatches) costs 154.3 vs the
   compiled reference (1 dispatch) 118.4: ~36 us for ~6 extra
   dispatches, ~6 us/dispatch. This attempt runs 4 kernels + link =
   5 dispatches per layer vs fused 2, ~+18 us. The remaining ~98 us
   is K0/K1 work slower than the fused chain.
2. **Per-group setup x144 vs amortized x4 (prime suspect for the
   rest).** The fused kernel pays one tg_x load + barriers per pair
   amortized over the full 36-step chain. K0/K1 pay a tg load +
   barrier per block per phase: 48 x (tg_x load + barrier) for 24
   short steps, 96 x (tg_h load + barrier) for 12 — each group does
   little work (24x16 / 12x8 MACs per thread), so scheduling,
   pipeline fill and setup dominate; plus the h_small device
   round-trip and K3's copy.
3. **The reference's speed comes from one dispatch, not just its
   shape.** The compiled reference's 7 short kernels run as ONE
   mx.compile dispatch; regrouping the candidate into 4 separate
   metal_kernel launches pays per-dispatch and per-group setup that
   exceeds the chain-latency saving. Reference-shaped is not
   reference-priced outside mx.compile.

Lesson for `ineff-baseline-experts-2`: the 1-row gap now survives
both fused packings (round-2 plateau 92-110) and unfused regrouping
(+100 us dispatch/setup). Note appended to the ineff entry.

## Next change

Do not build on attempt-5. The evidence now bounds both sides: one
dispatch plateaus at 92-110 (six fused packings), several dispatches
pay ~+100 us in dispatch + per-group setup. No further kernel
structure is indicated on this line — bank attempt-1 for hyp-2 (its
claim is "faster decode, unchanged prefill" modulo the 1-row gap)
and let hyp-4 decide whether any micro-lever inside one dispatch
(vectorized tg_x loads, barrier placement) is worth an attempt; my
read is the gap is structural and should be accepted.
