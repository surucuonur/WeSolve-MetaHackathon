# Hypothesis 2: experts prefill at compiled speed

- **Owner:** lead
- **Status:** building
- **Version:** 2
- **Updated:** 2026-11-04T17:55:21Z
- **Next:** builder-2-a (max_rows hand-off), builder-2-b (vectorized per-pair scalar) and builder-2-c (grouped/tiled replica) grade their attempts

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

(none started)

## Boundary checks

(none yet)

## Decision

(none yet)

## Failure ledger

(none yet)

## Unverified

(at least the boundary's *can't see* list, once written)

## Checks

(none yet)
