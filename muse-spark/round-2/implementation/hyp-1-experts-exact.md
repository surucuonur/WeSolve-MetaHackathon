# Hypothesis 1: experts kernel exact at every size, faster chained at 1 row

- **Owner:** lead
- **Status:** building
- **Version:** 3
- **Updated:** 2026-11-04T03:43:13Z
- **Next:** boundary checks for attempt-1 and attempt-2; hardening from attempt-1

## Inefficiency

Slot `experts`. Baseline: `none` (no earlier attempt passes the current G2). Starting point: `muse-spark/kernels/experts/attempt-4`.

One combined fix for three confirmed entries (savings do not add; the gate must be removed before any timing counts):

- `ineff-baseline-experts-1` (gate, ranked 8): starting point matches `gather_qvm` but prefill runs `gather_qmm_rhs` — prefill differ 62.4–62.6%, up to 110.5 ULP; 0 µs saving, blocks all timing.
- `ineff-baseline-experts-2` (ranked 2, mean 7.4%): chained 1-row latency 132.9 vs 80.3 µs/layer (+52.6, 1.05 ms/step); 4 threadgroups with wave/barrier serialisation.
- `ineff-baseline-experts-3` (ranked 3, mean 3.7% floors): per-pair re-reads persist — 908 vs 443 MB at 32 rows, 2.33 ms floor, 0 banked on round-1 grouped evidence.

Design: a kernel (or size-branched kernels) that is bit-exact at decode sizes (matching `gather_qvm`: float32 dequant `x·(scale·q+bias)`, strided-lane accumulation + `simd_sum`, full-bfloat16 silu) and at prefill sizes (matching `gather_qmm_rhs`: bfloat16-rounded dequantized tiles, `BK=32` MMA-order accumulation), while exposing more concurrent threadgroups per call at 1 row so the chained latency falls to or below the reference's, without spending the 8/32-row chained wins (63.7 / 210.0 µs/layer ahead). Must keep the G2 boundary (bit-exact at all six sizes, fp32 accumulation, bfloat16 rounding points).

## Boundary

(written by boundary-1; delta only — every line carries its source)

**Cited boundary, checked line by line.** Round 1's boundary for this slot is
[`muse-spark/implementation/hyp-1-fused-experts.md`](../../implementation/hyp-1-fused-experts.md)
*Boundary* (inputs/outputs table, neighbours A/B, identical-math, grader,
GPU limits). I checked each line against the starting point
`muse-spark/kernels/experts/attempt-4` (`block-0.metal`, `block-1.metal`,
`spec.json`) and against the current G2
(`claude/muse-agent-brief.md:179,182-188`; `claude/adr/0010-exact-at-every-size.md:22-29`;
`bench/grader/check_candidate.py:90-99,153-169,172-175,272-308`).
Still holding, not rewritten below: input/output names, dtypes, layouts and
weight shapes/bytes; `y[r,j]` positional in `inds` order; neighbour
contracts; the four bfloat16 rounding points in kind; launch-spec surface;
GPU limits. Baseline is `none` — no earlier attempt passes the current G2
(this entry, *Inefficiency*; `claude/adr/0010-exact-at-every-size.md:38`).
Only what changed follows.

### Inputs and outputs — changed: prefill row counts added

Unchanged shapes now also run at prefill sizes. `rows` is still read from
the first slot input's `shape[0]` (`bench/grader/slots.py:323`); the grader
adds prefill `127 / 1,016 / 2,048` rows (1/8/32 requests × 128-token
prompts, first chunk; `bench/grader/slots.py:211,214-220,240-254`;
`bench/grader/check_candidate.py:272-276`):

| Name | Shape at 127 / 1,016 / 2,048 rows | Dtype |
|---|---|---|
| `x` | [127,768] / [1016,768] / [2048,768] | bfloat16 (`claude/muse-agent-brief.md:74`) |
| `inds` | [127,4] / [1016,4] / [2048,4] | uint32 (`claude/muse-agent-brief.md:75`) |
| `gate_w`, `up_w`, `gate_scales`, `gate_biases`, `up_scales`, `up_biases`, `down_w`, `down_scales`, `down_biases` | rows-independent, as round 1 | as round 1 (`claude/muse-agent-brief.md:76-79`; `bench/grader/slots.py:33-36`) |
| `y` (output) | [127,4,768] / [1016,4,768] / [2048,4,768] | bfloat16 (`claude/muse-agent-brief.md:80`) |

Pairs `rows*4` are therefore 508 / 4,064 / 8,192 at prefill (arithmetic
from `rows` above), against 4 / 32 / 128 at decode 1/8/32 rows (round-1
boundary). A decode step can still have any 1–32 rows and a prefill chunk
other sizes again (`claude/muse-agent-brief.md:24`); the kernel must be
valid at every `rows`, not only the six tested ones (see *can't see*).

### Before and after — changed: same neighbours, now also in prefill

Neighbour code is unchanged: post-attention layernorm → `x`; router →
`inds`, `weights`; `experts(x, inds)` → `y`; `shared_expert(x)` independent;
combine pairs `y[r,j]` with `weights[r,j]` positionally → residual
(`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`;
`bench/grader/slots.py:97-119,105-118`). What changed is the invocation
domain: the same `MoE.__call__` runs the prefill first chunk, so the slot
is now called at the prefill row counts above with the same expectations —
`inds` any order, values in [0,60); `y` exact shape/dtype, positional
(`bench/grader/slots.py:217-218,231-254`;
`claude/adr/0010-exact-at-every-size.md:16`).

### What must stay identical — changed: bit-exact, with two arithmetics

- **Tolerance removed.** Round 1 allowed every element within 2 bfloat16
  ULP; now every element must equal the reference bit for bit
  (`differ == 0` in every layer at all six sizes;
  `claude/muse-agent-brief.md:186-188`;
  `bench/grader/check_candidate.py:90-99,153-169,172-175,308`).
  `max_ulp` is diagnostic only (`bench/grader/check_candidate.py:84-99`).
- **Summation order and dequantization are no longer free.** Round 1
  allowed any fp32 summation/dequantization order; that line is superseded.
  The reference itself changes kernel at `rows >= 60` — dispatch
  `M==1 && B>=16 && right_sorted_ && B/E>=4` with `B = rows*4`, `E = 60`,
  i.e. `B >= 240` (`muse-spark/round-2/exploration/territory-quantized-kernels.md`
  §A; `muse-spark/round-2/exploration/ineff-baseline-experts-1.md:22-32`).
  Sorting from `inds.size >= 64` ⇔ rows ≥ 16 is a distinct, earlier
  threshold and does not switch the kernel: decode 32 rows is sorted and
  still `gather_qvm` (`bench/grader/slots.py:62`;
  `third_party/mlx-lm/mlx_lm/models/switch_layers.py:181`).

  | Rows | Reference kernel | Weight value | K accumulation |
  |---|---|---|---|
  | < 60 (all decode) | `gather_qvm` | `float(scale)·q + float(bias)` kept in float32 through the product (`qouter`) | strided lanes `k≡lane (mod 32)` serial, then `simd_sum` tree |
  | ≥ 60 (all tested prefill) | `gather_qmm_rhs` | same value rounded to bfloat16 in threadgroup `Ws` before the multiply (`dequantize`: `static_cast<bfloat16>(s·q+b)`) | contiguous 8-wide frag dots, sequential over `BK=32` tiles (`BlockMMA`, `AccumType=float`) |

  Sources: `muse-spark/round-2/exploration/territory-quantized-kernels.md`
  §D (dequant rounding), §E (MMA order), §F (side-by-side table);
  `muse-spark/round-2/exploration/territory-baseline-experts.md` §F.
  Either difference alone breaks bit-equality (down outputs near
  cancellation amplify a 1-ULP hidden flip, round-1 boundary builders'
  notes; measured 62.4–62.6% differ, max 43/65/110.5 ULP at prefill,
  `muse-spark/round-2/exploration/ineff-baseline-experts-1.md:62-71`).
  A size branch is therefore required — `qvm` arithmetic below 60 rows,
  `qmm_rhs` arithmetic at/above — branched on `rows >= 60` from the
  dispatch source, not on the grader's bracketing sizes 32/127; rows
  33–59 in the real model still run `qvm`
  (`territory-quantized-kernels.md` §F conclusion). Row *grouping* inside
  the prefill branch is free (padded rows occupy separate accumulators
  and are discarded); per-row K order is not
  (`territory-quantized-kernels.md` §C invariant, §E invariant).
- **Unchanged in kind, now exact:** each gate/up/down product still
  accumulates in float32 and rounds once to bfloat16; `silu(gate)*up` is
  still computed on those bfloat16 values and rounded to bfloat16 before
  down (`claude/muse-agent-brief.md:90`; `bench/grader/slots.py:67-79`).
  The compiled `swiglu` (full-bfloat16 per-op silu) is shared by both
  reference paths, so it is not the divergence
  (`ineff-baseline-experts-1.md:56-58`). At prefill the gate/up values
  arriving at `swiglu` already differ by the two rows above.
- **Tie rule and pair order — unchanged, restated because G2 is exact:**
  this slot makes no 4th/5th-expert tie choice; it inherits routing's
  output as given and must accept any 4 valid ids per row in any order,
  including either side of a near-tie. The order of the 4 pairs is not
  free: `y[r,j]` must stay the output of expert `inds[r,j]`, because
  combine sums `y[...,j,:] * weights[...,j]` positionally
  (`bench/grader/slots.py:51-52,116`;
  `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:230-232`). Routing's own
  ordered tie-exactness is the `routing` slot's contract
  (`claude/muse-agent-brief.md:187`), not this slot's.
- **Starting point against this boundary:** attempt-4 replicates only
  `gather_qvm` — strided-lane `qvm_block` + `simd_sum`, float32 dequant
  `x·(s·q+bias)` (`muse-spark/kernels/experts/attempt-4/block-1.metal:20-40`),
  full-bfloat16 silu (`block-0.metal:53-64`), one kernel with grid
  `rows*4` groups at every row count (`spec.json`; `block-0.metal:6`).
  Decode 1/8/32 therefore passes bit-exact and every prefill size fails
  (`ineff-baseline-experts-1.md:62-71`); closing that gap without losing
  the decode bits is this hypothesis's correctness work.

### What the grader checks, and what it can't see — changed

Checks (`bench/grader/check_candidate.py`): G1 is true only after the
whole six-size correctness loop completes without exception (`:306`);
per-layer shape/dtype faults fail G2 with G1 true (`:158-159,288-304`).
G2 for `experts` requires exact shape/dtype and `differ == 0` in all 20
layers at decode 1/8/32 and prefill 127/1,016/2,048 (`:153-169,272-308`).
Piece timing runs only if G2 passes, only at decode 1/8/32, now chained
through the `link` kernel so each layer waits for the previous one
(`:178-213,309-337`); `beats_compiled` needs chained candidate <
compiled at all three decode sizes (`:334-335`). Prefill speed is not
timed — a slow-but-exact prefill branch is free apart from time to first
token (`claude/adr/0010-exact-at-every-size.md:42`). No `--baseline` bar
exists this round (`muse-spark/round-2/exploration/territory-grader.md` §5).
What it still can't see (checked by hand later — *check an attempt*):
row counts between the six tested ones (2–7, 9–31, 33–126, 128–1,015,
1,017–2,047, >2,048), with both thresholds (sort 16, kernel switch 60)
only bracketed by tested sizes; prefill beyond the first chunk of seeded
128-token random prompts (later chunks, other lengths/distributions);
reads of unwritten threadgroup memory, thread-timing dependence, and
the 32 KB limit below; NaN-vs-NaN counts as equal (`:93`)
(`territory-grader.md`, *What the grader still cannot see* 1–6).

### The GPU limits — unchanged

At most 1,024 threads per threadgroup (count enforced,
`bench/grader/slots.py:327-328`; `claude/muse-agent-brief.md:98`) and
32 KB of threadgroup memory (not enforced by `build` — verify by hand
from the Metal source; `territory-grader.md` §3). Starting point for
scale: 256 threads/group and 3,840 B threadgroup memory
(`muse-spark/kernels/experts/attempt-4/spec.json`;
`block-0.metal:13-16`), valid at every row count including 8,192
threadgroups at 2,048 rows
(`muse-spark/round-2/exploration/territory-baseline-experts.md` §C).
Any prefill branch must stay inside both limits at all six sizes.

## Builders

### builder-1-b — direction (b): branched single kernel, 512 threads/pair

- Attempt: `muse-spark/round-2/kernels/experts/attempt-1` (starting
  point `muse-spark/kernels/experts/attempt-4`, copied qvm path).
  Single fused kernel branched in-Metal on `rows >= 60`: below, the
  starting point's qvm arithmetic unchanged per block, widened to 512
  threads/pair (gate/up 1 wave, down 2 waves); at/above, per-pair
  bfloat16-dequant + contiguous BK=32 accumulation (`qmm_accum`), same
  silu/rounding points.
- Predicted (chained µs/layer at 1/8/32): 85 / 165 / 565. Measured
  (grader, this run): **104.3 / 193.3 / 535.7**, vs reference
  87.6 / 231.2 / 820.6 and compiled 77.6 / 235.7 / 791.6 in the same
  run. Starting point chained (other harness): 132.9 / 175.9 / 587.6.
- **G1 pass; G2 pass — differ 0, max_ulp 0.0 at all six sizes**
  (decode exact via unchanged qvm blocks; prefill exact via the
  scalar BK=32 mimic). `beats_compiled` false: wins 8/32 rows, loses
  1 row (104.3 vs compiled 77.6). Baseline `none` — no
  `wins_vs_baseline`/`beats_baseline`.
- Learned: threads-per-pair widening is nearly exhausted at 1 row
  (256→512 bought ~29 µs chained; prediction missed +23%). The binding
  limit is 4 threadgroups per call on 16 cores, which a fused per-pair
  kernel cannot raise; and territory §E's frag-tree warning did not
  bite — serial BK-order adds matched BlockMMA bitwise on all tested
  layers (see attempt notes).

### builder-1-a — direction (a): size-branched kernels, decode block-split

- Attempt: `muse-spark/round-2/kernels/experts/attempt-2` (starting
  point `muse-spark/kernels/experts/attempt-4`). Two kernels, each
  branched in-Metal on `rows >= 60`: gate/up kernel with one
  32-thread threadgroup per (pair, 32-output block) (48 groups at
  1 row) and down kernel likewise (96 groups), decode keeping
  attempt-4's `qvm_block` + full-bf16 silu per block; prefill using
  per-pair `qmm_block` (bfloat16-rounded dequantized weights,
  BK=32 sequential 8-wide `simdgroup_multiply_accumulate` steps).
- Predicted (chained µs/layer at 1/8/32): 40 / 165 / 610. Measured
  (grader, this run): **103.6 / 326.8 / 887.6**, vs reference
  76.1 / 244.6 / 797.1 and compiled 73.1 / 232.0 / 804.5 in the same
  run. Starting point chained (other harness): 132.9 / 175.9 / 587.6.
- **G1 pass; G2 pass — differ 0, max_ulp 0.0 at all six sizes**
  (per-pair MMA-order replication is bit-exact at prefill).
  `beats_compiled` false at every decode size. Baseline `none` —
  no `wins_vs_baseline`/`beats_baseline`.
- Learned: block-splitting to 1-simdgroup groups fixes only the
  1-row group starvation (132.9 → 103.6) and spends the 8/32-row
  wins: per-group memory parallelism collapses (one block's serial
  K chain per group, vs 8 concurrent in attempt-4) and the prefill
  tiles' 6 KB of threadgroup memory is paid in the decode branch
  too, capping resident groups. Next: 4-simdgroup groups (4 blocks
  each) with tile memory overlaid on the decode buffers.

## Boundary checks

*(none yet)*

## Decision

*(not yet)*
