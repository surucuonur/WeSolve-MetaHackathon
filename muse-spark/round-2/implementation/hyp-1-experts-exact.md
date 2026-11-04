# Hypothesis 1: experts kernel exact at every size, faster chained at 1 row

- **Owner:** lead
- **Status:** hardening
- **Version:** 5
- **Updated:** 2026-11-04T03:49:13Z
- **Next:** hardening from attempt-1: block-parallel decode, 4 blocks per group

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

### attempt-2 (`muse-spark/round-2/kernels/experts/attempt-2`) — boundary-1 · 2026-11-04T03:45:19Z — **pass**

Checked against the Boundary above for what G2 cannot see. No defect found.

- **Row counts on each side of 60.** The branch is on the dispatch threshold itself, not the tested sizes: both kernels read `rows` from `inds_shape[0]` and take `qvm_block` below 60 and `qmm_block` at/above (`block-0.metal:11,26,49`; `block-1.metal:9,21,34`). This matches `B = rows*4 >= 240 ⇔ B/E >= 4` at exactly 60 rows; 33–59 therefore stay on `gather_qvm` and 60–126 on `gather_qmm_rhs`, as the Boundary requires. The sort threshold (rows ≥ 16) is correctly ignored — the kernel is positional in `inds` order (`block-0.metal:12-15`; `block-1.metal:10-11`), so reference-internal sorting cannot affect its bits at 16–59 rows either. Nothing in the kernel depends on the row count beyond this branch and the linear grids (`rows*1536` / `rows*3072` threads = `rows*48` / `rows*96` groups of 32, `spec.json:1`): pair/block decomposition (`tg/12`, `tg%12`; `tg/24`, `tg%24`) covers every `rows >= 1`, including untested 2–7, 9–31, 61–126, 128–1,015 and >2,048, and prefill chunks other than the first. Per-pair independence (no offsets, no cross-pair grouping) means chunk position and prompt length cannot change a row's arithmetic.
- **Unwritten threadgroup reads.** None. Decode: `tg_x[768]` is fully written (`i = lane; i < 768; i += 32`) and barriered before `qvm_block` reads it, and `tg_gate`/`tg_up[32]` are fully written by lane 0 and barriered before the silu read (`block-0.metal:19-21,27-48`); `tg_h[384]` likewise (`block-1.metal:16,22-25`). Prefill: each `BK=32` tile fully writes `Xs[256]` (row 0 = the pair's row, rows 1–7 zero — discarded padding in separate accumulators, as the Boundary allows) and `Ws[1024]` before the barrier, the MMA loads stay inside those extents, and `tmp[256]` is fully written by the four `simdgroup_store`s and barriered before `o32[j] = tmp[j]` reads row 0 (`block-2.metal:79-116`). The second `qmm_block` call re-writes `Xs`/`Ws` tile-by-tile behind the same barriers, so no stale or unwritten value is read. Device-side, `hidden` is written at every `(pair, block*32+lane)` (`block-0.metal:47-48,56-57`) before the down kernel reads it, and `y` at every output element in both branches (`block-1.metal:29-32,39`), so the uninitialised intermediate is never read.
- **Threadgroup memory vs the 32 KB limit.** Not close. Gate/up: `768*2 + 32*2 + 32*2 + 256*4 + 1024*4 + 256*4 = 7,808 B` (23.8% of 32,768 B; declarations at `block-0.metal:19-24`). Down: `384*2 + 256*4 + 1024*4 + 256*4 = 6,912 B` (21.1%; `block-1.metal:16-19`). The allocations are rows-independent, so this holds at every row count. (The prefill tiles are allocated in the decode branch too — a residency cost the builder's notes already record, not a limit risk: even so the decode group carries 7,808 B against a 32 KB cap.) Threads per threadgroup are 32 ≤ 1,024 (`spec.json:1`).
- **Thread-timing dependence.** None found. All threadgroup sharing is ordered by `threadgroup_barrier` between write and read (`block-0.metal:30,46`; `block-1.metal:25`; `block-2.metal:98,108,113`); the two reused tile buffers are only overwritten after those barriers. Each threadgroup writes a disjoint slice of `hidden`/`y` (unique `pair`/`blk`/`lane`), `atomic_outputs` is false and no atomics are used (`spec.json:1`), and the only reductions are the fixed `simd_sum` tree (`block-2.metal:41-44`) and fixed-order sequential `BK=32` fragment accumulation (`block-2.metal:79-107`) — neither order can vary with scheduling. The gate/up → down dependency is kernel-launch order in the spec list plus the `hidden` data dependency, not timing.
- **Captured-input assumptions.** None beyond the slot contract. `qvm_block` is body-identical to the starting point's proven helper (checked against `muse-spark/kernels/experts/attempt-4/block-1.metal`), keeping float32 dequant and the full-bfloat16 silu (`block-2.metal:9-45,122-130`); `qmm_block` rounds dequantized weights to bfloat16 before the multiply (`block-2.metal:94-96`). Hard-coded constants are the model's fixed slot dimensions (768/384 widths, 12/24 output blocks, per-expert strides `36864`/`9216`, bits=4 nibble dequant), not properties of the captured arrays. The kernel assumes only what the Boundary guarantees — `inds` values in [0,60), flat row-major `[rows,4]` order, contiguous weights — and makes no assumption about expert order, duplicates, popularity, `x` distribution, or prompt content: every pair is computed independently from its own `inds[pair]` (`block-0.metal:15-17`), so skewed or repeated experts and later prefill chunks take the same path as the captured ones.

### attempt-1 (`muse-spark/round-2/kernels/experts/attempt-1`) — boundary-1 · 2026-11-04T03:46:11Z — **pass**

Checked `block-0.metal` (B0), `block-1.metal` (B1) and `spec.json` against the Boundary above for what G2 cannot see. No defect found.

- **Row counts on each side of 60.** The branch is on the dispatch threshold itself, not the tested bracket: `prefill = x_shape[0] >= 60` (B0:19) is exactly `B = rows*4 >= 240 ⇔ B/E >= 4` (`E = 60`), so 33–59 rows stay on the `gather_qvm` arithmetic and 60–126 take the `gather_qmm_rhs` arithmetic, as the Boundary requires; at 59 rows `B/E = 3.93` and at 60 it is `4.0`. The sort threshold (rows ≥ 16) is correctly ignored — the kernel is positional, `pair = threadgroup_position_in_grid.x`, `r = pair >> 2`, `e = inds[pair]` (B0:13,17–18), so reference-internal sorting cannot change its bits at 16–59 rows. Nothing else depends on the row count: the grid is `rows*2048` threads = `rows*4` groups of 512 (`spec.json:1`), one group per (row, slot) pair, with per-pair indices in range for any `rows >= 1` (`inds[pair]` of `rows*4`, `x[r*768+i]` with `i < 768`, `y[pair*768 + b*32 + j/lane]` with the offset `< 768`; B0:26,61,67,93,104) and index maxima inside the per-expert extents (gate/up and down alike: `W` word `36863 < 36864`, scale/bias `9215 < 9216`; B1:33–37,76–78, with `eW = e*36864`, `eS = e*9216` at B0:31–32). This covers untested 2–7, 9–31, 61–126, 128–1,015, 1,017–2,047 and >2,048 rows, and prefill chunks other than the first: each pair is computed independently (no offsets, no cross-pair grouping, no chunk state), and the per-output K order in both branches is rows-independent, so chunk position and prompt length cannot change a row's arithmetic. A short later chunk of <60 rows correctly falls back to the `qvm` branch, matching the reference's per-call dispatch.
- **Unwritten threadgroup reads.** None. `tg_x[768]` is written exactly once by `for (i = tid; i < 768; i += 512)` (B0:26–28) and barriered (B0:29) before any `qvm_block`/`qmm_accum` read of it (B1:28,75–79). `tg_gate`/`tg_up[384]` are fully covered before the barrier at B0:70 and the silu reads at B0:75,81: in decode, simdgroups 0–11 each have lane 0 store their block's 32 outputs (B0:37–44,47–54); in prefill, each lane stores one output, `b*32 + lane` (B0:58–67) — 12 blocks × 32 = 384 either way. `tg_h[384]` is written exactly once by tids 0–383 in the silu loop (B0:74–81) and barriered (B0:83) before the down reads (B0:90,103). Idle simdgroups (12–15 in gate/up; 8–15 in down wave 2, B0:88–96,101–105) write nothing, and their stale or uninitialised `p[32]` is never stored — the store sits inside the same `simd < 12` / `b < 24` guard as the compute; `qvm_block` itself zero-initialises `p` before accumulating (B1:24–26).
- **Threadgroup memory vs the 32 KB limit.** Not close. `tg_x[768] + tg_gate[384] + tg_up[384] + tg_h[384]` = 1,920 bfloat16 = 3,840 B (B0:21–24), 11.7% of the 32,768 B limit, rows-independent so it holds at every row count. Threads per threadgroup are 512 ≤ 1,024 (`spec.json:1`).
- **Thread-timing dependence.** None found. All cross-thread sharing is ordered by `threadgroup_barrier` between write and read (B0:29,70,83), and all three barriers sit outside any lane- or simd-divergent control flow — `prefill` (B0:19) is uniform across the group, so every thread reaches each barrier. Each threadgroup writes a disjoint `y` slice (unique `pair`; B0:93,104), each `tg_gate`/`tg_up`/`tg_h` element is written exactly once (previous bullet), `atomic_outputs` is false and no atomics are used (`spec.json:1`). The only reductions are the fixed `simd_sum` tree (B1:44–47), entered only under simd-uniform guards (`simd < 12` at B0:37,47,58,64; `b = v*16 + simd < 24` at B0:88,101), so a simdgroup always reduces wholly or skips wholly, and the fixed-order serial `BK=32` accumulation in `qmm_accum` (B1:73–80) — neither order can vary with scheduling.
- **Captured-input assumptions.** None beyond the slot contract. The decode helper `qvm_block` is body-identical to the starting point's proven helper (verified against `muse-spark/kernels/experts/attempt-4/block-1.metal`), keeping float32 dequant `x·(scale·q + bias)` and `simd_sum` order (B1:12–48), and the silu is the starting point's full-bfloat16 lowering, shared by both branches (B0:74–81). In the prefill branch every weight is rounded to bfloat16 before the multiply, `(bfloat)(scale·q + bias)` (B1:78–79); its one written equivalence — MLX's odd-nibble form `(s/16)·(byte & 0xf0) + b` — is exact, not sampled: scaling by a power of two is exact and `s·q` is exact in float32 for 4-bit `q` (B1:56–58). Hard-coded constants are the model's fixed slot dimensions (768/384 widths, 12/24 output blocks, `G = 12/24`, `Ww = 48/96`, per-expert strides 36864/9216), not properties of the captured arrays. The kernel assumes only what the Boundary guarantees — `inds` values in [0,60), flat row-major `[rows,768]` `x` and `[rows,4]` `inds`, contiguous weights — and no property of the captured distributions: `e = inds[pair]` is used directly per pair (B0:18), so any expert order, duplicates, skewed popularity (including an expert absent or chosen by every row), near-tie ids on either side, and later prefill chunks all take the same path; there are no value-based early exits. Residual, not a defect: `qmm_accum` reproduces `BlockMMA`'s 8-wide fragment dots with a single thread's serial adds in `BK=32` order (B1:73–80) rather than by construction with `simdgroup_multiply_accumulate` as attempt-2 does, and territory-quantized-kernels §E notes no source-level proof that the hardware order coincides. Its equality therefore rests on the G2 evidence (differ 0 at 127/1,016/2,048 rows over all 20 layers) plus the same full-bfloat16 silu residual already recorded for the starting point in round 1; since the per-output order is independent of row count and expert distribution, other row counts add no risk to it, but an unseen activation value cannot be ruled out from source alone.

## Decision

*(not yet)*
