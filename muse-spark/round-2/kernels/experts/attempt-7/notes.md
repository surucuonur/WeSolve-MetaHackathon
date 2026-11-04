# attempt-7 (round 2): hardening — interleaved down, shared second accumulator (builder-1-g)

Hardening attempt for `hyp-1-experts-exact` (slot `experts`, baseline
`none`). Builds on the best attempt
`muse-spark/round-2/kernels/experts/attempt-5` (fused single kernel,
512 threads/pair, interleaved gate/up + parallel block stores, G2 pass
all six, 92.1 / 162.0 / 534.8 µs/layer chained vs compiled
75.9 / 234.6 / 798.2 — loses only at 1 row by 16.2 µs) with the
down-interleave `muse-spark/round-2/kernels/experts/attempt-6` tried
(one simdgroup, two down blocks, merged K loop), made to compile
within the pipeline threadgroup limit. Attempt-6 failed G1:
`Thread group size (512) is greater than the maximum allowed
threads per threadgroup (448)` — its compiled pipeline's
`maxTotalThreadsPerThreadgroup` fell to 448 for the identical
512-thread launch.

## The one main change

Attempt-6's merged down loop, with the second accumulator's storage
shared so the kernel compiles at 512 threads. Everything else is
copied from attempt-5/6, not redesigned (`spec.json` byte-identical
to attempt-5/6; `block-1.metal` helpers `qvm_block`,
`qvm_gate_up_pair`, `qvm_down_pair`, `qmm_accum` bodies byte-identical
to attempt-6 by diff; `block-0.metal` gate/up, silu and prefill
branches otherwise byte-identical):

- **Before (attempt-5):** down ran 24 blocks of 32 outputs in 2
  sequential waves of 16 simdgroups (`b = v*16 + simd`, `b < 24`):
  wave 0 computed blocks 0–15, wave 1 blocks 16–23 (simdgroups
  8–15 idle), each block a separate `qvm_block` call — 12 strided
  K steps per lane (`k = lane, lane+32, …, lane+352` over `K=384`),
  then `simd_sum` and the parallel store
  `y[pair*768 + b*32 + lane] = (bfloat)p[lane]`. The second wave's
  first load waited for the first wave's last accumulate.
- **Attempt-6's down change (kept):** one simdgroup computes two
  down blocks in one merged K loop (`qvm_down_pair`, block-1).
  Simdgroup `s < 12` owns the consecutive pair `b0 = 2s`,
  `b1 = 2s+1` (12 pairs, one wave; simdgroups 12–15 idle, as in
  gate/up). For each strided `k` both blocks' loads are issued
  before either accumulates — block 0's scale/bias and 4 weight
  words, then block 1's scale/bias and 4 weight words (plus the
  shared `tg_h[k]`, read once) — and only then do the two
  accumulation passes run, block 0's 32 terms then block 1's, into
  separate accumulators. The 24 down blocks therefore run as 12
  interleaved pairs instead of 2 sequential waves.
- **This attempt's fix vs attempt-6:** attempt-6 declared three
  distinct 32-float thread arrays — kernel-scope `p[32]`, a
  gate/up-scoped `pu[32]` and a down-scoped `q[32]` — and the
  compiler charged all three against the per-threadgroup budget
  (`maxTotalThreadsPerThreadgroup` 448 < 512) even though `pu` and
  `q` are live in disjoint phases. Attempt-7 hoists the second
  accumulator to kernel scope: `float p[32]; float q[32];` beside
  the existing buffers in `block-0`, and passes the same `q` as
  gate/up's second accumulator (`qvm_gate_up_pair(..., p, q)`) and
  as down's second accumulator (`qvm_down_pair(..., p, q)`),
  deleting both scoped declarations. The kernel again has two
  distinct 32-float thread arrays, exactly attempt-5's footprint,
  at unchanged 512 threads.

Everything else is kept identical, as the task requires:

- **Per-block bits unchanged.** In `qvm_down_pair` each block's adds
  run in exactly `qvm_block`'s order: the same strided `k` sequence,
  the same float32 dequant terms
  `x * (s0 * (by & 0x0f) + bias)` / `x * (s1 * (by & 0xf0) + bias)`
  with `s1 = scale/16`, in the same per-output sequence; the two
  accumulators are independent, so interleaving cannot reorder
  either block's own adds. Each accumulator then gets the identical
  `#pragma clang loop unroll(full)` `simd_sum` tree as `qvm_block`.
  The `(bfloat)` rounding at the store is unchanged.
- **Sharing is a pure storage rename.** Gate/up and down run in
  disjoint phases separated by two `threadgroup_barrier`s (after
  gate/up and after silu): gate/up's `q` values are stored to
  `tg_up` before the first barrier, silu consumes only
  `tg_gate`/`tg_up`/`tg_h`, and `qvm_down_pair` re-zeroes both
  `p[0..31]` and `q[0..31]` on entry (as `qvm_gate_up_pair` does),
  so no value can leak across the barrier through the shared array.
  Peak live floats per phase is unchanged (64 in gate/up, 64 in
  down); only the distinct-array count charged by the compiler
  drops from three to two.
- **Parallel lane stores kept.** After the two `simd_sum`s every
  lane holds both blocks' full reduced 32-vectors, so lane `j`
  stores element `j` of each block — one store per lane per block:
  `y[pair*768 + b0*32 + lane] = (bfloat)p[lane]`,
  `y[pair*768 + b1*32 + lane] = (bfloat)q[lane]`.
- **Gate/up interleave, silu, prefill scalar branch, grids and
  rounding points identical.** Gate/up still calls
  `qvm_gate_up_pair` with attempt-5's parallel `tg_gate`/`tg_up`
  stores; the full-bfloat16 silu-mul is byte-identical; the
  `rows >= 60` prefill branch still uses the scalar `qmm_accum`
  (bfloat16-rounded dequant, contiguous `BK=32` order) for gate,
  up and down, with the same `x_shape[0] >= 60` dispatch
  threshold; `spec.json` (one kernel, `rows*2048` threads =
  `rows*4` groups of 512) is byte-identical. No kernel split, no
  new device-memory intermediate, same four bfloat16 rounding
  points and positional `y[r,j]`.

## Which inefficiency it removes, and how

- `ineff-baseline-experts-2` (chained 1-row latency): attempt-5's
  miss analysis left the still-sequential down chain as the
  largest named remainder at 1 row — 12 strided steps × 2 waves
  behind the silu barrier, plus the `tg_x` load, silu, barriers
  and chained fixed cost. Attempts 1–3 showed threadgroup count
  past 4 is not the lever and attempt-4/5 showed overlapping the
  per-block serial K chain is. The down product was the last
  product still running its blocks in fully sequential waves with
  only one block's loads in flight per simdgroup; merging two
  blocks per simdgroup puts both blocks' loads (8 weight words +
  2 scales/biases per step) in flight in every step, so each
  block's load latency has the other block's loads/compute to
  hide behind, without touching any block's `simd_sum` order.
  The shared-`q` fix is what lets this merged loop launch at all:
  attempt-6 proved the three-array form exceeds the pipeline
  threadgroup budget at 512 threads.
- `ineff-baseline-experts-1` (prefill G2 gate): already removed by
  attempt-5's branch; kept by reusing its scalar `qmm_accum`
  verbatim — the interleave does not touch prefill arithmetic.
- `ineff-baseline-experts-3` (per-pair re-reads): deliberately
  kept, as in attempts 1–6 — weight bytes per pair stay 552,960 B;
  the interleave changes load scheduling, not bytes.

## Limits (by hand; the grader does not check the 32 KB cap)

512 threads/group ≤ 1,024 (`spec.json` identical to attempt-5/6).
Threadgroup memory unchanged: `tg_x[768] + tg_gate[384] +
tg_up[384] + tg_h[384]` bfloat16 = **3,840 B** of 32,768 B,
rows-independent. Distinct thread arrays are two 32-float
accumulators (`p`, shared `q`) — attempt-5's count, one fewer than
attempt-6's three. Grid scales as `rows*4` groups, so the kernel
is valid at every `rows ≥ 1`, on both sides of 60. Decode coverage:
12 simdgroups × 2 blocks × 32 lanes = 768 outputs per pair, each
written exactly once, as before.

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Attempt-5 (other run): 92.1 / 162.0 / 534.8 µs/layer vs compiled
75.9 / 234.6 / 798.2 and reference 95.7 / 232.4 / 802.3. Attempt-6
predicted 85 / 156 / 540 but never launched, so that prediction is
untested, not missed. Baseline `none`, so no
`wins_vs_baseline` / `beats_baseline`. Weight bytes per layer
(552,960 B/pair): 2.21 / 17.69 / 70.78 MB at 4 / 32 / 128 pairs;
bandwidth floors at 200 GB/s: 11.1 / 88.5 / 353.9 µs.
Attempt-5 effective: 24.0 / 109.2 / 132.3 GB/s.

Down arithmetic: `K = 384` → 12 strided steps per lane per block.
Attempt-5 pays 2 waves × 12 steps sequentially after the silu
barrier; this attempt pays 12 merged steps. A merged step issues
twice the loads (8 weight words, 2 scales) for twice the
accumulate work, so it costs more than a single-block step but
less than two — the gate/up precedent (attempt-4) hid ≈ 20% of
the merged chain's exposed latency at 1 row.

- **1 row: 85.** Latency-bound (floor 11.1 µs). The down phase
  sits alone behind the silu barrier; at attempt-5's 92.1 µs the
  product chains (gate/up merged 24 steps + down 2×12 steps)
  dominate, so the down waves are roughly a third of the layer
  (≈ 30 µs). Collapsing 24 sequential steps into 12 merged ones
  at ≈ 1.3× the per-step cost (8 loads share the `tg_h[k]` read
  and the loop control) cuts the down phase to ≈ 23 µs, saving
  ≈ 7 µs → 92.1 − 7 ≈ **85 µs**. Still above compiled 75.9 — the
  `tg_x` load, silu, barriers and chained launch/link cost
  remain untouched — but it should roughly halve the 16.2 µs gap.
  The shared `q` itself is arithmetic-neutral; its only timing
  effect is restoring a legal 512-thread pipeline.
- **8 rows: 156.** 32 groups per call partly hide one group's
  down tail behind other groups' K loops (attempt-5 already at
  109.2 GB/s), so only part of the 1-row saving survives:
  162.0 − 6 ≈ **156 µs**, keeping the win over compiled 234.6.
- **32 rows: 540.** Streaming-bound (attempt-5: 132.3 GB/s).
  Group parallelism already hides load latency, so the merged
  loop buys nothing; the peak live accumulator count is the same
  64 floats gate/up already held in attempt-5, so no new spill
  tax is expected beyond attempt-6's identical merged step:
  534.8 + 5 ≈ **540 µs**, keeping the win over compiled 798.2.

G1 prediction: pass — the launch is attempt-5's (512 threads,
3,840 B threadgroup memory) and the only change from attempt-5's
register footprint is that down now also holds a second live
accumulator, which gate/up already held in the same kernel that
launched at 512; the distinct-array count is attempt-5's two,
not attempt-6's three.
G2 prediction: pass at all six sizes with `differ == 0` — each
down block's dequant terms, add order and `simd_sum` are
attempt-5's by construction (separate logical accumulators,
shared storage re-zeroed per call), gate/up and prefill are
byte-identical to attempt-5's, and the parallel store writes the
same `(bfloat)p[j]` values, just one merged loop earlier.

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** (decode 1/8/32 and prefill 127/1,016/2,048 — no first
difference, `experts` reports the first differing element only on
failure and there is none) — the shared-storage rename preserves
attempt-5/6 bits by construction and the grader confirms it: the
merged down loop's per-block order and the parallel stores are
bit-exact, and the `rows >= 60` scalar `qmm_accum` branch is
untouched. The 512-thread launch now compiles: sharing one
kernel-scope `q[32]` between gate/up's second accumulator and
down's second block restores the pipeline's threadgroup budget
that attempt-6's three distinct arrays exceeded (512 > 448).
Chained timing µs/layer (this run; candidate vs reference vs
compiled): **1 row 93.8 / 78.9 / 74.8 (loses to compiled by
19.0, loses to reference by 14.9)**; 8 rows 160.1 / 235.8 /
236.5 (wins); 32 rows 510.1 / 797.0 / 792.7 (wins).
`beats_compiled` false — only the 1-row size misses, as in
attempt-5. Baseline `none` — no `wins_vs_baseline` /
`beats_baseline`.

Against attempt-5 (different run; compiled shifted 75.9 → 74.8
at 1 row and 798.2 → 792.7 at 32 rows between runs, so treat
cross-run deltas as indicative): 93.8 vs 92.1 at 1 row
(**+1.8%**), 160.1 vs 162.0 at 8 rows (**−1.2%**), 510.1 vs
534.8 at 32 rows (**−4.6%**). Effective bandwidth on pair
bytes: 23.6 / 110.5 / 138.8 GB/s (attempt-5: 24.0 / 109.2 /
132.3). The down interleave therefore helps where group
parallelism already hides load latency (32 rows, +4.9% GB/s,
new best 32-row time in this hypothesis) and is neutral at
8 rows, but it does not move the latency-bound 1-row size.

## Miss analysis (prediction 85 / 156 / 540, measured 93.8 / 160.1 / 510.1)

Misses are +10.4% / +2.6% / −5.5% — all inside the 20% bar, so
no formal miss, but the size pattern was inverted versus the
prediction (saving predicted largest at 1 row, actually largest
at 32 rows) and that needs explaining:

1. **The compilation fix worked exactly as predicted, the
   latency saving did not.** G1 passes at 512 threads with the
   same 3,840 B of threadgroup memory, confirming attempt-6's
   failure was the third distinct accumulator array, not
   threadgroup memory or the merged loop itself: two distinct
   arrays (`p`, shared `q`) fit, three do not. That part of the
   prediction (pure rename, identical bits) is confirmed by
   G2's differ 0 at all six sizes.
2. **At 1 row the down waves were not a third of the layer's
   latency in the way the arithmetic assumed.** The prediction
   priced the down phase at ≈ 30 µs and the merged step at
   1.3× a single-block step. Measured 93.8 ≈ attempt-5's 92.1
   says the merged step costs closer to 2× at this size: with
   only 4 threadgroups (one per pair) on 16 cores, the down
   phase's latency was already hidden *across* simdgroups —
   attempt-5's wave 0 runs 16 blocks concurrently per group,
   attempt-7 runs 12 simdgroups each serially merging two
   blocks, so concurrent in-flight loads per group fall
   (48 vs 64 simdgroup-block chains per call at 1 row) while
   each chain's step doubles its loads. The gate/up precedent
   hid ≈ 20% because gate and up *share* the `tg_x[k]` read and
   both blocks' loads were previously fully sequential phases;
   down's two blocks share only `tg_h[k]` and already overlapped
   across the 16-simdgroup wave. The remaining ≈ 19 µs gap to
   compiled (93.8 vs 74.8) is therefore still the `tg_x` load,
   silu, the two barriers and the chained launch/link cost —
   none of which this change touches.
3. **At 32 rows the merged loop pays for a different reason.**
   One wave instead of two removes a wave-quantization tail
   (wave 1 ran 8 of 16 simdgroups) on every pair and halves the
   down loop's barrier-free step count, which at 128 groups
   shows up as throughput: 138.8 vs 132.3 GB/s. The predicted
   +5 µs register tax did not appear (measured −24.7 µs vs
   attempt-5 cross-run) — consistent with the shared `q`: peak
   live accumulators never exceed gate/up's 64 floats, so there
   is no new spill, and claim 2's cross-simdgroup overlap loss
   is hidden by group parallelism at this size.

Lesson for `ineff-baseline-experts-2`: a merged pair helps only
where the two blocks were previously *phases* (gate then up,
fully sequential, shared activation read); where the blocks
already ran as concurrent simdgroups in one wave (down's 24
blocks), merging trades cross-simdgroup parallelism for
in-step parallelism and is latency-neutral at 4 groups — the
1-row remainder is fixed cost and the serial gate/up chain,
not the down wave count. Register headroom at 512 threads is
exactly two 32-float accumulators; a third distinct array,
however short-lived, fails G1.

## Next change

Leave the down interleave in (it is the best 32-row time and
G2-exact) and attack the actual 1-row remainder named above:
the `tg_x` load + silu + barrier chain and the chained fixed
cost. Concretely, overlap the down product's first K steps with
the silu tail (down blocks whose `tg_h` slice is already
written) or narrow the silu barrier to a simdgroup-local one —
but any such change must keep the full-bfloat16 silu bits and
every accumulation order, and must not add a third accumulator
array (this attempt's G1 lesson).
