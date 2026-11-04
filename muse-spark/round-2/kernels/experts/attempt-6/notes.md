# attempt-6 (round 2): hardening — interleaved down block pair (builder-1-f)

Hardening attempt for `hyp-1-experts-exact` (slot `experts`, baseline
`none`). Builds on the best attempt
`muse-spark/round-2/kernels/experts/attempt-5` (fused single kernel,
512 threads/pair, interleaved gate/up + parallel block stores, G2 pass
all six, 92.1 / 162.0 / 534.8 µs/layer chained vs compiled
75.9 / 234.6 / 798.2 — loses only at 1 row by 16.2 µs). This is the
one main change named in attempt-5's notes (*Next change*): make the
down product's two waves overlap like gate/up.

## The one main change

Interleave the down product like gate/up, decode branch only.
Everything else is copied from attempt-5, not redesigned
(`spec.json` byte-identical; `block-1.metal` helpers `qvm_block`,
`qvm_gate_up_pair`, `qmm_accum` bodies byte-identical by diff;
`block-0.metal` gate/up, silu and prefill branches byte-identical):

- **Before (attempt-5):** down ran 24 blocks of 32 outputs in 2
  sequential waves of 16 simdgroups (`b = v*16 + simd`, `b < 24`):
  wave 0 computed blocks 0–15, wave 1 blocks 16–23 (simdgroups
  8–15 idle), each block a separate `qvm_block` call — 12 strided
  K steps per lane (`k = lane, lane+32, …, lane+352` over `K=384`),
  then `simd_sum` and the parallel store
  `y[pair*768 + b*32 + lane] = (bfloat)p[lane]`. The second wave's
  first load waited for the first wave's last accumulate.
- **Now:** one simdgroup computes two down blocks in one merged K
  loop (`qvm_down_pair`, block-1). Simdgroup `s < 12` owns the
  consecutive pair `b0 = 2s`, `b1 = 2s+1` (12 pairs, one wave;
  simdgroups 12–15 idle, as in gate/up). For each strided `k` both
  blocks' loads are issued before either accumulates — block 0's
  scale/bias and 4 weight words, then block 1's scale/bias and 4
  weight words (plus the shared `tg_h[k]`, read once) — and only
  then do the two accumulation passes run, block 0's 32 terms then
  block 1's, into separate accumulators `p`/`q`. The 24 down blocks
  therefore run as 12 interleaved pairs instead of 2 sequential
  waves.

Everything else is kept identical, as the task requires:

- **Per-block bits unchanged.** In `qvm_down_pair` each block's adds
  run in exactly `qvm_block`'s order: the same strided `k` sequence,
  the same float32 dequant terms
  `x * (s0 * (by & 0x0f) + bias)` / `x * (s1 * (by & 0xf0) + bias)`
  with `s1 = scale/16`, in the same per-output sequence; `p` and
  `q` are independent accumulators, so interleaving cannot reorder
  either block's own adds. Each accumulator then gets the identical
  `#pragma clang loop unroll(full)` `simd_sum` tree as `qvm_block`.
  The `(bfloat)` rounding at the store is unchanged.
- **Parallel lane stores kept.** After the two `simd_sum`s every
  lane holds both blocks' full reduced 32-vectors, so lane `j`
  stores element `j` of each block — one store per lane per block:
  `y[pair*768 + b0*32 + lane] = (bfloat)p[lane]`,
  `y[pair*768 + b1*32 + lane] = (bfloat)q[lane]`.
- **Gate/up interleave, silu, prefill scalar branch, grids and
  rounding points identical.** Gate/up still calls
  `qvm_gate_up_pair` verbatim with attempt-5's parallel
  `tg_gate`/`tg_up` stores; the full-bfloat16 silu-mul is
  byte-identical; the `rows >= 60` prefill branch still uses the
  scalar `qmm_accum` (bfloat16-rounded dequant, contiguous `BK=32`
  order) for gate, up and down, with the same `x_shape[0] >= 60`
  dispatch threshold; `spec.json` (one kernel, `rows*2048` threads
  = `rows*4` groups of 512) is byte-identical. No kernel split, no
  new device-memory intermediate, same four bfloat16 rounding
  points and positional `y[r,j]`.

## Which inefficiency it removes, and how

- `ineff-baseline-experts-2` (chained 1-row latency): attempt-5's
  miss analysis left the still-sequential down chain as the largest
  named remainder at 1 row — 12 strided steps × 2 waves behind the
  silu barrier, plus the `tg_x` load, silu, barriers and chained
  fixed cost. Attempts 1–3 showed threadgroup count past 4 is not
  the lever and attempt-4/5 showed overlapping the per-block serial
  K chain is. The down product was the last product still running
  its blocks in fully sequential waves with only one block's loads
  in flight per simdgroup; merging two blocks per simdgroup puts
  both blocks' loads (8 weight words + 2 scales/biases per step)
  in flight in every step, so each block's load latency has the
  other block's loads/compute to hide behind, without touching
  any block's `simd_sum` order.
- `ineff-baseline-experts-1` (prefill G2 gate): already removed by
  attempt-5's branch; kept by reusing its scalar `qmm_accum`
  verbatim — the interleave does not touch prefill arithmetic.
- `ineff-baseline-experts-3` (per-pair re-reads): deliberately
  kept, as in attempts 1–5 — weight bytes per pair stay 552,960 B;
  the interleave changes load scheduling, not bytes.

## Limits (by hand; the grader does not check the 32 KB cap)

512 threads/group ≤ 1,024 (`spec.json` identical to attempt-5).
Threadgroup memory unchanged: `tg_x[768] + tg_gate[384] +
tg_up[384] + tg_h[384]` bfloat16 = **3,840 B** of 32,768 B,
rows-independent. The only new state is register-local: a second
32-float accumulator `q[32]` beside `p[32]` and a second 4-word
load buffer inside `qvm_down_pair` — the register cost is the risk
noted below. Grid scales as `rows*4` groups, so the kernel is
valid at every `rows ≥ 1`, on both sides of 60. Decode coverage:
12 simdgroups × 2 blocks × 32 lanes = 768 outputs per pair, each
written exactly once, as before.

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Attempt-5 (other run): 92.1 / 162.0 / 534.8 µs/layer vs compiled
75.9 / 234.6 / 798.2 and reference 95.7 / 232.4 / 802.3. Baseline
`none`, so no `wins_vs_baseline` / `beats_baseline`. Weight bytes
per layer (552,960 B/pair): 2.21 / 17.69 / 70.78 MB at 4 / 32 /
128 pairs; bandwidth floors at 200 GB/s: 11.1 / 88.5 / 353.9 µs.
Attempt-5 effective: 24.0 / 109.2 / 132.3 GB/s.

Down arithmetic: `K = 384` → 12 strided steps per lane per block.
Attempt-5 pays 2 waves × 12 steps sequentially after the silu
barrier; attempt-6 pays 12 merged steps. A merged step issues
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
- **8 rows: 156.** 32 groups per call partly hide one group's
  down tail behind other groups' K loops (attempt-5 already at
  109.2 GB/s), so only part of the 1-row saving survives:
  162.0 − 6 ≈ **156 µs**, keeping the win over compiled 234.6.
- **32 rows: 540.** Streaming-bound (attempt-5: 132.3 GB/s).
  Group parallelism already hides load latency, so the merged
  loop buys nothing; the risk is the second live 32-float
  accumulator taxing occupancy, as the gate/up interleave did in
  attempt-4 before attempt-5's store fix (+10.4%). With the
  parallel stores already in place that tax measured ≈ 0 for
  gate/up (534.8 ≈ attempt-1's 535.7), so the down interleave is
  predicted ≈ neutral: 534.8 + 5 ≈ **540 µs**, keeping the win
  over compiled 798.2. Register spill remains the failure mode
  to watch.

G2 prediction: pass at all six sizes with `differ == 0` — each
down block's dequant terms, add order and `simd_sum` are
attempt-5's by construction (separate accumulators), gate/up and
prefill are byte-identical to attempt-5's, and the parallel store
writes the same `(bfloat)p[j]` values, just one merged loop
earlier.

## Verdict

**G1 fail** — the grader run
(`.venv/bin/python bench/grader/check_candidate.py
muse-spark/round-2/kernels/experts/attempt-6 experts`) returned:

```json
{"slot": "experts", "g1": false, "g2": false, "beats_compiled": false,
 "error": "ValueError: Thread group size (512) is greater than  the maximum allowed threads per threadgroup (448)."}
```

No correctness or timing was reached: G2 was not evaluated at any
of the six sizes and there is no `timing_us_per_layer`, so there
are no `differ`/`max_ulp` figures and no candidate/reference/
compiled times from this run to report. Baseline `none` — no
`wins_vs_baseline`/`beats_baseline`. The prediction
(85 / 156 / 540 µs/layer) is therefore untested, not missed on
time: the kernel never launched.

## Failure analysis (why G1 failed)

Attempt-5 launches the identical launch spec — one kernel,
`rows*2048` threads = `rows*4` groups of 512, 3,840 B of
threadgroup memory — and passed G1/G2 in its run, so the 512-thread
launch itself is legal for attempt-5's compiled pipeline. The only
delta in this attempt is the down interleave, and threadgroup
memory is unchanged (still 3,840 B, verified from the source: no
new `threadgroup` arrays), so the drop of the pipeline's
`maxTotalThreadsPerThreadgroup` to 448 must come from per-thread
register/thread storage, which Metal folds into that limit. The
inferred cause (from the code, not a register dump — no second GPU
job was run to measure it): attempt-5 has two distinct 32-float
thread arrays (`p[32]` at kernel scope, `pu[32]` scoped to the
gate/up branch; down reuses `p` alone), while attempt-6 has three
(`p`, gate/up's `pu`, down's new `q[32]` scoped to the down
branch), plus `qvm_down_pair`'s second 4-word load buffer live
alongside the first. Even though `pu` and `q` are live in disjoint
phases, the compiled kernel's per-thread allocation evidently grew
enough that 512 threads no longer fit the threadgroup budget
(448 = 512 × 7/8). The interleave arithmetic itself was never
executed, so this attempt says nothing about whether the merged
down loop preserves bits or saves time — G2's by-construction
argument above still stands untested.

Lesson for `ineff-baseline-experts-2`: the 512-thread fused kernel
sits at the edge of the GPU's per-threadgroup register budget —
any new simultaneously-live accumulator must reuse an existing
accumulator's storage, not add a third distinct thread array, or
the launch itself becomes illegal. Peak live floats per phase did
not grow (gate/up already held 64), but distinct-array count did,
and that is what the compiler charged for.

## Next change

Keep the identical merged down loop, accumulators' arithmetic and
parallel stores, but reuse storage so the kernel again has only
two distinct 32-float thread arrays: hoist the second accumulator
to kernel scope (`float p[32], q[32]` beside the existing buffers)
and pass `q` as gate/up's second accumulator (`pu`) and as down's
second accumulator, instead of declaring a fresh `pu` in the
gate/up branch and a fresh `q` in the down branch. That is a pure
storage rename — no add order, load order, grid, silu, prefill or
rounding change — and should restore attempt-5's 512-thread
launch headroom while finally testing the down interleave. If the
pipeline limit still reports < 512, the fallback is to free the
first down accumulator earlier by storing block `b0` through
threadgroup memory before the second `simd_sum`, at the cost of a
barrier the current design avoids.
