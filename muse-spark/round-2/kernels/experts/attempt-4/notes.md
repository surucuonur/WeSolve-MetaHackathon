# attempt-4 (round 2): hardening — fused interleave gate/up (builder-1-d)

Hardening attempt for `hyp-1-experts-exact` (slot `experts`, baseline
`none`). Builds on the best attempt
`muse-spark/round-2/kernels/experts/attempt-1` (fused single kernel,
512 threads/pair, G2 pass all six, 104.3/193.3/535.7 µs/layer chained
vs compiled 77.6/235.7/791.6 — loses only at 1 row). This is the one
main change named by attempt-3's miss analysis
(`attempt-3/notes.md`, *Next change*): attempt-3 split into two
kernels and plateaued at 103.6–109.5 µs/layer at 1 row across three
packings, so group count is not the lever; the next lever is to
overlap the per-block serial K chain itself inside the fused kernel.

## The one main change

Stay in attempt-1's fused single kernel — same one kernel, same
512 threads per (row, expert-slot) pair, same grid `rows*2048`
threads = `rows*4` groups, same branch on `rows >= 60`, same
threadgroup buffers — and interleave gate and up in the decode
branch only:

- **Before (attempt-1):** simdgroup `b` (`b < 12`) ran
  `qvm_block(gate, block b)` to completion (24 strided K steps per
  lane: `k = lane, lane+32, ..., lane+744`; each step loads 4 weight
  words + scale/bias for that `k`, then accumulates 32 outputs),
  stored `tg_gate`, then ran `qvm_block(up, block b)` the same way
  and stored `tg_up`. The two blocks' K loops were fully sequential:
  up's first load waited for gate's last accumulate.
- **Now:** simdgroup `b` runs one merged loop,
  `qvm_gate_up_pair` (block-1). For each strided `k`, both blocks'
  loads are issued first — gate's 4 weight words, scale and bias,
  then up's 4 weight words, scale and bias (plus the shared
  `tg_x[k]`, read once) — and only then do the two accumulation
  passes run, gate's 32 terms then up's 32 terms, into separate
  accumulators `pg`/`pu`. The next `k`'s loads therefore sit behind
  twice the issued load traffic per step, and each block's load
  latency has the other block's loads/compute to hide behind
  (double-buffering the two blocks' weight/scale/x loads within
  the step). Lane 0 then stores both blocks in one loop:
  `tg_gate[b*32+j] = (bfloat)pg[j]`, `tg_up[b*32+j] = (bfloat)pu[j]`.

Everything else is copied, not redesigned:

- **Per-block bits unchanged.** In `qvm_gate_up_pair`, each block's
  adds run in exactly `qvm_block`'s order: the same strided `k`
  sequence, the same float32 dequant terms
  `x * (s0 * (by & 0x0f) + bias)` /
  `x * (s1 * (by & 0xf0) + bias)` with `s1 = scale/16`, in the same
  per-output sequence; `pg` and `pu` are independent accumulators,
  so interleaving the two blocks cannot reorder either block's own
  adds. Each accumulator then gets the identical
  `#pragma clang loop unroll(full)` `simd_sum` tree as `qvm_block`.
  Down still calls `qvm_block` verbatim (verified by diff against
  attempt-1: `qvm_block` and `qmm_accum` bodies byte-identical;
  block-0 differs only in the header comment and the decode
  gate/up call site).
- **Prefill (`rows >= 60`) unchanged.** Attempt-1's scalar
  `qmm_accum` (weight rounded to bfloat16 first,
  `float(bfloat(scale·q+bias))`, then K accumulated contiguously in
  `BK=32` tile order, one output per lane) is copied byte-identical
  for gate, up and down, and block-0's prefill branch is untouched —
  the interleaving does not touch the prefill arithmetic.
- Same four bfloat16 rounding points (gate, up, hidden via the
  shared full-bfloat16 silu-mul, down — silu code byte-identical),
  fp32 accumulation, positional `y[r,j]` (`pair = r*4 + j`), and
  the same `rows >= 60` dispatch threshold from `x_shape[0]`, not
  the grader's bracketing sizes. No kernel split, no threadgroup
  count change, no new device-memory intermediate.

## Which inefficiency it removes, and how

- `ineff-baseline-experts-2` (chained 1-row latency): attempts 1–3
  all plateau at 103.6–109.5 µs/layer at 1 row despite 4, 12+24 and
  48+96 threadgroups per call, so the binding cost past 4 groups is
  each block's serial K chain (24 + 24 + 12 strided
  load-accumulate steps, gate→up→down ordered by the hidden
  dependency) executed with only one block's loads in flight per
  simdgroup at a time. Interleaving gate and up — the two
  independent, equal-length (24-step) chains — puts both blocks'
  loads in flight in every step without touching any block's
  `simd_sum` order, the only lever attempt-3's analysis left inside
  the fused kernel. Down (12 steps) is left sequential: it cannot
  start before the silu barrier in any case.
- `ineff-baseline-experts-1` (prefill G2 gate): already removed by
  attempt-1's branch; kept by reusing its scalar `qmm_accum`
  verbatim.
- `ineff-baseline-experts-3` (per-pair re-reads): deliberately
  kept, as in attempts 1–3 — weight bytes per pair stay
  552,960 B; the interleaving changes load scheduling, not bytes.

## Limits (by hand; the grader does not check the 32 KB cap)

512 threads/group ≤ 1,024 (spec unchanged, verified identical).
Threadgroup memory unchanged from attempt-1: `tg_x[768] +
tg_gate[384] + tg_up[384] + tg_h[384]` bfloat16 = **3,840 B** of
32,768 B, rows-independent. The only new state is register-local:
a second 32-float accumulator `pu[32]` beside `p[32]` and a second
4-word load buffer inside `qvm_gate_up_pair` — the register cost
is the risk noted below. Grid scales as `rows*4` groups, so the
kernel is valid at every `rows ≥ 1`, on both sides of 60.

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Weight bytes per layer (552,960 B/pair, of which gate+up are
368,640 B = 2/3): 2.21 / 17.69 / 70.78 MB at 4 / 32 / 128 pairs.
Attempt-1 chained (this round's grader): 104.3 / 193.3 / 535.7
µs/layer = effective 21.2 / 91.5 / 132.1 GB/s; compiled (same run)
77.6 / 235.7 / 791.6.

- **1 row: 90.** Latency-bound (floor 2.21 MB ÷ 200 GB/s =
  11.1 µs). Gate+up are 48 of the 60 serial product steps and 2/3
  of the pair bytes, so ≈ 2/3 × 104.3 ≈ 69 µs of attempt-1's time
  is the gate/up chains (the rest: down's 12 steps, `tg_x` load,
  silu, barriers, chained launch/link). Issuing both blocks' loads
  before accumulating doubles the loads outstanding per step; if
  that hides ≈ 20% of the gate/up chain's exposed latency
  (0.20 × 69 ≈ 14 µs), the layer lands at ≈ **90 µs** — still
  above the compiled 77.6, closing about half the remaining gap.
  Risk: the second accumulator's register pressure could spill or
  cut resident groups and hand the saving back.
- **8 rows: 178.** Streaming: 17.69 MB. Attempt-1 ran 91.5 GB/s
  with one block's 4 words in flight per simdgroup step; 8 words
  (plus two scale/bias pairs) per step should lift memory-level
  parallelism to ≈ 99 GB/s → 17.69 MB ÷ 99 GB/s ≈ **178 µs**,
  keeping attempt-1's win over compiled 235.7.
- **32 rows: 505.** Already near-saturated at attempt-1's
  132.1 GB/s; the same MLP lift, partly offset by register
  pressure at 128 groups, to ≈ 140 GB/s → 70.78 MB ÷ 140 GB/s ≈
  **505 µs**, keeping the win over compiled 791.6.

G2 prediction: pass at all six sizes with `differ == 0` — decode
interleaving reorders only *between* blocks, never within a
block's add sequence, and the prefill path is byte-identical to
attempt-1's, which already matched `gather_qmm_rhs` bit for bit.

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** (decode 1/8/32 and prefill 127/1,016/2,048) — the
interleaving kept attempt-1's bits in both regimes: per-block
`qvm_block` add order unchanged by the within-step interleave, and
the byte-identical scalar `qmm_accum` prefill path matched
`gather_qmm_rhs` again. Chained timing µs/layer (this run;
candidate vs reference vs compiled): **1 row 94.0 / 81.2 / 73.5
(loses both)**; 8 rows 177.0 / 252.1 / 239.0 (wins); 32 rows
591.3 / 799.2 / 796.3 (wins). `beats_compiled` false — only the
1-row size misses, as in attempt-1. Baseline `none` — no
`wins_vs_baseline`/`beats_baseline`.

Against attempt-1 (different run; compiled shifted 77.6 → 73.5
at 1 row between runs, so treat cross-run deltas as indicative):
94.0 vs 104.3 at 1 row (**−9.9%**), 177.0 vs 193.3 at 8 rows
(**−8.4%**), 591.3 vs 535.7 at 32 rows (**+10.4% regression**).
This is the first attempt to move below attempt-1's 103.6–109.5
µs/layer 1-row plateau (attempts 1–3) at any size, and it does so
at 1 and 8 rows — but it spends part of attempt-1's 32-row win.

## Miss analysis (prediction 90 / 178 / 505, measured 94.0 / 177.0 / 591.3)

Misses are +4.4% / −0.6% / +17.1% — all inside the 20% bar, so no
formal miss, but the 32-row *direction* was wrong (predicted
−6% vs attempt-1, measured +10.4%), and that needs explaining:

1. **The interleave works where latency binds.** Effective
   bandwidth on pair bytes: 23.5 GB/s at 1 row (attempt-1: 21.2)
   and 99.9 GB/s at 8 rows (attempt-1: 91.5). Doubling the loads
   outstanding per K step (8 weight words + two scale/bias pairs
   issued before either accumulation) hid enough of each block's
   load latency to break the 103.6–109.5 plateau that three
   threadgroup packings could not — confirming attempt-3's
   diagnosis that the per-block serial chain, not group count,
   was the 1-row lever. The gain is real but partial: 94.0 still
   loses to compiled 73.5, so roughly 20 µs/layer of the 1-row
   cost (chained launch/link, `tg_x` load, silu, barriers, down's
   still-sequential 12-step chain and its two waves) remains
   unoverlapped.
2. **At 32 rows the second accumulator is the tax.** 119.7 GB/s
   vs attempt-1's 132.1 GB/s. Once 128 groups saturate the GPU,
   exposed load latency is already hidden by group-level
   parallelism, so the interleave buys nothing — and its cost,
   a second live 32-float accumulator plus a second 4-word load
   buffer per thread for the whole K loop, is paid in register
   pressure and therefore resident-group occupancy on every
   streaming step. The prediction's assumed MLP lift to
   140 GB/s double-counted parallelism the group count already
   provided.
3. **The 8-row point separates the two regimes:** same register
   cost as 32 rows, but only 32 groups per call, so per-group
   load overlap still pays (91.5 → 99.9 GB/s, prediction exact).
   The crossover between "latency-bound, interleave wins" and
   "occupancy-bound, interleave taxes" sits between 8 and 32
   rows in this run.

Lesson for `ineff-baseline-experts-2`: the serial-chain lever is
real (first sub-plateau 1-row result: 94.0), but a uniform
interleave trades it against streaming occupancy. The two
regimes want different code, and both arithmetics are
bit-identical, so the choice between them is free.

**Next change (one):** make the interleave row-conditional
inside the same fused kernel — `qvm_gate_up_pair` below the
latency/occupancy crossover (e.g. `rows <= 8`), attempt-1's
sequential `qvm_block` gate then up above it — keeping the
1/8-row wins (94.0/177.0) while restoring attempt-1's 32-row
register profile (535.7). Both branches keep every block's
add order, so G2 is unaffected by where the crossover sits.
