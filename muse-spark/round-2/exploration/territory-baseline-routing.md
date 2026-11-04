# Territory: baseline-routing — the routing slot's starting point under bit-exact G2

- **Owner:** explorer-baseline-routing
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T03:20:20Z
- **Next:** none — checklist filled for all pieces, ineff-baseline-routing-1..4 posted, grader and g3 notes folded

## What it does

In every decode step and every prefill chunk, each of the 20 layers runs
`MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`): cast the
layer input to float32, multiply by the router weight transposed, sigmoid, add the
expert bias only for selection, take the top-4 experts per row, gather their
*unbiased* scores, normalise them (divide by their sum + 1e-20) and scale by 2.5.
The slot outputs `inds` [rows,4] uint32 and `weights` [rows,4] float32; `inds`
feeds the experts slot, `weights` feeds combine, which sums the 4 (expert,
weight) pairs *in the emitted order* — which is why round 2's G2 requires the
same experts, the same weights and the same order, bit for bit
(`claude/adr/0010-exact-at-every-size.md:25-27`,
`bench/grader/check_candidate.py:119-121`).

The starting point is round 1's `muse-spark/kernels/routing/attempt-1`: one fused
Metal kernel per layer (`block-0.metal`, 78 lines; `spec.json`: one kernel
`fused_routing`, grid `rows*32`, threadgroup 32). One threadgroup per row stages
`x[row]` (768 bfloat16→float32) in threadgroup memory, each of the 32 threads
computes 1–2 experts' 768-term dot products serially, writes
`sigmoid(acc)` to `scores[60]`, and thread 0 alone does a serial insertion
top-4 scan over the biased scores and writes `inds`/`weights`
(`block-0.metal:10-78`).

## The headline finding (details in ineff-baseline-routing-1 and -2)

Under the new grader the starting point fails G2 at **every** size while the
experts and their order always match (`rows_other_experts` 0 everywhere):
rows differing 20/159/638 at decode 1/8/32 rows (of 20/160/640 rows summed over
20 layers — 100% / 99.4% / 99.7% of rows) and 2,518/16,187/32,743 at prefill
127/1,016/2,048 rows (of 2,540/20,320/40,960 — 99.1% / 79.7% / 79.9%), max
relative error 2.6e-06 at decode and up to 8.7e-06 in the prefill
(`results/runs/2026-11-04-muse-round-1/grade-exact-routing.json`, quoted in
`muse-spark/round-2/exploration/README.md:39`). The experts match because
selection is insensitive to last-bit score differences (no captured row is a
near-tie at the 4th/5th place); the weights expose every last-bit difference
in the 4 selected scores. Section 3 below pins down, op by op, which of the
reference's operations the kernel reproduces exactly and which it does not:
**the divergence is in the scores — the router matmul's float32 accumulation
order (which changes with `rows`) and MLX's sigmoid formula — not in the
4-term weight sum, whose order the kernel already matches.**

## Checklist

### 1. The starting-point kernel — `muse-spark/kernels/routing/attempt-1/{block-0.metal,spec.json,notes.md}`

- **Callers:** the grader builds it via `slots.build` into one
  `mx.fast.metal_kernel` (`bench/grader/slots.py:303-345`) and calls it once per
  layer per size in G2 (`bench/grader/check_candidate.py:286`) and in the
  chained piece timing (`check_candidate.py:184-213`). In the model it would
  replace the routing part of `make_moe_call` (`slots.py:97-119`). Shapes:
  rows = 1/8/32 (decode), 127/1,016/2,048 (prefill); grid `rows*32` threads,
  one 32-thread threadgroup per row at every size (`spec.json`, block-0:10-11).
- **Callees:** none — one Metal kernel. It launches 1 kernel/layer (20/step)
  where the reference launches ~12/layer (240/step; brief:47).
- **Data:** in: `x` [rows,768] bfloat16, `w_router` [60,768] float32, `bias`
  [60] float32; out: `inds` [rows,4] uint32, `weights` [rows,4] float32
  (`spec.json`; brief:30-36). Threadgroup memory: `xsh[768]` float +
  `scores[60]` float = 3,312 B of 32 KB (`block-0.metal:13-14`).
- **Neighbours:** before: post-attention layernorm output reshaped to
  [rows,768] (`mobilemoe.py:258`, `slots.py:107`); after: experts gather by
  `inds`, combine sums `(y * weights[...,None])` in emitted order
  (`mobilemoe.py:230-231`). A replacement must keep shapes, dtypes, the
  unbiased-score rule and — new in round 2 — the reference's exact per-row
  order (see Invariants).
- **Cost:** measured (old, independent timing; Claude re-grade,
  `claude/findings/muse-round-1.md:47`): candidate 25.3/24.3/68.9 µs per layer
  at 1/8/32 rows vs compiled reference 31.5/42.7/87.6 and reference
  39.9/83.5/101.2. Round 1's own run: 28.5/23.6/41.8 (`notes.md:33-37`).
  No chained figure exists for this slot yet (round-2 README:36) — the chained
  timing is the bar a round-2 attempt must beat. End to end (G3, vLLM,
  `muse-round-1.md:85-86`): attempt alone +0.185/−0.397/−0.490 ms/step vs the
  compiled reference's −0.220/−0.183/−0.157, i.e. the attempt is ~0.41 ms/step
  *slower* than the compiled reference at 1 request and ~0.22/0.33 ms/step
  faster at 8/32. Traffic: `w_router` 60×768×4 B = 184 KB read **once per row**
  (each threadgroup streams its own copy, `block-0.metal:30-34`), `x` 1.5 KB/row
  staged once per row; see ineff-baseline-routing-4 for the 32-row cost.
  Bytes at 1 row are negligible; the slot is overhead-bound
  (`claude/GLOSSARY.md:39-42`; round 1's `ineff-runs-2`, confirmed).
- **Invariants (bit-exact G2):** everything in the slot is float32 after the
  exact bf16→f32 cast of `x` (`block-0.metal:20`; `slots.py:45`). The kernel
  matches the reference on: the cast; the bias add for selection
  (`scores[e] + bias[e]`, block-0:45 — same single f32 add as
  `slots.py:46`); the gather of unbiased scores; the 4-term sum order, the
  `+1e-20`, the division and the ×2.5 (section 3). It does **not** match: the
  768-term dot-product accumulation order, nor the sigmoid's formula/exp
  (section 3). Tie rule: thread 0's insertion uses strictly-greater compares,
  so the earliest index wins ties and the emitted order is descending biased
  score (`block-0.metal:40-61`) — the same set+order the reference's
  sort-based argpartition produces on every graded input so far
  (`rows_other_experts` 0 at all six sizes, grade-exact-routing.json); a
  replacement must not change this behaviour, because combine consumes the
  pairs in this order (`check_candidate.py:119-120`).

### 2. The reference — `bench/grader/slots.py:44-48` (`ref_routing`) and `mobilemoe.py:217-232` (`MoE.__call__`)

```python
scores = mx.sigmoid(x.astype(mx.float32) @ w_router.T)          # [rows, 60]
inds = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[:, :K]
w = mx.take_along_axis(scores, inds, axis=-1)
return inds, w / (w.sum(axis=-1, keepdims=True) + 1e-20) * SCALE   # SCALE=2.5
```

- **Callers:** `MoE.__call__` in all 20 layers, decode and prefill alike
  (`mobilemoe.py:219-228`); the grader's `check_routing` calls `ref_routing`
  per layer (`check_candidate.py:102-103`); `capture_prefill` records prefill
  inputs through the same split (`slots.py:214-254`). The model's own code is
  the same arithmetic with `self.router(x.astype(mx.float32))` (an
  `nn.Linear`, i.e. `x @ W.T`, W [60,768]) and `expert_bias` added for
  selection only (`mobilemoe.py:197,219-228`); `slots.py:108` converts the
  model's bfloat16 `expert_bias` to float32 exactly, and `check_identity`
  asserts the split equals the port bit for bit at decode sizes
  (`slots.py:122-135`).
- **Callees (MLX ops → Metal kernels, MLX 0.32.4):** `astype` (copy kernel,
  exact); `@` → `Matmul::eval_gpu` → GEMV / steel split-K GEMM / steel GEMM
  depending on rows (section 3.1); `sigmoid` → unary `Sigmoid` kernel
  (section 3.2); `scores + bias` → binary `Add` (`binary_ops.h:10-15`, plain
  `x + y`); `argpartition` → sort-based kernel (round 1:
  `ArgPartition::eval_gpu` directs argpartition to a full 60-element merge
  sort per row, `muse-spark/exploration/ineff-runs-2.md:94-98` citing
  mlx `sort.cpp:344-356` @v0.32.4); `take_along_axis` → gather (exact copy);
  `w.sum(axis=-1)` → `row_reduce_small` with `thread_reduce` (section 3.3);
  `/` and `*` → binary `Divide`/`Multiply` (`binary_ops.h:43-48,208-213`,
  plain `x / y`, `x * y`). Uncompiled, that is ~12 kernels/layer; under
  `mx.compile` (the timing bar, `check_candidate.py:322`) MLX fuses the
  elementwise chain but the matmul and reduction kernels keep their own
  accumulation orders — compilation does not change G2's target, which is the
  uncompiled `ref_routing` (`check_candidate.py:103`).
- **Data:** `scores` [rows,60] f32 → `inds` [rows,4] u32, `w`/`weights`
  [rows,4] f32. All row-contiguous (brief:24). `bias` in the slot is already
  float32 (`slots.py:108`).
- **Neighbours:** identical to item 1 (the kernel replaces exactly this).
- **Cost:** reference 39.9/83.5/101.2 µs/layer, compiled 31.5/42.7/87.6
  (independent timing, `muse-round-1.md:47`); knock-out cost 1.02 ms/step at
  1 row and 1.68 ms at 32 rows for ~4 MB of traffic — overhead-bound
  (`muse-spark/exploration/ineff-runs-2.md:15-27`, confirmed).
- **Invariants:** the reference *is* the invariant for round 2: same 4 experts
  in the same order with bit-equal weights at 1/8/32 decode rows and
  127/1,016/2,048 prefill rows (`check_candidate.py:120-129`: a row differs if
  any expert or any weight differs; `max_rel_err` is reported only to help
  find the cause). The reference checked against itself shows no difference
  at any size (ADR 0010:29).

### 3. Op-by-op: where the kernel's arithmetic matches the reference's, and where it does not

#### 3.1 The router product `x.astype(f32) @ w_router.T` — DOES NOT MATCH (main cause)

The kernel accumulates each expert's dot product **serially, in one thread,
in increasing `d` order**: `acc += xsh[d] * w_router[wbase + d]` for
`d = 0..767` (`block-0.metal:31-34`; the Metal compiler may contract each
mul+add to an FMA, but the grouping stays a 768-long left-to-right chain).

The reference never does that. `Matmul::eval_gpu` (MLX v0.32.4,
`mlx/backend/metal/matmul.cpp`, fetched 2026-11-04) sees A = `x` [rows,768]
f32 (not transposed), B = `w_router.T` [768,60] (transposed view), so
`transpose_a=false, transpose_b=true`, M=rows, N=60, K=768:

- **rows = 1 → GEMV.** `min(M,N)==1` routes to `gemv`
  (matmul.cpp, `Matmul::eval_gpu`); the `gemv_wide` shortcut is skipped
  because the output is float32 (`gemv_wide_config` requires float16/bfloat16
  out). Dispatch picks `GEMVKernel<float, BM=4, BN=1, SM=1, SN=32, TM=4,
  TN=4>`: each output is accumulated by one simdgroup whose 32 lanes each
  serially sum 24 products — lane `l` takes `d = i*128 + l*4 + (0..3)` for
  `i = 0..5` (`kernels/gemv.h:162-182`) — and the 32 partials are then combined
  by a shuffle tree, `result += simd_shuffle_down(result, sn)` for
  sn = 16,8,4,2,1 (`gemv.h:202-208`). Grouping: 32 blocked partials + pairwise
  tree, not a serial chain.
- **rows = 8, 32, 127 → steel split-K GEMM.** In `steel_matmul_axpby` the
  split-K case fires when `_tm*_tn <= 2048` (this device's architecture string
  ends in `s`; brief:99 names it `applegpu_g13s`), `_tk = 768/16 = 48 >= 8`
  and `K >= max(M,N)` (matmul.cpp, Case 1). `steel_gemm_splitk_axpby` uses
  BM=16 (M<40) or 32, BN=32, BK=16, WM=WN=2 (steel/gemm split-K dispatch):
  partitions = min(max(2, next_pow2(48/(_tm*_tn))), 32) with
  `_tm=(M+31)/32, _tn=2` → **32 partitions at rows 8 and 32** (the last
  partition also absorbs the remaining 16 BK-iterations,
  `steel/gemm/kernels/steel_gemm_splitk.h:140-155`), **8 partitions at rows
  127**. Each partition accumulates its K-slice with simdgroup 8×8 matrix
  multiply-accumulate over BK=16 chunks in order (GEMMKernel/mma in
  `steel/gemm/gemm.h`, `mma.h`), writes a float32 partial, and
  `gemm_splitk_accum` sums the partials sequentially in partition order
  (`steel_gemm_splitk.h:186-189`).
- **rows = 1,016, 2,048 → regular steel GEMM.** Split-K no longer fires
  (768 < M), so `steel_matmul_regular_axpby` runs with the device tile from
  `GEMM_TPARAM_MACRO` (for an architecture ending in `s` the macro's fallback:
  BM=64, BN=64, BK=16, WM=2, WN=2): K is walked in BK=16 chunks staged through
  threadgroup memory, accumulated by simdgroup matrix ops chunk after chunk.

So the reference has **three different float32 accumulation orders for the
same mathematical product, selected by `rows`** — and none of them is the
kernel's serial order. Every score therefore differs from the reference's in
the last bits with high probability, which is exactly the observed pattern:
~100% of decode rows and ~80–99% of prefill rows have at least one differing
weight, while no row picks different experts (the biased-score gaps at the
4th/5th place are far larger than 1-ULP score noise on these inputs).
The drop from ~100% (decode, GEMV/split-K regimes) to ~80% (1,016/2,048,
regular GEMM) is consistent with a per-regime last-bit flip rate; that last
step is an inference from the counts, not a measurement.
A scalar re-implementation cannot reproduce the steel regimes' rounding by
reordering a serial loop: within each 8×8 simdgroup matrix op the 8-product
accumulation is the hardware's. A bit-exact candidate must either reuse
`simdgroup_float8x8` multiply-accumulate with the same tiling/K-order per
regime (branching on `rows` inside the kernel, or one kernel per regime in
the spec), or find an accumulation that provably rounds identically — none
is known.

#### 3.2 The sigmoid — DOES NOT MATCH (independent second cause)

Kernel (`block-0.metal:35`): `1.0f / (1.0f + metal::exp(-acc))` — the fast
(non-`precise`) `exp`, applied to `-acc`, one division.

Reference (`mx.sigmoid` → `Sigmoid` in
`mlx/backend/metal/kernels/unary_ops.h:308-314`, installed headers):

```metal
auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
return (x < 0) ? y : 1 - y;
```

Three differences, each able to move the result by 1 ULP: (a) `precise::exp`
vs `metal::exp`; (b) the argument is `|x|`, never negative; (c) for `x >= 0`
the result is `1 - y` (a subtraction that rounds) instead of the direct
division the kernel computes for all signs. Even a kernel whose matmul
matched bit for bit would still fail G2 on sigmoid alone for a large share
of scores. The fix is free in time: copy this formula verbatim.

#### 3.3 Selection, gather, and the weight normalisation — MATCH

- `scores + bias`: one f32 add in both (`block-0.metal:45`;
  `binary_ops.h:10-15`).
- Selection: the kernel's strictly-greater insertion scan emits descending
  biased order, earliest index winning ties (`block-0.metal:44-61`). The
  reference's argpartition is a full sort of `-(scores+bias)`
  (round-1 note, `ineff-runs-2.md:94-98`); on all six graded sizes the emitted
  sets *and* order agree (`rows_other_experts` 0, and the grader's worst-row
  examples show identical expert lists, grade-exact-routing.json). This is
  the part of Claude's round-1 reading ("the reference's order of
  operations … differ", round-2 README:39) that does **not** need fixing —
  but a builder must preserve the exact tie behaviour, since G2 now also
  compares order (`check_candidate.py:120`).
- The 4-term sum: the reference's `w.sum(axis=-1)` on a [rows,4] f32 array
  takes `row_reduce_small` (row_size 4 <= 64, MLX v0.32.4
  `mlx/backend/metal/reduce.cpp`, `row_reduce_general_dispatch` Case 1) in its
  per-thread branch (`non_row_reductions=1 < 32 && row_size <= 8`,
  `row_reduce_small` dispatch), i.e. `thread_reduce` with `N_READS=4`
  (`defines.h:12`): `total = (((0 + w0) + w1) + w2) + w3` in memory order —
  and `w`'s memory order is the `inds` order from argpartition, the same
  sequence the kernel sums as `s0 + s1 + s2 + s3` (`block-0.metal:64-68`;
  `kernels/reduction/reduce_row.h:164-181`). The kernel then adds `1e-20f`
  inside the same expression (`denom = s0+s1+s2+s3 + 1e-20f`), the reference
  as a separate add after the reduction (`slots.py:48`) — same association,
  and in float32 `sum + 1e-20` equals `sum` for any sum above ~1e-27, so the
  epsilon cannot be the cause either. Division and ×2.5 are the same single
  ops (`binary_ops.h:43-48,208-213`; `block-0.metal:74-77`).
  **Conclusion:** once the 4 selected scores are bit-equal and in the same
  order, the weights are bit-equal. Builders should spend their effort on
  3.1 and 3.2, not on the normalisation — the opposite emphasis to round 2
  README:39's example list, which names "the float32 sum order of
  `x @ w_router.T` and of `w.sum(...)`" as co-equal suspects.

### 4. `spec.json`, `notes.md`, and the round-1 hypothesis/boundary entries

- **Callers/callees/data/neighbours:** `spec.json` declares the single kernel
  above; `slots.build` enforces threadgroup <= 1024 threads (32 here) and
  turns grid expressions into thread counts (`slots.py:326-341`). The
  round-1 boundary (`muse-spark/implementation/hyp-2-fused-routing.md:47-52`)
  declared summation order "fine within tolerance" — true under round 1's
  1e-4 G2, false under bit-exact G2; that sentence is the one assumption
  round 2 must retire.
- **Cost:** the attempt's own verdict and miss analysis (`notes.md:30-68`):
  the piece harness's fixed floor is ~24 µs/layer, not one 6.81 µs launch;
  the reference's 12 kernels cost ~3 µs each when 20 layers are pipelined.
  Attempt-2 (two-kernel split with w tiling shared across 8 rows) was slower
  at every row count (36.3/34.9/49.0 vs attempt-1's 28.5/23.6/41.8,
  `muse-spark/kernels/routing/attempt-2/notes.md`) — direct evidence that at
  decode sizes dispatch/occupancy, not traffic, binds.

## Relations

- **Fed by:** the model territory (layernorm output `x`; router weight and
  `expert_bias` parameters, `mobilemoe.py:187-199,258`); the grader territory
  (captured inputs at 1/8/32 rows, `models/muse-inputs/`; prefill inputs
  recorded live, `slots.py:214-254`).
- **Feeds:** `baseline-experts` (`inds` chooses the (row, expert) pairs and
  their order); `baseline-combine` (`weights` multiply the experts' outputs
  in emitted order — the reason order is part of G2); `g3` (this slot alone
  was +0.19/−0.40/−0.49 ms/step in vLLM, and its 1-request loss to the
  compiled reference is one slice of the not-yet-explained end-to-end
  picture).
- **Shared cause:** `baseline-combine`'s exactness gap is the same *kind* of
  problem (reduction association vs the reference's kernel) but a different
  op and fix; the two territories' builders should not share a design.

## Inefficiencies

- `ineff-baseline-routing-1` — the router product's accumulation order:
  three row-count regimes (GEMV tree / split-K MMA / steel MMA) vs the
  kernel's serial 768-term chain; blocks G2 at every size.
- `ineff-baseline-routing-2` — the sigmoid's formula and exp: fast direct
  `1/(1+exp(-x))` vs MLX's `precise::exp(|x|)` branch form; independently
  blocks G2 even if the product matches.
- `ineff-baseline-routing-3` — one simdgroup per row and a serial thread-0
  top-4 scan: occupancy and idle threads, worst at 1 row where the slot
  already loses to the compiled reference end to end.
- `ineff-baseline-routing-4` — `w_router` re-read once per row: negligible
  at 1 row, a 5.9 MB/layer traffic floor at 32 rows; bounded saving, with
  round-1 attempt-2 as counter-evidence at decode sizes.

## Files read

- `muse-spark/kernels/routing/attempt-1/block-0.metal` — the starting-point
  kernel: serial dot, fast sigmoid, serial top-4, LTR weight sum.
- `muse-spark/kernels/routing/attempt-1/spec.json` — one kernel, grid
  `rows*32`, threadgroup 32.
- `muse-spark/kernels/routing/attempt-1/notes.md` — prediction, verdict,
  the ~24 µs fixed-floor finding.
- `muse-spark/kernels/routing/attempt-2/{spec.json,notes.md}` — the
  two-kernel control: w tiling across rows did not pay at decode sizes.
- `bench/grader/slots.py` — `ref_routing` (44-48), split MoE, capture and
  prefill capture, launch-spec build.
- `bench/grader/check_candidate.py` — bit-exact `check_routing` (102-136),
  six G2 sizes, chained timing (178-213), `link` kernel.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py` — `MoE.__call__`
  (217-232) and the router/expert-bias parameters (187-199).
- `claude/muse-agent-brief.md` — slot table, G1/G2/G3 rules.
- `claude/GLOSSARY.md` — overhead/bandwidth/compute-bound terms.
- `claude/adr/0010-exact-at-every-size.md` — why G2 is bit-exact at six
  sizes and the timing chained.
- `claude/findings/muse-round-1.md` — re-grades (§1), token flips (§2),
  G3 (§3), new-grader counts (§6), starting points.
- `results/runs/2026-11-04-muse-round-1/grade-exact-routing.json` — per-size
  rows_differ / max_rel_err / worst-row examples quoted throughout.
- `muse-spark/round-2/exploration/README.md` — the starting point for this
  territory (routing section, lines 33-40) and *What is left* item 2.
- `muse-spark/implementation/hyp-2-fused-routing.md` — round-1 boundary and
  both builders' verdicts.
- `muse-spark/exploration/ineff-runs-2.md` — the confirmed overhead entry
  and the argpartition/reduction kernel notes appended by other explorers.
- MLX v0.32.4 sources (installed version, `.venv` python reports 0.32.4):
  installed headers `kernels/gemv.h`, `kernels/unary_ops.h` (Sigmoid),
  `kernels/binary_ops.h`, `kernels/reduction/reduce_row.h`,
  `kernels/defines.h`, `kernels/steel/gemm/kernels/steel_gemm_splitk.h`;
  fetched from `github.com/ml-explore/mlx` tag `v0.32.4`:
  `mlx/backend/metal/matmul.cpp` (GEMV/split-K/steel dispatch + tile
  params), `mlx/backend/metal/reduce.cpp` (`row_reduce_small` dispatch),
  `mlx/backend/common/reduce.cpp` (reduction plan for [rows,4]).

## From other agents

- explorer-grader, held in `territory-grader.md` (*Inefficiencies*) and folded
  here 2026-11-04T03:20:20Z: G2 routing is ordered bitwise equality including
  the reference's tie choice; `max_rel_err` is computed only over same-order
  rows (`check_candidate.py:122`), so a permuted row would fail
  `rows_differ` while staying invisible in that diagnostic. **Folded:** agreed
  — sections 1-2 already state the ordered-equality gate, and the
  "experts and order match" claim rests on `rows_other_experts` 0 at all six
  sizes plus the grader's worst-row examples (which print candidate and
  reference expert lists side by side), not on `max_rel_err`. A builder must
  reproduce the reference's 4th/5th tie choice exactly; the starting point's
  strictly-greater insertion does so on all graded inputs (section 3.3).
- explorer-g3, held in `territory-g3.md` (*Notes for other territories*) and
  folded here 2026-11-04T03:20:20Z: the routing attempt is 0.405 ms/step
  slower than its compiled reference in-step at 1 request in vLLM (+0.185 vs
  −0.220 ms) against a piece promise of −0.124 ms/step — a 0.529 ms/step
  swing — and no chained routing timing exists yet to say why. **Folded:**
  agreed and already carried in section 1 (Cost) and
  `ineff-baseline-routing-3`, whose occupancy account is offered as the
  explanation to be tested by the new grader's chained timing, not as a
  measured cause. The chained routing figure is this territory's open debt
  to the implementation phase.
