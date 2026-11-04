# Hypothesis 2: routing kernel with the reference's exact float32 arithmetic

- **Owner:** lead
- **Status:** building
- **Version:** 3
- **Updated:** 2026-11-04T03:43:13Z
- **Next:** boundary check for attempt-1; builder-2-b still running; hardening from attempt-1

## Inefficiency

Slot `routing`. Baseline: `none`. Starting point: `muse-spark/kernels/routing/attempt-1`.

One combined fix for four confirmed entries:

- `ineff-baseline-routing-1` (gate, ranked 5): router-product accumulation order matches MLX in no regime (GEMV at 1 row, split-K steel at 8/32/127, steel GEMM at 1,016/2,048); blocks G2, headroom 0.12 / 0.37 / 0.37 ms/step.
- `ineff-baseline-routing-2` (gate, joint): sigmoid `precise::exp(|x|)` branch vs fast `exp(-x)`; independent blocker, 0 µs direct.
- `ineff-baseline-routing-3` (ranked 4): one simdgroup per row + serial thread-0 scan; up to 0.4 ms/step at 1 row.
- `ineff-baseline-routing-4` (ranked 7): `w_router` re-read per row; upper bound 0 / 0.13 / 0.57 ms/step, unbanked on round-1 counter-evidence.

Design: a kernel that reproduces the reference's scores bit for bit at every size — the same simdgroup matrix accumulation and tiling per `rows` regime, MLX's sigmoid formula verbatim, the already-matching 4-term normalise sum and tie rule — with each expert's dot product sharded across lanes and a parallel top-4, so the 1-row occupancy deficit is removed together with the exactness gates. Must keep the G2 boundary (same experts, weights and order bit for bit at all six sizes).

## Boundary

*written by boundary-2, 2026-11-04T03:31:18Z — delta to round 1's boundary. Only what changed is written here; every round-1 line not listed below still holds verbatim.*

Round 1's boundary is [hyp-2-fused-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-2-fused-routing.md:19) (`muse-spark/implementation/hyp-2-fused-routing.md:19-62`). I checked each of its lines against the starting point ([spec.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/spec.json), [block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:10)) and against the current grader ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:44), [check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:102), [ADR 0010](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md:22)). The slot's baseline is `none` (this entry, line 11), so there is no `beats_baseline` bar; the bar is the compiled reference. Four round-1 conclusions are retired: the 1e-4 weight tolerance, expert-set (unordered) equality, the 1e-6 near-tie exemption, and "order of the 4 pairs is free". Everything else in round 1 — shapes/dtypes/layouts at decode sizes, the producer, the bias rule, the unbiased gather, the epsilon and the 2.5 scale, the GPU limits — holds as written, extended below where the size set grew.

### Inputs and outputs — extended to the prefill sizes (changed)

Sources: the brief's routing table ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:28)), `D, E, K = 768, 60, 4` ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:19)), [ref_routing](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:44) and the shape/dtype enforcement in [check_routing](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:107). All arrays stay row-contiguous ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:24)). `rows` is now six graded values, not three: decode 1 / 8 / 32 and prefill 127 / 1,016 / 2,048 ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:272), [capture_prefill](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:214)).

| | Name | Shape at decode 1 / 8 / 32 rows | Shape at prefill 127 / 1,016 / 2,048 rows | Dtype |
|---|---|---|---|---|
| input | `x` | [1,768] / [8,768] / [32,768] | [127,768] / [1016,768] / [2048,768] | bfloat16 |
| input | `w_router` | [60,768] at all row counts | [60,768] at all row counts | float32 |
| input | `bias` | [60] at all row counts | [60] at all row counts | float32 |
| output | `inds` | [1,4] / [8,4] / [32,4] | [127,4] / [1016,4] / [2048,4] | uint32, each < 60 |
| output | `weights` | [1,4] / [8,4] / [32,4] | [127,4] / [1016,4] / [2048,4] | float32 |

`weights[r,j]` still belongs to `inds[r,j]` — now at the same position `j` as the reference, not merely as a pair (see *What must stay identical*).

### Before and after — the neighbour contract is narrowed, not moved (one line changed)

- **Before (holds, at all six sizes).** The producer is unchanged: `x` is `post_attention_layernorm(h)` reshaped to flat `[rows,768]` ([mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:258), [slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:105)); `w_router` is `self.router.weight` and `bias` is `self.expert_bias.astype(float32)` ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:108)). The same `MoE.__call__` runs the prefill ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:217), [mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217)), so the prefill sizes feed the slot through the identical path with larger `rows`. The slot still reads `x` as bfloat16; the reference casts to float32 itself ([mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:219)).
- **After (1), the experts call — round 1's "order is free" is retired.** `SwitchGLU` still gathers by index and, from `indices.size >= 64` (rows ≥ 16, so at every prefill size), sorts and unscatters internally so that `y[...,j,:]` corresponds to `inds[...,j]` ([switch_layers.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/switch_layers.py:181)). Functionally a permutation would still be gathered correctly, but it no longer satisfies the boundary: G2 compares `inds` in order ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:119), [ADR 0010](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md:25)). The experts neighbour still expects 4 indices per row, each < 60 (checked at [check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:117)); distinctness follows from matching the reference's `argpartition` output, which is distinct by construction ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:46)).
- **After (2), combine — the pair order is now load-bearing.** Combine still computes `(y * weights[..., None]).sum(axis=-2).astype(bfloat16) + shared` ([mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:231), [slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:51)), summing the 4 pairs in the emitted order, which is why the reference order is part of G2 ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:119)). Round 1's condition "`inds[r,j]` ↔ `weights[r,j]` pairing preserved, order free" is therefore insufficient: both the pairing and the position `j` must match the reference. The bfloat16 cast still happens in combine, outside the slot; the slot still outputs pure float32.

Checklist delta (per [tracking.md](/Users/onur/Desktop/Github/Hackathon-muse/.agents/skills/muse-round/tracking.md:115)): producer — holds; `SwitchGLU` consumer — only the permutation line changed; combine consumer — only the order line changed. Callers, callees, data shapes/dtypes and costs are as in round 1, with `rows` extended as above.

### What must stay identical — bit for bit at all six sizes (changed)

G2 now passes only when a row is bitwise identical, in order: `differ = (inds != ref_inds).any(axis=1) | (w != ref_w).any(axis=1)` and `passes` requires `rows_differ == 0` ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:120), [check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:172)). `max_rel_err` is reported only over same-order rows as a diagnostic ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:122)); it cannot pass a differing row. The reference is [ref_routing](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:44):

```python
scores = mx.sigmoid(x.astype(mx.float32) @ w_router.T)
inds = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[:, :K]
w = mx.take_along_axis(scores, inds, axis=-1)
return inds, w / (w.sum(axis=-1, keepdims=True) + 1e-20) * SCALE
```

- **Rounding points (form holds, tolerance retired).** Everything inside the slot is still float32 after the exact bf16→f32 cast of `x`; there is still no bfloat16 rounding inside the slot ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:45), [mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:219)). Retired is round 1's "reduced precision anywhere … is allowed only if weights still land within 1e-4 relative … a different summation order is fine within that tolerance" ([hyp-2-fused-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-2-fused-routing.md:49)): any last-bit difference in any of the 4 weights, at any of the six sizes, now fails the row.
- **The router product — three accumulation regimes, none of them serial (new).** The reference's MLX kernel, and with it the float32 accumulation order of `x.astype(f32) @ w_router.T`, changes with `rows` (MLX 0.32.4 `Matmul::eval_gpu`, M=rows, N=60, K=768; derivation in [territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md:153) §3.1, confirmed in [ineff-baseline-routing-1.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-routing-1.md:24)):

| `rows` | MLX kernel | float32 accumulation order |
|---|---|---|
| 1 | GEMV (`gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4`) | 32 lanes × serial 24 products (lane `l`: `d = i*128 + l*4 + 0..3`, `i=0..5`), then shuffle tree `sn = 16,8,4,2,1` |
| 8, 32, 127 | steel split-K GEMM | 32 K-partitions at 8/32 rows, 8 at 127; each partition: simdgroup 8×8 MMA over BK=16 chunks; partials summed sequentially in partition order |
| 1,016, 2,048 | regular steel GEMM | BM=64/BN=64/BK=16 tiles, K walked in BK=16 chunks, simdgroup MMA accumulation chunk after chunk |

  A candidate must reproduce the regime that applies at the `rows` it is launched with — branching on `rows` generally (or one spec kernel per regime), not special-casing the six graded values: the dispatch is a function of `rows` (GEMV only at `rows = 1`; split-K while `768 >= rows`; regular GEMM above), and tile/partition counts vary inside the split-K regime. A serial reordering cannot reproduce the MMA regimes' rounding, since the 8-product accumulation inside each `simdgroup_float8x8` multiply-accumulate is the hardware's ([ineff-baseline-routing-1.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-routing-1.md:30)).
- **The sigmoid — MLX's formula verbatim (new).** The reference's `mx.sigmoid` is `y = 1 / (1 + metal::precise::exp(metal::abs(x)))`, returning `y` for `x < 0` and `1 - y` for `x >= 0` ([territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md:213) §3.2, [ineff-baseline-routing-2.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-routing-2.md:15)). Any of `precise::exp` vs `exp`, the `|x|` argument, or the `1 - y` subtraction differing moves a score by 1 ULP and fails the row; this is independent of the product regime and both must be fixed.
- **Bias, gather and the 4-term normalise sum (arithmetic holds, now load-bearing).** The bias rule still holds: selection uses `scores + bias` (one f32 add), weights use the unbiased `scores` gathered at the selected indices ([mobilemoe.py](/Users/onur/Desktop/Github/Hackathon-muse/third_party/mlx-lm/mlx_lm/models/mobilemoe.py:221), [slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:46)). The 4-term sum still holds and the starting point already matches it: `(((w0 + w1) + w2) + w3)` in `inds` order, then `+ 1e-20`, divide, `× 2.5` ([territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md:247) §3.3, [slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:48)). What changed is only that this exact association is now required bit for bit, where round 1 allowed any order within tolerance; once the 4 selected scores are bit-equal and in the same order, the weights are bit-equal, so builders should not spend effort here.
- **Tie rule for the 4th and 5th expert, and the order of the 4 pairs (changed).** Retired is round 1's set equality with a 1e-6 near-tie exemption ([hyp-2-fused-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-2-fused-routing.md:51)). The reference's choice between tied 4th and 5th biased scores, including exact ties, and the descending-biased-score order of positions 0..3 must be reproduced exactly at every size ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:187), [check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:120)). No reordering of the 4 pairs is allowed (retires [hyp-2-fused-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-2-fused-routing.md:52)); shapes, dtypes, the epsilon, the 2.5 scale and the unbiased-score rule still may not move.
- **Starting point against this boundary.** Checked line by line, [attempt-1](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:16) matches: the bf16→f32 cast ([block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:20)), the bias add ([block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:45)), the unbiased gather, the 4-term sum order with `+1e-20`, divide and `×2.5` ([block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:64)), and — on every graded input so far — the tie behaviour: the strictly-greater insertion scan (earliest index wins ties, descending biased order, [block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:44)) gives `rows_other_experts` 0 at all six sizes, so its tie rule must be preserved exactly (a `>=` compare or a differently-ordered parallel top-4 would break it). It does not match: the product, accumulated as one thread's serial 768-term chain ([block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:31)), and the sigmoid `1.0f / (1.0f + metal::exp(-acc))` ([block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:35)). Measured under the current grader it therefore fails G2 at every size while the experts and their order always match: rows differing 20/159/638 at decode 1/8/32 (100% / 99.4% / 99.7% of 20/160/640 rows over 20 layers) and 2,518/16,187/32,743 at prefill 127/1,016/2,048 (99.1% / 79.7% / 79.9%), max relative error 2.6e-06 at decode and 8.7e-06 at prefill ([territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md:34)).

### What the grader checks, and what it can't see (changed)

- **G1** compiles the Metal and runs it on all six sizes' inputs; `g1` is set only after every size's checks have run ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:267)).
- **G2** checks, per layer on all 20 layers: exact shapes/dtypes ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:107)), `inds < 60` ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:117)), and ordered bitwise equality of `inds` and `weights` ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:120)); `g2` requires every size to pass ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:308)). Decode inputs are captured from all 20 layers of one real decode step per row count ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:137)); prefill inputs are recorded live by `capture_prefill` with fixed seeds ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:240)). The G2 target is the uncompiled `ref_routing` ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:103)); `mx.compile` changes only the timing bar, not the target.
- **Piece timing (changed to chained).** Timing runs at decode 1 / 8 / 32 rows only, on the 20 layers' chained calls — each call data-dependent on the previous one through the `link` kernel ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:178)) — median of 30 repeats per layer ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:204)), and `beats_compiled` requires beating `mx.compile(ref_routing)` at all three row counts ([check_candidate.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:335)). The slot stops at the first attempt passing G1 and G2 and beating the compiled reference at all three row counts, or after 5 attempts ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:201)).
- **What it can't see.** Row counts other than the six graded: any decode count from 1 to 32 ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:24)) — 2–7 and 9–31 rows are ungraded, on both sides of the GEMV/split-K boundary at `rows = 1` — and prefill chunks of other sizes from other prompt counts or chunkings, on both sides of the split-K/regular boundary at `rows = 768` and of the tile/partition changes inside split-K; a kernel that hard-codes the six sizes passes G2 and breaks this boundary. Inputs other than the captured ones: other prompts, later decode steps, other layers' drift, unseen near-tie or exactly-tied 4th/5th biased scores (no graded row is a near-tie, so the tie rule is exercised only where the starting point already agrees), and score magnitudes that saturate the sigmoid differently. Threadgroup memory near the 32 KB limit: `slots.build` enforces only the 1,024-thread cap ([slots.py](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:326)), so the 32 KB limit is unchecked by the grader. Reads of threadgroup memory that was never written, and any result that depends on thread timing or atomic order, which a single equality run per layer cannot expose. Prefill speed ([ADR 0010](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md:42)), the end-to-end greedy-token check (24 tokens, plain MLX and vLLM) and G3 full-step timing, which the team runs after merging ([ADR 0010](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md:32)). Boundary checks will probe exactly these.

### The GPU limits (unchanged)

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:98)); `grid` counts threads, not threadgroups. Hardware: Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` available in Metal ([muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:99)). As in round 1; only the grader's enforcement gap above is new information.

## Builders

### builder-2-a — direction (a): faithful MLX-regime replica

- Attempt `muse-spark/round-2/kernels/routing/attempt-1`: one fused kernel,
  256 threads/row, in-Metal branch on `rows` (GEMV shuffle replica at
  rows==1; split-K partitions + `simdgroup_float8x8` MMA, BK=16, partials
  in partition order at 1<rows<=768; BK=16 sequential MMA above), MLX
  verbatim sigmoid, starting point's strictly-greater top-4 and
  `((w0+w1)+w2)+w3 +1e-20` weights. Predicted 27/26/40 µs/layer at
  1/8/32 rows. **G1 pass, G2 pass at all six sizes (0 rows differ).**
  Chained µs/layer (reference / compiled / candidate): 1 row
  77.8 / 56.2 / 54.9; 8 rows 113.1 / 91.2 / 91.4; 32 rows
  136.3 / 98.9 / 114.8 — beats compiled only at 1 row, so
  `beats_compiled` false (no baseline). Taught: the 8x8 hardware MMA
  rounding is tile-position independent, so a per-row 8x8 emulation is
  bit-exact for the steel regimes; the remaining loss is staging cost
  (48x2 barriers/row, 7/8 tile waste, per-row w re-read), worst at
  32 rows. Next: same orders, fewer barriers, real rows in all tile rows.

## Boundary checks

*(none yet)*

## Decision

*(not yet)*
