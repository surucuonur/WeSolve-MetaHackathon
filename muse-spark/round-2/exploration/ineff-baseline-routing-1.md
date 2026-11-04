# Inefficiency: the router product's float32 accumulation order matches MLX at no row count

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:23:45Z
- **Next:** verifier to confirm the three dispatch regimes against MLX v0.32.4 `matmul.cpp`

## Slot

`routing`

## What happens

The starting point (`muse-spark/kernels/routing/attempt-1/block-0.metal:31-34`)
computes each of the 60 router scores as one thread's serial float32 chain,
`acc += xsh[d] * w_router[e*768 + d]` for d = 0..767 (compiler FMA contraction
does not change the left-to-right grouping). The reference,
`mx.sigmoid(x.astype(mx.float32) @ w_router.T)` (`bench/grader/slots.py:45`),
never accumulates that way. In MLX 0.32.4's `Matmul::eval_gpu`
(`mlx/backend/metal/matmul.cpp`, tag v0.32.4) with M=rows, N=60, K=768,
A not transposed, B transposed, the accumulation order is chosen by `rows`:

| rows | MLX kernel | float32 accumulation order |
|---|---|---|
| 1 | GEMV `gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4` (`min(M,N)==1` → `gemv`; `gemv_wide` skipped, f32 out) | 32 lanes × serial 24 products (lane l: d = i*128 + l*4 + 0..3, i=0..5, `kernels/gemv.h:162-182`), then shuffle tree sn=16,8,4,2,1 (`gemv.h:202-208`) |
| 8, 32, 127 | steel split-K GEMM (Case 1: `_tm*_tn <= 2048` on this `…s` device, `_tk=48 >= 8`, `K >= max(M,N)`) | 32 K-partitions (8/32 rows) or 8 (127 rows); each partition: simdgroup 8×8 MMA over BK=16 chunks (BM=16/32, BN=32); partials summed sequentially in partition order (`steel_gemm_splitk.h:186-189`); last partition absorbs the remainder iterations (`splitk.h:140-155`) |
| 1,016, 2,048 | regular steel GEMM (split-K off: 768 < M) | BM=64/BN=64/BK=16 tiles (`GEMM_TPARAM_MACRO` fallback for `…s`), K walked in BK=16 chunks, simdgroup MMA accumulation chunk after chunk |

A serial scalar loop cannot reproduce the two MMA regimes' rounding by
reordering: the 8-product accumulation inside each `simdgroup_float8x8`
multiply-accumulate is the hardware's. Bit-exactness therefore needs either
the same simdgroup matrix ops with the same tiling and K-order per regime
(branch on `rows`, or separate spec kernels), or a proof that another order
rounds identically (none known). Full derivation:
`territory-baseline-routing.md` section 3.1.

## Cost

This is a correctness gate, not a speed cost: it makes G2 fail at every size,
so the slot's fused-kernel saving can never be banked. Measured
(`results/runs/2026-11-04-muse-round-1/grade-exact-routing.json`): rows
differing 20/159/638 at decode 1/8/32 rows (100%/99.4%/99.7% of the
20/160/640 rows over 20 layers) and 2,518/16,187/32,743 at prefill
127/1,016/2,048 (99.1%/79.7%/79.9% of 2,540/20,320/40,960); experts and
order always match (`rows_other_experts` 0); max relative error 2.6e-06
decode, 8.7e-06 prefill. In time terms the blocked amount is the whole
fused-vs-compiled headroom below.

## Kind

Overhead-bound slot blocked by a correctness (rounding-order) cause; in the
terms of `claude/GLOSSARY.md` the slot itself is overhead-bound (1.02 ms/step
at 1 row for ~4 MB, `ineff-runs-2` confirmed).

## Spans

`baseline-routing`; feeds `baseline-experts` and `baseline-combine` (both
consume this slot's outputs in the model).

## Predicted saving

Removing this cause (an exact product at the starting point's speed) unlocks
the fused kernel's measured gap to the compiled reference. Independent piece
timing (Claude re-grade, `claude/findings/muse-round-1.md:47`): candidate
25.3/24.3/68.9 vs compiled 31.5/42.7/87.6 µs/layer → 6.2/18.4/18.7 µs/layer ×
20 layers = **0.12 / 0.37 / 0.37 ms per step at 1/8/32 rows**. Caveats, both
measured: the round-2 bar is the *chained* timing, for which no routing
figure exists yet (round-2 README:36), and end-to-end the starting point is
~0.41 ms/step *slower* than the compiled reference at 1 request (G3:
+0.185 vs −0.220 ms, `muse-round-1.md:85-86`), so at 1 row exactness alone
banks nothing — it must arrive with the occupancy fix in
`ineff-baseline-routing-3`. The exact product itself may cost more than the
serial loop (MMA tiling at decode sizes has little work to amortise); the
saving above is the headroom it makes claimable, not a promise.

## Verdict

verifier, 2026-11-04T03:23:45Z — **confirmed**. Checked: serial chain at [block-0.metal:31-34](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:31) (`acc += xsh[d]·w_router[e·768+d]`, d=0..767, left-to-right grouping) vs reference `mx.sigmoid(x.astype(f32) @ w_router.T)` ([slots.py:45](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:45)) ✓. Three regimes in MLX v0.32.4 `Matmul::eval_gpu` (M=rows, N=60, K=768, A not transposed, B transposed; fetched `matmul.cpp` tag v0.32.4 + installed headers, derivation in [territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md) §3.1): rows=1 → GEMV (`min(M,N)==1` → `gemv`, `gemv_wide` skipped for f32 out), kernel `gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4` — 32 lanes × 24 products (lane l: d=i·128+l·4+0..3, i=0..5, [gemv.h:162-182](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/gemv.h:162)) then shuffle tree sn=16,8,4,2,1 ([gemv.h:202-208](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/gemv.h:202)) ✓; rows 8/32/127 → steel split-K Case 1 (`_tm·_tn≤2048` on `…s`, `_tk=48≥8`, `K≥max(M,N)`), partitions `min(max(2,next_pow2(_tk/(_tm·_tn))),32)` = 32 at 8/32, 8 at 127, BK=16 MMA chunks, partials summed sequentially ([steel_gemm_splitk.h:186-189](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h:186)), last partition absorbs remainder (:140-155) ✓; rows 1,016/2,048 → regular steel GEMM (768<M, split-K off), BM=64/BN=64/BK=16 fallback ✓. Serial reordering cannot reproduce the MMA 8-product hardware accumulation — bit-exactness needs the same ops/tiling per regime, as written ✓. Differ in [grade-exact-routing.json](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/grade-exact-routing.json): 20/159/638 decode (100%/99.4%/99.7% of 20/160/640), 2,518/16,187/32,743 prefill (99.1%/79.7%/79.9% of 2,540/20,320/40,960), `rows_other_experts` 0 everywhere, max rel err 2.6e-06 decode / 8.7e-06 prefill ✓. Headroom (31.5−25.3)/(42.7−24.3)/(87.6−68.9)=6.2/18.4/18.7 µs/layer ×20 = 0.12/0.37/0.37 ms/step ([muse-round-1.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/findings/muse-round-1.md) §1) ✓, with chained-timing and 1-row G3 (+0.185 vs −0.220 ms, ≈0.41 ms/step slower than compiled) caveats as written — exactness alone banks nothing at 1 row without [ineff-baseline-routing-3](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-routing-3.md).

## From other agents

- builder-2-a, 2026-11-04T03:37:08Z: round-2 routing attempt-1
  (`muse-spark/round-2/kernels/routing/attempt-1`) passes G2 at all six
  sizes with a per-row 8x8 `simdgroup_float8x8` emulation (not steel's
  BM/BN tiles): the hardware 8-product rounding is tile-position
  independent, so matching K chunk (BK=16, two 8-wide ops), split-K
  partition and sequential-sum order suffices bit for bit. Chained cost
  is staging-bound (96 barriers/row, 7/8 tile waste): 54.9/91.4/114.8
  vs compiled 56.2/91.2/98.9 µs/layer — exactness is solved; speed at
  8/32 rows is now a tiling/barrier problem, not a rounding problem.
