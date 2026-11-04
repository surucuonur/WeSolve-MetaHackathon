# Inefficiency: starting point matches `gather_qvm` but prefill runs `gather_qmm_rhs` — G2 fails at every prefill size

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:22:51Z
- **Next:** verifier to confirm dispatch threshold and the two rounding differences

## Slot

`experts`

## What happens

Attempt-4's `qvm_block` (`muse-spark/kernels/experts/attempt-4/block-1.metal:20-40`)
is a bit-faithful replica of MLX `affine_gather_qvm`/`qvm_impl`: each lane
accumulates its strided `K` elements serially in float32, dequantizing on
the fly as `x * (scale * q + bias)` (`block-1.metal:33-34`), then a
`simd_sum` tree reduces across the 32 lanes. That is exactly the kernel
MLX dispatches at decode sizes, so decode G2 is bit-exact.

From 60 rows the reference dispatches a different kernel. Dispatch
(MLX v0.32.4 `quantized.cpp:2005,2012,2073-2093`, via round 1
`muse-spark/exploration/territory-quantized-kernels.md` §A/§C and
`territory-switch-layers.md` §G): `gather_qmm_rhs` needs
`M==1 && B>=16 && right_sorted_ && B/E>=4`, `B = rows*4`, `E = 60`, i.e.
rows ≥ 60. Decode 1/8/32 rows (`B/E` ≤ 2.13) never fire it; prefill
127/1,016/2,048 rows (`B/E` = 8.5/67.7/136.5) always do. Note the two
distinct thresholds: the reference sorts from `inds.size >= 64` (rows ≥
16, `bench/grader/slots.py:62`, `third_party/mlx-lm/mlx_lm/models/switch_layers.py:181`),
but sorting alone does not switch the kernel below 60 rows — decode at
32 rows is sorted and still `gather_qvm`.

`affine_gather_qmm_rhs` (`quantized.h:2580-2718`, installed 0.32.4
headers) rounds differently in two independent ways, both new readings
this round (round 1 mapped the kernel as never-run and did not read its
arithmetic):

1. **Dequantized weight rounded to bfloat16 before the multiply.**
   `QuantizedBlockLoader` (`quantized.h:674-791`) dequantizes into
   threadgroup memory via `dequantize<T, pack_factor, bits>`
   (`quantized.h:557-663`; bits=4 path `:614-620`):
   `w_local = static_cast<U>(s * q + b)` with `U = T` = bfloat16, stored
   in `Ws` (`quantized.h:2623-2624`). `qvm` instead keeps `scale*q+bias`
   in float32 registers through `qouter` (`quantized.h:452-553`).
2. **Different accumulation order.** `gather_qmm_rhs` tiles `K` in
   `BK=32` blocks through `gemm_loop_aligned` and simdgroup-matrix
   fragments (`BlockMMA`, `AccumType=float`, `steel/gemm/mma.h:451,512-538`;
   tile mapping by `schedule_row_tile<BM=16>`, `utils.h:501-542`), vs
   `qvm`'s strided-lane serial sum + `simd_sum` tree
   (`quantized.h:1317-1321`). Round 1 proved the order is load-bearing:
   down outputs near cancellation amplify a 1-ULP hidden difference
   (`muse-spark/implementation/hyp-1-fused-experts.md:103-108`), so
   either difference alone breaks bit-exactness.

The silu lowering is not the cause: both paths feed the same compiled
`swiglu` (full-bfloat16 per-op rounding, attempt-4 `block-0.metal:53-64`),
but with differently-rounded gate/up inputs, so hidden and `y` diverge.
Attempt-4 has no size branch: one kernel, grid `rows*4` groups at every
row count (`spec.json`; territory §C), always computing `qvm` arithmetic.

## Cost

Correctness, not time. Under the new grader
(`results/runs/2026-11-04-muse-round-1/grade-exact-experts.json`, summed
over 20 layers; totals = rows×4×768×20):

- decode 1/8/32 rows: differ 0/0/0, max 0.0 ULP — pass.
- prefill 127 rows: differ 4,868,667 of 7,802,880 (62.4%), max 43.0 ULP.
- prefill 1,016 rows: differ 39,078,912 of 62,423,040 (62.6%), max 65.0 ULP.
- prefill 2,048 rows: differ 78,741,557 of 125,829,120 (62.6%), max 110.5 ULP.

Kernel/byte accounting at the prefill sizes (per layer, per-pair traffic
552,960 B/pair, territory §A): 508/4,064/8,192 pairs read
281 MB / 2.25 GB / 4.53 GB per layer per call through the per-pair grid,
where the reference's grouped `gather_qmm_rhs` reads each appearing
expert's tile once per `BM=16`-row tile. Prefill speed is not gated
(G3 times decode only, `claude/adr/0010-exact-at-every-size.md:42`), so
this traffic is not itself the loss — the loss is that G2 fails, so no
chained-timing win at any size can be banked (`check_candidate.py:308-337`
times only when G2 passes).

At decode 1/8/32 rows this entry costs 0 kernels and 0 bytes: the
starting point already matches. The reference at those sizes launches
7/7/13 kernels/layer (round 1 `territory-switch-layers.md` §A); the
candidate launches 1.

## Kind

Overhead-bound framing does not apply: far below both limits, the cost
is a correctness gate. In `claude/GLOSSARY.md` terms the slot at decode
sizes is overhead-bound; this entry removes no overhead — it unblocks
the slot.

## Spans

`baseline-experts` (attempt-4 `block-1.metal`), `quantized-kernels`
(`affine_gather_qvm` vs `affine_gather_qmm_rhs` arithmetic),
`switch-layers` (sort flag sets `right_sorted_`), `grader` (six-size
bit-exact G2).

## Predicted saving

0 µs per decode step at 1/8/32 rows — removing this inefficiency saves
no time; it makes the slot's existing chained wins countable
(175.9 vs 239.6 µs/layer at 8 rows, 587.6 vs 797.6 at 32 rows,
`console-chain.txt`). Against the compiled reference the same holds:
the old-grader compiled times (139.5/221.4/762.4 µs/layer independent,
`grade-experts.json`) are not the round-2 bar; the round-2 bar is the
compiled reference under chained timing (not yet published for this
slot), and no attempt reaches that comparison until G2 passes at all
six sizes.

Fix shape implied by the sources (for builders, not costed here): a
size branch at rows ≥ 60 (pairs ≥ 240) replicating `gather_qmm_rhs`
arithmetic — bfloat16-rounded dequantized tiles + `BK=32` MMA-order
accumulation — or a single arithmetic provably equal to both MLX
kernels. The launch-spec surface allows only an in-Metal branch or a
second listed kernel with early return (territory §C).

## Verdict

verifier, 2026-11-04T03:22:51Z — **confirmed**. Checked: `qvm_block` replica at [block-1.metal:20-40](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/block-1.metal:20) (strided `k=lane; k<K; k+=32`, dequant `x·(s·q+bias)` at :33-34, `simd_sum` at :37-40) matches `qvm_impl`/`qouter` ✓. Dispatch: `gather_qmm_rhs` needs `M==1 && B>=16 && right_sorted_ && B/E>=4` with `B=rows·4, E=60` ⇔ rows≥60 (round-1 [territory-quantized-kernels.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/territory-quantized-kernels.md) §A/§C, [territory-switch-layers.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/territory-switch-layers.md) §G; MLX v0.32.4 `quantized.cpp:2005,2012,2073-2093`) — decode B/E 0.07/0.53/2.13 never fires, prefill 8.47/67.73/136.53 always does ✓. Sort threshold `inds.size>=64` ⇔ rows≥16 ([slots.py:62](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:62), `switch_layers.py:181`) is distinct and does not switch the kernel at 32 rows ✓. Prefill rounding (installed 0.32.4 headers): `affine_gather_qmm_rhs` ([quantized.h:2580](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:2580)) dequantizes to bfloat16 via `QuantizedBlockLoader` ([quantized.h:674](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:674)) → `dequantize` `static_cast<U>(s·q+b)`, `U=T` ([quantized.h:557](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:557), bits=4 path) stored in `Ws` ([quantized.h:2624](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:2624)), vs `qvm` float32 through `qouter` ([quantized.h:454](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:454)) ✓; accumulation `BK=32` `gemm_loop_aligned` + `BlockMMA` `AccumType=float` ([mma.h](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/mma.h), `schedule_row_tile<BM=16>` [utils.h](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/utils.h)) vs strided-lane + `simd_sum` ([quantized.h:1320](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:1320)) ✓ — either difference breaks exactness (order is load-bearing, [hyp-1-fused-experts.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-1-fused-experts.md)). Silu at [block-0.metal:53-64](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/block-0.metal:53) is the shared compiled `swiglu`, not the cause ✓; no size branch, grid `rows·4` groups ([spec.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/spec.json)) ✓. Differ in [grade-exact-experts.json](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/grade-exact-experts.json): decode 0/0/0 pass; prefill 4,868,667/7,802,880=62.4% (43.0), 39,078,912/62,423,040=62.6% (65.0), 78,741,557/125,829,120=62.6% (110.5) ✓. Per-pair 552,960 B ×508/4,064/8,192 = 281 MB/2.25 GB/4.53 GB per layer ✓; prefill speed untimed (G3 decode only, [0010-exact-at-every-size.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md)), timing gated on G2 ([check_candidate.py:309](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:309)) ✓. Predicted saving 0 µs — a correctness gate, as written.

## From other agents

(none yet)
