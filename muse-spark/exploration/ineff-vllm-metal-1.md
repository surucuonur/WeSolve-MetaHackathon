- **Owner:** explorer-vllm-metal
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:35Z
- **Next:** verifier to confirm; note overlap with routing/combine slot work

## Slot

routing + combine (engine-side: the fix is dispatch policy, but any
routing/combine slot kernel subsumes it — see Spans).

## What happens

`CompiledMLPBlocks` fuses a whole MoE/MLP block's elementwise glue with
`mx.compile` on decode-shaped calls (`third_party/vllm-metal/vllm_metal/compiled_mlp.py:174-196`).
But `_target_policies` lists only Qwen3Next/Qwen3.5 block types
(`compiled_mlp.py:115-131`): MobileMoE-S's MoE block is never wrapped, even
with `VLLM_METAL_COMPILED_MLP=1` (default off, `envs.py:121`). And
`MAX_COMPILED_TOKENS = 4` (`compiled_mlp.py:49`) would route 8/32-row steps
eager anyway. So on the benchmark model every layer runs routing (~12 kernels:
matmul, sigmoid, bias add, argpartition, take, sum, div, scale) and combine
(~5 kernels: mul, sum, casts, add shared) fully eager.

## Cost

Kernels per step: routing 240, combine 100 (both engines, 1 and 32 rows).
Fixed cost per dependent kernel ~4-7 µs at 1 row. Knock-out savings
(step-breakdown §3, plain MLX — vLLM runs the same kernels):

- 1 row: routing 1.02 ms + combine 0.61 ms = 1.63 ms of a 4.75 ms step.
- 8 rows: no knock-out measured; step is 13.37 ms (step-breakdown §1).
- 32 rows: routing 1.68 ms + combine 0.33 ms = 2.01 ms of a 28.19 ms step.

Bytes ÷ peak bandwidth is not the cost here: routing reads 3.7 MB over 240
kernels (4 GB/s, 2% of peak) — pure launch overhead.

## Kind

Overhead-bound (glossary): far below bandwidth and compute limits; time goes
to launching hundreds of tiny dependent kernels.

## Spans

vllm-metal (dispatch policy) + model (the MoE block source) + the routing and
combine slots (a fused slot kernel removes the same time; whoever builds
first subsumes the other — lead to settle).

## Predicted saving

If the MoE block compiled like the Qwen blocks, each layer's routing glue
(~10 kernels) fuses to ~2-3 and combine (~4 glue) to ~1-2; the router matmul
and the expert products stay. At 1 row, routing glue ≈ 200 kernels × 4.3 µs
≈ 0.86 ms → fused ≈ 50 × 4.3 ≈ 0.22 ms, saving ≈ 0.65 ms; combine glue ≈ 80
× 6.1 µs ≈ 0.49 ms → fused ≈ 20 × 6.1 ≈ 0.12 ms, saving ≈ 0.37 ms.

- 1 row: ≈ 1.0 ms per step (0.65 + 0.37, estimate).
- 8 rows: ≈ 0.9 ms per step (interpolated between 1 and 32 rows, estimate).
- 32 rows: ≈ 0.7 ms per step (routing 200 glue × 7 µs ≈ 1.4 ms → fused ≈ 0.4,
  saving ≈ 1.0? No: GPU-bound at 32 rows, fusion removes launches but the
  chain overlaps less; halve the 1-row-style figure → ≈ 0.7 ms, estimate).

Caveat: `mx.compile` of the reference is G3's baseline, so a slot kernel must
beat this saving, not add to it.

## Verdict

(verifier writes)
