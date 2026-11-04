# Inefficiency: routing+combine bookkeeping dominates the 1-row step

- **Owner:** explorer-findings
- **Status:** superseded by ineff-runs-2 and ineff-model-2
- **Version:** 3
- **Updated:** 2026-11-03T21:27:27Z
- **Next:** settled by lead: joint routing+combine view; per-slot entries carry the causes to hypotheses

## Slot

`routing` + `combine` (the brief's `routing+combine` slot).

## What happens

Routing (cast, router product, sigmoid, bias add, negate, argpartition, gather,
weight normalisation) launches 240 kernels per step and combine (weight, sum, cast
back, add shared) 100 more — 340 of the step's 867 kernels at 1 row
(`results/runs/2026-11-01-breakdown/tables.md`, "Each piece at 1 row", lines
358-375; op mix in "Operations per step by type", lines 149-159). The router
product is the only real math; the rest is selection and weighting bookkeeping
(`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`, as split in
step-breakdown "How the step was cut into pieces").

## Cost

Plain MLX, knock-out savings (tables.md "What removing a piece saves", lines
320-356) over bare step periods (4.75 / 13.37 / 28.19 ms):

- **1 row:** 340 kernels. Routing 1.02 ms / 240 = 4.3 µs per kernel; combine
  0.61 ms / 100 = 6.1 µs per kernel. Bytes: routing must read 3.7 MB/step, so
  3.7 MB / 200 GB/s = 19 µs; combine reads activations only. Measured total:
  **1.63 ms of the 4.75 ms step (34%)**.
- **8 rows:** knock-outs did not run at 8 rows (step-breakdown: knock-outs at 1
  and 32 rows only), and the kernel count is identical to 1 row (867, same op
  mix). Estimate by linear interpolation in rows: routing
  1.02 + (1.68 − 1.02) × 7/31 = 1.17 ms; combine
  0.61 + (0.33 − 0.61) × 7/31 = 0.55 ms; total **≈1.72 ms of 13.37 ms (13%)**.
- **32 rows:** routing 1.68 ms / 240 = 7.0 µs per kernel; combine 0.33 ms / 100 =
  3.3 µs per kernel. Measured total: **2.01 ms of the 28.19 ms step (7%)**.

## Kind

Overhead-bound (glossary: far below both limits). Routing moves 2-4 GB/s; combine
reads no weights. The cost is 340 fixed kernel costs plus their CPU encode.

## Spans

`findings` (measured costs), `model` (MoE block code), `quantized-kernels` (the
router product kernel). Overlaps ineff-runs-2 (routing) and ineff-model-2
(combine); kept as one entry because the brief's `routing+combine` candidate
builds both jointly — lead to settle (notes appended on both).

## Predicted saving

If each slot fuses to 1 kernel per layer (20 + 20 = 40 kernels vs 340), saving
300 fixed costs per step:

- **1 row:** routing (240 − 20) × 4.3 = 946 µs; combine (100 − 20) × 6.1 =
  488 µs; total **≈1420 µs per step**.
- **8 rows (estimate):** routing 220 × (1.17 ms / 240 = 4.9 µs) = 1078 µs;
  combine 80 × (0.55 ms / 100 = 5.5 µs) = 440 µs; total **≈1500 µs per step**.
- **32 rows:** routing 220 × 7.0 = 1540 µs; combine 80 × 3.3 = 264 µs; total
  **≈1800 µs per step**.

Supporting evidence: compiling the whole 20-layer MoE stack as one function cut
the block 6.02 → 4.27 ms (−29%) at 1 row in the chained test
(decode-bottlenecks §3) — a different measurement (synchronous, includes
build+encode), so an existence proof, not a prediction. Caveats: knock-out
savings include CPU encode, which fusion also cuts (favourable); but a faster
version gains less than the full saving (step-breakdown Limits).

Folded from explorer-vllm-metal's note (ineff-vllm-metal-1, now superseded by
this entry): extending vllm-metal's `CompiledMLPBlocks` dispatch to MobileMoE is
an alternative fix for this same time — it currently never wraps (gate default
off, `_target_policies` lists only Qwen3Next/Qwen3.5, `MAX_COMPILED_TOKENS = 4`;
`compiled_mlp.py:49,115-131`) — but it is roughly G3's compiled-reference
baseline, so a slot kernel must beat it, not add to it. Their glue-only 1-row
figure (≈1.0 ms) agrees with the ≈1.42 ms full-fusion figure within the
matmul+casts that stay.

## Verdict

(not yet written — the verifier writes this)

## From other agents

explorer-vllm-metal, 2026-11-03T21:45Z — same cause from the engine side; I
marked my ineff-vllm-metal-1 superseded by this entry. Engine facts for the
fix direction: vllm-metal already has a whole-block `mx.compile` dispatch,
`CompiledMLPBlocks` (`third_party/vllm-metal/vllm_metal/compiled_mlp.py`),
installed after load (`v1/model_lifecycle.py:228-239`), but (a) it is gated
by `VLLM_METAL_COMPILED_MLP`, default off (`envs.py:121`); (b)
`_target_policies` lists only Qwen3Next/Qwen3.5 block types
(`compiled_mlp.py:115-131`), so MobileMoE-S's MoE block is never wrapped even
with the flag on; (c) `MAX_COMPILED_TOKENS = 4` (`compiled_mlp.py:49`) routes
8/32-row steps eager anyway. So extending the engine's compile dispatch to
MobileMoE is an alternative fix for this same time — but note it is roughly
G3's compiled-reference baseline, and any fused routing/combine slot kernel
subsumes it. My 1-row saving arithmetic for the glue alone (≈1.0 ms/step:
routing ≈0.65 + combine ≈0.37) agrees with yours (≈1.42 ms for full fusion
to 1 kernel/layer) within the matmul+Casts that stay.
