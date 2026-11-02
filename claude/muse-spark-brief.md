# Brief for Muse Spark: write a faster MobileMoE-S decode kernel

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur, and committed before Muse Spark's first call. [Who made what](../PROVENANCE.md).*

This brief is sent to Muse Spark (`muse-spark-1.3-contributor`) at the start of every call. It fixes the task, what can be changed, the answer format and how results are judged. None of it changes after the first call. The harness that applies it is in [bench/muse/](../bench/muse/).

## The task

Meta's MobileMoE-S, int4, is served on an Apple M1 Pro (16-core GPU, 200 GB/s, 16 GB) by vLLM through vllm-metal, which runs the model with MLX. The attached findings measure where each decode step's time goes. A step is about 870–1,090 small GPU kernels and is far from both the bandwidth and compute limits. The findings name the pieces that cost the most.

You will:

1. **Stage 1, choose.** Read the findings and choose one or two of the slots below to replace with your own fused Metal kernels. Explain why, and estimate what each saves per step at 1 and 32 requests.
2. **Stage 2, write.** For each chosen slot, write the kernels. The harness compiles them, checks them against the original on real inputs, times them, and sends back only what it measured. You get up to **5 attempts per slot**.

## The slots

Each slot is a piece of the mixture-of-experts block, `MoE.__call__` in `mobilemoe.py`, which runs in all 20 layers of every step. `rows` is the number of tokens in the step: 1 to 32 in decode. The harness flattens the inputs to the shapes below and restores the model's shapes afterwards. All arrays are row-contiguous.

The model's constants: hidden size `D = 768`, experts `E = 60`, experts per token `K = 4`, expert hidden size `H = 384`, `routed_scaling_factor = 2.5`, `norm_topk_prob = true`.

### `routing`: router, top-4 selection and weights

| | Name | Shape | Dtype |
|---|---|---|---|
| input | `x` | [rows, 768] | bfloat16 |
| input | `w_router` | [60, 768] | float32 |
| input | `bias` | [60] | float32 (the model's bfloat16 `expert_bias`, converted exactly) |
| output | `inds` | [rows, 4] | uint32 |
| output | `weights` | [rows, 4] | float32 |

Reference:

```python
scores = mx.sigmoid(x.astype(mx.float32) @ w_router.T)          # [rows, 60]
inds = mx.argpartition(-(scores + bias), kth=3, axis=-1)[:, :4]  # the 4 largest biased scores
w = mx.take_along_axis(scores, inds, axis=-1)                    # unbiased scores of those 4
weights = w / (w.sum(axis=-1, keepdims=True) + 1e-20) * 2.5
```

The order of the 4 experts within a row is free, as long as `weights[r, j]` belongs to `inds[r, j]`. Today this slot is about 12 kernels per layer, 240 per step. It costs 1.02 ms of a 4.75 ms step at 1 request and 1.68 ms at 32 requests.

### `combine`: weighting the routed experts' outputs and adding the shared expert

| | Name | Shape | Dtype |
|---|---|---|---|
| input | `y` | [rows, 4, 768] | bfloat16 (the 4 experts' outputs, in `inds` order) |
| input | `weights` | [rows, 4] | float32 |
| input | `shared` | [rows, 768] | bfloat16 (the shared expert's output) |
| output | `out` | [rows, 768] | bfloat16 |

Reference:

```python
out = (y * weights[..., None]).sum(axis=-2).astype(mx.bfloat16) + shared
```

Today this slot is 5 kernels per layer, 100 per step. It costs 0.61 ms at 1 request and 0.33 ms at 32 requests.

### `routing+combine`

Both slots above, each with its own kernels, in one candidate.

### `experts`: the routed experts' three quantized products

| | Name | Shape | Dtype |
|---|---|---|---|
| input | `x` | [rows, 768] | bfloat16 |
| input | `inds` | [rows, 4] | uint32 |
| input | `gate_w`, `up_w` | [60, 768, 48] | uint32 (4-bit weights in [expert, in, out] layout, 8 per word along out) |
| input | `gate_scales`, `gate_biases`, `up_scales`, `up_biases` | [60, 768, 12] | bfloat16 (one scale and bias per 32 outputs) |
| input | `down_w` | [60, 384, 96] | uint32 |
| input | `down_scales`, `down_biases` | [60, 384, 24] | bfloat16 |
| output | `y` | [rows, 4, 768] | bfloat16 |

Each weight is `w = scale * q + bias`, with `q` the 4-bit value. In MLX's affine quantization, the value for output column `o` sits in word `o / 8` at bits `4 * (o % 8)`.

Reference: for each row `r` and each `j`, with `e = inds[r, j]`:

```python
y[r, j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]
```

As in MLX's `gather_qmm(..., transpose=False)`, each product accumulates in float32 and rounds its output to bfloat16. `silu(gate) * up` is computed on those bfloat16 values, and its result is rounded to bfloat16 before the down product.

Today this slot is 7 kernels per layer at 1 request and 13 at 32 requests (from 16 rows, MLX also sorts the pairs by expert). It costs 0.92 ms at 1 request and 14.67 ms of a 28.19 ms step at 32 requests, half the step. It reads each expert once per (token, expert) pair.

## Tools and limits

- **Only `mx.fast.metal_kernel`, MLX 0.32.4.** You write the Metal body; MLX generates the signature. Inputs arrive as `const device T* name` and outputs as `device T* name`. Shapes and strides come as `name_shape` and `name_strides` when the source uses them. The thread attributes are available, as in MLX's documentation: `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`, `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup`, and so on. The full documentation string is attached.
- **No other code runs.** The harness runs no Python from your answer. It reads your launch spec as data.
- **Stay within the GPU's limits.** At most 1,024 threads per threadgroup and 32 KB of threadgroup memory. Each attempt runs in its own process with a 180-second limit.
- **Hardware.** The Apple M1 Pro GPU (`applegpu_g13s`) has 16 cores and a SIMD width of 32. bfloat16 is available in Metal as `bfloat`.

## Answer format

**Stage 1.** Explain your reasoning freely. End with exactly one ` ```json ` block:

```json
{
  "targets": [
    {"slot": "routing+combine", "why": "...", "expected_saving_ms": {"rows_1": 0.0, "rows_32": 0.0}, "design": "..."}
  ]
}
```

List one or two targets, in the order you want to work on them. `slot` must be one of `routing`, `combine`, `routing+combine`, `experts`.

**Stage 2.** Explain freely. Then give:

- one ` ```metal ` block per kernel body, and one per optional header, numbered in order of appearance from 0;
- exactly one ` ```json ` launch spec:

```json
{
  "kernels": [
    {
      "name": "fused_routing",
      "source_block": 0,
      "header_block": null,
      "inputs": ["x", "w_router", "bias"],
      "outputs": [
        {"name": "inds", "shape": ["rows", 4], "dtype": "uint32"},
        {"name": "weights", "shape": ["rows", 4], "dtype": "float32"}
      ],
      "grid": ["rows*32", 1, 1],
      "threadgroup": [32, 1, 1],
      "template": {"T": "bfloat16"},
      "init_value": null,
      "atomic_outputs": false
    }
  ],
  "slot_outputs": {"inds": "inds", "weights": "weights"}
}
```

- **Kernels run in list order.** A kernel's `inputs` may name slot inputs or outputs of earlier kernels.
- **`slot_outputs`** maps each output of the slot to the kernel output that provides it.
- **For `routing+combine`,** give one JSON object with two specs, each in the format above for its own slot: `{"routing": {"kernels": [...], "slot_outputs": {...}}, "combine": {"kernels": [...], "slot_outputs": {...}}}`. The harness runs the original experts and shared expert between the two parts. It tests each part on its own, and the combine part receives the reference routing weights.
- **Shape and grid entries** are integers, or products of integers and the names `rows`, `D`, `E`, `K` and `H`, such as `"rows*32"`. Nothing else is evaluated.
- **dtypes** are `float32`, `bfloat16`, `float16`, `uint32` or `int32`.
- **Template values** are one of those dtypes, an integer or a boolean.
- **`grid`** counts threads, not threadgroups, as in `mx.fast.metal_kernel`.

## How each attempt is judged

The pass criteria were fixed before the first call, as written here.

1. **G1, runs.** It compiles and runs on real inputs at 1, 8 and 32 rows.
2. **G2, correct.** It is tested on inputs captured from all 20 layers during a real decode step, at 1, 8 and 32 rows.
   - **`routing`:** each row selects the same 4 experts. The only exception is a row whose 4th and 5th biased scores in the reference are within 1e-6 of each other; such rows are counted and reported. The weights, matched by expert, are within 1e-4 relative.
   - **`combine` and `experts`:** every element is within 2 bfloat16 units in the last place of the reference value. For values smaller than 0.001 in magnitude, the unit at 0.001 applies, so summing in a different order near zero is not a failure.
   - **Then, end to end:** the same greedy tokens as the original model over 24 generated tokens, at 1 and 32 requests, in both plain MLX and vLLM.
3. **G3, faster.** The decode step's period is the median over 3 repeats in shuffled order, with 128-token prompts, at 1, 8 and 32 requests. It must be lower than both the original model's and an `mx.compile` of the same reference code. The gap must exceed the spread between repeats.

**What the harness sends back after each attempt:**
- the compiler's error message;
- or the correctness failures, with counts and the worst examples;
- or, when the attempt is correct, its time per call next to the original's and the compiled reference's.

The piece timing runs the slot on all 20 layers' captured inputs and waits for the GPU once at the end. It takes the median of 30 repeats, per layer. It is a guide; G3 decides.

**When a slot stops.** Work on a slot stops at the first attempt that passes G1 and G2 and beats the compiled reference in the piece timing at all three row counts, or after 5 attempts. The best passing attempt, by piece timing, goes to the end-to-end and G3 checks.
