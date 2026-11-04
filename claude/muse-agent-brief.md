# Brief for the Muse Spark agent: write faster MobileMoE-S kernels, for decoding and prompt processing

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Committed and tagged `agent-era` before the Muse Spark session started. [Who made what](../PROVENANCE.md).*

You are a Claude Code session powered by Meta's Muse Spark, working in your own git worktree on the branch `muse-agent`. This brief fixes your task, what you may change, where your work goes and how it is judged. It does not change during your session.

## The task

Meta's MobileMoE-S, int4, is served on an Apple M1 Pro (16-core GPU, 200 GB/s, 16 GB) by vLLM through vllm-metal, which runs the model with MLX. The findings below measure where each decode step's time goes. A step is about 870–1,090 small GPU kernels and is far from both the bandwidth and compute limits. The findings name the pieces that cost the most.

A request goes through two phases: **prompt processing** (the prefill), which runs its whole prompt through the model, and **decoding**, one step per generated token. An improvement must hold in both: faster in one, and no slower in the other ([ADR 0011](adr/0011-decode-and-prefill.md)). A kernel that speeds up the decode step but slows the prefill is not an improvement.

1. **Choose.** Read the findings and choose one or two of the slots below to replace with your own fused Metal kernels. Write `muse-spark/strategy.md`: why, your design, and what each saves per step at 1 and 32 requests (your estimate). End it with the JSON block described under *Your files*. Commit it.
2. **Write and check.** For each chosen slot, write the kernels as files, run the grader on them, read its verdict, and improve. Up to **5 attempts per slot**; an attempt is a folder you graded and committed.
3. **Finish.** Write `muse-spark/summary.md`: your final attempt per slot and what you learned. Commit it.

**Read these first** (all in your worktree):
- `claude/findings/bottleneck-explained.md` and `claude/findings/step-breakdown.md`: the measured findings;
- `results/runs/2026-11-01-breakdown/tables.md` and `results/runs/2026-11-02-gpu-counters/tables.md`: the raw tables;
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py` and `switch_layers.py`: the model code;
- MLX's custom-kernel documentation: `.venv/bin/python -c "import mlx.core as mx; help(mx.fast.metal_kernel)"`;
- `bench/grader/check_candidate.py` and `bench/grader/slots.py`: how you are graded.

## The slots

Each slot is a piece of the mixture-of-experts block, `MoE.__call__` in `mobilemoe.py`, which runs in all 20 layers of every step. `rows` is the number of tokens in the call: 1 to 32 in a decode step, up to 2,048 in a prefill chunk. The grader flattens the inputs to the shapes below and restores the model's shapes afterwards. All arrays are row-contiguous.

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

- **Only `mx.fast.metal_kernel`, MLX 0.32.4.** You write the Metal body; MLX generates the signature. Inputs arrive as `const device T* name` and outputs as `device T* name`. Shapes and strides come as `name_shape` and `name_strides` when the source uses them. The thread attributes are available, as in MLX's documentation: `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`, `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup`, and so on.
- **The grader runs no Python of yours.** It reads your launch spec as data and compiles only your Metal.
- **Stay within the GPU's limits.** At most 1,024 threads per threadgroup and 32 KB of threadgroup memory. A grader run should take under 3 minutes, longer for an attempt much slower than the reference at the prefill sizes; stop one that runs past 15.
- **Hardware.** The Apple M1 Pro GPU (`applegpu_g13s`) has 16 cores and a SIMD width of 32. bfloat16 is available in Metal as `bfloat`.

## Rules of this session

These keep your work apart from everyone else's. Git hooks and an audit script enforce them, and a commit that breaks them is refused.

- **Write only inside `muse-spark/`.** Never change `bench/`, `claude/`, `results/`, `tools/`, `third_party/`, or any other file. A copy of the grader you edit is never used.
- **Commit only on `muse-agent`,** with the co-author trailer your settings add: `Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>`. Start each subject with `Muse Spark:`. No other co-author.
- **Never push, merge, rebase, change git config, or skip hooks.**
- **Don't read `.env`** or any credentials file.
- One commit per attempt, after you've run the grader on it, so the history shows every attempt in order.

## Your files

```
muse-spark/
  strategy.md                                your choice and design, ending with the targets JSON
  kernels/<slot-folder>/attempt-<N>/
    block-0.metal, block-1.metal, ...        Metal kernel bodies (and headers), numbered from 0
    spec.json                                the launch spec
    notes.md                                 optional: what you tried and why
  summary.md                                 your final attempt per slot
```

`<slot-folder>` is `routing`, `combine`, `routing-combine` or `experts`. Grade an attempt with:

```bash
.venv/bin/python bench/grader/check_candidate.py muse-spark/kernels/<slot-folder>/attempt-<N> <slot>
```

where `<slot>` is `routing`, `combine`, `routing+combine` or `experts`.

**The targets JSON** at the end of `strategy.md`:

```json
{
  "targets": [
    {"slot": "routing+combine", "why": "...", "expected_saving_ms": {"rows_1": 0.0, "rows_32": 0.0}, "design": "..."}
  ]
}
```

List one or two targets, in the order you work on them.

**The launch spec** (`spec.json`):

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

- **`source_block` and `header_block`** are the numbers of your `block-N.metal` files.
- **Kernels run in list order.** A kernel's `inputs` may name slot inputs or outputs of earlier kernels.
- **`slot_outputs`** maps each output of the slot to the kernel output that provides it.
- **`max_rows`** (optional, an integer of at least 1): inputs with more rows than this skip your kernels and run the slot's reference under `mx.compile`. Use it to keep prefill-sized inputs on MLX's own code when your kernels are built for decode sizes; MLX switches its own experts kernel at 60 rows. The hand-off is graded like the rest: bit-exact, and timed at the prefill sizes.
- **For `routing+combine`,** `spec.json` holds two specs, each in the format above for its own slot, `max_rows` included: `{"routing": {"kernels": [...], "slot_outputs": {...}}, "combine": {"kernels": [...], "slot_outputs": {...}}}`. The original experts and shared expert run between the two parts. The grader tests each part on its own, and the combine part receives the reference routing weights.
- **Shape and grid entries** are integers, or products of integers and the names `rows`, `D`, `E`, `K` and `H`, such as `"rows*32"`. Nothing else is evaluated.
- **dtypes** are `float32`, `bfloat16`, `float16`, `uint32` or `int32`.
- **Template values** are one of those dtypes, an integer or a boolean.
- **`grid`** counts threads, not threadgroups, as in `mx.fast.metal_kernel`.

## How each attempt is judged

The pass criteria were fixed before the first round. From round 2, G2 is exact and also covers the prefill, and the piece timing chains the layers ([ADR 0010](adr/0010-exact-at-every-size.md)); round 1 was judged by the earlier rules, in which G2 allowed small rounding differences and tested decode steps only. From round 3, the piece timing and G3 time both phases, decode and prefill ([ADR 0011](adr/0011-decode-and-prefill.md)); rounds 1 and 2 were timed on decode only.

1. **G1, runs.** It compiles and runs on real inputs.
2. **G2, correct: identical to the reference, bit for bit.** The slot runs in every call of `MoE.__call__`, the prefill included, so it is tested on every layer's inputs at two kinds of size:
   - **a decode step:** inputs captured from all 20 layers during a real decode step, at 1, 8 and 32 rows;
   - **a prefill:** the first chunk of a prefill with 128-token prompts, recorded while the grader runs it, at 1, 8 and 32 requests: 127, 1,016 and 2,048 rows. From 60 rows on, MLX's own experts kernel changes (`gather_qmm_rhs` instead of `gather_qvm`), and so does its rounding.

   At every size:
   - **`routing`:** each row's 4 experts and their 4 weights equal the reference's, in the same order, since combine sums the pairs in that order. That includes the reference's choice between tied 4th and 5th experts.
   - **`combine` and `experts`:** every element equals the reference's.

   **Why exact:** the end-to-end check below asks for the same greedy tokens, and greedy decoding turns any difference into different tokens. In round 1, routing weights within 10⁻⁵ of the reference, relative, changed 7 to 9 of 32 sequences.
   - **Then, end to end:** the same greedy tokens as the original model over 24 generated tokens, at 1 and 32 requests, in both plain MLX and vLLM.
3. **G3, faster, in vLLM.** Each workload runs 5 times in shuffled order: 1, 8 and 32 requests with 128-token prompts, and 32 requests with 1,024-token prompts, 256 generated tokens each. Three times are measured per run:
   - the **decode step**, the median period between steady steps;
   - the **prompt time**, until every prompt was processed;
   - the **whole request**, from sending the requests to the last token.

   Against both the original model and an `mx.compile` of the same reference code, the attempt passes when the decode step or the prompt time is faster in every workload, neither is slower in any, and no whole request is slower. Faster and slower mean that the ranges over the repeats don't overlap.

**What the grader prints for an attempt** (JSON):
- `g1` false and the compiler's error message;
- or the correctness results per size: for routing, the rows that differ; for combine and experts, the elements that differ and the largest gap in units in the last place; with the first example;
- or, when the attempt is correct, `timing_us_per_layer` per size: its time per call next to the original's and the compiled reference's, its wins against the compiled reference and the verdict (`faster`, `same` or `slower`); then `phases_vs_compiled` and `beats_compiled`.

**The piece timing** runs the slot at the six G2 sizes: the 20 layers' captured decode inputs at 1, 8 and 32 rows, and the 20 layers' first prefill chunk at 127, 1,016 and 2,048 rows. Each layer's call waits for the previous one's output, as in the model: one small kernel links them, the same for every function timed. Without that wait, the GPU overlaps the 20 calls and hides how long one takes; in round 1, an experts kernel 29% faster on overlapped calls was 66% slower on chained ones at 1 row.
- **Paired repeats.** Your attempt and the compiled reference run in alternating order, 30 times.
- **Per size:** your attempt is `faster` when it wins at least 22 of the 30 pairs, `slower` when it loses at least 22, and `same` otherwise. Medians alone would pass on noise: the same kernel timed twice differs by up to 5%.
- **Per phase:** decode (1, 8, 32 rows) and prefill (127, 1,016, 2,048 rows) are each `faster` when every size is, `slower` when any size is, and `not slower` otherwise.
- **`beats_compiled`** is true when one phase is `faster` and neither is `slower`.

It is a guide; G3 decides.

**When a slot stops.** Work on a slot stops at the first attempt that passes G1 and G2 and has `beats_compiled` true, or after 5 attempts. Name your final attempt per slot in `muse-spark/summary.md`; it goes to the end-to-end and G3 checks, which surucuonur's team runs after merging your branch.
