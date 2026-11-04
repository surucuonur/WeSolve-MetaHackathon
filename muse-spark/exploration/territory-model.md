- **Owner:** explorer-model
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:25Z
- **Next:** none — checklist sourced, ineff-model-2 open, ineff-model-1 superseded by ineff-runs-2

## What it does

`DecoderLayer.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:251-258`) runs one
decode layer: attention + residual, then the MoE block + residual. `MoE.__call__`
(`mobilemoe.py:217-232`) is the whole territory core: router matmul + sigmoid, biased
top-4 selection, weight normalisation, the routed experts (`SwitchGLU`), the
weight-and-sum combine, and the shared expert. All 20 layers are MoE layers
(`interleave_moe_layer_step: 1` in
`models/MobileMoE-S-QAT-mlx-bfloat16/config.json`, `mobilemoe.py:240-244`).
Constants: D=768, E=60, K=4, H=384, routed_scaling_factor=2.5, norm_topk_prob=true,
int4 affine group_size=32, expert layout [E, in, out] (`transposed_experts: true`).

## Checklist

### A. `MoE.__call__`, router: cast + matmul + sigmoid (`mobilemoe.py:219`)

- **Callers:** `DecoderLayer.__call__` (`mobilemoe.py:258`) with `x = post_attention_layernorm(h)`.
  Decode shapes (plain MLX): `x` [rows, 1, 768] bf16 — decode feeds [rows, 1] ids
  (`bench/micro/breakdown.py:442-460`, greedy keeps `[:, None]`), so at 1/8/32 rows:
  [1,1,768] / [8,1,768] / [32,1,768]. The slots flatten lead dims to [rows, 768]
  (`bench/grader/slots.py:107`).
- **Callees:** `x.astype(float32)` → AsType kernel; `self.router` (`nn.Linear(768, 60, bias=False)`,
  weight [60, 768] float32, `mobilemoe.py:197`) → Matmul kernel; `mx.sigmoid` → Sigmoid kernel.
  3 kernels/layer, 60/step at every row count
  (`results/runs/2026-11-01-breakdown/tables.md:133`: AsType 20, Matmul 20, Sigmoid 20).
- **Data:** `x` [rows,1,768] bf16 (from post_norm, consumed here and by experts/shared);
  `w_router` [60, 768] float32, 184 KB/layer, 3.7 MB/step (`claude/findings/step-breakdown.md:250`);
  `scores` [rows,1,60] float32, consumed by top-4 (B) and weight gather (C).
- **Neighbours:** before: post_attention_layernorm + attention residual (other-kernels territory).
  After: top-4 selection (B). A replacement must keep `scores` bit-exact through sigmoid in
  fp32 — G2 matches weights 1e-4 relative by expert and grants freedom only on tie rows
  (`claude/muse-agent-brief.md:182-184`).
- **Cost:** part of routing: 12 kernels/layer, 240/step; knock-out 1.02 ms @1 row,
  1.68 ms @32 rows plain MLX (`step-breakdown.md:169-170`). CPU record 39 µs @1, 77 µs @32
  per step (`tables.md:96`). Bytes 3.7 MB/step → 0.02 ms at peak; 4 GB/s achieved
  (`step-breakdown.md:250`). Serialized GPU 0.98 ms @1, 2.11 ms @32 (`tables.md:207,256`).
- **Invariants:** router computed in float32 (comment `mobilemoe.py:218`); cast bf16→fp32 exact;
  sigmoid in fp32; tie rule for 4th/5th expert (within 1e-6 biased) is the only selection freedom.

### B. `MoE.__call__`, top-4: bias add + argpartition + gather (`mobilemoe.py:222-225`)

- **Callers:** straight-line in `MoE.__call__` after (A).
- **Callees:** `self.expert_bias` is bfloat16 [60] (`mobilemoe.py:198`) → AsType to fp32;
  `scores + bias` → Add; `-(...)` → Negative; `mx.argpartition(..., kth=3)` → ArgPartition;
  `[..., :k]` slice is a view (free); `mx.take_along_axis` → GatherAxis.
  5 kernels/layer, 100/step (`tables.md:134`: AsType 20, Add 20, Negative 20, ArgPartition 20,
  GatherAxis 20). Note the bias AsType runs in all 20 layers every step although the value
  never changes (`step-breakdown.md:424`: storing it fp32 is exact but saved nothing measurable —
  it runs alongside the chain).
- **Data:** `biased` [rows,1,60] fp32 (selection only, discarded); `inds` [rows,1,4] uint32,
  consumed by experts (D) and required as a slot output; gathered `scores` [rows,1,4] fp32,
  consumed by (C).
- **Neighbours:** before (A); after (C) weights and (D) experts. Replacement must keep the same
  4 experts per row up to order and the tie exception (brief G2).
- **Cost:** part of routing (see A). CPU record 69 µs @1, 127 µs @32 (`tables.md:97`).
  Serialized GPU 1.30 ms @1, 4.11 ms @32 (`tables.md:208,259`) — the largest serialized piece
  of routing at 32 rows, i.e. argpartition/gather dominate the routing chain there.
- **Invariants:** bias changes selection, not weights (comment `mobilemoe.py:221`); order of the
  4 within a row is free as long as weights stay paired (`muse-agent-brief.md:47`).

### C. `MoE.__call__`, weight norm + scale (`mobilemoe.py:226-228`)

- **Callers:** straight-line after (B); `norm_topk_prob` true, factor 2.5 (config + `mobilemoe.py:194-195`).
- **Callees:** `scores.sum(-1, keepdims)` → Sum; `+ 1e-20` → Add; `/` → Divide; `* 2.5` →
  Multiply. 4 kernels/layer, 80/step (`tables.md:135`: Sum 20, Add 20, Divide 20, Multiply 20).
- **Data:** `scores` [rows,1,4] fp32 in and out; consumed by combine (D). Slot output `weights`
  [rows, 4] fp32.
- **Neighbours:** before (B); after combine (D). Replacement keeps the `+1e-20` guard and fp32
  arithmetic; G2 1e-4 relative by expert.
- **Cost:** part of routing (see A). CPU record 65 µs @1, 135 µs @32 (`tables.md:98`).
- **Invariants:** normalise-then-scale order; fp32 throughout (no rounding point here).

### D. `MoE.__call__`, experts call + weighting (`mobilemoe.py:230-231`)

- **Callers:** straight-line after (C); `inds` reshaped to lead dims for the port, flat [rows,4]
  for the slot (`slots.py:111-114`).
- **Callees:** `self.experts(x, inds)` → `SwitchGLU.__call__` (switch-layers territory):
  7 kernels/layer @1 row (140/step: 3×(Arange+GatherQMM) + fused silu-mul), 13 @32 rows
  (260/step: + sort/unsort: Gather 60, ArgSort 40, FloorDivide 20, `tables.md:136-139`).
  `(y * scores[..., None])` → bf16→fp32 AsType + Multiply; `.sum(-2)` → Sum;
  `.astype(bf16)` → AsType. The final `+ shared` → Add (see E).
  Combine piece totals 5 kernels/layer, 100/step (`tables.md:140`: AsType 40, Multiply 20,
  Sum 20, Add 20).
- **Data:** `x` [rows,1,768] bf16 reused (third consumer after router and shared);
  `y` [rows,1,4,768] bf16 expert outputs in `inds` order; `scores` [rows,1,4] fp32;
  weighted sum fp32 → bf16; `out` [rows,1,768] bf16. Slot `combine` inputs/outputs exactly
  (`muse-agent-brief.md:49-64`).
- **Neighbours:** before (C) + experts; after residual add in `DecoderLayer.__call__`.
  Replacement keeps: fp32 accumulation of the weighted sum, single bf16 rounding before adding
  `shared` (`slots.py:52` — the `+ shared` is inside the slot).
- **Cost:** combine knock-out 0.61 ms @1 row, 0.33 ms @32 rows (`step-breakdown.md:171,212`);
  routed experts 0.92 ms @1, 14.67 ms @32 (`step-breakdown.md:170,211`). Combine CPU record
  75 µs @1, 135 µs @32 (`tables.md:103`).
- **Invariants:** G2 for combine: every element within 2 bf16 ulp of reference (floor at 0.001);
  end-to-end same greedy tokens over 24 tokens (`muse-agent-brief.md:184-185`).

### E. `MoE.__call__`, shared expert + residual (`mobilemoe.py:232, MLP:131-139`)

- **Callers:** `MLP(dim=768, hidden=1536)` (`mobilemoe.py:215`), dense int4: gate/up/down
  QuantizedMatmul + one fused silu-mul (`tables.md:141-144`: 4 kernels/layer, 80/step).
  Runs on the same `x` as the routed path; result added inside combine's Add kernel (D).
- **Callees:** `swiglu` is `@mx.compile` fused (`activations.py:9-11`); the three products are
  MLX split-K quantized matmuls from 14 rows up (quantized-kernels territory).
- **Data:** `shared` [rows,1,768] bf16 → combine Add. Weights 44.2 MB/step (`step-breakdown.md:252`).
- **Neighbours:** inside combine slot boundary as input; outside all slots as computation —
  a slot kernel may not change it. Final `return h + feed_forward(...)` residual add lives in
  `DecoderLayer.__call__` (`mobilemoe.py:258`).
- **Cost:** shared knock-out 0.51 ms @1, 2.90 ms @32 (`step-breakdown.md:172,213`).
- **Invariants:** bf16 I/O; int4 dequant `w = scale*q + bias`, group 32 (`muse-agent-brief.md:82`).

### F. `TransposedQuantizedSwitchLinear.__call__` (`mobilemoe.py:172-184`)

- **Callers:** `SwitchGLU.__call__` for gate/up/down (`switch_layers.py:188-194`); built per
  projection in `MoE.__init__` (`mobilemoe.py:200-214`) with group_size=32, bits=4, mode=affine.
- **Callees:** one `mx.gather_qmm(..., transpose=False, rhs_indices only)` per call → GatherQMM
  kernel + MLX's default `arange` lhs indices (`tables.md:137-139`: Arange 20 + GatherQMM 20
  per projection per step). No `lhs_indices` passed (`mobilemoe.py:172-184`).
- **Data:** weights [60, in, out] uint32 4-bit, 8/outputs per word along out; scales/biases
  [60, in, out/32] bf16 (`muse-agent-brief.md:72-80`); `x` expanded [rows,1,4,768]-ish
  (`switch_layers.py:177`); output bf16 after fp32 accumulate (brief §experts).
- **Neighbours:** before: expand_dims/sort in SwitchGLU; after: silu-mul (gate/up) or unsort (down).
  A replacement keeps fp32 accumulation + bf16 rounding per product, and the bf16 rounding of
  `silu(gate)*up` before the down product (`muse-agent-brief.md:90`).
- **Cost:** inside routed-experts cost (see D). The 60 Arange/step are MLX-internal (see note to
  quantized-kernels under Relations).
- **Invariants:** G2 experts: 2 bf16 ulp; layout [E, in, out] grouped along out (config
  `transposed_experts: true`, comment `mobilemoe.py:42`).

### G. Wiring: `DecoderLayer.__init__`, `MoE.__init__`, `sanitize`

- **Callers:** model build only; `sanitize` (`mobilemoe.py:309-325`) splits fused gate_up and
  swapaxes to [E, out, in] at load — not decode-time (note: with `transposed_experts` the
  checkpoint already stores [E, in, out]; sanitize's swapaxes branches only fire for legacy
  fused keys).
- **Callees:** `Attention` (other-kernels/vllm-metal territory), `nn.RMSNorm` ×2/layer
  (input + post norms: 40 kernels/step, knock-out 0.27 ms @1, 0.34 ms @32,
  `step-breakdown.md:167,208`), residual Adds (`tables.md:119`: Add 40/step).
- **Data:** norms + adds are bf16 on [rows,1,768]; outside all slots.
- **Neighbours:** attention output → input_norm path (before); next layer's input_norm (after).
- **Cost:** norms+adds ≈ 0.3 ms/step; not slot-replaceable — noted for other-kernels, not posted here.
- **Invariants:** none beyond bf16 elementwise exactness.

## Relations

- **Feeds from:** post_attention_layernorm (other-kernels territory) → `x`; attention block
  (other-kernels / vllm-metal) → `h` for the residual.
- **Feeds to:** next layer's input_norm; final_norm → lm_head (other-kernels).
- **Inside:** routing (A–C) → slot `routing`; experts call (D/F) → slot `experts`;
  weighting+add (D) → slot `combine`; shared expert (E) → combine input, computed outside slots.
- **Notes for other territories** (their entries do not exist yet, 21:20Z, so parked here):
  - *switch-layers:* sort threshold `indices.size >= 64` (`switch_layers.py:181`) trips at
    16+ rows (rows×4 ≥ 64) — matches "from 16 rows MLX also sorts" (brief §experts) and the
    kernel jump 140→260/step (`tables.md:136`). The `_gather_sort`/`_scatter_unsort` traffic
    (Gather 60, ArgSort 40, FloorDivide 20) is SwitchGLU glue, counted in `experts`
    (`tables.md:99,136`), not in the GatherQMM products.
  - *quantized-kernels:* call site `mobilemoe.py:172-184` passes only `rhs_indices`, so MLX
    builds `arange` lhs indices per product (60 Arange/step, `tables.md:137-139`,
    `step-breakdown.md:423`). Below 16 rows passing cached indices is exact; from 16 rows MLX
    only uses sorted kernels without lhs indices — fix belongs in MLX.
  - *quantized-kernels:* `swiglu` is already `@mx.compile` fused (`activations.py:9-11`),
    one kernel per use (`tables.md:136,141`); nothing to win there from this territory.
  - *grader:* slot I/O confirmed against code — routing outputs `inds` uint32 + `weights` fp32
    (`slots.py:44-48` == `mobilemoe.py:219-228` flattened); combine includes the `+ shared` add
    (`slots.py:52` == `mobilemoe.py:231-232`); experts rounding points in brief §experts match
    `gather_qmm` fp32-accumulate/bf16-out semantics.

## Inefficiencies

- `ineff-model-1` (superseded by `ineff-runs-2`): routing is 12 tiny dependent kernels
  per layer — same cause as explorer-runs' entry, code-side detail appended there.
- `ineff-model-2` (open): combine is 5 tiny kernels per layer with bf16↔fp32 casts.

## Files read

- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py` (all 329 lines): the territory itself.
- `third_party/mlx-lm/mlx_lm/models/switch_layers.py:160-199`: SwitchGLU call sequence, sort
  threshold, expand/squeeze around the territory's call site.
- `third_party/mlx-lm/mlx_lm/models/activations.py:9-11`: swiglu already compile-fused.
- `models/MobileMoE-S-QAT-mlx-bfloat16/config.json`: D/E/K/H, 20 MoE layers, int4 group-32
  affine, transposed_experts, 2.5 / norm_topk_prob.
- `bench/grader/slots.py:44-119`: reference slot code == port code flattened; slot I/O shapes.
- `bench/micro/breakdown.py:442-460`: decode shapes [rows, 1] ids → [rows, 1, 768] activations.
- `claude/muse-agent-brief.md` (slots, G1–G3, per-slot kernel counts and costs).
- `claude/findings/step-breakdown.md` (§3 knock-outs, §5 repeated work) and
  `claude/findings/bottleneck-explained.md` (§3–4, fixed cost, expert byte counts).
- `results/runs/2026-11-01-breakdown/tables.md:96-147,173-264,333-336`: per-piece CPU µs,
  kernel-op inventory by type, serialized encode/GPU, knock-out savings.

## From other agents

(none yet)
