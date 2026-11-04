- **Owner:** explorer-vllm-metal
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:46Z
- **Next:** none (checklist sourced; ineff-1/2 superseded into findings-1/runs-4, ineff-3/4 open)

## What it does

vllm-metal's model runner turns each decode step's scheduled requests into one
lazy MLX graph — decode rows first, then prefill rows — submits it with
`mx.async_eval`, and samples the next tokens. Around the model it owns: batch
assembly and paged-attention metadata, the per-layer patched attention path
(QKV prep, RoPE, fused cache write, paged-attention kernel), the decode
dispatch policy (compiled vs eager MLP/MoE blocks), selective lm_head
projection, and sampling (native greedy argmax or a torch-CPU bridge).

## The checklist

### 1. `execute_model` → `_start_paged_forward` (graph build + submit)

- **Callers:** `MetalWorker.execute_model` (`v1/worker.py:315`), called by the
  vLLM engine once per step; on the paged path it returns `None` and the engine
  later calls `sample_tokens` (`v1/worker.py:328`).
- **Callees:** `_handle_new_requests`, `_collect_cached_requests`,
  `prepare_grouped` (attention metadata), `_target_forward` (the model),
  `_submit_paged_forward_outputs` → `mx.async_eval`
  (`v1/model_runner.py:3257-3376,1480-1800,891-900`).
- **Data:** in: `SchedulerOutput` (request ids, block tables, seq lens). Built:
  `input_ids` [1, rows] int32 (`model_runner.py:1654-1658`); `OffsetCache(0)`
  per layer; `cu_seqlens`. Out: `logits` [1, rows, vocab] (float, vocab 256k?
  — MobileMoE vocab; see model territory), stashed in `_PagedForwardState`.
- **Neighbours:** before: scheduler (`schedule`, 0.06/0.12/0.30 ms at 1/8/32
  rows) and the decode-pipeline gate (`begin_step`). After: `sample_tokens` →
  `_sample_paged_batch`. A replacement must keep the packed [decode][prefill]
  row order and `logits_cu_seqlens` indexing the sampler uses.
- **Cost:** runner work outside the model 0.5/0.9/1.6 ms per step at 1/8/32
  rows; graph record 2.41/2.45/3.12 ms; encode 4.10/4.70/6.46 ms
  (`claude/findings/step-breakdown.md` §2 table). No kernels of its own except
  one `AsType` on the inputs (step-breakdown §4 table).
- **Invariants:** pure-decode steps keep the model's own `__call__` and packed
  boundaries (no selective-logits gather); the pipeline gate guarantees a
  deferred step's membership can only shrink (`decode_pipeline.py:237-249`).

### 2. `_sample_paged_batch` + `DecodePipeline` (sampling)

- **Callers:** `sample_tokens` after `execute_model` returned `None`
  (`v1/model_runner.py:3398-3450`); or the deferred path
  `_submit_deferred_decode_sample` on pipeline-eligible steps.
- **Callees:** greedy: `SamplingBatch.native_decode_tokens` → `mx.argmax`
  (1 kernel) (`v1/sampling_batch.py:221,644-649`); else `mx.eval` + `mlx_to_torch`
  (fp32 cast) + vLLM torch sampler on CPU (`sampling_batch.py:651-663`).
- **Data:** in: `logits` [1, rows, vocab]; out: one token id per row (+ optional
  logprobs). The deferred path feeds lazy tokens as next step's `input_ids` via
  device gather, no host round-trip (`decode_pipeline.py:287-324`).
- **Neighbours:** after the forward's `async_eval`; before scheduler output
  processing. G2's end-to-end check (same 24 greedy tokens) pins this path.
- **Cost:** sampling + bookkeeping 0.10/0.13/0.20 ms CPU at 1/8/32 rows; the
  native-greedy sample is one argmax kernel (step-breakdown §2; benchmark
  workload is greedy so the torch bridge never runs there).
- **Invariants:** native greedy requires temp≈0, no top-k/p, no penalties, no
  logprobs (`sampling_batch.py:173-194`); seeded/non-greedy stays on torch.

### 3. Patched attention path per layer (`sdpa_forward`)

- **Callers:** the model's attention layers, wrapped by `patch_sdpa_attention`
  at load (`attention/impls/sdpa_wrapper.py:258-286`, `runtime/sdpa.py:128-133`).
  MobileMoE-S (12 q heads / 4 kv heads) takes the SDPA runtime.
- **Callees:** `prepare_sdpa_qkv` (q/k/v projections, q/k norms, RoPE) →
  `queries[0]` gathers + `astype` + `mx.contiguous` → fused `reshape_and_cache`
  kernel → `paged_attention_primitive` → o_proj
  (`attention/impls/sdpa.py:669-679,722-724`, cache write ~`:800-815`,
  attention dispatch ~`:830+`).
- **Data:** q/k/v [1, L, heads, 64] bfloat16 → contiguous [L, heads, 64] cache
  dtype; slot_mapping/block_tables/seq_lens metadata memoized per forward
  (`_kernel_metadata`); KV cache [blocks, 16, 4, 64] bfloat16.
- **Neighbours:** between post-norm... actually input_norm → attention → residual
  add → post_norm → MoE. Must keep: RoPE positions per request, cache-write
  visibility to the read in the same step (graph provenance on the aliased
  cache buffers), output dtype = query dtype.
- **Cost:** vLLM attention vs plain MLX: 320 vs 220 kernels/step (+60 Gather,
  +60 Contiguous from 2 rows up; cache write 20 fused vs 40); knock-out
  7.48 vs 5.60 ms at 32 rows, of which the kernel is 4.33 vs 3.12 ms and the
  copies 0.24 ms; record CPU 1,716 vs 1,062 µs (step-breakdown §3-4).
- **Invariants:** G2 pins greedy tokens end to end; attention numerics must
  match MLX SDPA within that.

### 4. Paged-attention decode kernel selection (GQA miss)

- **Callers:** `sdpa_forward` via `ops.paged_attention_primitive` → C++
  dispatch `paged_attention.cpp`.
- **Callees:** GQA decode kernel only if
  `gqa_decode_geometry_supported` admits the layout
  (`metal/ops_internal.h:47-54`, `metal/paged_attention.cpp:67-76,156`);
  MobileMoE-S 12/4/64/16 is not among the six listed layouts, so decode uses
  the per-token kernel (one threadgroup per query head × token,
  `paged_attention.cpp:1026-1028`), where each group reads its KV head via
  `head_idx / num_queries_per_kv` (`metal/kernels_v2/pagedattention.metal:859-860`).
- **Data:** reads each KV head 3× (once per sharing query head), partly from
  cache (decode-bottlenecks §6 table: 12/4 takes 2.3× the time of 4/4 for the
  same cache bytes).
- **Neighbours:** after `reshape_and_cache`, before o_proj. A GQA replacement
  must reproduce the per-token kernel's softmax numerics exactly enough for G2's
  token check.
- **Cost:** direct kernel timing at 32 rows: 125 µs/layer at 256-token ctx
  (vs 55 for 4/4), 569 µs at 1,152-token ctx (vs 247); 1-row: 26 vs 19 µs
  (decode-bottlenecks §6). In-step attention kernel knock-out: 4.33 ms at
  32×128 (step-breakdown §3).
- **Invariants:** none beyond G2 tokens; nothing logs the fallback today.

### 5. Compiled-MLP decode dispatch (`CompiledMLPBlocks`)

- **Callers:** `ModelLifecycle.install_decode_dispatch` after load+split
  (`v1/model_lifecycle.py:228-239`), via `load_model`
  (`v1/model_runner.py:714`). Gated by `VLLM_METAL_COMPILED_MLP` (default
  **False**, `envs.py:121,170`); skipped for pooling and LoRA serves.
- **Callees:** wraps target blocks so decode-shaped calls
  (`x` [1, ≤4, D], fp16/bf16) run `mx.compile(inner.__call__)`
  (`compiled_mlp.py:174-196`).
- **Data:** unchanged shapes/dtypes; the trace captures load-final weights.
- **Neighbours:** wraps whole MoE/MLP blocks; the MoE block's shared expert
  compiles as part of the outer trace. Must keep bit-identical outputs on the
  quantized path (pinned by tests/parity harness per the module docstring).
- **Cost:** `_target_policies` lists only Qwen3Next/Qwen3.5 block types
  (`compiled_mlp.py:115-131`) — MobileMoE-S's MoE block is **not wrapped** even
  with the flag on, so its routing (240 kernels/step) + combine (100/step) run
  fully eager: knock-out 1.02 + 0.61 ms at 1 row, 1.68 + 0.33 ms at 32 rows
  (step-breakdown §3). Also `MAX_COMPILED_TOKENS = 4` would exclude 8/32-row
  steps anyway.
- **Invariants:** G2 routing (same 4 experts, weights within 1e-4 relative) and
  combine (within 2 bfloat16 ulps) tolerances.

### 6. Selective logits / lm_head layout

- **Callers:** `_start_paged_forward` (`model_runner.py:1742-1756`) via
  `_paged_logits_layout` (`:936-970`) and adapter `project_logits`
  (`v1/model_adapter.py:794-808`).
- **Callees:** pure-decode steps use the model's own `__call__` (no gather);
  mixed steps gather [decode rows + last row of each prefill] then project.
- **Data:** vLLM hands the model one packed sequence (1 × rows); plain MLX
  hands rows sequences of 1. `supports_selective_logits` probes bit-exact
  head agreement once after load (`model_adapter.py:810-844`).
- **Neighbours:** final_norm → lm_head → sampler indexing `logits[0, row]`.
- **Cost:** no extra kernels on pure-decode steps. Open question (not an
  inefficiency entry): vLLM's output-layer knock-out saves 5.08 ms vs plain
  MLX's 2.50 ms at 32 rows although the kernel takes 2.35 ms in either layout
  alone (`lm_head_shapes.py`; step-breakdown §3 "vLLM's output layer saves
  twice as much"). At least one vLLM 32-row knock-out removes more than its
  own piece (top-level savings sum to 33.3 ms > 30.3 ms step).
- **Invariants:** bit-exact head probe at load; G2 token check.

### 7. MoE offload (`ExpertStore`) — not on the benchmark path

- Swaps `SwitchGLU` for `OffloadedSwitchGLU` streaming experts from disk
  through an LRU (`moe_offload.py:1-22`); installed only when offloading is
  enabled (`model_lifecycle.py:421`). Benchmark serves the resident model, so
  no cost here. Noted so nobody re-explores it.

## Relations

- Feeds from: scheduler (vLLM core, outside territory) → batch + block tables;
  checkpoint conversion (`expert_bias` bfloat16, model territory).
- Feeds to: model forward (model/switch-layers territories) via `input_ids` +
  `OffsetCache`s + paged context; sampler outputs → scheduler.
- Shares the decode step's kernel chain with: MLX kernels
  (quantized-kernels/other-kernels territories) and the model block
  (model territory). Engine-attention findings overlap the runs/findings
  territories' tables (cited, not duplicated).

## Inefficiencies

- `ineff-vllm-metal-1` (superseded by `ineff-findings-1`): MobileMoE MoE
  block excluded from compiled-MLP dispatch — same routing+combine cause;
  engine mechanism appended there as a note.
- `ineff-vllm-metal-2` (superseded by `ineff-runs-4`): GQA decode kernel lacks
  the 12/4/64/16 layout — same KV re-read cause; code mechanism + admission
  caveat appended there as a note.
- `ineff-vllm-metal-3` (open): 60 q/k/v `[0]` gathers (+60 contiguous copies
  from 2 rows) per step.
- `ineff-vllm-metal-4` (open): ~2 ms of engine CPU per 1-row step while the
  step is CPU-bound (attention-path Python + extra encode).

## Files read

- `v1/model_runner.py` (`execute_model` :3257, `_start_paged_forward` :1480,
  `_sample_paged_batch` :1885, `_paged_logits_layout` :936, `_submit…` :891):
  the decode step's engine path.
- `v1/decode_pipeline.py` (all 380 lines): one-step-ahead deferred sampling.
- `v1/sampling_batch.py` (`sample_from_logits` :628, `sample_decode_tokens`
  :666, native-greedy gate :173): sampling costs.
- `v1/model_adapter.py` (`target_forward` :728, `project_logits` :794,
  selective-logits probe :810): head projection paths.
- `v1/worker.py` (`execute_model` :315, `init_device` :131): engine boundary.
- `v1/model_lifecycle.py` (`install_decode_dispatch` :228): dispatch install.
- `compiled_mlp.py` (all 209 lines): policies exclude MobileMoE; flag default
  off; ≤4-token routing.
- `envs.py:121-122,170-178`: flag defaults.
- `attention/runtime/sdpa.py`, `attention/impls/sdpa.py` (`prepare_sdpa_qkv`
  :273, `sdpa_forward` :606), `attention/impls/sdpa_wrapper.py`,
  `attention/context.py` (`prepare_grouped` :222): per-layer attention path.
- `metal/ops_internal.h:47-54`, `metal/kernels_v2/pagedattention.metal:859-860`:
  GQA layout list and per-group KV read.
- `moe_offload.py:1-80`: offload scope (not benchmark path).
- `claude/findings/step-breakdown.md` (§1-5), `claude/findings/decode-bottlenecks.md`
  (§6): all measured costs.

## From other agents

(none yet)
