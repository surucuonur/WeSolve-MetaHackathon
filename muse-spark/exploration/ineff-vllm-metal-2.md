- **Owner:** explorer-vllm-metal
- **Status:** superseded by ineff-runs-4
- **Version:** 2
- **Updated:** 2026-11-03T21:45Z
- **Next:** none (code mechanism appended as a note on ineff-runs-4)

## Slot

outside the slots (attention kernel dispatch).

## What happens

vllm-metal has a GQA decode kernel that reads each KV head once for all
query heads sharing it, but it admits only six head layouts
(`third_party/vllm-metal/vllm_metal/metal/ops_internal.h:47-54`). MobileMoE-S
is 12 query / 4 KV heads / head size 64 / block 16, which is not listed, so
`gqa_decode_geometry_supported` fails (`metal/paged_attention.cpp:67-76`) and
decode silently falls back to the per-token kernel: one threadgroup per
(query head, token) (`paged_attention.cpp:1026-1028`), each computing
`kv_head_idx = head_idx / num_queries_per_kv` and reading that KV head itself
(`metal/kernels_v2/pagedattention.metal:859-860`). Each KV head is read 3×
(once per sharing query head), partly absorbed by cache.

## Cost

Direct kernel timing at 32 rows, block 16, head 64 (decode-bottlenecks §6,
`attention_heads.py`):

- 256-token ctx (≈ the 128-token-prompt sweep): 125 µs/layer for 12/4 vs
  55 µs for 4/4 (same cache bytes) → 70 µs/layer × 20 = 1.40 ms/step.
- 1,152-token ctx (≈ the 1,024-token-prompt sweep): 569 vs 247 µs/layer →
  322 µs × 20 = 6.44 ms/step (upper estimate; 4/4 also does 1/3 the
  arithmetic).
- 1 row: 26 vs 19 µs/layer at 256 ctx → 7 µs × 20 = 0.14 ms/step.
- In-step attention-kernel knock-out at 32×128: 4.33 ms vLLM vs 3.12 ms
  plain MLX (step-breakdown §3) — consistent with ~1.2 ms of repeat reads.

Bytes: at 32×128 the kernel reads ~3× the 160 MB cache ≈ 480 MB; at
200 GB/s that is 2.4 ms of pure reread traffic, of which caches absorb
roughly half (measured cost follows query heads at 2.3×, not 3×).

## Kind

Between bandwidth-bound and overhead-bound: the rereads are real memory
traffic (GPU counters show repeat reads reaching memory), but each
threadgroup's reads are small and occupancy-bound. Call it bandwidth-bound
reread + overhead-bound dispatch; the fix (one read per KV head) cuts both.

## Spans

vllm-metal (kernel + dispatch in `metal/`).

## Predicted saving

A kernel reading each KV head once should cost ≈ what 4/4 costs:

- 1 row (128-token prompts): (26 − 19) µs × 20 ≈ 0.14 ms/step.
- 8 rows: ≈ 0.35 ms/step (scales with rows: 8/32 × 1.40 ms, estimate).
- 32 rows (128-token prompts): (125 − 55) µs × 20 ≈ 1.40 ms/step.
- 32 rows (1,024-token prompts): (569 − 247) µs × 20 ≈ 6.4 ms/step (upper).

Caveat (from the code): even with the layout added, the GQA kernel admits
only batches with full-partitions × query-heads ≥ 33 × GPU cores
(`paged_attention.cpp:156`); at 128-token prompts (ctx 256 < partition 512)
there are no full partitions, so the gain shows mainly at long contexts
unless admission changes too.

## Verdict

(verifier writes)
