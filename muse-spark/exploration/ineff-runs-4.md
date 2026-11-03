# Inefficiency: attention re-reads K/V per query head (outside the slots)

- **Owner:** explorer-runs
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to confirm against attention_heads.txt

## Slot

Outside the slots (for surucuonur's team, not a builder target).

## What happens

With 12 query heads over 4 KV heads, each query head re-reads the shared K/V instead of reading it once: `results/runs/2026-10-30-micro/attention_heads.txt` shows rows 32, ctx 1152 at 568.7 µs/layer for 12q/4kv (66.4 GB/s on unique K/V) vs 247.1 µs/layer for 4q/4kv (152.8 GB/s) — same K/V bytes, ~2.3× the time. At rows 1, ctx 256: 26.0 vs 18.7 µs/layer. The knock-out table (`results/runs/2026-11-02-gpu-counters/tables.md`, `attn_kernel` row) bounds the whole attention prize: 0.41 ms saved at 1 row, 3.12 ms at 32 rows (unrecorded).

## Cost

Per decode step:

- 1 row: attention knock-out saves 0.41 ms; micro excess (12q−4q) is 7.3 µs/layer × 20 = 0.15 ms.
- 8 rows: no knock-out or micro row — unmeasured.
- 32 rows: attention knock-out saves 3.12 ms; micro excess is 321.6 µs/layer × 20 = 6.4 ms, which exceeds the knock-out because the micro ctx (1152) is hotter than serving's mean ctx — the knock-out caps the claim.

## Kind

Bandwidth-bound: the same K/V bytes fetched ~3× (12 query heads / 4 KV heads).

## Spans

`runs` (mechanism would belong to whichever territory owns attention kernels)

## Predicted saving

Sharing one K/V read across the query heads in a group, per decode step:

- 1 row: ≈ 0.1 ms (micro 0.15 ms, capped by knock-out 0.41 ms).
- 8 rows: unmeasured (no 8-row attention data in this territory).
- 32 rows: up to ≈ 2 ms (a share of the 3.12 ms knock-out; the 6.4 ms micro excess is at higher ctx and cannot all be claimed).

## Verdict

(verifier to write)

## From other agents

explorer-vllm-metal, 2026-11-03T21:45Z — the code mechanism for this entry
(the attention kernels are my territory); I marked my ineff-vllm-metal-2
superseded by this entry. Mechanism: vllm-metal has a GQA decode kernel that
reads each KV head once, but it admits only six layouts
(`third_party/vllm-metal/vllm_metal/metal/ops_internal.h:47-54`):
32/8/128/16, 24/4/256/16, 24/4/256/32, 16/2/128/16, 16/2/256/16, 16/2/256/32.
MobileMoE-S is 12/4/64/16, so `gqa_decode_geometry_supported` fails
(`metal/paged_attention.cpp:67-76`) and decode silently (nothing logs it)
falls back to the per-token kernel — one threadgroup per (query head, token)
(`paged_attention.cpp:1026-1028`), each reading its KV head via
`kv_head_idx = head_idx / num_queries_per_kv`
(`metal/kernels_v2/pagedattention.metal:859-860`). Two caveats for the saving:
(a) at 128-token prompts (ctx ~256 < partition 512) the GQA kernel's
admission rule (full-partitions × query-heads ≥ 33 × GPU cores,
`paged_attention.cpp:156`) admits nothing, so the fix needs an admission
change too at short contexts; (b) supporting your knock-out cap: the in-step
kernel knock-out at 32×128 is 4.33 ms vLLM vs 3.12 ms plain MLX
(step-breakdown §3), so ~1.2 ms of the 12/4-vs-4/4 micro gap is visible
in-step at serving ctx. My 8-row interpolation (≈0.35 ms, scaling 8/32 of the
1.40 ms 32-row micro excess) is yours to use or drop.
