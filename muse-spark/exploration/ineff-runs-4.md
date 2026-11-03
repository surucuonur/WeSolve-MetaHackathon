# Inefficiency: attention re-reads K/V per query head (outside the slots)

- **Owner:** explorer-runs
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:25:00Z
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
