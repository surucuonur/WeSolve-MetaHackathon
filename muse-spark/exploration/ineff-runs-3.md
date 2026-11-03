# Inefficiency: (token, expert) pairs unsorted below 16 rows

- **Owner:** explorer-runs
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:25:00Z
- **Next:** verifier to confirm against moe_parts.txt

## Slot

`experts`

## What happens

MLX sorts pairs by expert only from 16 rows (brief). The sort experiment in `results/runs/2026-10-30-micro/moe_parts.txt` (kernel A/B, gate+up+down of one layer chained ×20) shows sorting wins at every row count: rows 8, ours sorted=False 5.08 ms vs sorted=True 3.82 ms per 20 layers; rows 16: 8.12 vs 5.65; rows 32: 15.02 vs 9.63. Below 16 rows serving pays the unsorted price; at 8 rows that price is directly measured.

## Cost

Per decode step (micro chained-layer setup, plain MLX):

- 1 row: no experiment row; 4 pairs need little sorting — cost ≈ 0.
- 8 rows: unsorted 5.08 ms vs sorted 3.82 ms per 20 layers — excess 1.26 ms.
- 32 rows: already sorted in serving — excess 0 (the 15.02 → 9.63 gap measures what sorting is worth, already collected).

## Kind

Bandwidth-bound: unsorted pairs scatter each expert's reads across DRAM instead of streaming each expert once.

## Spans

`runs`, `switch-layers`, `quantized-kernels`

## Predicted saving

Sorting the pairs at 8 rows (or an equivalent layout that streams each expert once), per decode step:

- 1 row: 0.0 ms (no data; 4 pairs, nothing to sort).
- 8 rows: 5.08 − 3.82 = 1.3 ms.
- 32 rows: 0.0 ms (already sorted).

Caveat: the micro setup's distinct/layer (up to 47.8 at R=32) exceeds serving's (38.2 at c32, `results/runs/2026-10-29-baseline/summary.md`), so the 8-row prize in serving may be smaller; the sort itself also costs a few kernels (other-kernels territory).

## Verdict

(verifier to write)
