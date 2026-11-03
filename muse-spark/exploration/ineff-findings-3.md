# Inefficiency: routed experts read per (token, expert) pair and run slowly

- **Owner:** explorer-findings
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:35Z
- **Next:** verifier to check the arithmetic against tables.md

## Slot

`experts`.

## What happens

Below 60 rows per step MLX runs the per-pair kernel (`gather_qvm`), which handles
each (token, expert) pair on its own and reads the expert's weights once per
pair; the grouped kernel (`gather_qmm_rhs`), which reads each expert once, needs
240+ pairs and never runs in decode (MLX 0.32.4 `quantized.cpp:2011-2012`, via
decode-bottlenecks §5). At 32 rows the 2,560 pairs ask for 1,416 MB while the 802
distinct (layer, expert) pairs need 443 MB (bottleneck-explained §4; per-layer
distinct counts in `runs-plain.jsonl` `distinct_experts`). Sorting from 16 rows
(120 extra sort kernels/step: 60 Gather, 40 ArgSort, 20 FloorDivide — tables.md
lines 136-139) only reorders the per-pair kernel's visits. One expert is
0.553 MB (3 × 294,912 int4 weights + scales/offsets).

## Cost

Plain MLX, knock-out savings (tables.md lines 320-356) over bare steps
(4.75 / 13.37 / 28.19 ms); byte totals from the GPU counters (2026-11-02) as
quoted in the findings:

- **1 row:** 140 kernels × (0.92 ms / 140 = 6.6 µs). Bytes: 51 MB read /
  200 GB/s = 255 µs (44 MB needed → 220 µs). Measured: **0.92 ms of 4.75 ms
  (19%)**, 48 GB/s on needed bytes (24% of peak). Each weight already read once.
- **8 rows:** not measured (no knock-outs, no per-piece counters at 8 rows).
  Rough estimate: cost per pair interpolates 11.5 µs (0.92 ms / 80 pairs at 1
  row) → 5.7 µs (14.67 ms / 2,560 pairs at 32 rows); 640 pairs × ~8 µs ≈
  **≈5.1 ms of 13.37 ms (38%)**. Asked 354 MB, needed 221 MB (20.0 distinct
  experts/layer × 20 × 0.553 MB).
- **32 rows:** 260 kernels. Bytes: 908 MB read / 200 GB/s = 4.54 ms. Measured:
  **14.67 ms of 28.19 ms (52%)**, 62 GB/s (31% of peak). The 465 MB of repeat
  reads that reach memory are 30% of the step's 1,557 MB reads; the cache serves
  the other half of the 973 MB repeat reads.

## Kind

Overhead-bound at all row counts (6% of peak compute, 30 GB/s on needed bytes at
32 rows — far below both limits), with a bandwidth-waste component at 32 rows
(465 MB extra reads). Most of the 14.67 ms is kernel efficiency, not bytes:
908 MB at peak = 4.5 ms, 443 MB at peak = 2.2 ms (bottleneck-explained §4,
estimate).

## Spans

`findings` (measured costs), `quantized-kernels` (gather_qmm kernel choice),
`switch-layers` (pair sorting from 16 rows), `model` (expert call).

## Predicted saving

Basis: the grouped kernel is 25-36% faster on equal balanced data (4-5
pairs/expert): 15.02 → 9.63 ms at 32 rows chained over 20 layers
(decode-bottlenecks §5). Real routing shares less (3.2 pairs/expert at 32 rows,
128/40.1; 2.2 at 16 rows), where the grouped win is untested — so 25% is the
planning figure and 36% the upper end:

- **1 row:** weights already read once; saving comes from fusing 7 kernels/layer
  (brief) toward 1-2: (140 − 40) × 6.6 µs ≈ **660 µs per step**.
- **8 rows (rough estimate):** 25% × ~5.1 ms ≈ **1300 µs per step**; at 1.6
  pairs/expert (32/20.0) the grouped kernel is untested and the gain could be
  near zero — fusion only.
- **32 rows:** 25% × 14.67 ms = **3670 µs per step** (upper 36% ≈ 5280 µs).
  Per layer: 733 µs → save ~183 µs.

## Verdict

(not yet written — the verifier writes this)
