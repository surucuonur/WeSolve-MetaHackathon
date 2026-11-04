# Inefficiency: routed experts read per (token, expert) pair and run slowly

- **Owner:** explorer-findings
- **Status:** confirmed
- **Version:** 4
- **Updated:** 2026-11-03T21:27:42Z
- **Next:** verifier to check arithmetic and reconcile the @32 saving basis with ineff-quantized-kernels-1

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
`switch-layers` (pair sorting from 16 rows), `model` (expert call). Complements
ineff-runs-1 (excess-bytes floor, 2.3 ms @32 rows) — this entry adds kernel
efficiency and the grouped-kernel measurement (3.67 ms @32 rows); the two are
compatible (floor vs measured-kernel basis). See my note on ineff-runs-1.
ineff-quantized-kernels-1 gives the in-kernel mechanism (strided lanes, x
re-reads); its @32 saving basis (same 908 MB at 150 GB/s streaming = 8.6 ms)
vs this entry's (measured grouped kernel = 3.67 ms) needs verifier
reconciliation — see my note there.

## Predicted saving

Basis: the grouped kernel is 25-36% faster on equal balanced data (4-5
pairs/expert): 15.02 → 9.63 ms at 32 rows chained over 20 layers
(decode-bottlenecks §5). Real routing shares less (3.2 pairs/expert at 32 rows,
128/40.1; 2.2 at 16 rows), where the grouped win is untested — so 25% is the
planning figure and 36% the upper end:

- **1 row:** weights already read once; saving comes from fusing the on-chain
  kernels (60 products + 20 activations = 80/step; the 60 Aranges are off-chain
  and ~free, step-breakdown §5) toward 1/layer: (80 − 20) × 6.6 µs ≈ **400 µs
  per step** (corrected from 660 — was counting off-chain Aranges;
  agrees with ineff-switch-layers-2's 0.4 ms).
- **8 rows (rough estimate):** 25% × ~5.1 ms ≈ **1300 µs per step**; at 1.6
  pairs/expert (32/20.0) the grouped kernel is untested and the gain could be
  near zero — fusion only.
- **32 rows:** 25% × 14.67 ms = **3670 µs per step** (upper 36% ≈ 5280 µs).
  Per layer: 733 µs → save ~183 µs.

## Verdict

verifier, 2026-11-03T21:27:42Z — **confirmed** (v3 text, incl. the corrected 1-row saving). Grouped kernel needs 240+ pairs / 60+ rows; per-pair `gather_qvm` runs in decode (decode-bottlenecks §5 quoting quantized.cpp:2011-2012; kernel names in quantized.h:2356,2580 with the `affine_` prefix §5 omits) ✓. 2560 pairs; 1416 asked / 802 distinct / 443 MB needed (bottleneck-explained §4; runs-plain.jsonl `distinct_experts`: 32-row sum 802, 8-row mean 20.0 ✓). Sort glue 120 (tables.md:136) ✓. 0.553 MB/expert (§4: 552,960 B) ✓. 1-row: 140 kernels (:170), 6.6 µs (:182), 255/220 µs at peak, 19%, 48 GB/s / 24% ✓. 8-row: 11.5/5.7 µs per pair, 640×8 µs ≈ 5.1 (38%), asked 1416/4 = 354, needed 20.0×20×0.553 = 221 ✓. 32-row: 260 kernels (:211), 908/200 = 4.54, 52%, 62 GB/s / 31%, 465/1557 = 30% (1557 per §4's knock-out recording), 973−465 ≈ half cached ✓. Kind: 6% compute, 443.5/14.67 = 30 GB/s, 4.5/2.2 at peak ✓. Saving basis −25/−30/−36 (§5 A/B table), 128/40.1 = 3.2, 2.2 at 16 rows, win-below-4-untested ✓ (§5); the A/B reading (kernel choice on identically ordered pairs) re-verified in moe_parts.py:149-179. 8-row 25%×5.1 ≈ 1300 (1.6 pairs/expert caveat ✓); 32-row 25%×14.67 = 3670, upper 36% ≈ 5280, 733/183 per layer ✓. 1-row (v3 correction): on-chain 60 products + 20 activations = 80 (GatherQMM 60 in tables.md:152-154; Compiled 20 in :136; 60 Aranges off-chain and ~free per §5 repeat-fixes −0.02/−0.04 ✓); (80−20)×6.6 = 396 ≈ 400 ✓, agreeing with switch-layers-2's 0.4. Reconciliation with ineff-quantized-kernels-1 (lead settlement — complementary): this entry's 3.67 is the MEASURED grouped kernel (removes re-reads at demonstrated efficiency); qk-1's 8.6 is an IDEAL (same 908 MB at 150 GB/s, keeps re-reads). Do not add either to runs-1's 2.3 floor; the joint ceiling is 443 MB at streaming rate ≈ 11.5 ms.
