# Hypothesis 1: fused grouped experts kernel

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-03T21:32:00Z
- **Next:** boundary-1 to write the boundary

## Inefficiency

Slot `experts`. One combined fix for five confirmed entries (savings do not add; joint ceiling 443 MB at streaming rate ≈ 11.5 ms at 32 rows, per the verifier):

- `ineff-quantized-kernels-1` (rank 1, mean 19.9% of step): per-pair qvm grids at ~1/3 peak; ideal 0.5 / 2.5 / 8.6 ms at 1/8/32 rows.
- `ineff-findings-2` (rank 3, mean 10.4%): measured 25% grouped-kernel win; 0.40 / 1.30 / 3.67 ms.
- `ineff-switch-layers-2` (rank 4, mean 4.3%): unfused products+activation; ~0.4 ms at all row counts.
- `ineff-runs-1` (rank 6, mean 4.2%): 465 MB excess re-reads; floor 0.0 / 0.6 / 2.3 ms.
- `ineff-switch-layers-1` (rank 7, mean 0.8%): sort glue 0.7 ms at 32 rows, removable only inside this fix.

Design: one Metal kernel that batches same-expert pairs with coalesced reads, runs gate/up/down plus the activation with intermediates on-chip, and needs no sort/unsort. Must keep the G2 boundary (2 bf16 ULP, fp32 accumulation, bf16 rounding points).

## Boundary

(boundary-1 to write)

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
