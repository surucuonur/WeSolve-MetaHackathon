# Hypothesis 2: fused routing kernel

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-03T21:32:00Z
- **Next:** boundary-2 to write the boundary

## Inefficiency

Slot `routing`. One entry:

- `ineff-runs-2` (rank 2, mean 11.0% of step): routing costs 1.02 ms at 1 row and 1.68 ms at 32 rows while reading ~4 MB — 12 tiny kernels/layer of pure overhead (argpartition runs a full 60-element sort per row; the weights-norm sum runs on 1 thread at 1 row). Fusion to ~1 kernel/layer saves 0.9 / 1.2 / 1.4 ms at 1/8/32 rows.

Design: one Metal kernel per layer that computes router scores (matmul + sigmoid + bias), selects the top-4 experts, and normalizes/scales the weights, with the 4 experts in any order per row. G2 gives freedom only on within-1e-6 tie rows and 1e-4 relative weights by expert.

## Boundary

(boundary-2 to write)

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
