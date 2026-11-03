# Hypothesis 3: fused combine kernel

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-03T21:52:00Z
- **Next:** boundary-3 to write the boundary

## Inefficiency

Slot `combine`. One entry:

- `ineff-model-2` (rank 5, mean 4.3% of step): each layer's weighting launches 5 kernels (bf16→fp32 cast, multiply, sum, fp32→bf16 cast, add shared) costing 0.61 ms at 1 row and 0.33 ms at 32 rows for ~6 MB of activations. Fusion to 1 kernel/layer saves 0.47 / 0.30 / 0.19 ms at 1/8/32 rows.

Design: one Metal kernel per layer that weights the 4 experts' bf16 outputs by the fp32 weights, sums in fp32, rounds once to bf16, and adds the shared expert. G2: every element within 2 bf16 ULP (floor at 0.001). Note the piece-harness lesson from hyp-2: the fused-kernel floor is ~24 µs/layer, not 6.81 µs — predict against the harness, and G3 decides the real saving.

## Boundary

(boundary-3 to write)

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(open)

## From other agents
