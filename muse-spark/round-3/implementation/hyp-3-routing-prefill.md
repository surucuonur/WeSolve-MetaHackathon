# Hypothesis 3: routing prefill at compiled speed

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T18:07:45Z
- **Next:** boundary-3 draws the boundary

## Inefficiency

[ineff-baseline-routing-1.md](../exploration/ineff-baseline-routing-1.md) (confirmed, traffic: per-row `w_router` re-read, excess 116.1/935.4/1,886.5 µs/layer at 127/1,016/2,048 rows) plus [ineff-baseline-routing-2.md](../exploration/ineff-baseline-routing-2.md) (confirmed, shape: 24 barriers, 28 KB residency cap, 23% single-partition duty; residual 24.1/142.7/336.8 µs/layer). Together they partition the starting point's prefill gap exactly (measured − compiled = 139.8/1,077.7/2,222.6 µs/layer; 2.6/5.9/8.7× compiled), which makes prompts 7–10% slower and costs −3.5% whole-request throughput at 32×1,024. Slot: `routing`. Baseline: `none` (starting point `muse-spark/round-2/kernels/routing/attempt-4`, which passes G1+G2 and wins every decode size but is `slower` at every prefill size).

Predicted effect if the prefill path matches compiled speed at equal bits: save 2.80/21.6/44.5 ms per chunk (traffic 2.32/18.7/37.7 + shape 0.48/2.85/6.74), keeping the decode wins (15.3/22.2/23.9 µs/layer). Fix shapes: (a) `max_rows` hand-off (decode stays on the staged scalar path, prefill on the compiled reference — banks the prize at `same`); (b) GEMM-shaped kernel sharing `w_router` across rows (removes traffic and shape together, but round-1 tiled sharing was slower everywhere, so the design needs proof on the chained timing); (c) narrower staged footprint restoring residency (fewer barriers, fuller lanes in the single-partition regime). Every direction must keep the exact `rows`-branched accumulation orders and add no fixed cost at decode sizes.

## Boundary

(boundary-3 to write)

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(none yet)

## Failure ledger

(none yet)

## Unverified

(at least the boundary's *can't see* list, once written)

## Checks

(none yet)
