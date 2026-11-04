# Hypothesis 1: experts kernel exact at every size, faster chained at 1 row

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T03:29:15Z
- **Next:** boundary-1 to draw the boundary

## Inefficiency

Slot `experts`. Baseline: `none` (no earlier attempt passes the current G2). Starting point: `muse-spark/kernels/experts/attempt-4`.

One combined fix for three confirmed entries (savings do not add; the gate must be removed before any timing counts):

- `ineff-baseline-experts-1` (gate, ranked 8): starting point matches `gather_qvm` but prefill runs `gather_qmm_rhs` — prefill differ 62.4–62.6%, up to 110.5 ULP; 0 µs saving, blocks all timing.
- `ineff-baseline-experts-2` (ranked 2, mean 7.4%): chained 1-row latency 132.9 vs 80.3 µs/layer (+52.6, 1.05 ms/step); 4 threadgroups with wave/barrier serialisation.
- `ineff-baseline-experts-3` (ranked 3, mean 3.7% floors): per-pair re-reads persist — 908 vs 443 MB at 32 rows, 2.33 ms floor, 0 banked on round-1 grouped evidence.

Design: a kernel (or size-branched kernels) that is bit-exact at decode sizes (matching `gather_qvm`: float32 dequant `x·(scale·q+bias)`, strided-lane accumulation + `simd_sum`, full-bfloat16 silu) and at prefill sizes (matching `gather_qmm_rhs`: bfloat16-rounded dequantized tiles, `BK=32` MMA-order accumulation), while exposing more concurrent threadgroups per call at 1 row so the chained latency falls to or below the reference's, without spending the 8/32-row chained wins (63.7 / 210.0 µs/layer ahead). Must keep the G2 boundary (bit-exact at all six sizes, fp32 accumulation, bfloat16 rounding points).

## Boundary

*(to be written by boundary-1)*

## Builders

*(none yet)*

## Boundary checks

*(none yet)*

## Decision

*(not yet)*
