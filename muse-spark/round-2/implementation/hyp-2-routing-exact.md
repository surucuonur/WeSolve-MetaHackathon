# Hypothesis 2: routing kernel with the reference's exact float32 arithmetic

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T03:29:15Z
- **Next:** boundary-2 to draw the boundary

## Inefficiency

Slot `routing`. Baseline: `none`. Starting point: `muse-spark/kernels/routing/attempt-1`.

One combined fix for four confirmed entries:

- `ineff-baseline-routing-1` (gate, ranked 5): router-product accumulation order matches MLX in no regime (GEMV at 1 row, split-K steel at 8/32/127, steel GEMM at 1,016/2,048); blocks G2, headroom 0.12 / 0.37 / 0.37 ms/step.
- `ineff-baseline-routing-2` (gate, joint): sigmoid `precise::exp(|x|)` branch vs fast `exp(-x)`; independent blocker, 0 µs direct.
- `ineff-baseline-routing-3` (ranked 4): one simdgroup per row + serial thread-0 scan; up to 0.4 ms/step at 1 row.
- `ineff-baseline-routing-4` (ranked 7): `w_router` re-read per row; upper bound 0 / 0.13 / 0.57 ms/step, unbanked on round-1 counter-evidence.

Design: a kernel that reproduces the reference's scores bit for bit at every size — the same simdgroup matrix accumulation and tiling per `rows` regime, MLX's sigmoid formula verbatim, the already-matching 4-term normalise sum and tie rule — with each expert's dot product sharded across lanes and a parallel top-4, so the 1-row occupancy deficit is removed together with the exactness gates. Must keep the G2 boundary (same experts, weights and order bit for bit at all six sizes).

## Boundary

*(to be written by boundary-2)*

## Builders

*(none yet)*

## Boundary checks

*(none yet)*

## Decision

*(not yet)*
