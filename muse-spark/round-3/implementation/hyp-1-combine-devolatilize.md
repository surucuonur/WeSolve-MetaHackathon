# Hypothesis 1: combine without volatiles

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T17:53:22Z
- **Next:** boundary-1 draws the boundary

## Inefficiency

[ineff-baseline-combine-1.md](../exploration/ineff-baseline-combine-1.md) (confirmed): the starting point's seven `volatile float` rounding barriers (`p0..p3`, `s01`, `s012`, `acc`) cost +0.142/+0.508/+1.441 ms per decode step in vLLM (+7.1/+25.4/+72.1 µs/layer, ±0.1 ms control) while the grader prices them near zero. Measurement side: [ineff-g3-1.md](../exploration/ineff-g3-1.md) (confirmed). The same barriers drive the +1,501.7 ms prompt twin on 32×128 (ranked prefill item 6), so one fix covers both phases. Slot: `combine`. Baseline: `none` (starting point `muse-spark/round-2/kernels/combine/attempt-1`, which passes G1+G2 and `beats_compiled` but fails G3).

Predicted effect if the volatiles go with exact per-op rounding kept: return to ≈ −0.51/−0.71/−0.39 ms/step (the volatiles-free twin's medians), i.e. recover ≈0.5 ms/step at 8 requests and ≈1.4 ms/step at 32 (≈0 claimed at 1 row, within noise), net vs compiled −0.2/−0.4/−0.1 ms/step (verifier-corrected), plus prompt recovery of order 1 s on combine-including 32×128 prompts. Suggested direction: `__attribute__((noinline))` device-function barriers for the mul and the adds (values in registers, no memory traffic; helpers in `header_block`).

## Boundary

(boundary-1 to write)

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
