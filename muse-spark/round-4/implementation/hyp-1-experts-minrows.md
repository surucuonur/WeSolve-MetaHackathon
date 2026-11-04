# Hypothesis 1: experts with `min_rows` — hand 1 row to the compiled reference

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T21:49:10Z
- **Next:** boundary-1 writes the Boundary section

**Inefficiency:** the experts starting point loses `beats_compiled` on the 1-row decode size alone (96.9 vs 86.6 µs/layer) while winning at 8/32 rows by 69.3 / 328.7 µs/layer (round 3 grade; ranked item 1 in `../exploration/README.md`). Round 3's kernel fixes for 1 row bought nothing (vectorised scalar) or 2× worse (unfused), both refuted. Round 4's `min_rows` (ADR 0015) hands 1-row inputs to the compiled reference instead.

**Slot:** `experts`. **Baseline:** none. **Starting point:** `muse-spark/round-3/kernels/experts/attempt-1` (byte-identical round-2 attempt-7 kernels, `max_rows: 59`; G1 pass, G2 differ 0 at all six sizes, prefill `same` ×3 by hand-off).

**Directions:** (a) pure spec change — copy the starting point, add `min_rows: 2`, keep `max_rows: 59`; (b) same hand-off with `max_rows: 32`, handing the 33–59 band to the compiled reference too. Both keep every kernel byte and accumulation order; they differ only in which ungraded band runs the candidate.

## Boundary

*(boundary-1 writes here.)*

## Builders

*(one section per builder, filled as attempts return.)*

## Boundary checks

*(boundary-1 appends here for each attempt that passed G2.)*

## Decision

*(lead writes at close.)*

## Failure ledger

*(lead keeps: each graded attempt that lost, the size or phase, its named cause, its approach.)*

## Unverified

*(starts from the boundary's can't-see list.)*

## Checks

*(checker-1 writes here before candidate.)*
