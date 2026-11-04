# Hypothesis 3: combine carried into the set, byte-identical

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T22:05:00Z
- **Next:** boundary-3 writes the Boundary section

**Inefficiency:** none inside the slot is left to remove: the starting point sits on the chained floor (~32 µs/layer at decode, bytes+compute ≤1.5 µs of it) and exploration confirms no bit-exact change widens its decode margin (`../exploration/ineff-baseline-combine-1.md`, `-2`, both confirmed with achievable saving 0 / below paired noise). What the round needs from this slot is a round-4 attempt the ready check can time as part of the set: the pair's measured whole-request gains exist only with combine replaced, and round 3's attempt-2 is another round's folder.

**Slot:** `combine`. **Baseline:** none. **Starting point:** `muse-spark/round-3/kernels/combine/attempt-2` (pragma FP_CONTRACT OFF devolatilized kernel, no `max_rows`; G2 exact, `faster` at all six sizes, `beats_compiled` true in both phases).

**Direction:** (a) byte-identical copy of the starting point (both blocks and `spec.json`), graded once in this round — the round-3 attempt-6 noise-test precedent. No kernel change is attempted; any margin claim for the set comes from the ready check, not from this grade.

## Boundary

*(boundary-3 writes here.)*

## Builders

*(one section per builder, filled as attempts return.)*

## Boundary checks

*(boundary-3 appends here for each attempt that passed G2.)*

## Decision

*(lead writes at close.)*

## Failure ledger

*(lead keeps: each graded attempt that lost, the size or phase, its named cause, its approach.)*

## Unverified

*(starts from the boundary's can't-see list.)*

## Checks

*(checker-3 writes here before candidate.)*
