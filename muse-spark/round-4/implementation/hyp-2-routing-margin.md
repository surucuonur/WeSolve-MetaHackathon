# Hypothesis 2: widen the routing decode margin

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T21:49:10Z
- **Next:** boundary-2 writes the Boundary section

**Inefficiency:** the routing+combine pair keeps only −0.36 of its −0.66 ms/step piece win vs the compiled reference at 32×1024 and the 0.9% median gap sits inside the repeat spread — ~0.17 ms/step more separation passes (`../exploration/ineff-g3-1.md`, confirmed). The priced pools inside this slot are the per-row `w_router` re-read (excess 0 / 6.5 / 28.6 µs per layer at 1 / 8 / 32 rows; `ineff-baseline-routing-1.md`) and the staged scalar path's shape cost (27 barriers, 23%-duty tail third; single-digit µs per layer; `ineff-baseline-routing-2.md`). The two overlap: do not add.

**Slot:** `routing`. **Baseline:** none. **Starting point:** `muse-spark/round-3/kernels/routing/attempt-6` (byte-identical round-2 attempt-4 staged scalar kernels, `max_rows: 32`; G2 exact, decode `faster` ×3, prefill handed off; Claude's re-grade margins 17.7 / 15.8 / 19.7 µs per layer over compiled).

**Directions:** (a) share each `w_router` byte across rows (attack the re-read) while keeping the scalar accumulation order bit-exact; (b) trim the staged path's shape cost (barriers, tail duty, dead zeroing at 1 row) without touching the re-read. Both keep `max_rows: 32` so prefill stays handed off, and both must stay bit-exact at all six sizes — round 1's tiled-sharing attempt was slower everywhere, so direction (a) banks only if it holds the chained rate.

## Boundary

*(boundary-2 writes here.)*

## Builders

*(one section per builder, filled as attempts return.)*

## Boundary checks

*(boundary-2 appends here for each attempt that passed G2.)*

## Decision

*(lead writes at close.)*

## Failure ledger

*(lead keeps: each graded attempt that lost, the size or phase, its named cause, its approach.)*

## Unverified

*(starts from the boundary's can't-see list.)*

## Checks

*(checker-2 writes here before candidate.)*
