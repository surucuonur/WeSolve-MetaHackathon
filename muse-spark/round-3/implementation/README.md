# Implementation: round 3

**Version:** v2 · 2026-11-04T17:55:21Z · skill version 8

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-combine-devolatilize` | `combine` | `builder-1-a`, `builder-1-b` | — | building |
| `hyp-2-experts-prefill` | `experts` | `builder-2-a`, `builder-2-b`, `builder-2-c` | — | building |

Ranked decode item 1 (`ineff-g3-2`, the three-way interaction at 8 requests) is not opened as a hypothesis: it spans all three slots, its mechanism is still open, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the ready check times each candidate alone and all together, so round 3 sees composition inside the round. Next in order after one of the two open hypotheses closes: routing prefill (ranked prefill item 5) as hyp-3, then experts 1-row decode (ranked decode item 3) as hyp-4, built on hyp-2's winner. Tile waste and redundant offsets (items 7–8) ride with hyp-2's fast-branch directions; they are not separate hypotheses.

## Versions

- v2 · 2026-11-04T17:55:21Z · boundaries drawn, 5 builders started (2 combine, 3 experts) · changed: hyp-1-combine-devolatilize v2, hyp-2-experts-prefill v2
- v1 · 2026-11-04T17:53:22Z · opened implementation, hyp-1 (combine) and hyp-2 (experts prefill) in boundary · changed: none
