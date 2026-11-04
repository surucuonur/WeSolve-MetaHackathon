# Implementation: round 3

**Version:** v11 · 2026-11-04T18:18:10Z · skill version 8

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-combine-devolatilize` | `combine` | `builder-1-a`, `builder-1-b` | 1: pass, pass, 46.8 / 42.0 / 49.2 (55.5 / 56.8 / 72.8); 2: pass, pass, 31.8 / 34.7 / 32.7 (36.9 / 41.1 / 45.7) | candidate |
| `hyp-2-experts-prefill` | `experts` | `builder-2-a`, `builder-2-b`, `builder-2-c`, `builder-2-d`, `builder-2-e` | 1: pass, pass, 96.9 / 168.9 / 554.7 (86.6 / 238.2 / 883.4); 2: pass, pass, 152.8 / 173.3 / 634.1 (113.0 / 240.4 / 896.3); 3: fail (G1), —, —; 4: pass, pass, 122.2 / 206.4 / 584.7 (77.9 / 237.6 / 807.4); 5: pass, pass, 234.3 / 252.7 / 560.4 (118.4 / 291.7 / 802.4) | dropped |
| `hyp-3-routing-prefill` | `routing` | `builder-3-a`, `builder-3-b`, `builder-3-c`, `builder-3-d` | 1: pass, pass, 103.6 / 129.9 / 126.5 (139.4 / 157.2 / 149.7); 2: fail (G1), —, — | building |

The experts 1-row decode gap ([ineff-baseline-experts-2](../exploration/ineff-baseline-experts-2.md)) folds into hyp-2's hardening: the close rule needs `beats_compiled`, which needs the 1-row fix on attempt-1's line, so no separate hyp-4 opens.

Ranked decode item 1 (`ineff-g3-2`, the three-way interaction at 8 requests) is not opened as a hypothesis: it spans all three slots, its mechanism is still open, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the ready check times each candidate alone and all together, so round 3 sees composition inside the round. Next in order after one of the two open hypotheses closes: routing prefill (ranked prefill item 5) as hyp-3, then experts 1-row decode (ranked decode item 3) as hyp-4, built on hyp-2's winner. Tile waste and redundant offsets (items 7–8) ride with hyp-2's fast-branch directions; they are not separate hypotheses.

## Versions

- v11 · 2026-11-04T18:18:10Z · hyp-2 dropped (1-row gap structural, no beats_compiled; slot keeps reference); routing attempt-2 G1 fail, attempt-1 boundary pass · changed: hyp-2-experts-prefill v8, hyp-3-routing-prefill v4
- v10 · 2026-11-04T18:13:54Z · routing attempt-1 graded (G2 pass, decode faster x3, prefill 2048 noise-slower); ledger · changed: hyp-3-routing-prefill v3
- v9 · 2026-11-04T18:10:45Z · checker-1 pass: hyp-1 is a candidate (combine attempt-2) · changed: hyp-1-combine-devolatilize v7
- v8 · 2026-11-04T18:10:09Z · experts attempt-4 graded (exact grouped path, prefill 2.3x off — replica line ends); hyp-2 hardening on attempt-1 (1-row fix folds in, no hyp-4), hyp-3 to building · changed: hyp-2-experts-prefill v7, hyp-3-routing-prefill v2
- v7 · 2026-11-04T18:07:45Z · hyp-1 to hardening (attempt-2 best, Decision written, checker started), hyp-3 opened in boundary · changed: hyp-1-combine-devolatilize v6, hyp-3-routing-prefill v1
- v6 · 2026-11-04T18:06:32Z · experts attempt-2 graded (G2 pass, prefill still slower — vectorized scalar buys nothing); ledger · changed: hyp-2-experts-prefill v6
- v5 · 2026-11-04T18:05:59Z · combine attempt-2 graded (G2 pass, beats compiled), experts attempt-3 graded (G1 fail, reserved word); ledger + fix builder · changed: hyp-1-combine-devolatilize v5, hyp-2-experts-prefill v5
- v4 · 2026-11-04T18:03:53Z · boundary checks pass for combine attempt-1 and experts attempt-1 · changed: hyp-1-combine-devolatilize v4, hyp-2-experts-prefill v4
- v3 · 2026-11-04T17:58:28Z · combine attempt-1 graded (G2 pass, beats compiled both phases), experts attempt-1 graded (G2 pass, prefill same x3, 1-row misses); boundary checks started · changed: hyp-1-combine-devolatilize v3, hyp-2-experts-prefill v3
- v2 · 2026-11-04T17:55:21Z · boundaries drawn, 5 builders started (2 combine, 3 experts) · changed: hyp-1-combine-devolatilize v2, hyp-2-experts-prefill v2
- v1 · 2026-11-04T17:53:22Z · opened implementation, hyp-1 (combine) and hyp-2 (experts prefill) in boundary · changed: none
