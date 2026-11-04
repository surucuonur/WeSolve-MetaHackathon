# Implementation: round 2

**Version:** v4 · 2026-11-04T03:47:53Z · skill version 5

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-experts-exact` | `experts` | `builder-1-a`, `builder-1-b` | 1: pass, pass, 104.3 / 193.3 / 535.7 (77.6 / 235.7 / 791.6); 2: pass, pass, 103.6 / 326.8 / 887.6 (73.1 / 232.0 / 804.5) | building |
| `hyp-2-routing-exact` | `routing` | `builder-2-a`, `builder-2-b` | 1: pass, pass, 54.9 / 91.4 / 114.8 (56.2 / 91.2 / 98.9) | building |

Ranked item 1 (`ineff-g3-1`, the three-together interaction) is not opened as a hypothesis: it spans all three slots, its mechanism (serial-latency composition vs three-kernel interaction) is not yet separated, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the pairwise G3 / chained routing-combine measurements it names remain owed by the team that runs G3. Building starts with the highest-ranked single-slot causes in two different slots.

## Versions

- v4 · 2026-11-04T03:47:53Z · boundary checks pass for experts attempt-1/2 and routing attempt-1 · changed: hyp-1-experts-exact v4, hyp-2-routing-exact v4
- v3 · 2026-11-04T03:43:13Z · experts attempts 1+2 and routing attempt 1 graded, all G2 pass · changed: hyp-1-experts-exact v3, hyp-2-routing-exact v3
- v2 · 2026-11-04T03:32:10Z · boundaries drawn, 4 builders started · changed: hyp-1-experts-exact v2, hyp-2-routing-exact v2
- v1 · 2026-11-04T03:29:15Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
