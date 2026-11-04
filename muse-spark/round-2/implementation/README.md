# Implementation: round 2

**Version:** v1 · 2026-11-04T03:29:15Z · skill version 5

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-experts-exact` | `experts` | — | — | boundary |
| `hyp-2-routing-exact` | `routing` | — | — | boundary |

Ranked item 1 (`ineff-g3-1`, the three-together interaction) is not opened as a hypothesis: it spans all three slots, its mechanism (serial-latency composition vs three-kernel interaction) is not yet separated, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the pairwise G3 / chained routing-combine measurements it names remain owed by the team that runs G3. Building starts with the highest-ranked single-slot causes in two different slots.

## Versions

- v1 · 2026-11-04T03:29:15Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
