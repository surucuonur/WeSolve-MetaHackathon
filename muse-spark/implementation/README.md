# Implementation: round 1

**Version:** v4 · 2026-11-03T21:52:00Z · skill version 3

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-fused-experts` | `experts` | `builder-1-a`, `builder-1-b`, `builder-1-c` | 0 (attempts 1/2/3 claimed, grading) | building |
| `hyp-2-fused-routing` | `routing` | `builder-2-a`, `builder-2-b` | 1: pass, pass, 28.5 / 23.6 / 41.8 (30.5 / 42.0 / 49.1); 2: pass, pass, 36.3 / 34.9 / 49.0 (29.3 / 42.9 / 46.6) | ready |
| `hyp-3-fused-combine` | `combine` | — | 0 | boundary |

## Versions

- v4 · 2026-11-03T21:52:00Z · hyp-2 ready (attempt-1 to G3), hyp-3 opened in boundary · changed: hyp-2-fused-routing v4, hyp-3-fused-combine v1
- v3 · 2026-11-03T21:41:00Z · routing attempts 1+2 graded (both pass G2, attempt-1 beats compiled everywhere), boundary checks started · changed: hyp-2-fused-routing v3
- v2 · 2026-11-03T21:35:00Z · boundaries drawn, 5 builders started · changed: hyp-1-fused-experts v2, hyp-2-fused-routing v2
- v1 · 2026-11-03T21:32:00Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
