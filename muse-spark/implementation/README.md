# Implementation: round 1

**Version:** v11 · 2026-11-03T22:55:00Z · skill version 3

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-fused-experts` | `experts` | `builder-1-a`, `builder-1-b`, `builder-1-c`, `builder-1-d` | 1: pass, pass, 79.3 / 260.3 / 743.7 (62.0 / 214.1 / 761.0); 2: pass, pass, 107.3 / 188.4 / 647.6 (135.0 / 216.9 / 761.7); 3: pass, pass, 68.1 / 238.0 / 654.0 (65.0 / 213.0 / 764.0); 4: pass, pass, 50.0 / 157.8 / 515.4 (65.3 / 217.4 / 760.2) | hardening |
| `hyp-2-fused-routing` | `routing` | `builder-2-a`, `builder-2-b` | 1: pass, pass, 28.5 / 23.6 / 41.8 (30.5 / 42.0 / 49.1); 2: pass, pass, 36.3 / 34.9 / 49.0 (29.3 / 42.9 / 46.6) | ready |
| `hyp-3-fused-combine` | `combine` | `builder-3-a`, `builder-3-b` | 1: pass, pass, 17.2 / 17.0 / 18.5 (22.9 / 26.2 / 28.0); 2: pass, pass, 17.9 / 16.1 / 18.1 (20.9 / 23.0 / 27.9) | ready |

## Versions

- v11 · 2026-11-03T22:55:00Z · experts attempt-4 graded (bit-exact, beats eager+compiled everywhere), attempt-1 boundary pass · changed: hyp-1-fused-experts v7
- v10 · 2026-11-03T22:45:00Z · experts attempt-1 graded (G2 pass, beats_compiled false); hyp-1 hardening from attempt-2 · changed: hyp-1-fused-experts v6
- v9 · 2026-11-03T22:35:00Z · hyp-3 ready (attempt-1 to G3) · changed: hyp-3-fused-combine v4
- v8 · 2026-11-03T22:25:00Z · experts attempt-3 boundary pass · changed: hyp-1-fused-experts v5
- v7 · 2026-11-03T22:15:00Z · experts attempt-3 graded (G2 pass, beats_compiled false), attempt-2 boundary pass · changed: hyp-1-fused-experts v4
- v6 · 2026-11-03T22:05:00Z · experts attempt-2 and combine attempts 1+2 graded (all G2 pass, all beat compiled); boundary checks started · changed: hyp-1-fused-experts v3, hyp-3-fused-combine v3
- v5 · 2026-11-03T21:55:00Z · hyp-3 boundary drawn, 2 combine builders started · changed: hyp-3-fused-combine v2
- v4 · 2026-11-03T21:52:00Z · hyp-2 ready (attempt-1 to G3), hyp-3 opened in boundary · changed: hyp-2-fused-routing v4, hyp-3-fused-combine v1
- v3 · 2026-11-03T21:41:00Z · routing attempts 1+2 graded (both pass G2, attempt-1 beats compiled everywhere), boundary checks started · changed: hyp-2-fused-routing v3
- v2 · 2026-11-03T21:35:00Z · boundaries drawn, 5 builders started · changed: hyp-1-fused-experts v2, hyp-2-fused-routing v2
- v1 · 2026-11-03T21:32:00Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
