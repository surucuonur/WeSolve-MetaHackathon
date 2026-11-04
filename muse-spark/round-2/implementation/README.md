# Implementation: round 2

**Version:** v7 · 2026-11-04T03:52:53Z · skill version 5

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-experts-exact` | `experts` | `builder-1-a`, `builder-1-b`, `builder-1-c` | 1: pass, pass, 104.3 / 193.3 / 535.7 (77.6 / 235.7 / 791.6); 2: pass, pass, 103.6 / 326.8 / 887.6 (73.1 / 232.0 / 804.5); 3: pass, pass, 109.5 / 195.0 / 637.7 (86.9 / 243.6 / 798.9) | hardening |
| `hyp-2-routing-exact` | `routing` | `builder-2-a`, `builder-2-b`, `builder-2-c` | 1: pass, pass, 54.9 / 91.4 / 114.8 (56.2 / 91.2 / 98.9); 2: pass, fail (decode pass, prefill fail), —; 3: pass, pass, 46.9 / 300.0 / 297.2 (58.0 / 84.5 / 90.9) | hardening |

Ranked item 1 (`ineff-g3-1`, the three-together interaction) is not opened as a hypothesis: it spans all three slots, its mechanism (serial-latency composition vs three-kernel interaction) is not yet separated, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the pairwise G3 / chained routing-combine measurements it names remain owed by the team that runs G3. Building starts with the highest-ranked single-slot causes in two different slots.

## Versions

- v7 · 2026-11-04T03:52:53Z · hardening attempts graded (experts-3, routing-3 both G2 pass, neither beats best) · changed: hyp-1-experts-exact v6, hyp-2-routing-exact v7
- v6 · 2026-11-04T03:49:13Z · both hyps to hardening, 2 hardening builders started · changed: hyp-1-experts-exact v5, hyp-2-routing-exact v6
- v5 · 2026-11-04T03:48:31Z · routing attempt-2 graded (scalar decode-exact, prefill fail by design) · changed: hyp-2-routing-exact v5
- v4 · 2026-11-04T03:47:53Z · boundary checks pass for experts attempt-1/2 and routing attempt-1 · changed: hyp-1-experts-exact v4, hyp-2-routing-exact v4
- v3 · 2026-11-04T03:43:13Z · experts attempts 1+2 and routing attempt 1 graded, all G2 pass · changed: hyp-1-experts-exact v3, hyp-2-routing-exact v3
- v2 · 2026-11-04T03:32:10Z · boundaries drawn, 4 builders started · changed: hyp-1-experts-exact v2, hyp-2-routing-exact v2
- v1 · 2026-11-04T03:29:15Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
