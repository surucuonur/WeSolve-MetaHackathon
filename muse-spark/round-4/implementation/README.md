# Implementation: round 4

**Version:** v11 · 2026-11-04T22:07:00Z · skill version 10

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-3-combine-setcopy` | `combine` | — | — | boundary |
| `hyp-1-experts-minrows` | `experts` | `builder-1-a`, `builder-1-b` | 1: pass, pass, handed off / 157.6 / 571.1 (232.1 / 832.6); 2: pass, pass, handed off / 170.5 / 538.1 (236.8 / 823.2) | candidate |
| `hyp-2-routing-margin` | `routing` | `builder-2-a`, `builder-2-b` | 1: pass, pass, 58.5 / 136.9 / 74.7 (86.1 / 157.3 / 92.1); 2: pass, pass, 44.4 / 71.6 / 83.8 (59.2 / 87.6 / 92.1) | building |

Ranked item 6 (combine at its floor) opens no kernel hypothesis: exploration confirms no bit-exact change widens its decode margin, so the slot is carried into the set by a byte-identical re-grade of its starting point under `hyp-3-combine-setcopy`. The three-slot interaction (`ineff-g3-2`) is carried as a constraint on every set claim and is measured by the ready check, not built against.

## Versions

- v11 · 2026-11-04T22:07:00Z · checker-1 pass: hyp-1 is a candidate (experts attempt-1) · changed: hyp-1-experts-minrows v9
- v10 · 2026-11-04T22:05:00Z · hyp-1 and hyp-2 closing (Decisions on experts attempt-1, routing attempt-1), hyp-3 combine set-copy opened in boundary · changed: hyp-1-experts-minrows v8, hyp-2-routing-margin v4, hyp-3-combine-setcopy v1
- v9 · 2026-11-04T22:04:30Z · hyp-1 checker defects fixed, re-check next · changed: hyp-1-experts-minrows v8
- v8 · 2026-11-04T22:04:00Z · hyp-2 closing on routing attempt-1 · changed: hyp-2-routing-margin v4
- v7 · 2026-11-04T22:03:30Z · hyp-1 Unverified written · changed: hyp-1-experts-minrows v7
- v6 · 2026-11-04T21:58:00Z · routing attempt-3 graded, hyp-1 closing on attempt-1 · changed: hyp-1-experts-minrows v6, hyp-2-routing-margin v3
- v5 · 2026-11-04T21:57:30Z · routing attempts 1+2 graded (both beats_compiled; direction a narrowed 32-row margin), experts boundary checks pass · changed: hyp-1-experts-minrows v5, hyp-2-routing-margin v3
- v4 · 2026-11-04T21:55:00Z · experts attempt-2 graded, beats_compiled true; boundary checks next · changed: hyp-1-experts-minrows v4
- v3 · 2026-11-04T21:53:17Z · experts attempt-1 graded, beats_compiled true (1-row and prefill handed off); boundary check next · changed: hyp-1-experts-minrows v3
- v2 · 2026-11-04T21:52:00Z · boundaries drawn, 4 builders started (2 experts, 2 routing) · changed: hyp-1-experts-minrows v2, hyp-2-routing-margin v2
- v1 · 2026-11-04T21:49:10Z · opened implementation, hyp-1 (experts min_rows) and hyp-2 (routing margin) in boundary · changed: hyp-1-experts-minrows v1, hyp-2-routing-margin v1
