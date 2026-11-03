# Implementation: round 1

**Version:** v12 · 2026-11-03T23:06:00Z · skill version 3

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-fused-experts` | `experts` | `builder-1-a`, `builder-1-b`, `builder-1-c`, `builder-1-d` | 1: pass, pass, 79.3 / 260.3 / 743.7 (62.0 / 214.1 / 761.0); 2: pass, pass, 107.3 / 188.4 / 647.6 (135.0 / 216.9 / 761.7); 3: pass, pass, 68.1 / 238.0 / 654.0 (65.0 / 213.0 / 764.0); 4: pass, pass, 50.0 / 157.8 / 515.4 (65.3 / 217.4 / 760.2) | ready |
| `hyp-2-fused-routing` | `routing` | `builder-2-a`, `builder-2-b` | 1: pass, pass, 28.5 / 23.6 / 41.8 (30.5 / 42.0 / 49.1); 2: pass, pass, 36.3 / 34.9 / 49.0 (29.3 / 42.9 / 46.6) | ready |
| `hyp-3-fused-combine` | `combine` | `builder-3-a`, `builder-3-b` | 1: pass, pass, 17.2 / 17.0 / 18.5 (22.9 / 26.2 / 28.0); 2: pass, pass, 17.9 / 16.1 / 18.1 (20.9 / 23.0 / 27.9) | ready |

No confirmed in-slot inefficiency is left that would pay: all 7 ranked entries fed the three hypotheses above (5 experts entries → hyp-1, routing → hyp-2, combine → hyp-3), and each closed `ready` with its best attempt beating the compiled reference at all three row counts.

## Hypotheses

**hyp-1-fused-experts** ([hyp-1-fused-experts.md](hyp-1-fused-experts.md), `experts`, ready): four builders competed. The grouped designs lost — attempt-1 (full fused grouped single kernel, 15 internal versions) and attempt-3 (grouped two-kernel split) pass G2 but beat compiled only at 32 rows, because grouping's byte savings don't convert in the piece harness (duty-cycle ceiling ~55 GB/s; scan/chunk/barrier overhead). Attempt-2's simpler per-pair fusion (one kernel, gate/up/down + bf16 silu, qvm-exact numerics) beat compiled everywhere but regressed vs eager at 1 row on occupancy; hardening (attempt-4, 256 threads/pair) fixed that and added memory-level parallelism, beating both eager and compiled at all rows, bit-exact. Best: attempt-4.

**hyp-2-fused-routing** ([hyp-2-fused-routing.md](hyp-2-fused-routing.md), `routing`, ready): attempt-1's single fused kernel (scores + top-4 + norm, one 32-thread group per row) beat compiled at all rows with 40× G2 margin; attempt-2's two-kernel split won only at 8 rows. Both passed boundary checks. No hardening: one minimal launch per layer, residual ~24 µs/layer is dispatch/eval floor. Best: attempt-1.

**hyp-3-fused-combine** ([hyp-3-fused-combine.md](hyp-3-fused-combine.md), `combine`, ready): both single-kernel directions (thread-per-output and threadgroup-per-row) beat compiled at all rows and sit on the same ~17 µs/layer dispatch floor; attempt-1 wins on wider relative margins and simplicity. The 8-row 2.0-ULP touch was proven to be FMA contraction, max exactly 2.0 on realistic inputs. No hardening: at the bare dispatch floor. Best: attempt-1.

## Ready to test

Final attempts for G3: `experts` → `muse-spark/kernels/experts/attempt-4`; `routing` → `muse-spark/kernels/routing/attempt-1`; `combine` → `muse-spark/kernels/combine/attempt-1`.

Expected change in decode step period (faster is negative), worked out two ways. Piece basis: (reference − candidate) µs/layer from each best attempt's own grading run × 20 layers. Knock-out basis: the inefficiency entries' predicted savings. G3 decides between them.

| Slot | Piece basis at 1 / 8 / 32 requests | Knock-out basis at 1 / 8 / 32 |
|---|---|---|
| `experts` (attempt-4) | −0.29 / −1.22 / −5.01 ms | −0.40 / −1.30 / −3.67 ms (measured grouped) to −0.5 / −2.5 / −8.6 ms (ideal) |
| `routing` (attempt-1) | −0.15 / −0.49 / −0.47 ms | −0.9 / −1.2 / −1.4 ms |
| `combine` (attempt-1) | −0.10 / −0.10 / −0.24 ms | −0.47 / −0.30 / −0.19 ms |

Arithmetic: experts (64.3−50.0), (218.6−157.8), (765.7−515.4) × 20; routing (36.1−28.5), (48.0−23.6), (65.4−41.8) × 20; combine (22.4−17.2), (22.1−17.0), (30.7−18.5) × 20 — all µs/layer to ms/step. Caveats for G3: 1-row piece-harness baselines shift between runs (experts eager 64.3 here vs 99.8 in attempt-2's run; compiled 62–135 across runs), so the 1-row margins need the spread check; 8/32-row baselines are stable. The piece timing sums isolated 20-layer evals while G3 times full steps, so cache/overlap effects can move the truth either way.

## Versions

- v12 · 2026-11-03T23:06:00Z · final: tracker, hypothesis paragraphs, Ready to test · changed: hyp-1-fused-experts v8
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
