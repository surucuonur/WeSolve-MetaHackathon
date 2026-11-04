# Implementation: round 2

**Version:** v10 · 2026-11-04T04:15:27Z · skill version 5

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-experts-exact` | `experts` | `builder-1-a`, `builder-1-b`, `builder-1-c`, `builder-1-d`, `builder-1-e`, `builder-1-f`, `builder-1-g` | 1: pass, pass, 104.3 / 193.3 / 535.7 (77.6 / 235.7 / 791.6); 2: pass, pass, 103.6 / 326.8 / 887.6 (73.1 / 232.0 / 804.5); 3: pass, pass, 109.5 / 195.0 / 637.7 (86.9 / 243.6 / 798.9); 4: pass, pass, 94.0 / 177.0 / 591.3 (73.5 / 239.0 / 796.3); 5: pass, pass, 92.1 / 162.0 / 534.8 (75.9 / 234.6 / 798.2); 6: fail (G1), —, —; 7: pass, pass, 93.8 / 160.1 / 510.1 (74.8 / 236.5 / 792.7) | dropped |
| `hyp-2-routing-exact` | `routing` | `builder-2-a`, `builder-2-b`, `builder-2-c`, `builder-2-d` | 1: pass, pass, 54.9 / 91.4 / 114.8 (56.2 / 91.2 / 98.9); 2: pass, fail (decode pass, prefill fail), —; 3: pass, pass, 46.9 / 300.0 / 297.2 (58.0 / 84.5 / 90.9); 4: pass, pass, 47.5 / 69.1 / 72.9 (60.7 / 84.9 / 89.5) | ready |
| `hyp-3-combine-exact` | `combine` | `builder-3-a`, `builder-3-b` | 1: pass, pass, 31.8 / 33.6 / 39.5 (35.4 / 38.2 / 43.4); 2: pass, pass, 45.2 / 37.6 / 33.8 (47.3 / 55.6 / 43.0) | ready |

Ranked item 1 (`ineff-g3-1`, the three-together interaction) is not opened as a hypothesis: it spans all three slots, its mechanism (serial-latency composition vs three-kernel interaction) is not yet separated, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose — and the pairwise G3 / chained routing-combine measurements it names remain owed by the team that runs G3. Building starts with the highest-ranked single-slot causes in two different slots.

No confirmed inefficiency is left that I judge will pay. The experts re-read floor (`ineff-baseline-experts-3`, 2.33 ms at 32 rows) banks only if a grouped design holds ≥120 GB/s on distinct bytes, which neither round-1 grouped attempt nor this round's packings achieved; the routing `w_router` re-read upper bound (`ineff-baseline-routing-4`) is already beaten in spite of itself by routing attempt-4, and round-1's tiled-sharing counter-evidence stands; the prefill-only items (`ineff-quantized-kernels-1`, `-2`) save 0 µs of decode time; the combine floor (`ineff-baseline-combine-2`) is kept, not removable by a kernel body. `routing+combine` as one candidate folder was not opened: the grader tests its parts independently (combine on reference weights), both parts are already `ready` as separate candidates, and a joint folder would add packaging, not a kernel change or a predicted saving.

## Hypotheses

**hyp-1-experts-exact** ([hyp-1-experts-exact.md](hyp-1-experts-exact.md), `experts`, dropped): seven graded attempts, six passing G2 bit-exact at all six sizes — the prefill gate is solved (scalar bfloat16-dequant + BK=32 order and MMA-tile forms both reproduce `gather_qmm_rhs`). None beats the compiled reference chained at 1 row: the fused designs plateau at 92–110 µs/layer at 1 row across per-pair, block-split and interleaved packings, while winning at 8/32 rows. Best attempt-7 (93.8 / 160.1 / 510.1) still loses at 1 row, so the hypothesis is dropped and the slot keeps the reference. Attempt-6 failed G1 (pipeline threadgroup limit).

**hyp-2-routing-exact** ([hyp-2-routing-exact.md](hyp-2-routing-exact.md), `routing`, ready): four graded attempts. Attempt-2 proved scalar sums reproduce MMA rounding at decode sizes; attempt-3 extended that to all six sizes but was latency-bound unstaged (300 µs at 8/32 rows); attempt-4 stages `w_router` coalesced and wins everywhere (47.5 / 69.1 / 72.9 vs compiled 60.7 / 84.9 / 89.5), G2 exact, boundary pass. Best: attempt-4.

**hyp-3-combine-exact** ([hyp-3-combine-exact.md](hyp-3-combine-exact.md), `combine`, ready): two graded attempts, both exact (volatile-barrier stepwise fold; the `precise` spelling does not compile) and both beating compiled chained. Attempt-1 (31.8 / 33.6 / 39.5 vs 35.4 / 38.2 / 43.4) wins at 1/8 rows and is simplest; attempt-2 is fastest at 32 rows. Best: attempt-1.

## Ready to test

Final attempts for G3: `experts` → the reference (no round-2 attempt; hyp-1 dropped, baseline `none` so the compiled reference is the bar and the unchanged model's experts run); `routing` → `muse-spark/round-2/kernels/routing/attempt-4`; `combine` → `muse-spark/round-2/kernels/combine/attempt-1`.

Expected change in decode step period (faster is negative). Worked out from the chained piece timing in each best attempt's own grading run — (compiled − candidate) µs/layer × 20 layers — plus the compiled references' measured G3 offsets in vLLM (`claude/findings/muse-round-1.md` §3: routing compiled −0.220 / −0.183 / −0.157 ms, combine compiled −0.288 / −0.260 / −0.272 ms against the unchanged model), summed per slot (the compiled references compose to within ±0.29 ms, same source). G3 decides; `ineff-g3-1` warns the sum may not compose.

| Slot | Piece basis vs compiled at 1 / 8 / 32 | Expected vs unchanged (piece + compiled G3 offset) |
|---|---|---|
| `routing` (attempt-4) | (60.7−47.5), (84.9−69.1), (89.5−72.9) ×20 = −0.26 / −0.32 / −0.33 ms | −0.48 / −0.50 / −0.49 ms |
| `combine` (attempt-1) | (35.4−31.8), (38.2−33.6), (43.4−39.5) ×20 = −0.07 / −0.09 / −0.08 ms | −0.36 / −0.35 / −0.35 ms |
| `experts` (reference) | — | 0.00 ms |
| **Together (sum)** | | **−0.84 / −0.85 / −0.84 ms vs unchanged** |

Against the previous round's measured all-three-attempts step (+0.98 / +0.16 / −1.76 ms in vLLM, same source), the expected change is therefore −1.82 / −1.01 / +0.92 ms at 1 / 8 / 32 requests: faster at 1 and 8 (the old experts 1-row penalty and the inexact kernels are gone), slower at 32, where round 1's experts kernel alone saved 4.15 ms and this round banks no experts saving because no exact attempt beats the compiled reference at 1 row. Caveats: each piece figure comes from its own grading run (1-row chained baselines shift between runs; within-run vs-compiled is the valid comparison), the piece timing sums isolated 20-layer chains while G3 times full steps, and the interaction term is unmeasured for the new pair — watch the repeat spread, thinnest at 1 row.

## Versions

- v10 · 2026-11-04T04:15:27Z · final: tracker, hypothesis paragraphs, Ready to test · changed: none (README only)
- v9 · 2026-11-04T04:14:47Z · experts attempts 6 (G1 fail) and 7 graded; hyp-1 dropped, best loses at 1 row · changed: hyp-1-experts-exact v8
- v8 · 2026-11-04T04:04:00Z · routing attempt-4 and combine attempts graded; hyp-2 and hyp-3 ready · changed: hyp-1-experts-exact v7, hyp-2-routing-exact v8, hyp-3-combine-exact v2
- v7 · 2026-11-04T03:52:53Z · hardening attempts graded (experts-3, routing-3 both G2 pass, neither beats best) · changed: hyp-1-experts-exact v6, hyp-2-routing-exact v7
- v6 · 2026-11-04T03:49:13Z · both hyps to hardening, 2 hardening builders started · changed: hyp-1-experts-exact v5, hyp-2-routing-exact v6
- v5 · 2026-11-04T03:48:31Z · routing attempt-2 graded (scalar decode-exact, prefill fail by design) · changed: hyp-2-routing-exact v5
- v4 · 2026-11-04T03:47:53Z · boundary checks pass for experts attempt-1/2 and routing attempt-1 · changed: hyp-1-experts-exact v4, hyp-2-routing-exact v4
- v3 · 2026-11-04T03:43:13Z · experts attempts 1+2 and routing attempt 1 graded, all G2 pass · changed: hyp-1-experts-exact v3, hyp-2-routing-exact v3
- v2 · 2026-11-04T03:32:10Z · boundaries drawn, 4 builders started · changed: hyp-1-experts-exact v2, hyp-2-routing-exact v2
- v1 · 2026-11-04T03:29:15Z · opened implementation, hyp-1 (experts) and hyp-2 (routing) in boundary · changed: none
