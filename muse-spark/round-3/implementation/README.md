# Implementation: round 3

**Version:** v21 · 2026-11-04T18:59:13Z · skill version 8

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-combine-devolatilize` | `combine` | `builder-1-a`, `builder-1-b` | 1: pass, pass, 46.8 / 42.0 / 49.2 (55.5 / 56.8 / 72.8); 2: pass, pass, 31.8 / 34.7 / 32.7 (36.9 / 41.1 / 45.7) | dropped |
| `hyp-2-experts-prefill` | `experts` | `builder-2-a`, `builder-2-b`, `builder-2-c`, `builder-2-d`, `builder-2-e` | 1: pass, pass, 96.9 / 168.9 / 554.7 (86.6 / 238.2 / 883.4); 2: pass, pass, 152.8 / 173.3 / 634.1 (113.0 / 240.4 / 896.3); 3: fail (G1), —, —; 4: pass, pass, 122.2 / 206.4 / 584.7 (77.9 / 237.6 / 807.4); 5: pass, pass, 234.3 / 252.7 / 560.4 (118.4 / 291.7 / 802.4) | dropped |
| `hyp-3-routing-prefill` | `routing` | `builder-3-a`, `builder-3-b`, `builder-3-c`, `builder-3-d`, `builder-3-e`, `builder-3-f` | 1: pass, pass, 103.6 / 129.9 / 126.5 (139.4 / 157.2 / 149.7); 2: fail (G1), —, —; 3: pass, pass, 66.1 / 96.1 / 109.5 (64.7 / 95.4 / 103.9); 4: fail (G1), —, —; 5: pass, pass, 72.6 / 123.3 / 111.9 (76.2 / 120.4 / 107.7); 6: pass, pass, 51.7 / 90.9 / 132.2 (66.3 / 104.0 / 152.0) | dropped |

The experts 1-row decode gap ([ineff-baseline-experts-2](../exploration/ineff-baseline-experts-2.md)) folds into hyp-2's hardening: the close rule needs `beats_compiled`, which needs the 1-row fix on attempt-1's line, so no separate hyp-4 opens.

Ranked decode item 1 (`ineff-g3-2`, the three-way interaction at 8 requests) is not opened as a hypothesis: it spans all three slots, its mechanism is still open, and no single-slot kernel change is predicted to remove it. It is carried as a constraint — per-slot wins are not assumed to compose. Tile waste and redundant offsets (items 7–8) ride with fast-branch directions; they are not separate hypotheses.

No confirmed inefficiency is left that I judge will pay. Combine's volatiles are removed by attempt-2 (exact everywhere, faster everywhere in-grader, medians show the +1.05 ms gone) but the ready check fails it alone on paired spread, and no kernel change can tighten spread on 7 plain ops. Routing's prefill gap is banked at `same` by the `max_rows` hand-off; the GEMM-sharing line proved exact but stays 1.44–3.50× off with decode `same` at 8/32 rows, and the narrower-footprint line added a copy launch that flipped decode to `same`. Experts' prefill is banked at `same` by its hand-off but the 1-row decode gap blocks the slot, and the gap is structural (fused plateau plus unfused 2× worse). Tile waste and redundant offsets need a fast prefill branch and none exists. The three-way 8-request term still has no mechanism, and this round installed no experts candidate so it stays untested. A `routing+combine` joint folder would be packaging (parts grade independently) with no new kernel content.

## Hypotheses

**hyp-1-combine-devolatilize** ([hyp-1-combine-devolatilize.md](hyp-1-combine-devolatilize.md), `combine`, dropped): two builders, two exact zero-traffic decontractions (noinline barriers, pragma FP_CONTRACT OFF), both G2-exact at all six sizes with boundary passes and `beats_compiled` true. Best attempt-2 (simplest). Checker-1 passed it as a candidate; the ready check failed it alone (vs base and vs compiled-combine fail, all phases `not slower`, one 1×128 outlier rep overlapping base) while every decode median favored it (−0.53/−0.59/−0.44/−0.26 ms vs base). Dropped: spread names no change. Slot keeps the reference.

**hyp-2-experts-prefill** ([hyp-2-experts-prefill.md](hyp-2-experts-prefill.md), `experts`, dropped): five attempts. Attempt-1 (`max_rows` 59) banks prefill `same` × 3 with decode wins at 8/32 rows but misses 1 row, so `beats_compiled` is false. The vectorized scalar (attempt-2) buys nothing, the grouped replica (attempt-4) is bit-exact but 2.3× off, and the unfused 1-row path (attempt-5) is 2× worse than fused — the reference's speed is one `mx.compile` dispatch, not just its shape. Dropped: the 1-row gap survives both approaches and no third exists in the spec surface. Slot keeps the reference.

**hyp-3-routing-prefill** ([hyp-3-routing-prefill.md](hyp-3-routing-prefill.md), `routing`, dropped): six attempts. Attempt-1/6 (`max_rows` 32) banks decode `faster` × 3 and prefill `same` × 3 (`beats_compiled` true; the first run's 2,048-row `slower` on identical code did not reproduce in the committed one-run noise retest). Checker-3 passed it as a candidate after one Unverified fix. The ready check failed it alone (vs base pass; vs compiled-routing fail on the 8×128 range overlap) with no phase slower anywhere. Dropped: byte-identical decode to a G3-passing kernel and prefill that IS compiled name no change. Slot keeps the reference.

## Ready to test

Final attempts for G3: `experts` → the reference (hyp-2 dropped); `routing` → the reference (hyp-3 dropped); `combine` → the reference (hyp-1 dropped). No slot has a `ready` attempt and no slot has a baseline, so every slot keeps the reference. The ready check ([ready-verdict.json](../checks/ready-1/ready-verdict.json), exit 1, verdict stands) timed the two dropped candidates alone and together, 5 repeats, same tokens everywhere:

| Variant | Decode step median vs base at 1×128 / 8×128 / 32×128 / 32×1024 | Prompt median vs base | Whole request median vs base | Verdict |
|---|---|---|---|---|
| `muse-routing` (attempt-6) | −0.38 / −0.31 / −0.59 / −0.48 ms | +1.8 / +0.5 / −3.6 / −10.8 ms | −92 / −70 / −117 / −130 ms | FAIL (vs base pass; vs compiled-routing fail — 8×128 ranges overlap) |
| `muse-combine` (attempt-2) | −0.53 / −0.59 / −0.44 / −0.26 ms | +1.4 / −8.0 / −39.7 / −180.0 ms | −112 / −154 / −103 / −250 ms | FAIL (vs base fail — 1×128 overlap on one 5.892 rep; vs compiled-combine fail) |
| `muse-all` (both together) | −0.92 / −0.89 / −0.73 / −0.78 ms | −0.9 / −6.2 / −38.0 / −174.0 ms | −231 / −227 / −208 / −390 ms | PASS (vs base pass and vs compiled-all pass; decode `faster`, request `faster`, prompt `not slower`) |

Against the earlier predictions (PREDICTION): combine predicted decode ≈ −0.51/−0.71/−0.39 ms/step at 1/8/32 requests (twin-swing recovery) — measured −0.53/−0.59/−0.44 ms at 1×128/8×128/32×128 in medians, the prediction held where the verdict failed on spread; its prompt twin (order 1 s on 32×128) measured −39.6 ms here because this run pairs combine with reference experts, not the 14.5×-slow experts branch the twin was measured against. Routing predicted keeping −0.47/−0.43/−0.54 ms decode and prompt to parity — measured −0.38/−0.31/−0.59/−0.48 ms decode and +1.8/+0.5/−3.6/−10.8 ms prompt (the starting point's +3.3/+19.1/+77.7/+662.5 ms prompt excess is gone). No phase of any variant is `slower` anywhere; every failure is `not slower` on range overlap.

**Unverified:** the 2,048-row identical-code `slower` rate (one 8/30 event in two grader runs); the 1×128 outlier rep's cause (machine hiccup vs kernel, single event); the three-way 8-request term (untestable without an experts candidate); `muse-all` is measured but not submitted (no per-slot `ready` exists, so per-slot submission is all-reference); the second ready check is unused.

**Check** (checker-final, 2026-11-04T18:58:35Z): 7 defects — last-digit rounding slips in the median-vs-base table, all redone from `muse-spark/round-3/checks/ready-1/ready-verdict.json` medians (difference-of-medians; median-of-paired-diffs does not reproduce the table, so difference-of-medians is the method). No verdict, comparison, or conclusion changes; fix the digits and re-check.
- `implementation/README.md:33` (`muse-routing`): decode 1×128 −0.39 → −0.38 (5.085−5.470=−0.3846); request 8×128 −69 → −70 (3830.59−3900.13=−69.54); request 32×128 −116 → −117 (8364.72−8481.37=−116.65).
- `implementation/README.md:34` (`muse-combine`): prompt 32×128 −39.6 → −39.7 (757.77−797.43=−39.66); request 8×128 −153 → −154 (3746.50−3900.13=−153.63).
- `implementation/README.md:35` (`muse-all`): prompt 32×128 −37.9 → −38.0 (759.47−797.43=−37.96); request 32×128 −207 → −208 (8273.75−8481.37=−207.62).
- `implementation/README.md:37`: routing measured decode −0.39 → −0.38 (same cell as L33).
- `summary.md:8`: −0.39 → −0.38; −37.9 → −38.0; −207 → −208 (same three cells).
- Fix-consistency note (outside scope): `hyp-3-routing-prefill.md:263` carries −0.39 in the same routing decode list.
- Passing: `--audit` on `checks/ready-1` prints "ready-verdict.json agrees with this audit" (exit 1, FAIL stands); all 29 other table cells reproduce; every verdict string matches the JSON (`muse-routing` fail on the 8×128 overlap 14.451>14.312; `muse-combine` fail on the single 5.892 rep vs base min 5.430; `muse-all` pass, decode/request `faster`, prompt `not slower`); no phase `slower` anywhere; `muse-all` cites the `muse-all` variant, not a sum; predictions labelled (PREDICTION) with hyp-entry bases already checker-passed; Unverified covers the noise/outlier/untestable/not-submitted items with no free-cost language; all three slots keep the reference with zero `ready` attempts (ready-folder check vacuous); attempt counts 16/12/6 and round-1 carried recommendations re-verified.

**Re-check** (checker-final, 2026-11-04T18:59:45Z): check: pass. All 7 fixed digits verified against `muse-spark/round-3/checks/ready-1/ready-verdict.json` (difference-of-medians): L33 −0.38/−70/−117, L34 −39.7/−154, L35 −38.0/−208, L37 −0.38, `summary.md:8` −0.38/−38.0/−208, `hyp-3-routing-prefill.md:263` −0.38 — and all 36 table cells now reproduce. `--audit` re-run: "ready-verdict.json agrees with this audit" (exit 1, FAIL stands); all other final-claims checks re-run unchanged.

## Versions

- v21 · 2026-11-04T18:59:13Z · checker-final's 7 rounding digits fixed across Ready to test, summary and hyp-3 Decision · changed: hyp-3-routing-prefill v13
- v20 · 2026-11-04T18:54:08Z · final: tracker, hypothesis paragraphs, Ready to test (all reference; muse-all measured passing) · changed: none (README only)
- v19 · 2026-11-04T18:53:18Z · ready-1 verdict stands (both fail alone, muse-all passes); hyp-1 and hyp-3 dropped, slots keep reference · changed: hyp-1-combine-devolatilize v8, hyp-3-routing-prefill v12
- v18 · 2026-11-04T18:31:55Z · checker-3 re-check pass: hyp-3 is a candidate (routing attempt-6); ready check next · changed: hyp-3-routing-prefill v11
- v17 · 2026-11-04T18:30:28Z · checker-3 found 1 defect (Unverified omitted can't-see items); fixed, re-check next · changed: hyp-3-routing-prefill v10
- v16 · 2026-11-04T18:27:11Z · hyp-3 closing (attempt-6 best, boundary pass, Decision written, checker started) · changed: hyp-3-routing-prefill v9
- v15 · 2026-11-04T18:26:21Z · routing attempt-6 graded (identical copy, 2048 slower not reproduced, beats_compiled true — best); boundary check next · changed: hyp-3-routing-prefill v8
- v14 · 2026-11-04T18:24:53Z · routing attempt-5 graded (GEMM exact, prefill 1.4-3.5x off — line ends); hyp-3 hardening on attempt-1, one re-grade to test noise · changed: hyp-3-routing-prefill v7
- v13 · 2026-11-04T18:22:08Z · routing attempt-4 graded (G1 fail, pipeline cap moved 640→576); ledger, one more fix at 8 rows/group · changed: hyp-3-routing-prefill v6
- v12 · 2026-11-04T18:19:19Z · routing attempt-3 graded (G2 pass, decode same x3, prefill still slower — re-read dominates); ledger · changed: hyp-3-routing-prefill v5
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
