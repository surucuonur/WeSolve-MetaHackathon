# Implementation: round 4

**Version:** v21 · 2026-11-04T23:07:00Z · skill version 10

## Tracker

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-experts-minrows` | `experts` | `builder-1-a`, `builder-1-b` | 1: pass, pass, handed off / 157.6 / 571.1 (232.1 / 832.6); 2: pass, pass, handed off / 170.5 / 538.1 (236.8 / 823.2) | dropped |
| `hyp-2-routing-margin` | `routing` | `builder-2-a`, `builder-2-b`, `builder-2-c` | 1: pass, pass, 58.5 / 136.9 / 74.7 (86.1 / 157.3 / 92.1); 2: pass, pass, 44.4 / 71.6 / 83.8 (59.2 / 87.6 / 92.1); 3: pass, pass, 49.3 / 74.5 / 81.8 (66.2 / 86.1 / 94.8) | dropped |
| `hyp-3-combine-setcopy` | `combine` | `builder-3-a` | 1: pass, pass, 39.3 / 45.5 / 43.2 (47.1 / 60.5 / 69.6) | dropped |

No confirmed inefficiency is left that I judge will pay. Every graded attempt this round passes G1, G2 and `beats_compiled`, and every one fails the ready check on 5-repeat range overlap with no phase slower anywhere in either run — the loss mode is spread, and no kernel change in the confirmed set addresses spread: combine is at its chained floor, the routing pools each had their simplest exact attack (sharing narrowed the margin; shape trims did not widen it), and the experts hand-off works exactly as designed but its slot can pass only inside a set, and both sets it was timed in failed — the three-slot set on its 8×128 decode (+0.32 ms/step vs base, the interaction rounds 1–2 saw), the pair on overlap at 32 rows. Both permitted ready checks are used. No slot has a `ready` attempt and no slot has a baseline, so every slot keeps the reference.

## Hypotheses

**hyp-1-experts-minrows** ([hyp-1-experts-minrows.md](hyp-1-experts-minrows.md), `experts`, dropped): the `min_rows: 2` hand-off turns round 3's structural 1-row loss into a `handed off` size; both attempts (windows 2–59 and 2–32) are bit-exact with `beats_compiled` true and boundary passes, checker-1 passed attempt-1 as a candidate. The ready check fails it alone (+0.06 ms/step at 1×128, as predicted — it runs MLX's own code there) and in the three-slot set. Slot keeps the reference.

**hyp-2-routing-margin** ([hyp-2-routing-margin.md](hyp-2-routing-margin.md), `routing`, dropped): three bit-exact attempts, all `beats_compiled` true with boundary passes; attempt-1 (dead zeroing removed) is best on same-run margins, direction-(a) sharing (attempt-2) narrowed the 32-row margin to 8.3 µs/layer, and tail-group feeding (attempt-3) widened nothing. Checker-2 passed attempt-1 as a candidate. Both ready checks fail it alone and in set/pair on overlap. Slot keeps the reference.

**hyp-3-combine-setcopy** ([hyp-3-combine-setcopy.md](hyp-3-combine-setcopy.md), `combine`, dropped): the byte-identical set member (verified by `diff`/`cmp`, boundary pass, checker-3 pass as candidate) is `faster` at all six sizes in this round's grade. Both ready checks fail it alone and in set/pair on overlap. Slot keeps the reference.

## Ready to test

Final attempts for G3: `experts` → the reference (hyp-1 dropped); `routing` → the reference (hyp-2 dropped); `combine` → the reference (hyp-3 dropped). No slot has a `ready` attempt and no slot has a baseline, so every slot keeps the reference. Two ready checks ran (the round's maximum), both exit 1 with verdicts that stand, same greedy tokens everywhere. Ready-1 ([ready-verdict.json](../checks/ready-1/ready-verdict.json)) timed each candidate alone and all three together; ready-2 ([ready-verdict.json](../checks/ready-2/ready-verdict.json)) timed routing+combine alone and as a pair. Difference-of-medians against the unchanged model (base), ms:

| Check / variant | Decode step at 1×128 / 8×128 / 32×128 / 32×1024 | Prompt | Whole request | Verdict |
|---|---|---|---|---|
| ready-1 `muse-experts` (attempt-1) | +0.06 / −1.57 / −4.02 / −5.09 | −2.2 / −15.0 / +37.8 / +136.4 | +9 / −405 / −990 / −946 | FAIL (vs base and vs compiled-experts, all `not slower`) |
| ready-1 `muse-routing` (attempt-1) | −0.45 / −0.14 / +0.51 / +0.33 | −1.2 / −1.3 / +29.4 / +21.9 | −116 / −46 / +121 / +44 | FAIL (vs base and vs compiled-routing, all `not slower`) |
| ready-1 `muse-combine` (attempt-1) | +0.03 / −1.15 / −0.36 / +1.12 | +9.4 / −15.6 / −45.4 / −109.0 | +11 / −303 / −136 / +466 | FAIL (vs base and vs compiled-combine, all `not slower`) |
| ready-1 `muse-all` (all three) | −0.74 / +0.32 / −4.31 / −4.00 | +0.3 / −16.3 / −34.4 / −137.3 | −209 / +56 / −1147 / −1005 | FAIL (vs base and vs compiled-all, all `not slower`) |
| ready-2 `muse-routing` (attempt-1) | −0.78 / −0.46 / −0.42 / −1.28 | −8.4 / −1.9 / +0.4 / −145.7 | −160 / +9 / −74 / −663 | FAIL (vs base and vs compiled-routing, all `not slower`) |
| ready-2 `muse-combine` (attempt-1) | −0.44 / −1.04 / −1.21 / +0.58 | −9.5 / −18.0 / −33.0 / +31.0 | −114 / −280 / −308 / +268 | FAIL (vs base and vs compiled-combine, all `not slower`) |
| ready-2 `muse-all` (routing+combine) | −1.37 / −0.46 / −1.62 / +0.86 | −12.0 / −5.0 / −3.2 / +106.7 | −340 / −168 / −355 / −76 | FAIL (vs base and vs compiled-all, all `not slower`) |

Against the earlier predictions (PREDICTION): experts predicted −1.49 ms/step at 8 requests and −5.23 at 32 against compiled — measured vs base −1.57 at 8×128 and −4.02 / −5.09 at the 32-request workloads, so the kernel's speed carried end to end and only the verdict (spread, plus the set's 8×128 loss) failed. Routing predicted −0.55 / −0.41 / −0.35 at 1 / 8 / 32 — measured medians sit in that band in ready-2 (−0.78 / −0.46 / −0.42) and flip sign at 32 rows in ready-1 (+0.51 / +0.33): prediction-sized gains inside repeat noise. Combine predicted −0.16 / −0.30 / −0.53 — measured 8-request medians beat it in both runs (−1.15, −1.04 vs −0.30) and the 32×128 median beats it in ready-2 (−1.21) but not ready-1 (−0.36 vs −0.53); the 1-request median flips sign between runs (+0.03 → −0.44) and the 32×1024 median is positive in both (+1.12, +0.58). The three-slot set at 8×128 measured +0.32 ms/step vs base where the additive prediction was ≈−2.16 ms/step (`ineff-g3-2`): the interaction is confirmed live for these kernels, at roughly +2.5 ms/step against the parts. No phase of any variant is judged `slower` anywhere in either run; every failure is `not slower` on range overlap except as noted by the medians.

**Unverified:** why the three replaced slots lose at 8 requests (mechanism still unknown after three rounds; now measured at +0.32 ms/step vs base, ≈+2.5 ms/step vs the additive parts); whether a quieter machine or more repeats would separate any variant's ranges (the verdict rule is fixed at 5); kernel speed at ungraded row counts 2–7, 9–31 and 33–59; sizes above 2,048 rows and prompt lengths beyond 128/1,024; the pair at 32×1024 across runs (median −4.00 as triple member, +0.86 as pair — composition and noise are entangled there).

**Check** (checker-final, 2026-11-04T23:04:37Z): 1 defect. No verdict or table number changes; fix the sentence and re-check.
- `implementation/README.md:37` (combine, decode): the claim that the 8/32-request medians “meet or beat” the −0.16 / −0.30 / −0.53 PREDICTION in both runs and that the 32×1024 median flips sign between runs does not reproduce from the cited files. Ready-1 `muse-combine` at 32×128 is −0.36 ms/step vs base ([ready-verdict.json](../checks/ready-1/ready-verdict.json), difference-of-medians; table at `implementation/README.md:31`), which does not meet the −0.53 prediction at 32; and the 32×1024 decode median is positive in both runs (+1.12 at `implementation/README.md:31`, +0.58 at `implementation/README.md:34`), so it does not flip sign (the 1-request decode does: +0.03 → −0.44). Fix-consistency note (outside scope): `hyp-3-combine-setcopy.md:109` carries the same claim.
- Passing: both `--audit` runs print “ready-verdict.json agrees with this audit” (ready-1 and ready-2; each exit 1, FAIL is the standing verdict, not a defect); all 84 table cells in *Ready to test* and every ready-check number in `summary.md` reproduce by difference-of-medians against that file's `base` medians; every variant's phases are `not slower` vs base and vs compiled as stated, same tokens everywhere; `muse-all` numbers cite the `muse-all` variant, not a sum of single slots (the ≈−2.16 / +2.5 interaction is the cited `ineff-g3-2` additive PREDICTION beside its +0.32 measurement); predictions in *Ready to test* (`implementation/README.md:37`) and `summary.md:8` are labelled PREDICTION and stand beside their measurements; every slot named for G3 is the reference, with no `ready` attempt and no baseline (all hypotheses `dropped`); attempt counts recount from the kernel folders — this round experts 2 / routing 3 / combine 1, totals 18 / 15 / 7 (rounds 1–3 totals 16 / 12 / 6 from `muse-spark/round-3/summary.md:7` plus this round).
- **Re-check** (checker-final, 2026-11-04T23:06:20Z): `check: pass` — the first-check defect is fixed and the full final check (steps 1–5 plus ready/baseline/audit/end-to-end) was re-run. Fix verified at `implementation/README.md:37` against `ready-1/ready-verdict.json` and `ready-2/ready-verdict.json` (difference-of-medians vs `base`): the 8-request medians beat the −0.30 PREDICTION in both runs, the 32×128 median beats −0.53 in ready-2 but not ready-1, the 1-request median flips sign and the 32×1024 median is positive in both; the same fix at `hyp-3-combine-setcopy.md:109` reproduces identically. Both `--audit` runs again print “ready-verdict.json agrees with this audit” (each exit 1, FAIL stands); all 84 table cells and every ready-check number in `summary.md` reproduce; no remaining defect.

## Versions

- v21 · 2026-11-04T23:07:00Z · checker-final re-check pass: round 4 finished, all slots keep reference · changed: none
- v20 · 2026-11-04T23:05:30Z · checker-final defect fixed (combine prediction sentence), re-check next · changed: hyp-3-combine-setcopy v6
- v19 · 2026-11-04T23:01:20Z · final: tracker, hypothesis paragraphs, Ready to test (all reference; both ready checks FAIL, verdicts stand) · changed: hyp-1-experts-minrows v10, hyp-2-routing-margin v8, hyp-3-combine-setcopy v5
- v18 · 2026-11-04T23:00:30Z · all hypotheses dropped on ready verdicts · changed: hyp-1-experts-minrows v10, hyp-2-routing-margin v8, hyp-3-combine-setcopy v5
- v17 · 2026-11-04T22:59:00Z · ready-1 verdict committed · changed: none
- v16 · 2026-11-04T22:13:00Z · checker-3 pass: hyp-3 is a candidate (combine attempt-1); three candidates, ready check next · changed: hyp-3-combine-setcopy v4
- v14 · 2026-11-04T22:10:00Z · checker-2 pass: hyp-2 is a candidate (routing attempt-1); combine boundary check pass, checker-3 next · changed: hyp-2-routing-margin v7, hyp-3-combine-setcopy v2
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
