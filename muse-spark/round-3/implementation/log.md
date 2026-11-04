2026-11-04T17:53:37Z | lead | opened hyp-1 (combine devolatilize) and hyp-2 (experts prefill), both in boundary
2026-11-04T17:54:36Z | boundary-2 | started: experts prefill boundary from round-2 hyp-1-experts-exact, starting point attempt-7
2026-11-04T17:54:36Z | boundary-2 | posted hyp-2-experts-prefill Boundary: delta on round-2 boundary, max_rows window 32-59 (decode on candidate, hand-off below MLX switch at 60)
2026-11-04T17:55:11Z | boundary-1 | posted hyp-1 Boundary: delta to round-2 hyp-3 (arithmetic contract unchanged, starting point now G2-passing; new: two-phase paired timing bar, max_rows, ready-check/G3, stall-cost can't-see)
2026-11-04T17:55:31Z | lead | v2: both hyps to building, 5 builders started (1-a/b combine, 2-a/b/c experts)
2026-11-04T17:56:00Z | builder-2-a | claimed attempt-1 for hyp-2 direction (a) max_rows hand-off on round-2 attempt-7
2026-11-04T17:56:12Z | builder-1-a | claimed muse-spark/round-3/kernels/combine/attempt-1 for hyp-1 direction (a) noinline barriers
2026-11-04T17:57:18Z | builder-2-b | claimed attempt-2 for hyp-2 direction (b) vectorized per-pair scalar prefill
2026-11-04T17:57:26Z | builder-2-a | graded attempt-1: G1+G2 pass, decode slower/faster/faster, prefill same x3, beats_compiled false — prefill prize banked
2026-11-04T17:57:27Z | builder-1-b | claimed muse-spark/round-3/kernels/combine/attempt-2 for hyp-1 direction (b) fp-contract-off pragma barrier
2026-11-04T17:58:15Z | builder-1-a | graded attempt-1: G1 pass, G2 pass all six sizes (differ 0), beats_compiled true (both phases faster)
2026-11-04T17:58:15Z | builder-1-a | done: attempt-1 recorded in notes.md, hyp-1 section, ineff-baseline-combine-1 note (grader sees prefill part of prize)
2026-11-04T17:58:41Z | lead | v3: combine-1 + experts-1 graded, both G2 pass; boundary checks started
2026-11-04T17:59:26Z | boundary-2 | checked experts attempt-1 vs hyp-2 boundary: pass (max_rows 59 in 32-59 window, decode on fused path, hand-off below MLX switch at 60; 3,840 B tg, no unwritten reads, no timing dependence)
2026-11-04T17:59:39Z | boundary-1 | boundary-checked combine attempt-1: pass (all six can't-see probes; check_sizes not run, model-loading GPU job)
2026-11-04T18:00:47Z | builder-2-c | claimed attempt-3 for hyp-2 direction (c) grouped/tiled prefill replica on round-2 attempt-7 (attempt-2 taken by builder-2-b)
2026-11-04T18:04:05Z | lead | v4: both attempt-1 boundary checks pass
2026-11-04T18:05:37Z | builder-2-c | graded attempt-3: G1 fail (Metal reserved word 'half' in block-3.metal), no G2/timing; recorded in notes.md, hyp-2 section, ineff-baseline-experts-1 note
2026-11-04T18:05:37Z | builder-2-c | done: one attempt graded and recorded; next change is rename half->part in a copied new attempt
2026-11-04T18:05:45Z | builder-1-b | graded muse-spark/round-3/kernels/combine/attempt-2: G1 pass, G2 pass all six sizes, faster everywhere (24-30/30), beats_compiled true
2026-11-04T18:05:45Z | builder-1-b | done: pragma-barrier devolatilization recorded in notes.md, hyp-1 Builders, ineff-baseline-combine-1 notes; miss (>20% vs cross-run anchor) explained as machine-state drift of chained absolutes
2026-11-04T18:06:14Z | builder-2-b | graded attempt-2: G1+G2 pass (differ 0 all six), decode slower/faster/faster, prefill slower x3, beats_compiled false — direction (b) buys nothing
2026-11-04T18:06:14Z | builder-2-b | done: verdict in notes.md + hyp-2 section, miss analysis in notes.md + ineff-baseline-experts-1 note
2026-11-04T18:06:25Z | lead | v5: combine-2 G2 pass, experts-3 G1 fail (ledger); boundary check + fix builder next
2026-11-04T18:06:45Z | lead | v6: experts-2 G2 pass but prefill slower (ledger); best remains attempt-1
2026-11-04T18:07:07Z | builder-2-d | claimed attempt-4 (experts): copy of attempt-3 with half->part rename
2026-11-04T18:07:17Z | boundary-1 | boundary-checked attempt-2 (hyp-1-combine-devolatilize): pass on all six can't-see items, appended to Boundary checks
2026-11-04T18:08:08Z | lead | v7: hyp-1 hardening (attempt-2 best, Decision), hyp-3 opened; checker-1 + boundary-3 next
2026-11-04T18:09:42Z | builder-2-d | graded attempt-4: G1+G2 pass (differ 0 all six), decode slower/faster/faster, prefill slower x3 (2.67/2.29/2.32x), beats_compiled false
2026-11-04T18:09:42Z | builder-2-d | done: verdict in notes.md + hyp-2 section, miss analysis in notes.md + ineff-baseline-experts-1 note
2026-11-04T18:09:55Z | boundary-3 | posted hyp-3-routing-prefill Boundary: delta on round-2 hyp-2 boundary, starting point attempt-4 G2-passing, two-phase bar, max_rows window 32-126 (no 60-row switch for routing)
2026-11-04T18:10:26Z | checker-1 | checked hyp-1-combine-devolatilize: pass (all 6 steps; 2 non-blocking observations: L53 cross-run deltas diagnostic-only, L87 -35% decode-only)
2026-11-04T18:10:28Z | lead | v8: experts-4 exact but 2.3x off (ledger, line ends); hyp-2 hardening (1-row fix), hyp-3 building
2026-11-04T18:10:55Z | lead | v9: hyp-1 candidate (combine attempt-2, checker pass)
2026-11-04T18:11:38Z | builder-3-a | claimed muse-spark/round-3/kernels/routing/attempt-1 for hyp-3 direction (a) max_rows hand-off on round-2 attempt-4
2026-11-04T18:13:40Z | builder-3-a | graded attempt-1: G1+G2 pass, decode faster x3, prefill same/same/slower (2048: 8 wins on identical code — paired noise), beats_compiled false
2026-11-04T18:13:40Z | builder-3-a | done: verdict in notes.md + hyp-3 section, noise/drift analysis in notes.md + ineff-baseline-routing-1 note; next is re-grade of byte-identical copy
2026-11-04T18:14:04Z | lead | v10: routing-1 G2 pass, decode faster x3, 2048 noise-slower (ledger)
2026-11-04T18:14:11Z | builder-2-e | claimed attempt-5 (experts) for hyp-2 direction (e): reference-shaped unfused 1-row path on attempt-1
2026-11-04T18:14:37Z | builder-3-b | claimed muse-spark/round-3/kernels/routing/attempt-2 for hyp-3 direction (b) GEMM-shaped prefill sharing on round-2 attempt-4
2026-11-04T18:15:06Z | boundary-3 | boundary check attempt-1: pass — max_rows 32 holds every decode step at or below it, kernels byte-identical, 33-126 band empty, no unwritten TG reads or timing dependence; 2048-row slower is identical-code noise, not a defect
2026-11-04T18:15:24Z | builder-3-c | claimed routing attempt (hyp-3 direction c: drop parts where partitions=1 via two-kernel split)
2026-11-04T18:16:56Z | builder-2-e | graded attempt-5: G1+G2 pass (differ 0 all six), decode slower/faster/faster (1-row 234.3 vs 118.4, 0 wins), prefill same x3, beats_compiled false — unfused 1-row path slower than fused
2026-11-04T18:16:56Z | builder-2-e | done: verdict in notes.md + hyp-2 section, miss analysis in notes.md + ineff-baseline-experts-2 note; next is bank attempt-1, gap looks structural
2026-11-04T18:17:45Z | builder-3-b | graded attempt-2: G1 fail (threadgroup 960 > pipeline max 640), no G2/timing; recorded in notes.md, hyp-3 section, ineff-baseline-routing-1 note
2026-11-04T18:17:45Z | builder-3-b | done: one attempt graded and recorded; next change is 10-rows-per-group (600 threads) copy in a new attempt
2026-11-04T18:18:33Z | lead | v11: hyp-2 dropped (slot keeps reference); routing-2 G1 fail (ledger), routing-1 boundary pass
2026-11-04T18:19:07Z | builder-3-c | graded attempt-3: G1+G2 pass (differ 0 all six), decode same x3 (14/15/9 wins), prefill slower x3, beats_compiled false
2026-11-04T18:19:07Z | builder-3-c | done: verdict in notes.md + hyp-3 section, drift/wiring miss analysis in notes.md + ineff-baseline-routing-2 note; next is same-run --baseline re-grade of byte-identical copy
2026-11-04T18:19:22Z | builder-3-d | start: hyp-3 direction (b) fix — copy attempt-2 to attempt-4 with 10 rows/group
2026-11-04T18:19:22Z | builder-3-d | claimed muse-spark/round-3/kernels/routing/attempt-4 for hyp-3 direction (b) 10-rows/group fix of attempt-2
2026-11-04T18:19:37Z | lead | v12: routing-3 G2 pass, decode same x3, prefill slower (ledger)
2026-11-04T18:21:55Z | builder-3-d | graded attempt-4: G1 fail (threadgroup 600 > pipeline max 576 — cap moved from attempt-2's 640), no G2/timing; recorded in notes.md, hyp-3 section, ineff-baseline-routing-1 note
2026-11-04T18:21:55Z | builder-3-d | done: one attempt graded and recorded; next change is 8-rows-per-group (480 threads) copy in a new attempt
2026-11-04T18:22:23Z | lead | v13: routing-4 G1 fail (cap 576); ledger, fix at 8 rows/group next
