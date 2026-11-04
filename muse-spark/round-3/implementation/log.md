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
