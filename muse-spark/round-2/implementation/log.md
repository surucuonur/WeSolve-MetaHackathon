2026-11-04T03:29:43Z | lead | opened hyp-1 experts-exact and hyp-2 routing-exact in boundary; g3-1 interaction carried as constraint, not a hypothesis
2026-11-04T03:31:03Z | boundary-2 | started hyp-2 routing-exact boundary: checking round-1 boundary against starting point and bit-exact G2
2026-11-04T03:31:12Z | boundary-1 | wrote hyp-1-experts-exact Boundary: delta vs round-1 boundary — bit-exact G2 at six sizes, gather_qvm<60 vs gather_qmm_rhs>=60 (bf16-rounded dequant + BK=32 MMA order), branch on rows>=60
2026-11-04T03:31:59Z | boundary-2 | wrote hyp-2 routing-exact Boundary: delta to round-1 boundary, bit-exact ordered equality at six sizes, three product regimes, verbatim sigmoid, starting point fails on product+sigmoid only
2026-11-04T03:32:30Z | lead | boundaries drawn for hyp-1 and hyp-2; both to building, 4 builders started
2026-11-04T03:34:50Z | builder-1-b | claimed round-2 kernels/experts/attempt-1 for hyp-1-experts-exact direction (b): single fused kernel, rows>=60 branch
2026-11-04T03:35:30Z | builder-2-a | claimed round-2 routing attempt-1 for hyp-2-routing-exact direction (a) MLX-regime replica
2026-11-04T03:35:46Z | builder-1-a | claimed round-2 kernels/experts/attempt-2 for hyp-1-experts-exact direction (a): size-branched GU+Down kernels, decode block-split qvm (48+96 groups at 1 row), prefill per-pair BK32 MMA
2026-11-04T03:36:20Z | builder-1-b | graded round-2 kernels/experts/attempt-1: G1 pass, G2 pass all six sizes (differ 0), chained 104.3/193.3/535.7 us/layer vs compiled 77.6/235.7/791.6 — beats_compiled false (1-row only); recorded in hyp-1-experts-exact and ineff-baseline-experts-2
2026-11-04T03:36:37Z | builder-2-a | wrote round-2 routing attempt-1: GEMV/split-K/steel branch + simdgroup MMA + verbatim sigmoid; grading
2026-11-04T03:37:31Z | builder-2-a | graded round-2 routing attempt-1: G1 pass, G2 pass all six sizes, chained 54.9/91.4/114.8 vs compiled 56.2/91.2/98.9 us/layer (beats_compiled false); timing miss explained (barrier staging); recorded in hyp-2-routing-exact and ineff-baseline-routing-1
2026-11-04T03:37:42Z | builder-1-a | graded round-2 kernels/experts/attempt-2: G1 pass, G2 pass differ 0 at all six sizes, chained 103.6/326.8/887.6 us/layer vs compiled 73.1/232.0/804.5, beats_compiled false; miss explained in notes (1-simdgroup groups lose in-group load overlap, prefill tile memory caps decode residency)
2026-11-04T03:43:31Z | lead | checkpoint v3: experts attempt-1/2 and routing attempt-1 all G2 pass, none beats compiled yet; boundary checks next
