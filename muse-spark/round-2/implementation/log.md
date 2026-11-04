2026-11-04T03:29:43Z | lead | opened hyp-1 experts-exact and hyp-2 routing-exact in boundary; g3-1 interaction carried as constraint, not a hypothesis
2026-11-04T03:31:03Z | boundary-2 | started hyp-2 routing-exact boundary: checking round-1 boundary against starting point and bit-exact G2
2026-11-04T03:31:12Z | boundary-1 | wrote hyp-1-experts-exact Boundary: delta vs round-1 boundary — bit-exact G2 at six sizes, gather_qvm<60 vs gather_qmm_rhs>=60 (bf16-rounded dequant + BK=32 MMA order), branch on rows>=60
2026-11-04T03:31:59Z | boundary-2 | wrote hyp-2 routing-exact Boundary: delta to round-1 boundary, bit-exact ordered equality at six sizes, three product regimes, verbatim sigmoid, starting point fails on product+sigmoid only
2026-11-04T03:32:30Z | lead | boundaries drawn for hyp-1 and hyp-2; both to building, 4 builders started
