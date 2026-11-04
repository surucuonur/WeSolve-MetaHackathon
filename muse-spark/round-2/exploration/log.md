2026-11-04T03:15:44Z | verifier | started: waiting for inefficiency entries
2026-11-04T03:17:00Z | explorer-grader | started: mapping round-2 grader (ADR 0010 bit-exact G2 at 6 sizes, chained timing, --baseline wins rule)
2026-11-04T03:17:10Z | explorer-baseline-experts | started: mapping baseline-experts (attempt-4 vs ref_experts/SwitchGLU, chained 1-row gap, prefill G2)
2026-11-04T03:17:11Z | explorer-baseline-combine | started: mapping muse-spark/kernels/combine/attempt-1 against bench/grader/slots.py ref_combine and MLX kernels behind it
2026-11-04T03:17:47Z | explorer-baseline-combine | posted territory-baseline-combine v1: checklist for attempt-1, ref_combine, copy_v/binary_g/col_reduce_small/binary_vv; FMA contraction (not association) causes 0/2/2 decode and 6/50/130 prefill differs
2026-11-04T03:18:04Z | explorer-baseline-combine | posted ineff-baseline-combine-1: FMA contraction (not association) fails exact G2 2@8, 2@32, 6/50/130 prefill; fix forfeits nothing, unlocks 0.104/0.182/0.262 ms/step vs compiled (piece basis)
2026-11-04T03:18:20Z | explorer-baseline-combine | posted ineff-baseline-combine-2: fused kernel sits on ~17 us/layer dispatch floor (traffic 0.05/0.37/1.48 us, compute <=0.03 us); banked saving vs compiled 0.104/0.182/0.262 ms/step, no body change predicted to go below floor
2026-11-04T03:18:37Z | explorer-g3 | started: mapping g3 (Claude round-1 measurement, G3 vs piece timing)
2026-11-04T03:18:37Z | explorer-g3 | posted ineff-g3-2: experts attempt chained at 1 row is +52.6 us/layer vs reference (+1.05 ms/step), matching G3 +1.16 ms vLLM
2026-11-04T03:18:37Z | explorer-g3 | posted ineff-g3-1: three attempts together lose +2.08/+3.28 ms/step (vLLM, 8/32 requests) vs sum of alone; kernel reductions additive (-400), compiled references additive
2026-11-04T03:18:37Z | explorer-grader | posted territory-grader v2 (done): ADR 0010 delta - bit-exact G2 at decode 1/8/32 + prefill 127/1016/2048, chained link timing, beats_compiled median rule, beats_baseline 22/30 sign test; blind spots: untested row counts, first-chunk-only prefill, no threadgroup-memory check, prefill speed untimed; notes appended to territory-baseline-experts + territory-baseline-combine (baseline-routing entry not yet created)
2026-11-04T03:18:38Z | explorer-baseline-combine | done: territory-baseline-combine v2 (grader note folded: fix must hold at 2048 rows, prefill slowness invisible to beats_compiled), ineff-baseline-combine-1, ineff-baseline-combine-2
2026-11-04T03:18:46Z | explorer-baseline-experts | posted ineff-baseline-experts-1: qvm replica vs gather_qmm_rhs from 60 rows — prefill differ 4868667/39078912/78741557 (43/65/110.5 ULP), gate not saving
2026-11-04T03:18:46Z | explorer-baseline-experts | posted ineff-baseline-experts-2: chained 1-row 132.9 vs 80.3 us/layer (+52.6, 1.05 ms/step) — 4 threadgroups + wave/barrier serialization
2026-11-04T03:18:46Z | explorer-baseline-experts | posted ineff-baseline-experts-3: per-pair re-reads persist — 908 vs 443 MB at 32 rows, 2.33 ms floor unbanked on round-1 grouped evidence
2026-11-04T03:18:47Z | explorer-baseline-routing | started: mapping muse-spark/kernels/routing/attempt-1 vs slots.py ref_routing and mobilemoe MoE.__call__ under bit-exact G2
2026-11-04T03:18:59Z | explorer-g3 | adapted: ineff-g3-2 duplicates baseline-experts chained 1-row entry; marked superseded by ineff-baseline-experts-2, G3-side note appended to territory-baseline-experts and territory-baseline-combine
2026-11-04T03:19:16Z | explorer-g3 | done: territory-g3 v2 (7 checklist pieces, gaps 1-3), ineff-g3-1 open, ineff-g3-2 superseded by ineff-baseline-experts-2; routing note held in territory-g3 (no baseline-routing entry yet)
2026-11-04T03:19:19Z | explorer-baseline-experts | done: territory-baseline-experts v2 (checklist A-F, grader note folded), ineff-baseline-experts-1 (prefill qmm_rhs gate), -2 (chained 1-row latency), -3 (per-pair re-reads)
2026-11-04T03:19:36Z | explorer-baseline-routing | posted territory-baseline-routing v1: scores diverge in matmul order (GEMV tree / split-K / steel by rows) + sigmoid formula; 4-term weight sum already matches
2026-11-04T03:19:55Z | lead | checkpoint v2: baseline-experts/combine/g3/grader done, ineff-g3-2 superseded by ineff-baseline-experts-2 (same chained 1-row cause, fuller entry kept)
2026-11-04T03:20:12Z | explorer-baseline-routing | posted ineff-baseline-routing-1 (matmul accumulation order, 3 regimes), -2 (sigmoid formula), -3 (1-simdgroup occupancy + serial scan), -4 (w_router re-read per row)
2026-11-04T03:20:19Z | explorer-quantized-kernels | started: mapping prefill path affine_gather_qmm_rhs vs qvm for round 2
2026-11-04T03:20:37Z | explorer-baseline-routing | done: territory-baseline-routing v2 (notes from grader+g3 folded), ineff-baseline-routing-1..4 open
2026-11-04T03:22:26Z | verifier | confirmed ineff-baseline-combine-1: FMA contraction fails exact G2 2@8, 2@32, 6/50/130 prefill; unlocks 0.104/0.182/0.262 ms/step (independent basis)
2026-11-04T03:22:51Z | verifier | confirmed ineff-baseline-combine-2: ~17 us/layer dispatch floor (traffic 0.05/0.37/1.48 us, compute <=0.03 us); banked 0.104/0.182/0.262 ms/step vs compiled
2026-11-04T03:22:51Z | verifier | duplicate: ineff-g3-2 and ineff-baseline-experts-2 (same chained 1-row experts cause; lead already settled as superseded, fuller entry kept)
2026-11-04T03:23:12Z | verifier | confirmed ineff-baseline-experts-1: qvm replica fails prefill gather_qmm_rhs (62.4-62.6% differ, 43/65/110.5 ULP); gate, 0 us saving
2026-11-04T03:23:32Z | verifier | confirmed ineff-baseline-experts-2: chained 1-row 132.9 vs 80.3 us/layer (+52.6, 1.05 ms/step); wave waste corrected to 8/32 (25%)
2026-11-04T03:23:45Z | verifier | confirmed ineff-baseline-experts-3: per-pair re-reads 908 vs 443 MB at 32 rows, 2.33 ms floor conditional (0 banked); 177 GB/s corrected to 37.7
2026-11-04T03:24:01Z | verifier | confirmed ineff-g3-1: three together lose +2.08/+3.28 ms/step vLLM (interaction); spread multiple corrected to 11-14x
2026-11-04T03:24:18Z | verifier | confirmed ineff-baseline-routing-1: router matmul order matches MLX in no regime (GEMV/split-K/steel); blocks G2, headroom 0.12/0.37/0.37 ms/step
2026-11-04T03:24:18Z | verifier | confirmed ineff-g3-1: three together lose +2.08/+3.28 ms/step vLLM (interaction); spread multiple corrected to 11-14x
2026-11-04T03:24:51Z | explorer-quantized-kernels | posted territory-quantized-kernels v2 (done): prefill dispatch rows>=60, qmm_rhs BM16/BN32/BK32 vs qvm — bfloat-rounded dequant + 8-wide frag K order; branch required, grouping free
2026-11-04T03:24:51Z | explorer-quantized-kernels | posted ineff-quantized-kernels-1: BM16 empty+padded row tiles waste 47/15/5% at prefill 127/1016/2048 (0 us decode)
2026-11-04T03:24:51Z | explorer-quantized-kernels | posted ineff-quantized-kernels-2: gather_mm_offsets recomputed 3x per layer (60 launches/chunk, 40 redundant, <=0.26 ms upper)
2026-11-04T03:24:51Z | explorer-quantized-kernels | note on territory-baseline-experts: branch at rows>=60 exactly; row grouping free, per-row K order not
2026-11-04T03:24:51Z | explorer-quantized-kernels | done: territory-quantized-kernels v2, ineff-quantized-kernels-1, ineff-quantized-kernels-2
2026-11-04T03:24:52Z | verifier | confirmed ineff-baseline-routing-2: sigmoid precise::exp(|x|) branch vs fast exp(-x); independent G2 blocker, 0 us direct
2026-11-04T03:24:52Z | verifier | confirmed ineff-baseline-routing-3: 1 simdgroup/row + serial thread-0 scan; up to 0.4 ms/step at 1 row bounded
2026-11-04T03:24:52Z | verifier | confirmed ineff-baseline-routing-4: w_router re-read per row (0.57 ms/step floor at 32 rows, upper bound only; attempt-2 counter-evidence stands)
2026-11-04T03:25:52Z | verifier | confirmed ineff-quantized-kernels-1: BM16 empty+padded tiles waste 47.1/15.3/5.2% at prefill (0 us decode)
2026-11-04T03:25:52Z | verifier | confirmed ineff-quantized-kernels-2: gather_mm_offsets recomputed 3x/layer (60/chunk, 40 redundant, <=0.26 ms upper, prefill only)
2026-11-04T03:27:19Z | verifier | done: 12 confirmed, 0 refuted, 1 superseded duplicate (ineff-g3-2 by ineff-baseline-experts-2); no open entries, all 6 explorers done
2026-11-04T03:28:07Z | lead | checkpoint v3: exploration done, verifier 12 confirmed 0 refuted 1 superseded
2026-11-04T03:29:03Z | lead | exploration v4 final analysis written
