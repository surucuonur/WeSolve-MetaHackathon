2026-11-03T21:31:52Z | lead | opened hyp-1-fused-experts (experts, 5 entries) and hyp-2-fused-routing (routing, runs-2); starting boundary-1 and boundary-2
2026-11-03T21:33:09Z | boundary-2 | wrote hyp-2-fused-routing Boundary: routing IO table, MoE neighbours + checklist, f32 rounding, bias/tie rules, G1/G2 vs unseen rows, 1024-thread/32KB limits
2026-11-03T21:33:38Z | boundary-1 | started hyp-1-fused-experts boundary (experts slot)
2026-11-03T21:33:38Z | boundary-1 | wrote hyp-1-fused-experts Boundary: slot I/O at 1/8/32 rows, neighbours A/B checklists, rounding points, G1/G2 vs blind spots, GPU limits
2026-11-03T21:34:02Z | lead | checkpoint v2: boundaries drawn for hyp-1 and hyp-2, both building with 5 builders
2026-11-03T21:34:50Z | builder-1-a | started; claimed muse-spark/kernels/experts/attempt-1 for hyp-1 direction (a) full fused grouped kernel
2026-11-03T21:35:19Z | builder-2-a | claimed muse-spark/kernels/routing/attempt-1 (single fused kernel direction)
2026-11-03T21:35:46Z | builder-1-b | started; claimed experts/attempt-2 (direction b: fusion first, per-pair reads); attempt-1 taken
2026-11-03T21:36:26Z | builder-2-b | started: two-kernel split (scores matmul+sigmoid+bias / select+norm) for hyp-2-fused-routing
2026-11-03T21:36:36Z | builder-2-b | claimed muse-spark/kernels/routing/attempt-1 for direction (b) two-kernel split
2026-11-03T21:36:42Z | builder-2-b | attempt-1 taken by builder-2-a, claimed muse-spark/kernels/routing/attempt-2 instead
2026-11-03T21:36:54Z | builder-2-a | graded routing/attempt-1: G1 pass, G2 pass, 28.5/23.6/41.8 vs compiled 30.5/42.0/49.1 — beats compiled at all rows, stops slot per brief rule
2026-11-03T21:36:54Z | builder-2-a | done: verdict + >20% miss explanation in attempt-1/notes.md, section in hyp-2, note on ineff-runs-2
2026-11-03T21:37:00Z | builder-1-c | started: two-kernel grouped experts split (direction c)
2026-11-03T21:37:22Z | builder-1-c | claimed muse-spark/kernels/experts/attempt-3 (1,2 taken by siblings)
2026-11-03T21:38:30Z | builder-2-b | graded attempt-2: G1 pass, G2 pass (worst rel err 2.6e-6), 36.3/34.9/49.0 vs compiled 29.3/42.9/46.6, beats_compiled=false (wins only at 8 rows)
2026-11-03T21:38:30Z | builder-2-b | finished: miss explained (24-28us fused-kernel floor, not 6.81us/launch), section added to hyp-2, note added to ineff-runs-2
2026-11-03T21:39:58Z | builder-1-c | graded attempt-3 v1: G1 pass, G2 fail (~7pct bad, max 11/32/68 ULP); small-|y| worst cases -> silu intermediate-rounding suspect, trying per-op bf16 rounds
2026-11-03T21:40:13Z | lead | checkpoint v3: routing attempts 1+2 collected (both G2 pass, attempt-1 beats compiled at all rows); boundary-2 to check both
2026-11-03T21:41:28Z | boundary-2 | checked routing attempts 1+2 vs boundary: both pass (att-1: 3.3KB tg, 1 group/row all counts; att-2: 17.0KB tg, whole-rows partial groups all counts; no unwritten reads, races, or input assumptions)
2026-11-03T21:46:09Z | lead | checkpoint v4: hyp-2 ready with routing/attempt-1 (both boundary checks pass, no hardening — single minimal launch, residual is dispatch floor); hyp-3 opened in boundary
2026-11-03T21:47:42Z | boundary-3 | wrote hyp-3-fused-combine Boundary: combine IO table, MoE neighbours + checklist, rounding points, tie/positional pairing, G1/G2 vs unseen rows, 1024-thread/32KB limits
2026-11-03T21:51:53Z | lead | checkpoint v5: hyp-3 boundary drawn, building with 2 builders (thread-per-output vs threadgroup-per-row)
2026-11-03T21:52:22Z | builder-3-a | started; claimed muse-spark/kernels/combine/attempt-1 for hyp-3 direction (a) single kernel, thread-per-output
2026-11-03T21:53:13Z | builder-1-c | graded attempt-3 v3 (stable bf16 sigmoid, MLX bit-exact): G1 pass, G2 pass (bad 0, max 1/1/1.5 ULP), timing 150/659/2045us vs 88/223/766 ref -- 6-10x over prediction, diagnosing
2026-11-03T21:53:25Z | builder-3-b | started; attempt-1 taken by builder-3-a, claimed muse-spark/kernels/combine/attempt-2 for hyp-3 direction (b) threadgroup-per-row with staged weights/shared
2026-11-03T21:53:38Z | builder-3-a | graded combine/attempt-1: G1 pass, G2 pass (0 bad, max ULP 0/2/1), 17.2/17.0/18.5 vs compiled 22.9/26.2/28.0 — beats compiled at all rows, stops slot per brief rule
2026-11-03T21:53:38Z | builder-3-a | done: verdict + >20% miss explanation (routing-specific 24us floor vs 17us streaming floor) in attempt-1/notes.md, section in hyp-3, note on ineff-model-2
2026-11-03T21:54:15Z | builder-1-a | graded attempt-1 v2: G1 pass, G2 pass bit-exact (max_ulp 0.0 all rows), timing 4379.8/22507.8/47623.4 vs compiled 62.8/212.5/765.3 - 60x too slow, cache-thrash diagnosis, rewriting as v3 (k-outer loop interchange)
2026-11-03T21:54:50Z | builder-3-b | graded combine/attempt-2: G1 pass, G2 pass, 17.9/16.1/18.1 vs compiled 20.9/23.0/27.9 — beats compiled at all rows (slot already stopped at attempt-1)
2026-11-03T21:54:50Z | builder-3-b | done: verdict in attempt-2/notes.md, section in hyp-3-fused-combine; no >20% miss (15/15/10% high), staging+barrier cost nil vs direction (a)
2026-11-03T21:55:51Z | builder-1-b | graded experts/attempt-2 v1: G1 pass, G2 fail (~7% over 2 ULP, max 11/32/68); serial-vs-strided order ruled out
2026-11-03T21:55:51Z | builder-1-b | graded experts/attempt-2 v2 (bit-exact qvm order): G1 pass, G2 fail identical to v1 -> systematic, not order
2026-11-03T21:55:51Z | builder-1-b | CPU search over 64 silu rounding placements: only full-bf16 matches ref (bad=0); MLX compiled swiglu rounds every op
2026-11-03T21:55:51Z | builder-1-b | graded experts/attempt-2 v3: G1 pass, G2 pass max_ulp 0.0, beats_compiled true; 107.3/188.4/647.6 us/layer at 1/8/32 (orig 99.8/218.7/768.6); regresses 7.5% vs eager at 1 row; done
2026-11-03T21:57:06Z | builder-1-a | graded attempt-1 v3: G1/G2 pass bit-exact, 1236.0/5060.2/10979.8 vs compiled 70.3/215.9/764.2 (4x better, still 14x off) - L1-load-bound diagnosis, rewriting as v4 (load fan-out, split phases)
2026-11-03T21:57:41Z | lead | checkpoint v6: experts attempt-2 + combine attempts 1+2 collected (all G2 pass, all beat compiled); boundary-1/3 to check; 1-b numerics finding sent to 1-a/1-c
2026-11-03T21:58:58Z | boundary-1 | checked experts/attempt-2 vs boundary: pass (rows*4 groups valid all 1-32 rows, no unwritten tg reads, 3840B tg, barriers+exact-once writes, no input assumptions; stale fp32 comment B0:6 cosmetic)
2026-11-03T22:00:00Z | builder-1-a | graded attempt-1 v5: G1/G2 pass bit-exact, 256.7/1148.7/2430.9 (same as v4: not occupancy-bound) - prime suspect acc-array scratch spilling, trying forced unroll as v6
2026-11-03T22:00:30Z | builder-1-c | graded attempt-3 v7 FINAL: G1 pass, G2 pass (max ULP 1.0), 68/238/654us vs compiled 65/213/764 (beats at 32 rows only); removed probe-f64 scratch
2026-11-03T22:00:30Z | builder-1-c | done: verdict + miss analysis in attempt-3/notes.md, section in hyp-1, note on ineff-quantized-kernels-1; next change for hardening: hybridize attempt-2 (8-row) with attempt-3 grouping (32-row)
2026-11-03T22:02:00Z | builder-1-a | graded attempt-1 v6: G1/G2 pass bit-exact, 248.5/1148.9/2435.1 (same as v5: not spilling, issue-bound) - found it: 36-75pct dummy work for <4-pair groups; adding exact itb-skip as v7
2026-11-03T22:03:39Z | lead | checkpoint v7: experts attempt-3 collected (G2 pass, beats_compiled false); attempt-2 boundary pass; boundary-1 to check attempt-3
2026-11-03T22:05:02Z | boundary-1 | checked experts/attempt-3 vs boundary: pass (rows-independent 60-group grid valid all 1-32 rows, no unwritten tg reads, 16516B tg per kernel, uniform barriers + exact-once writes, no input assumptions; non-bitwise fp32 order is boundary-allowed residual)
2026-11-03T22:06:56Z | builder-1-a | graded attempt-1 v7: G1/G2 pass bit-exact, 81.6/416.2/1573.8 (matches prediction) - key insight: 16-multiple o0 alignment makes runs/pairs fully static; rewriting inner loops as v8 (predict ~35/170/600)
2026-11-03T22:08:58Z | lead | checkpoint v8: experts attempt-3 boundary pass; waiting on builder-1-a and boundary-3
