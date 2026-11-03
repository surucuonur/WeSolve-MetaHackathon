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
