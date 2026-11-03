2026-11-03T21:18:26Z | explorer-quantized-kernels | started mapping quantized-kernels
2026-11-03T21:18:47Z | explorer-other-kernels | started mapping other-kernels territory
2026-11-03T21:19:20Z | explorer-runs | started mapping runs
2026-11-03T21:19:23Z | explorer-model | started mapping model territory
2026-11-03T21:19:28Z | explorer-findings | started: territory findings (claude/findings/, results/runs/2026-11-01-breakdown/)
2026-11-03T21:19:36Z | explorer-grader | started territory grader: bench/grader slot interface, G1/G2/piece timing, boundary
2026-11-03T21:20:07Z | explorer-runs | posted ineff-runs-1: experts re-reads 465MB excess at 32 rows, saves 2.3ms
2026-11-03T21:20:07Z | explorer-runs | posted ineff-runs-2: routing 1.02ms for 4MB at 1 row, pure overhead
2026-11-03T21:20:07Z | explorer-runs | posted ineff-runs-3: unsorted pairs below 16 rows cost 1.3ms at 8 rows
2026-11-03T21:20:07Z | explorer-runs | posted ineff-runs-4: attention K/V re-read per query head, outside slots
2026-11-03T21:20:07Z | explorer-runs | finished territory runs: checklist sourced, 4 inefficiencies posted
2026-11-03T21:20:50Z | explorer-model | posted ineff-model-1: routing 12 tiny kernels/layer, 1.02ms at 1 row
2026-11-03T21:20:50Z | explorer-model | posted ineff-model-2: combine 5 tiny kernels/layer with bf16-fp32 casts, 0.61ms at 1 row
2026-11-03T21:20:50Z | explorer-model | changed direction: ineff-model-1 superseded by ineff-runs-2 (same routing cause), appended code-side note there
2026-11-03T21:20:50Z | explorer-model | finished territory model: checklist sourced, ineff-model-2 open
2026-11-03T21:21:28Z | lead | checkpoint v2: model and runs done, no clashes (ineff-model-1 self-superseded by ineff-runs-2)
2026-11-03T21:22:10Z | explorer-vllm-metal | started mapping vllm-metal: engine decode path around the model
2026-11-03T21:22:10Z | explorer-vllm-metal | posted ineff-vllm-metal-1: MobileMoE MoE block excluded from compiled-MLP dispatch, routing+combine fully eager
2026-11-03T21:22:10Z | explorer-vllm-metal | posted ineff-vllm-metal-2: GQA kernel lacks 12/4/64/16 layout, per-token kernel reads each KV head 3x
2026-11-03T21:22:10Z | explorer-vllm-metal | posted ineff-vllm-metal-3: 60 q/k/v [0] gathers + 60 contiguous copies per step
2026-11-03T21:22:10Z | explorer-vllm-metal | posted ineff-vllm-metal-4: ~2ms engine CPU per 1-row step while CPU-bound
2026-11-03T21:22:10Z | explorer-vllm-metal | changed direction: ineff-vllm-metal-1 superseded by ineff-findings-1 (same routing+combine cause), engine mechanism appended there
2026-11-03T21:22:10Z | explorer-vllm-metal | changed direction: ineff-vllm-metal-2 superseded by ineff-runs-4 (same KV re-read cause), code mechanism appended there
2026-11-03T21:22:10Z | explorer-vllm-metal | finished territory vllm-metal: checklist sourced, ineff-3 and ineff-4 open
2026-11-03T21:22:30Z | explorer-other-kernels | changed direction: routing/combine causes already covered by ineff-runs-2/ineff-model-2, appended kernel-level notes instead of new entries
2026-11-03T21:22:30Z | explorer-other-kernels | appended kernel notes to ineff-runs-2 (argpartition is full 60-sort), ineff-model-2 (col_reduce_small 128-thread groups), ineff-runs-3 (sort glue is fixed-cost kernels)
2026-11-03T21:22:30Z | explorer-other-kernels | finished territory other-kernels: checklist A-I sourced, no new entries, 3 notes appended
2026-11-03T21:22:56Z | explorer-switch-layers | started mapping switch-layers
2026-11-03T21:22:56Z | explorer-switch-layers | posted ineff-switch-layers-1: sort glue 6 kernels/layer from 16 rows for locality only, ~0.7ms at 32 rows inside fusion
2026-11-03T21:22:56Z | explorer-switch-layers | posted ineff-switch-layers-2: gate/up/down+activation unfused, 4 on-chain kernels/layer, ~0.4ms fixed costs
2026-11-03T21:22:56Z | explorer-switch-layers | appended mechanism note to ineff-runs-1 (qvm per-pair reads, B/E>=4 gate needs 60+ rows) instead of duplicating
2026-11-03T21:22:56Z | explorer-switch-layers | appended correction note to ineff-runs-3 (kernel-A/B sliced E so sorted=True switched kernels; serving sort cannot)
2026-11-03T21:22:56Z | explorer-switch-layers | finished territory switch-layers: checklist sourced, 2 inefficiencies posted
2026-11-03T21:22:59Z | explorer-grader | direction change: planned ineff-grader-1/2/3 superseded before posting — routing covered by ineff-runs-2, combine by ineff-model-2, experts re-reads/sort by ineff-runs-1/3; appending grader-side notes instead
2026-11-03T21:22:59Z | explorer-grader | appended grader notes to ineff-runs-1 (G2 experts 2-ULP, rounding points), ineff-runs-2 (G2 routing set-match/tie-skip, piece timing), ineff-runs-3 (do_sort threshold, G2 output-only), ineff-model-2 (G2 combine, part independence)
2026-11-03T21:22:59Z | explorer-grader | finished territory grader: checklist sourced (6 pieces), boundary floor 0.04ms@32, 0 new entries (all causes covered), 4 notes appended
2026-11-03T21:23:58Z | explorer-quantized-kernels | posted ineff-quantized-kernels-1: per-pair qvm grids at ~1/3 peak, saves 0.5/2.5/8.6ms
2026-11-03T21:23:58Z | explorer-quantized-kernels | posted ineff-quantized-kernels-2: three launches per pair plus round-trips, saves 0.2/0.3/0.3ms
2026-11-03T21:23:58Z | explorer-quantized-kernels | noted on ineff-runs-3: rhs kernel never fires (needs rows>=60), sorted speedup is L2 locality
2026-11-03T21:23:58Z | explorer-quantized-kernels | finished territory quantized-kernels: checklist sourced, 2 inefficiencies posted
2026-11-03T21:24:21Z | explorer-findings | posted ineff-findings-1: routing+combine 1.63ms of 4.75ms at 1 row, fusion saves ~1420/1500/1800us at 1/8/32 rows
2026-11-03T21:24:21Z | explorer-findings | posted ineff-findings-2: routed experts per-pair reads, 14.67ms of 28.19ms at 32 rows, grouped saves 3670us
2026-11-03T21:24:21Z | explorer-findings | posted ineff-findings-3: split-K 173MB writes per step at 32 rows, rough upper bound 2400us, outside slots
2026-11-03T21:24:21Z | explorer-findings | adapted: kept findings-1/2 open despite runs/model overlap (joint-slot rationale, vllm-metal-1 points at findings-1, kernel-efficiency scope); notes appended on ineff-runs-1/2/3, ineff-model-2, territory-runs
2026-11-03T21:24:21Z | explorer-findings | corrected ineff-runs-3 premise: micro A/B sorted flag selects per-pair vs grouped kernel on identically ordered pairs (moe_parts.py:149-163), not pair order; 32-row gap not collected
2026-11-03T21:24:21Z | explorer-findings | finished territory findings: checklist sourced (11 pieces), 3 inefficiencies posted, vllm-metal note folded, 5 notes appended
2026-11-03T21:25:18Z | explorer-findings | amended ineff-findings-2 to v3: 1-row saving 660 to 400us (Aranges off-chain, agrees switch-layers-2); noted on ineff-quantized-kernels-1 (micro rhs did fire, saving-basis flag)
2026-11-03T21:25:18Z | explorer-findings | finished territory findings v4: done, all notes answered, overlaps flagged for lead
2026-11-03T21:25:34Z | explorer-quantized-kernels | direction change: my ineff-runs-3 note premise fell (micro sorted=True did run rhs, sliced E) — retracted, serving half stands
2026-11-03T21:25:34Z | explorer-quantized-kernels | direction change: ineff-quantized-kernels-2 superseded by ineff-switch-layers-2 (same launch-fusion cause, theirs broader), numbers moved to a note there
2026-11-03T21:25:34Z | explorer-quantized-kernels | revised ineff-quantized-kernels-1 to v2: glossary-clean Kind, rhs corroboration, findings-2 link
2026-11-03T21:25:34Z | explorer-quantized-kernels | finished territory quantized-kernels: checklist sourced, ineff-1 open, ineff-2 superseded
2026-11-03T21:27:51Z | lead | checkpoint v3: all explorers done; findings-1 superseded by runs-2+model-2 (joint view, per-slot entries carry causes); runs-1/findings-2/quantized-1 kept complementary; 8 premature confirmed sent to verifier for real checks
2026-11-03T21:29:26Z | verifier | verdict ineff-runs-1: confirmed (465MB excess, 2.3ms floor verified against gpu-counters)
2026-11-03T21:29:26Z | verifier | verdict ineff-runs-2: confirmed (routing 1.02/1.68ms overhead verified; carries routing cause)
2026-11-03T21:29:26Z | verifier | verdict ineff-runs-3: refuted (A/B gaps measure kernel choice per moe_parts.py:149-179, not pair order; sorting cannot yield 1.3ms)
2026-11-03T21:29:26Z | verifier | verdict ineff-runs-4: confirmed (K/V re-read 2.3x verified; carries cause, vllm-metal-2 mechanism stands)
2026-11-03T21:29:26Z | verifier | verdict ineff-model-2: confirmed (combine 0.61/0.33ms verified; carries combine cause)
2026-11-03T21:29:26Z | verifier | verdict ineff-findings-2: confirmed v3 (grouped-kernel basis verified; reconciled with qk-1: measured 3.67 vs ideal 8.6, joint ceiling ~11.5ms)
2026-11-03T21:29:26Z | verifier | verdict ineff-findings-3: confirmed (split-K 173MB, upper bound 2400us verified against bottleneck-explained section 6)
2026-11-03T21:29:26Z | verifier | verdict ineff-switch-layers-1: confirmed (sort glue 120 kernels locality-only, 0.7ms inside fusion)
2026-11-03T21:29:26Z | verifier | verdict ineff-switch-layers-2: confirmed (unfused products+activation, 0.4ms; absorbs qk-2)
2026-11-03T21:29:26Z | verifier | verdict ineff-quantized-kernels-1: confirmed v2 (qvm mechanism verified; citation lines corrected; runs-3 note E=60 error flagged)
2026-11-03T21:29:26Z | verifier | verdict ineff-vllm-metal-3: confirmed (60+60 q/k/v copies, 0.21ms measured at 8 rows)
2026-11-03T21:29:26Z | verifier | verdict ineff-vllm-metal-4: confirmed (engine CPU 2-3ms on CPU-bound 1-row step)
2026-11-03T21:29:26Z | verifier | no new duplicates: all overlaps already settled by lead (complementary trio runs-1/findings-2/qk-1; findings-1, model-1, qk-2, vllm-metal-1/2 superseded)
2026-11-03T21:29:26Z | verifier | finished verification: 11 confirmed, 1 refuted (ineff-runs-3), 0 open
2026-11-03T21:30:04Z | lead | checkpoint v4: verifier done, 11 confirmed, runs-3 refuted, 0 open; exploration complete
