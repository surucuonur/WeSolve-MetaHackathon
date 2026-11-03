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
