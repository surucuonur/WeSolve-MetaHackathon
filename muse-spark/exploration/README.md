# Exploration: round 1 — analysis

**Version:** v5 · 2026-11-03T21:30:39Z · skill version 3

## Tracker

| Territory | Agent | Status | Inefficiencies | Updated |
|---|---|---|---|---|
| `model` | `explorer-model` | done | `ineff-model-1` (superseded by ineff-runs-2), `ineff-model-2` (confirmed) | 2026-11-03 21:30Z |
| `switch-layers` | `explorer-switch-layers` | done | `ineff-switch-layers-1` (confirmed), `-2` (confirmed) | 2026-11-03 21:30Z |
| `quantized-kernels` | `explorer-quantized-kernels` | done | `ineff-quantized-kernels-1` (confirmed), `-2` (superseded by ineff-switch-layers-2) | 2026-11-03 21:30Z |
| `other-kernels` | `explorer-other-kernels` | done | — (kernel notes on runs-2, model-2, runs-3) | 2026-11-03 21:30Z |
| `vllm-metal` | `explorer-vllm-metal` | done | `ineff-vllm-metal-1` (superseded by ineff-findings-1), `-2` (superseded by ineff-runs-4), `-3` (confirmed), `-4` (confirmed) | 2026-11-03 21:30Z |
| `grader` | `explorer-grader` | done | — (grader notes on runs-1/2/3, model-2) | 2026-11-03 21:30Z |
| `findings` | `explorer-findings` | done | `ineff-findings-1` (superseded by ineff-runs-2 and ineff-model-2), `-2` (confirmed), `-3` (confirmed) | 2026-11-03 21:30Z |
| `runs` | `explorer-runs` | done | `ineff-runs-1` (confirmed), `-2` (confirmed), `-3` (refuted), `-4` (confirmed) | 2026-11-03 21:30Z |

Verdicts: 11 confirmed, 1 refuted, 5 superseded, 0 open. All 8 territory entries `done`.

## Context

A decode step produces one token per request. On plain MLX with 128-token prompts the bare step takes 4.75 / 13.37 / 28.19 ms at 1 / 8 / 32 rows (Claude: `results/runs/2026-11-01-breakdown/tables.md:14-24`, via [territory-findings.md](territory-findings.md) §A); vLLM takes 5.64 / 14.30 / 30.47 ms. A step is 867 kernels at 1 row, 987 at 32 rows, and stays far below both hardware limits (whole-step 24–41 GB/s against 200 GB/s peak, ≤9% of peak compute), so it is overhead-bound throughout (Claude: step-breakdown §3, [territory-findings.md](territory-findings.md) §E). Each tiny kernel costs ~6.81 µs end to end (Claude: `results/runs/2026-10-30-micro/step_parts.txt`, via [territory-runs.md](territory-runs.md)).

The MoE block (`MoE.__call__`, `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`) runs in all 20 layers of every step ([territory-model.md](territory-model.md)). Per layer it launches: routing, 12 kernels (router cast+matmul+sigmoid, bias add, negate, argpartition, gather, weight norm); the routed experts, 7 kernels at 1 row (3 products + activation + 3 off-chain index ranges) and 13 from 16 rows (6 sort/unsort glue kernels); combine, 5 kernels (cast, multiply, sum, cast, add with the shared expert); plus the shared expert, 4 kernels. The three slots mirror this split: `routing` (240 kernels/step), `experts` (140–260/step), `combine` (100/step).

Where the time goes (Claude's knock-outs, plain MLX, via [territory-findings.md](territory-findings.md) §C–D and [territory-runs.md](territory-runs.md)): at 1 row, routing costs 1.02 ms and combine 0.61 ms — together 1.63 ms of the 4.75 ms step (34%), more than the routed experts' 0.92 ms. At 32 rows the experts cost 14.67 ms of 28.19 ms (52%) while routing falls to 1.68 ms (6%) and combine to 0.33 ms (1%). The experts read 908 MB at 32 rows against 443 MB for the distinct experts (Claude: `results/runs/2026-11-02-gpu-counters/tables.md`, via [territory-runs.md](territory-runs.md)) — half the bytes are repeat reads, and even so the 14.67 ms runs at only 62 GB/s (31% of peak), so most of it is kernel efficiency, not bytes (swarm reading: [ineff-findings-2.md](ineff-findings-2.md), [ineff-quantized-kernels-1.md](ineff-quantized-kernels-1.md)).

## Territory map

Measurements feed code: `runs` and `findings` map the raw tables and Claude's findings into the costs every inefficiency cites. `model` maps `MoE.__call__` and hands the experts path to `switch-layers`, the routing/combine chains to `other-kernels`, and the `gather_qmm` call to `quantized-kernels`. `vllm-metal` maps the engine around the model. `grader` maps the slot contract every solution keeps.

| Territory | Receives | Passes on | Measured cost |
|---|---|---|---|
| `model` | `x` from post-attention norm; `h` for the residual | routing → `inds`/`weights`; experts → `y`; combine → `out` to next layer | routing 1.02/1.68 ms, experts 0.92/14.67 ms, combine 0.61/0.33 ms at 1/32 rows (knock-outs) |
| `switch-layers` | `x` [rows,1,768], `inds` [rows,1,4] from `MoE.__call__` | `y` in `inds` order to combine | experts path: 140 kernels/0.92 ms at 1 row, 260/14.67 ms at 32 rows |
| `quantized-kernels` | pairs + expert weights via `gather_qmm` | products to activation (gate/up) and unsort (down) | 60 GatherQMM launches/step; experts traffic at 55–62 GB/s (31% of peak) |
| `other-kernels` | ops from `MoE.__call__`; `y` from experts; weights; shared expert | `inds`/`weights` to experts/combine; `out` onward | routing chain 240 kernels, combine chain 100 kernels; every kernel far below occupancy |
| `vllm-metal` | scheduled batch + block tables from the scheduler | packed forward → sampler → scheduler | runner outside model 0.5/0.9/1.6 ms at 1/8/32 rows; attention path +1 ms Python vs plain MLX |
| `grader` | references from model/switch-layers code; costs from runs/findings | spec format, G1/G2/timing verdicts, boundary floor to builders | reference slot costs (the headroom); slot-edge floor ≈0.04 ms at 32 rows |
| `findings` | raw run tables (via `runs`) | pieces C–J costs to every code territory | denominators: 4.75/13.37/28.19 ms steps; MoE 560–680 kernels |
| `runs` | raw counters/micro/baseline tables | fixed 6.81 µs/kernel, knock-out costs, byte totals to all | step reads 193/685/1561 MB vs 180/393/731 MB must-read at 1/8/32 rows |

## Inefficiencies, ranked

Confirmed inefficiencies inside the slots, ranked by predicted saving as a share of the decode step (plain MLX: 4.75 / 13.37 / 28.19 ms at 1 / 8 / 32 rows), averaged over the three row counts. Shares shown as 1-row / 8-row / 32-row (mean). **The five `experts` entries are one combined fix: their savings do not add** (verifier reconciliation; joint ceiling 443 MB at streaming rate ≈ 11.5 ms at 32 rows). Each item names whether its evidence is Claude's measurement or the swarm's own reading.

1. **Per-pair qvm grids run expert traffic at ~1/3 peak** — 10.5 / 18.7 / 30.5% (mean 19.9%). Cost: 0.92 ms at 1 row, ~5.1 ms at 8 rows, 14.67 ms at 32 rows (62 GB/s, 31% of peak). Kind: overhead-bound with a bandwidth-waste component. Entry: [ineff-quantized-kernels-1.md](ineff-quantized-kernels-1.md) (saves 0.5 / 2.5 / 8.6 ms at ideal 150 GB/s streaming). Costs are Claude's counters/knock-outs; the mechanism is the swarm's own: every decode product lands in `affine_gather_qvm` (M=1 rules out the matrix path, B/E≤2.1 rules out the grouped path below 60 rows — proved against MLX v0.32.4 dispatch source), with strided lanes, 12× `x` re-reads per pair, and occupancy-starved grids at 1 row.
2. **Routing is pure overhead: 1 ms for ~4 MB** — 18.9 / 9.0 / 5.0% (mean 11.0%). Cost: 1.02 ms at 1 row, 1.68 ms at 32 rows (4–40 GB/s). Kind: overhead-bound. Entry: [ineff-runs-2.md](ineff-runs-2.md) (saves 0.9 / 1.2 / 1.4 ms by fusing 12 kernels/layer to ~1). Costs are Claude's knock-outs; the swarm's kernel inventory adds that argpartition runs a full 60-element merge sort per row and the weights-norm sum runs on a single GPU thread at 1 row.
3. **Routed experts read per pair and run slowly (grouped-kernel basis)** — 8.4 / 9.7 / 13.0% (mean 10.4%). Cost: 14.67 ms of 28.19 ms at 32 rows. Kind: overhead-bound with a bandwidth-waste component. Entry: [ineff-findings-2.md](ineff-findings-2.md) (saves 0.40 / 1.30 / 3.67 ms on the measured 25% grouped-kernel win, upper 36%). The 25–36% gap is Claude's micro A/B measurement; the swarm's correction establishes it measures kernel choice (per-pair vs grouped), not pair order.
4. **Gate/up/down products and activation unfused** — 8.4 / 3.0 / 1.4% (mean 4.3%, tied with 5 within rounding). Cost: 4 on-chain kernels/layer, 80/step, ~0.54 ms fixed costs at every row count. Kind: overhead-bound. Entry: [ineff-switch-layers-2.md](ineff-switch-layers-2.md) (saves ~0.4 ms at all row counts; absorbs `ineff-quantized-kernels-2`). The kernel inventory is Claude's op table; the fusion design and the absorbed kernel-side traffic detail are the swarm's own.
5. **Combine is 5 tiny kernels with bf16↔fp32 casts** — 9.9 / 2.2 / 0.7% (mean 4.3%, tied with 4 within rounding). Cost: 0.61 ms at 1 row, 0.33 ms at 32 rows (~6 MB activations). Kind: overhead-bound. Entry: [ineff-model-2.md](ineff-model-2.md) (saves 0.47 / 0.30 / 0.19 ms fusing to 1 kernel/layer). Costs are Claude's knock-outs; the chain inventory, the 32-row overlap cap, and the launch-shape detail (128-thread `col_reduce_small`, cache-resident intermediates) are the swarm's own.
6. **Experts re-read weights per (token, expert) pair** — 0 / 4.5 / 8.2% (mean 4.2%). Cost: 908 MB read vs 443 MB must-read at 32 rows (465 MB excess). Kind: overhead-bound with a bandwidth-waste component. Entry: [ineff-runs-1.md](ineff-runs-1.md) (saves 0.0 / 0.6 / 2.3 ms: excess bytes at peak bandwidth, a floor). Bytes are Claude's counters; the per-pair-read mechanism (qvm grid per pair, B/E≥4 gate needing 60+ rows) is the swarm's dispatch proof.
7. **Sort glue costs 6 kernels/layer from 16 rows, for locality only** — 0 / 0 / 2.5% (mean 0.8%). Cost: 120 kernels/step, ~0.7 ms on-chain at 32 rows. Kind: overhead-bound. Entry: [ineff-switch-layers-1.md](ineff-switch-layers-1.md) (saves 0 / 0 / 0.7 ms, only inside a fused grouped kernel that needs no sort). The op counts are Claude's tables; the proof that the sort cannot change kernel choice below 60 rows (so standalone removal loses time) is the swarm's own.

Superseded (same causes, kept readable, not ranked): `ineff-model-1` → runs-2; `ineff-findings-1` → runs-2 + model-2 (its joint 1420/1500/1800 µs figure ≈ their sum; the brief tests `routing+combine` parts independently, so hypotheses stay per-slot); `ineff-quantized-kernels-2` → switch-layers-2; `ineff-vllm-metal-1` → findings-1; `ineff-vllm-metal-2` → runs-4.

## Recommendations

Confirmed inefficiencies outside the slots, for surucuonur's team. No slot covers them, so no hypothesis targets them.

- **Attention re-reads K/V per query head** ([ineff-runs-4.md](ineff-runs-4.md)): 12 query heads over 4 KV heads take 2.3× the 4/4 time for the same bytes (Claude: `attention_heads.txt` 568.7 vs 247.1 µs/layer at 32 rows; knock-out caps the prize at 0.41 ms at 1 row, 3.12 ms at 32 rows). The swarm's mechanism: vllm-metal's GQA decode kernel admits only six layouts, missing MobileMoE-S 12/4/64/16, so decode silently falls back to the per-token kernel — with the caveat that at 128-token prompts the GQA admission rule admits nothing either, so the fix needs an admission change too. Saving up to ~0.1 ms at 1 row, ~2 ms at 32 rows.
- **Split-K partial traffic from 14 rows** ([ineff-findings-3.md](ineff-findings-3.md)): 173 MB written per step at 32 rows plus read-back plus 140 add kernels, in the q/k/v/o projections and the shared expert (Claude: `bottleneck-explained.md` §6; knock-out write savings 57 + 102 MB corroborate). Rough upper bound ≈2.4 ms at 32 rows, 0 at 1/8 rows; the time effect is unmeasured and testing needs a patched MLX.
- **60 q/k/v `[0]` gathers + 60 contiguous copies per step** ([ineff-vllm-metal-3.md](ineff-vllm-metal-3.md)): every layer copies q, k, v twice before attention (`sdpa.py:722-724`); exact fix is `squeeze(0)` views. Measured −0.21 ms at 8 rows (Claude: `repeat_fixes.py`, beats the 0.03 ms spread); ~0.3 ms estimate at 1 row, 0.24 ms knock-out at 32 rows.
- **~2 ms engine CPU per 1-row step while CPU-bound** ([ineff-vllm-metal-4.md](ineff-vllm-metal-4.md)): vLLM spends 2–3 ms more CPU than plain MLX (0.5 ms runner, ~1 ms attention-path Python, +1.1–1.5 ms encode; Claude: step-breakdown §2+§4), and at 1 row the step waits only 0.41 ms for the GPU, so CPU sets the pace. Small Python-side fixes save ≈0.5 ms at 1 row; 0 at 8/32 rows (GPU-bound there).

## Refuted claims

- `ineff-runs-3` (refuted): the micro sorted/unsorted gaps (5.08→3.82, 8.12→5.65, 15.02→9.63 ms) measure per-pair vs grouped kernel choice on identically ordered pairs, not pair order — sorting serving's pairs at 8 rows cannot yield 1.3 ms, and the 32-row gap was never collected.

## Files read

Each territory entry lists what it read and what each file gave: [territory-model.md](territory-model.md), [territory-switch-layers.md](territory-switch-layers.md), [territory-quantized-kernels.md](territory-quantized-kernels.md), [territory-other-kernels.md](territory-other-kernels.md), [territory-vllm-metal.md](territory-vllm-metal.md), [territory-grader.md](territory-grader.md), [territory-findings.md](territory-findings.md), [territory-runs.md](territory-runs.md).

## Versions

- v5 · 2026-11-03T21:30:39Z · final analysis: context, territory map, 7 ranked inefficiencies, 4 recommendations, 1 refuted claim · changed: none (README only)
- v4 · 2026-11-03T21:30:00Z · verifier done: 11 confirmed, runs-3 refuted · changed: ineff-runs-1 v3, ineff-runs-2 v3, ineff-runs-3 v3, ineff-runs-4 v3, ineff-model-2 v3, ineff-findings-2 v4, ineff-findings-3 v3, ineff-switch-layers-1 v2, ineff-switch-layers-2 v2, ineff-quantized-kernels-1 v3, ineff-vllm-metal-3 v3, ineff-vllm-metal-4 v3
- v3 · 2026-11-03T21:27:27Z · all explorers done; findings-1 superseded by runs-2+model-2; 8 premature confirmations sent to verifier · changed: territory-switch-layers v2, territory-quantized-kernels v3, territory-other-kernels v2, territory-vllm-metal v2, territory-grader v2, territory-findings v4, ineff-findings-1 v3, ineff-findings-2 v3, ineff-quantized-kernels-1 v2
- v2 · 2026-11-03T21:21:13Z · model and runs done; 8 inefficiencies posted · changed: territory-model v2, territory-runs v2, ineff-model-1 v2, ineff-model-2 v1, ineff-runs-1 v1, ineff-runs-2 v1, ineff-runs-3 v1, ineff-runs-4 v1, territory-vllm-metal v1, ineff-vllm-metal-1 v1, ineff-vllm-metal-2 v1
- v1 · 2026-11-03T21:17:35Z · opened exploration, 8 territories started · changed: none
