# Inefficiency: `gather_qmm_rhs` BM=16 tile quantization wastes up to ~440/640/610 µs per prefill layer (repriced)

- **Owner:** explorer-quantized-kernels
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:50:16Z
- **Next:** done — confirmed by verifier (see Verdict), no corrections

## Slot

`experts`

## What happens

Revisit of round 2's confirmed
[`ineff-quantized-kernels-1`](../round-2/exploration/ineff-quantized-kernels-1.md)
(same mechanism, frozen — not re-derived here): MLX's prefill kernel
`affine_gather_qmm_rhs` (`quantized.h:2570-2718`) tiles sorted pairs
into `BM=16`-row tiles, over-provisions `grid_y =
min(B,ceil(B/16)+E-1)` (round-2 entry §*What happens*), and runs
partial tiles through zero-padding (`BlockLoader::load_safe`) plus
`store_result_safe`. What is NEW this round: (a) the waste is priced
in measured prompt µs against the chained two-phase timing, which did
not exist in round 2 (prefill was correctness-only); (b) a third
sub-mechanism — at 127 rows EVERY useful tile is partial
(`cnt_e≈8.5<16`, all 60 tiles), so 100% of useful work takes the slow
`gemm_loop_unaligned` + `load_safe` (per-element `tmp_idx/tmp_val`
bound checks, `steel/gemm/loader.h:83-128`) + `store_result_safe`
path (`quantized.h:2687-2695`). At 1,016 ≈20% of tiles are partial
(the 5th of 5 per expert), at 2,048 ≈11% (the 9th of 9) —
territory §S3. A regrouped exact branch (different BM, variable
tiles, or per-row accumulation with `qmm_rhs` per-row arithmetic —
regrouping is free, territory §S1) avoids all three without changing
any valid output bit.

## Cost

Decode 1/8/32 rows: **0 kernels, 0 bytes** — this kernel never runs
there (`B/E ≤ 2.13 < 4`, round-2 territory §A). Prefill cost table
(balanced experts, from the round-2 entry — verified there):

| prefill rows | B | dispatched `grid_y` | useful `T` | empty y-tiles | padded rows `T·16−B` |
|---|---|---|---|---|---|
| 127 | 508 | 91 | 60 | 31 (34.1%) | 452 / 960 (47.1%) |
| 1,016 | 4,064 | 313 | 300 | 13 (4.2%) | 736 / 4,800 (15.3%) |
| 2,048 | 8,192 | 571 | 540 | 31 (5.4%) | 448 / 8,640 (5.2%) |

Empty threadgroups per layer: 31/13/31 y-tiles × 48 N-tiles (12+12+24
across gate/up/down) = 1,488 / 624 / 1,488 groups of 64 threads that
run only the `schedule_row_tile` scan and exit (`quantized.h:2637-2640`).
Measured basis (new): chained compiled prefill per layer 960.8 /
4306.0 / 11998.9 µs
(`results/runs/2026-11-04-muse-round-2/
grade-phases-experts-attempt-7.json`); mains ≈ measured − glue
(sort+unsort 8/62/126 µs by byte floors, swiglu, offsets) ≈ 930 /
4,150 / 11,700 µs per layer (territory §S5).

## Kind

Overhead-bound (`claude/GLOSSARY.md`): issued work with no output
(padded MMA rows, empty groups, safe-path checks) inside a kernel
already at ~13–33% of either peak (territory §S5).

## Spans

`quantized-kernels` (host grid formula + `schedule_row_tile` +
`BlockMMA` partial-tile path) → slot `experts`. Neighbour:
`baseline-experts` (a grouped prefill branch chooses its own tiling;
this entry prices MLX's).

## Predicted saving

0 µs per decode step at 1, 8 and 32 rows (kernel never runs there).
At prefill, per layer, removing padding saves the padded fraction of
the mains' row work, bounded two ways:

- **Upper edge** (fraction × measured mains): 0.471×930 / 0.153×4150 /
  0.052×11700 ≈ **440 / 635 / 608 µs per layer** at 127 / 1,016 /
  2,048 rows (≈8.8 / 12.7 / 12.2 ms per 20-layer chunk). Realized
  only if the replacement matches MLX's per-group rate; if the mains
  are barrier-bound rather than work-bound, removing padded rows
  saves less (barriers stay).
- **Lower edge** (padded MMA at peak compute, round-2 territory §E:
  51.3/83.5/50.8 µs per layer-product × 3): **154 / 251 / 152 µs per
  layer** (≈3.1 / 5.0 / 3.0 ms per chunk). The kernel does not run at
  peak, so true waste sits above this and below the upper edge.
- Empty tiles add ≤ (empty share × 1% work share — a scan+exit is
  <1% of a 24-iteration K loop) ≈ **≤3 / ≤2 / ≤6 µs per layer**,
  negligible next to padding; the all-partial safe-path tax at 127
  rows is folded into the upper edge (no separate figure exists —
  no per-tile timing).
- **Total: up to ~440 / ~640 / ~610 µs per layer** (≈46% / 15% / 5%
  of compiled prefill) at 127 / 1,016 / 2,048 rows.

Taken ONLY by a fast exact prefill branch that matches MLX's grouped
rate — a `max_rows` hand-off does NOT take it (prefill stays on MLX's
kernels as-is, `same` as compiled). Saving in one phase that costs
the other is not one: the branch must also keep decode wins (or hand
off nothing at decode sizes, which `max_rows` 59 does by construction).
Caveat: a second run timed compiled 2,048-row prefill 39% lower
(7347.5 vs 11998.9 µs — territory §S5); the 2,048-row bound scaled
from that run would be ~370 µs upper edge instead of ~610.

## Verdict

verifier, 2026-11-04T17:50:16Z — **confirmed**. Repricing arithmetic re-derived from the chained basis; inherited mechanism covered by round-2's confirmed verdict.

- **confirmed:** round-2 `ineff-quantized-kernels-1` is `confirmed`, and its verdict covers every inherited cite — `affine_gather_qmm_rhs` BM=16 (quantized.h:2570-2718), host `grid_y` formula (quantized.cpp:1817-1819), empty-exit (:2637-2640), `load_safe` (loader.h:83-128), `store_result_safe` (:2687-2695), the full table (91/313/571, T 60/300/540, empty 31/13/31, padded 47.1/15.3/5.2%), 64-thread groups, 12+12+24 N-tiles, lower-edge 51.3/83.5/50.8 per layer-product, decode-zero (B/E ≤ 2.13 < 4). Table re-arithmeticked anyway ✓ (e.g. 60×16−508 = 452, 452/960 = 47.1%).
- **confirmed:** new sub-mechanism follows from confirmed premises — B/60 = 8.47 < 16 at 127 rows so all 60 tiles are partial and take the `tgp_bm<16` safe path (`load_safe` per-element bound checks + `store_result_safe`, territory §S3); 1/5 ≈ 20% at 1,016, 1/9 ≈ 11% at 2,048 ✓; regrouping free (territory §S1 + round-2 §C invariant) ✓.
- **confirmed:** empty groups 31/13/31 × 48 = 1,488/624/1,488 ✓ (matches §S1: 2,880+1,488 / 14,400+624 / 25,920+1,488); measured basis 960.8/4306.0/11998.9 (experts grade-phases JSON) ✓; mains ≈ 930/4,150/11,700 (territory §S5: sort+unsort floors 8/62/126 recomputed from 1.6/12.5/25.2 MB ✓); kind 13–33% (§S5 table: 12.8–33% ✓).
- **confirmed:** upper edge 0.471×930 = 438 → 440, 0.153×4150 = 635, 0.052×11700 = 608 ✓; ×20 = 8.8/12.7/12.2 ms ✓; lower edge (51.3/83.5/50.8)×3 = 154/251/152 ✓; ×20 = 3.1/5.0/3.0 ✓; empty-tile ≤3/≤2/≤6 orders ✓; total ~440/~640/~610 ≈ 46%/15%/5% of compiled prefill (45.8/14.7–14.9/5.1% ✓); safe-path tax honestly without a separate figure ✓.
- **confirmed:** bankability — fast-branch-only, `max_rows` does not take it (prefill stays on MLX kernels) ✓; `max_rows` 59 hands off nothing at decode sizes (decode ≤ 32 < 59) ✓; 2,048-row caveat — second run 7347.5 is 38.8% lower (territory §S5, handoff-check JSON), scaled bound 608×7347.5/11998.9 = 372 → ~370 ✓.
- No correction. Title "~440/640/610" is the body's 440/635/608 at ~ precision.

## From other agents

(none yet)
