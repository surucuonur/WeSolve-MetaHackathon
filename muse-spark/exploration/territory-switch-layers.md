- **Owner:** explorer-switch-layers
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:32Z
- **Next:** none — checklist sourced, 2 inefficiencies posted, 2 notes appended, no notes to answer

## What it does

`SwitchGLU.__call__` (`third_party/mlx-lm/mlx_lm/models/switch_layers.py:176-199`) runs the
routed experts of one MoE layer: it expands `x`, and from 16 rows up sorts the
(token, expert) pairs by expert (`_gather_sort`), runs the three expert products
(up, gate, down) through one `mx.gather_qmm` each, applies the compiled swiglu
activation, then un-sorts (`_scatter_unsort`). It is the `experts` slot's code path.

## Checklist

### A. `SwitchGLU.__call__` (`switch_layers.py:176-199`)

- **Callers:** `MoE.__call__` (`mobilemoe.py:230`, `y = self.experts(x, inds)`).
  Decode shapes (plain MLX): `x` [rows,1,768] bf16, `inds` [rows,1,4] uint32
  (`territory-model.md` §A from `bench/micro/breakdown.py:442-460`; slot flattens to
  [rows,768] / [rows,4] per `bench/grader/slots.py:107-114`). At 1/8/32 rows: x
  [1,1,768] / [8,1,768] / [32,1,768], inds [1,1,4] / [8,1,4] / [32,1,4].
- **Callees:** `mx.expand_dims` + `squeeze` (views, free); `_gather_sort` (B) and
  `_scatter_unsort` (C) only when `indices.size >= 64` (`:181`), i.e. rows×4 ≥ 64,
  i.e. 16+ rows; three `gather_qmm` calls via `TransposedQuantizedSwitchLinear`
  (`mobilemoe.py:172-184`, transpose=False, rhs_indices only) — one GatherQMM +
  one default-Arange kernel each; compiled `swiglu` (`activations.py:9-11`, one
  fused kernel). Kernel inventory matches exactly:
  `results/runs/2026-11-01-breakdown/tables.md:136-139` — 7/layer at 1–8 rows
  (3×(Arange+GatherQMM) + silu-mul), 13/layer at 32 rows (+ Gather 60, ArgSort 40,
  FloorDivide 20 per step). The 867→987 op jump at 32 rows is all here
  (`tables.md:152-154`: +60 Gather, +40 ArgSort, +20 FloorDivide).
- **Data:** gate_w/up_w [60,768,48] uint32, scales/biases [60,768,12] bf16;
  down_w [60,384,96] uint32, scales/biases [60,384,24] bf16 (brief §experts).
  Sorted path: x [32,1,768] → [128,1,768] bf16, idx [128] uint32; gate/up out
  [128,1,384] bf16; down out [128,1,768] → unsorted to y [32,1,4,768] bf16.
  One expert = 0.553 MB (`bottleneck-explained.md` §4 worked example).
- **Neighbours:** before: router/top-4/weights (model territory) produce `inds`;
  after: combine weighting (`mobilemoe.py:231`) consumes `y` in `inds` order, plus
  the shared expert add. A replacement must keep y's (row, expert-slot) positions
  and the brief §experts rounding points (see Invariants).
- **Cost:** routed-experts knock-out: 0.92 ms @1 row, 14.67 ms @32 rows plain MLX
  (`step-breakdown.md:170,211`); 140 kernels/step @1 (6.6 µs each), 260 @32.
  Reads 51 MB @1 (must-read 44), 908 MB @32 (must-read 443, per-pair 1416)
  (`2026-11-02-gpu-counters/tables.md:26,32`).
- **Invariants:** G2 experts: every element within 2 bf16 ulp of reference (floor at
  0.001); each product accumulates fp32 and rounds to bf16; `silu(gate)*up` is
  computed on bf16 values and rounded to bf16 before the down product (brief §experts);
  unsort must restore exact (row, slot) positions.

### B. `_gather_sort` (`switch_layers.py:12-17`)

- **Callers:** `SwitchGLU.__call__` (`:185`) when rows ≥ 16.
- **Callees:** `mx.argsort(indices)` → ArgSort (order); `mx.argsort(order)` →
  ArgSort (inv_order, a second full sort just to invert the permutation);
  `order // M` → FloorDivide; `x.flatten(0,-3)[order // M]` → Gather;
  `indices[order]` → Gather. Per layer at 32 rows: 2 ArgSort + 1 FloorDivide +
  2 Gather (`tables.md:136` ÷ 20 layers).
- **Data:** over 128 pairs: order/inv_order/idx [128] (uint32/int32), x gathered to
  [128,1,768] bf16 (196 KB/layer). rhs passed to `gather_qmm` ascending;
  `sorted_indices=True` sets MLX `right_sorted_` (`ops.cpp:5824-5825`, v0.32.4).
- **Neighbours:** before: expand_dims; after: the three products read sorted x/idx.
  On the critical chain (products wait for it).
- **Cost:** 5 kernels/layer on-chain (inv_order's argsort overlaps the products);
  no isolated knock-out — inside the 14.67 ms. CPU record for the whole experts
  glue rises 128 → 399 µs/step from 1 to 32 rows (`tables.md:99`).
- **Invariants:** sorted rhs must stay a permutation of the pairs (same multiset of
  (row, expert) pairs); stability across equal experts is irrelevant since pairs
  are distinguishable by row.

### C. `_scatter_unsort` (`switch_layers.py:20-24`)

- **Callers:** `SwitchGLU.__call__` (`:197`) when rows ≥ 16.
- **Callees:** `x[inv_order]` → Gather (1/layer, the third Gather in `tables.md:136`);
  `mx.unflatten` is a view (free).
- **Data:** [128,1,768] bf16 → [32,1,4,768] bf16 (196 KB/layer moved).
- **Neighbours:** before: down product; after: squeeze (view) → combine. On-chain.
- **Cost:** 1 kernel/layer, 20/step; ~6.81 µs each (`moe_parts` fixed-cost unit,
  `territory-runs.md` §`step_parts.txt`).
- **Invariants:** must restore exact pre-sort positions (combine pairs y with weights
  by slot).

### D. `QuantizedSwitchLinear.__call__` (`switch_layers.py:75-90`)

- **Callers:** nothing in the served int4 model — MobileMoE replaces all three
  projections with `TransposedQuantizedSwitchLinear` (`mobilemoe.py:200-214`).
  Same op, `transpose=True`, [E,out,in] layout; the live twin is
  `mobilemoe.py:172-184` (model territory §F).
- **Callees:** one `mx.gather_qmm(..., transpose=True, rhs_indices only,
  sorted_indices)` → GatherQMM + default lhs `arange` (`ops.cpp` `indices_or_default`,
  `tables.md:137-139` pattern). The bias add (`:88-89`) is absent in the served
  model (bias=False, `switch_layers.py:171-173`).
- **Data/Neighbours/Cost/Invariants:** as the live twin (see A/F of territory-model):
  fp32 accumulate, bf16 out; per-pair reads in MLX's qvm/qmv vector kernel below
  60 rows (see G).
- **Note:** kept for the unquantized→quantized conversion path (`to_quantized`,
  `:131-149`); not decode-time.

### E. `SwitchLinear` (`switch_layers.py:93-149`, unquantized)

- Not on the int4 decode path (`mx.gather_mm`, dense). Only its `to_quantized`
  constructor matters at conversion time. No checklist beyond this line.

### F. `SwitchMLP` (`switch_layers.py:202-236`)

- GELU variant; MobileMoE uses `SwitchGLU`. Same sort threshold (`:222`) and
  glue structure, but dead code for this model. No checklist beyond this line.

### G. MLX dispatch under `gather_qmm` (v0.32.4 source, read to fix the callees)

- `GatherQMM::eval_gpu` (`quantized.cpp:1980-2094`): the grouped path
  `gather_qmm_rhs` needs `M == 1 && B >= 16 && right_sorted_ && B/E >= 4`
  (`:2012`); else `M >= vector_limit` (4 when transpose=False, `:2005`) takes the
  matrix path; else the per-pair vector kernel (`gather_qvm`, `:2076-2093`).
- The port's call gives M=1, B=4·rows pairs, E=60, right_sorted_=True (sorted flag
  set, lhs not passed: `ops.cpp:5824-5825` + `primitives.h:1703-1710` wiring).
  `B/E >= 4` needs 240+ pairs = 60+ rows — never in decode (max 32). So **every**
  decode row count runs `gather_qvm`, whose grid is (M, N/64, B) threadgroups of
  64 threads (`quantized.cpp:1462-1500`): each pair's tiles read their expert's
  weights independently → per-pair reads. The sort changes locality, not kernel choice.
- The micro kernel-A/B experiment (`bench/micro/moe_parts.py:152-180`) slices
  weights to E experts, so B/E ≥ 4 there and sorted=True *does* switch kernels —
  its sorted/unsorted gaps measure the grouped kernel, not serving's sort. Noted
  on `ineff-runs-3`.

## Relations

- **Fed by:** model territory — `MoE.__call__` passes x [rows,1,768] bf16 and inds
  [rows,1,4] uint32 (`mobilemoe.py:230`).
- **Feeds:** model territory — y [rows,1,4,768] bf16 in inds order to combine
  (`mobilemoe.py:231`).
- **Callees in:** quantized-kernels territory — GatherQMM/ArgSort/Gather/FloorDivide
  Metal kernels and their fixed costs (entry does not exist yet, 21:25Z; note parked):
  *quantized-kernels:* every decode `gather_qmm` lands in `gather_qvm`
  (`quantized.cpp:2076-2093`, grid (M,N/64,B), 64-thread groups) because B/E<4
  (`:2012`); per-pair weight reads are structural there, not a flag the port could set.
- **Measurement in:** runs/findings territories — knock-outs, counters, micro tables
  (all inefficiency costs cite them; mechanism notes appended to `ineff-runs-1` and
  `ineff-runs-3` instead of duplicating).
- **Note folded:** model territory's parked note on the 16-row threshold
  (`territory-model.md` Relations) — incorporated in A/B above; kernel inventory agrees.

## Inefficiencies

- `ineff-switch-layers-1` (open): sort glue costs 6 kernels/layer from 16 rows for
  locality only — kernel choice is per-pair qvm regardless below 60 rows.
- `ineff-switch-layers-2` (open): gate/up/down + activation unfused — 4 on-chain
  kernels/layer plus bf16 intermediates written and re-read.
- Not posted: per-pair re-reads (covered by `ineff-runs-1`, mechanism appended);
  unsorted-below-16 (covered by `ineff-runs-3`, correction appended); the 60
  default-Arange kernels/step (measured ~free, `step-breakdown.md:438-463` — they
  run alongside the chain).

## Files read

- `third_party/mlx-lm/mlx_lm/models/switch_layers.py` (all 236 lines): the territory.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:142-232`: caller, live proj twin.
- `third_party/mlx-lm/mlx_lm/models/activations.py:9-11`: swiglu compile-fused.
- `bench/micro/moe_parts.py:149-218`: kernel-A/B setup (weights sliced to E).
- `bench/micro/repeat_fixes.py:10-11,71-72`: sorted-kernels-need-no-lhs remark.
- `claude/muse-agent-brief.md` (slots, G1–G3, slot costs).
- `claude/GLOSSARY.md` (kinds).
- `claude/findings/step-breakdown.md` + `bottleneck-explained.md` (knock-outs, bytes).
- `results/runs/2026-11-01-breakdown/tables.md:96-159` (CPU, kernel inventory).
- `results/runs/2026-11-02-gpu-counters/tables.md` (reads, knock-out savings).
- `results/runs/2026-10-30-micro/moe_parts.txt` + `step_parts.txt` (via territory-runs).
- MLX v0.32.4 source (web): `mlx/backend/metal/quantized.cpp:1980-2094,1462-1500`,
  `mlx/ops.cpp:5720-5835`, `mlx/primitives.h:1701-1717`.

## From other agents

(none yet)
