# Territory: other-kernels

- **Owner:** explorer-other-kernels
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:35Z
- **Next:** none — checklist sourced; kernel-level notes appended to ineff-runs-2, ineff-model-2, ineff-runs-3 instead of new entries (same causes already covered)

## What it does

The routing and combine slots are built from MLX's generic Metal kernels, not the
quantized ones: a small matmul plus sigmoid for router scores, bias add, negate,
`argpartition` top-4, `take_along_axis` gather, row reductions, broadcast
multiply/divide, astype copies, and the final add. This territory maps those ops to
the kernels in `.venv/.../mlx/backend/metal/kernels/` (headers, installed) and the
launch choices in MLX's C++ dispatch (read at tag v0.32.4 on GitHub — installed
version confirmed 0.32.4), with shapes, bytes, and launch structure at 1, 8, 32 rows.

## Checklist

### A. Router matmul (routing kernel 2 of 12)

- **Callers:** `MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:219`:
  `self.router(x.astype(mx.float32))`), via slot `routing` (`bench/grader/slots.py:45`).
- **Callees:** one `Matmul` kernel op per layer (20/step, `results/runs/2026-11-01-breakdown/tables.md:133`).
  Dense f32 `[rows,768] @ [768,60]`; MLX's steel/gemm path (`kernels/steel/gemm/`;
  exact kernel choice is in C++ `matmul.cpp`, outside the installed tree).
- **Data:** `x` f32 [rows,768] (from AsType of bf16); `w_router` f32 [60,768], 184,320 B/layer,
  3.7 MB/step reread per layer (`claude/findings/step-breakdown.md:250`); out `scores` f32
  [rows,60], consumed by sigmoid (B) and later by the bias add (C).
- **Neighbours:** before: `x.astype(f32)` (A0) and the bias AsType (off-chain,
  `step-breakdown.md:462-463` — runs alongside, saves ~nothing alone); after: sigmoid.
  A replacement must keep scores bit-exact through fp32 sigmoid (G2 ties rule,
  `claude/muse-agent-brief.md:182-184`).
- **Cost:** 1 kernel/layer; reads 184 KB weights + rows*3 KB activations; writes rows*240 B.
  Piece `router` (AsType+Matmul+Sigmoid) knock-out is inside routing's 1.02/1.68 ms
  (`step-breakdown.md:169,210`); serialized GPU 0.98 ms @1, 2.12 ms @32 for all 20 layers'
  router pieces (`tables.md:207,256` — serialized, not in-step).
- **Invariants:** fp32 accumulate exactness for the matmul; sigmoid in fp32.

### A0. AsType x bf16→f32 (routing kernel 1 of 12)

- **Callers/Callees:** `mobilemoe.py:219`, `slots.py:45`; `copy_v` in `kernels/copy.h:21-37`
  (one thread per N outputs, grid covers rows*768 elements: 768 / 6,144 / 24,576 threads).
- **Data:** read rows*1,536 B bf16, write rows*3,072 B f32. Consumed only by the matmul.
- **Cost:** 1 kernel/layer, pure fixed cost (~4–7 µs;
  `step-breakdown.md:179-185`) moving ≤24 KB.

### B. Sigmoid (routing kernel 3 of 12)

- **Callers/Callees:** same call sites; `unary_v` (`kernels/unary.h:3-19`) with
  `Sigmoid` (`kernels/unary_ops.h:308`); grid covers rows*60 elements (60/480/1,920 threads).
- **Data:** f32 [rows,60] in and out; consumed by bias add and the weight gather.
- **Cost:** 1 kernel/layer moving ≤7.7 KB. Fixed-cost-bound.

### C. Bias add + negate (routing kernels 5–6 of 12)

- **Callers/Callees:** `mobilemoe.py:223-224`, `slots.py:46`; `binary_vv`/`binary_g`
  (`kernels/binary.h:50-67,178-199`) for `scores + bias` (broadcast [60] over rows),
  then `unary_v` with negate; grids of rows*60 elements.
- **Data:** f32 [rows,60] throughout; bias f32 [60] (from the per-layer bf16→f32 AsType,
  routing kernel 4 — one of the 20 AsType/step in `tables.md:134`, off the critical chain).
- **Cost:** 2 kernels/layer moving ≤15 KB total. Both exist only to feed argpartition.

### D. ArgPartition → full 60-element argsort (routing kernel 7 of 12)

- **Callers/Callees:** `mx.argpartition(-biased, kth=3, axis=-1)` (`mobilemoe.py:224`,
  `slots.py:46`); `ArgPartition::eval_gpu` "direct[s] arg partition to sort"
  ([sort.cpp, v0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/sort.cpp)
  lines 344–356): a FULL merge sort of all 60 elements per row, not a top-4 selection.
  Size 60 → tn=4, potential_bn=15 → bn=32, single block (`sort.cpp:288-314`);
  `block_sort` (`kernels/sort.h:319-364`) with grid (1, rows, 1) threadgroups × 32 threads:
  32 / 256 / 1,024 threads total, each row fully sorted with per-thread sort + log2(32)=5
  merge rounds over 128 padded slots (`sort.h:158-243`).
- **Data:** in f32 [rows,60]; out uint32 [rows,60] (60 indices written per row, slice
  `[..., :4]` keeps 4 — a view, no kernel). 15× more index bytes written than used.
- **Neighbours:** before negate; after `take_along_axis`. G2 ties rule lives here
  (brief:182-184).
- **Cost:** 1 kernel/layer, the heaviest per-thread work in routing (full sort for top-4),
  still fixed-cost-bound: 32 threads/row cannot fill the GPU (occupancy 8–9% @1 row,
  `step-breakdown.md:351-357`).
- **Invariants:** top-4 set must match except within-1e-6 tie rows; order within the 4 is free.

### E. GatherAxis take_along_axis (routing kernel 8 of 12)

- **Callers/Callees:** `mobilemoe.py:225`, `slots.py:47`; `gather_axis`
  (`kernels/indexing/gather_axis.h:6-44`), one thread per output, rows*4 threads (4/32/128).
- **Data:** scores f32 [rows,60] + inds u32 [rows,4] → w f32 [rows,4]. Strided reads of 4 of 60.
- **Cost:** 1 kernel/layer; ≤2 KB moved. Pure fixed cost.

### F. Weights normalize: Sum + Add + Divide + Multiply (routing kernels 9–12 of 12)

- **Callers/Callees:** `mobilemoe.py:226-228`, `slots.py:48`; `Sum` 20/step, `Divide` 20/step
  (`tables.md:134,152-154`), scalar `Add` (+1e-20) and `Multiply` (×2.5) inside the Add-100 /
  Multiply-40 totals (`tables.md:152`).
- **Callees (launch):** the sum over rows of 4 → `row_reduce_small`
  ([reduce.cpp, v0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/reduce.cpp),
  row_size 4 ≤ 64 takes the small path; non_row_reductions 1 ≤ 8 → grid = rows threads in ONE
  threadgroup): **1 thread @1 row, 8 @8, 32 @32** (`kernels/reduction/reduce_row.h:199-257`,
  one thread loops the 4 elements). Add/Divide/Multiply are `binary_vs` grids of rows*4 elements.
- **Data:** f32 [rows,4] → [rows,1] → [rows,4] weights; consumed by experts (no — by combine)
  and slot output. G2: weights within 1e-4 relative by expert.
- **Cost:** 4 kernels/layer moving ≤1 KB. The sum at 1 row is a single GPU thread.

### G. Combine: AsType + Multiply + Sum + AsType + Add (5 kernels/layer)

- **Callers/Callees:** `mobilemoe.py:231-232`, `slots.py:52`; AsType 40, Multiply 20, Sum 20,
  Add 20 per step (`tables.md:140`), identical at 1/8/32 rows (`tables.md:152-154`).
- **Callees (launch):** y bf16→f32 `copy_v` grid of rows*4*768 (3,072 / 24,576 / 98,304 threads);
  broadcast multiply `binary_g` (`kernels/binary.h:178-199`), weights [rows,4,1] over y;
  sum over the middle axis (size 4, stride 768) → strided small path, `col_reduce_small`
  (4 < 32 threshold, `reduce.cpp` strided dispatch), group (32,4,1)=128 threads;
  result f32→bf16 `copy_v`; `+ shared` `binary_vv` grid of rows*768.
- **Data:** y bf16 [rows,4,768] (6 / 49 / 196 KB), weights f32 [rows,4], shared bf16 [rows,768],
  out bf16 [rows,768]. Unique activation bytes ~0.18 / 1.5 / 5.9 MB/step; kernel-touch bytes
  (every kernel's read+write) ~1.3 / 11 / 43 MB/step — but intermediates ≤393 KB stay in GPU
  cache between dependent kernels, so DRAM traffic is far below that; either way ≤0.22 ms
  at peak vs 0.61/0.33 ms measured → overhead-bound (agrees with ineff-model-2).
- **Neighbours:** before: experts' y (`quantized-kernels`/`switch-layers`), routing weights,
  shared expert (outside slots); after: residual add + next layer. G2: 2 bf16-ULP per element
  (brief:184).
- **Cost:** measured knock-out 0.61 ms @1, 0.33 ms @32 (`step-breakdown.md:171,212`);
  100 kernels × ~6.1/~3.3 µs. @8 rows not knocked out; same 100 kernels (`tables.md:153`).
- **Invariants:** fp32 accumulation order for the 4-term sum (near-zero 0.001 floor, brief:184);
  bf16 rounding points at the astype and the add.

### H. Softmax kernels (context — NOT launched by the slots)

- `kernels/softmax.h:10-98` `softmax_single_row` is a fused one-threadgroup-per-row kernel
  (max + exp + normalize in one launch) — the pattern a fused routing kernel would follow.
  Routing does not call softmax: sigmoid + manual top-4 + manual normalize (`mobilemoe.py:219-228`).
  No entry: nothing to remove, noted as the fusion template for builders.

### I. Neighbours outside the slots (noted, no entries)

- Input/post-attention RMSNorms (40 kernels/step) + residual adds: ≈0.3 ms/step
  (`territory-model.md` checklist G; `step-breakdown.md:167,208`). Outside all slots;
  `fast.rms_norm` is already one kernel per norm — nothing to win from this territory.
- `expert_bias` bf16→f32 AsType: off-chain, exact fix in `step-breakdown.md:422-424`
  saves ~nothing (`:437-460`). Covered there, not posted.
- Experts-gather ArgSort (40/step @32 rows, `tables.md:154`): same single-block merge sort
  as (D); ≤128-element rows → bn=32, one threadgroup per row — fixed-cost regime. Noted on
  ineff-runs-3's caveat; the sort-vs-unsorted call belongs to `switch-layers`/`quantized-kernels`.

## Relations

- Fed by: `model` (`MoE.__call__` launches every op here); `switch-layers`/`quantized-kernels`
  feed combine's y; shared expert (in `model`) feeds combine's add.
- Feeds: routing's inds/weights feed `experts` and combine; combine's out feeds the next layer.
- Spans asserted by others: ineff-runs-2 spans `runs, model, other-kernels` (routing);
  ineff-model-2 spans combine (kernel detail appended there).

## Inefficiencies

- None raised here: routing's chain is ineff-runs-2 (with ineff-model-1's code detail),
  combine's chain is ineff-model-2. Kernel-level confirmation appended to both (see log);
  sort-cost note appended to ineff-runs-3.

## Files read

- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232` (MoE.__call__: the 12+5 op chain).
- `bench/grader/slots.py:44-52` (slot references == port code flattened).
- `kernels/sort.h:1-364` (merge-sort structure, block_sort); `kernels/binary.h` (all);
  `kernels/unary.h` (all); `kernels/unary_ops.h:308` (Sigmoid); `kernels/copy.h:21-37` (copy_v);
  `kernels/softmax.h:10-98` (fused single-row template); `kernels/reduction/reduce_row.h:199-257`
  (row_reduce_small); `kernels/indexing/gather_axis.h:6-44`.
- [sort.cpp @ v0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/sort.cpp):
  ArgPartition→sort (:344-356), bn/tn selection (:288-314), single-block grid.
- [reduce.cpp @ v0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/reduce.cpp):
  row small path (:431-486, :576-601), strided small (:604-668, :957-986), eval dispatch (:989+).
- `results/runs/2026-11-01-breakdown/tables.md:96-103,133-140,152-159,207-214,333-335`
  (CPU µs, op inventory, serialized costs, knock-outs).
- `claude/findings/step-breakdown.md` (§3 knock-outs, §5 repeated work) and
  `claude/findings/bottleneck-explained.md` (§3 fixed cost).
- `muse-spark/exploration/ineff-runs-2.md`, `ineff-model-1.md`, `ineff-model-2.md`,
  `ineff-runs-3.md` (adaptation: causes already covered → appended notes instead).

## From other agents

(none yet)
