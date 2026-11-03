# Territory: quantized-kernels

- **Owner:** explorer-quantized-kernels
- **Status:** done
- **Version:** 3
- **Updated:** 2026-11-03T22:05Z
- **Next:** none (territory mapped; 1 inefficiency open, 1 superseded)

## What it does

This territory is the `affine_gather_*` Metal kernels behind `mx.gather_qmm`
(`quantized.h`, `quantized_utils.h`). In MobileMoE-S decode they run the routed
experts' gate/up/down products: one `gather_qmm` each per layer per step, always
dispatched to `affine_gather_qvm` (proof below). 60 `GatherQMM` launches per step
at every row count (`results/runs/2026-11-01-breakdown/tables.md:152-154`).

## The checklist

### A. `affine_gather_qvm` + `qvm_impl` — the kernel decode actually runs

- **Callers:** `GatherQMM::eval_gpu` in
  `github.com/ml-explore/mlx`, tag `v0.32.4`,
  `mlx/backend/metal/quantized.cpp:1980-2094`. MobileMoE calls
  `mx.gather_qmm(..., transpose=False, group_size=32, bits=4, mode="affine")`
  with `x` `[pairs,1,K]` so `M=1` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:172-184`,
  `switch_layers.py:176-199`). Dispatch: `M==1 < vector_limit==4`
  (`quantized.cpp:2005,2034`) rules out `gather_qmm`; the rhs fast path needs
  `M==1 && B>=16 && right_sorted_ && B/E>=4` (`quantized.cpp:2012`) but
  `B/E = rows*4/60 <= 2.1` at `rows<=32` (needs `rows>=60`), so it never fires;
  `transpose_==false` selects `gather_qvm` (`quantized.cpp:2073-2093`).
  Shapes at 1/8/32 rows: gate/up `M=1,N=384,K=768,B=rows*4`; down
  `M=1,N=768,K=384,B=rows*4`.
- **Callees:** `gather_qvm` launches `affine_gather_qvm` (`quantized.h:2355-2416`,
  via `quantized.cpp:1461-1529`) with `group_dims=(32,2,1)=64` threads,
  `bn = min(group_size,32)*2 = 64`, grid `(M,(N+63)/64,B)`: gate/up
  `(1,6,B)`, down `(1,12,B)`. Each threadgroup runs `qvm_impl`
  (`quantized.h:1222-1330`): 2 simdgroups x 32 lanes; each lane owns one K
  element per iteration and decodes 32 outputs via `qouter`
  (`quantized.h:452-553`, bits=4 path `:503-509`).
- **Data:** `w` uint32 `[E,in,out/8]` (4-bit, 8 per word along out),
  `scales`/`biases` bfloat16 `[E,in,out/32]`, `x` bfloat16 `[B,1,K]`,
  `rhs_indices` uint32 `[B]`, `y` bfloat16 `[B,1,N]`; per-pair offsets via
  `adjust_matrix_offsets` (`quantized.h:1636-1688`). Weights/scales read
  exactly once per pair: gate pair traffic = 6 tiles x 64 lanes x 24 iters x
  16 B weights = 144 KB = `768*384/2` B, scales+biases 36 KB; `x` row re-read
  12x per pair per product (each of 6 tiles' 2 simdgroups streams the full row:
  `qvm_impl:1275-1287`), ~55 KB x per pair over gate+up+down vs 3.75 KB needed.
- **Neighbours:** before: router `inds` (model territory) + `_gather_sort` at
  `indices.size>=64` (switch-layers; sort buys only L2 locality since rhs never
  fires). Between gate/up and down: activation (other-kernels). After: combine
  (model). Replacement must keep G2: every element within 2 bfloat16 ulp of the
  reference (1e-3 floor), float32 accumulation, bfloat16 rounding points.
- **Cost:** 60 launches/step (~6 us each: routed experts 140 kernels/0.92 ms @1,
  `claude/findings/step-breakdown.md:179-185`). Bytes (GPU counters,
  `results/runs/2026-11-02-gpu-counters/tables.md:17-18`): experts saves
  51 MB read in 0.92 ms @1 (55 GB/s), 908 MB in 14.67 ms @32 (62 GB/s, 31% of
  peak). Grids @1 row are 24/24/48 threadgroups (~1.5-3K threads, ~5-10% of
  slots; whole-step Compute Occupancy 9%, same table).
- **Invariants:** `w = scale*q + bias` per group of 32 along out; output column
  `o` in word `o/8` bits `4*(o%8)` (brief); float32 accumulate, bfloat16 store
  (`qvm_impl:1324-1329`); fully K/N-aligned launches (384/768 % 64 == 0,
  768/384 % 32 == 0), no tail waste.

### B. `affine_gather_qmm_n` / `affine_gather_qmm_t` + `qmm_n_impl` / `qmm_t_impl` — never run in decode

- **Callers:** only `GatherQMM::eval_gpu` when `M >= vector_limit`
  (`quantized.cpp:2034-2051`); decode has `M=1`. Documented so no one chases them.
- **Callees:** `affine_gather_qmm_n` (`quantized.h:2501-2568`) /
  `_t` (`:2418-2499`), `BM=BN=BK=32`, `QuantizedBlockLoader`
  (`quantized.h:665-791`) + `BlockMMA`, `gemm_loop_*` (`quantized_utils.h:6-90`).
- **Data / Neighbours / Cost / Invariants:** same dtypes and `scale*q+bias`
  decoding as (A); zero cost in decode.

### C. `affine_gather_qmm_rhs` + `schedule_row_tile` — never run in decode

- **Callers:** only when `M==1 && B>=16 && right_sorted_ && B/E>=4`
  (`quantized.cpp:2012-2032`); needs `rows>=60`, unreachable at `rows<=32`.
  This is the kernel that would batch same-expert pairs (offsets +
  `schedule_row_tile`, `utils.h:501-542`); its absence is why each pair keeps
  its own grid (see ineff-quantized-kernels-1).
- **Callees:** `affine_gather_qmm_rhs` (`quantized.h:2570-2718`), `BM=16,BN=32,
  BK=32`, same loader/MMA helpers as (B).
- **Data / Neighbours / Cost / Invariants:** same as (A); zero cost in decode.
  Note for switch-layers: model sorts at `>=16` rows but dispatch ignores
  sortedness below `B/E>=4`.

### D. `affine_gather_qmv_fast` / `affine_gather_qmv` — never run in decode

- **Callers:** only when `transpose_==true` and `M < vector_limit`
  (`quantized.cpp:2053-2071`); MobileMoE uses `transpose=False`.
- **Callees:** `qmv_fast_impl` (`quantized.h:850-962`) / `qmv_impl` (`:964-1123`)
  via `affine_gather_qmv_fast` (`:2231-2291`) / `affine_gather_qmv` (`:2293-2353`).
- **Data / Neighbours / Cost / Invariants:** non-transposed `[E,out,in]` layout
  only; zero cost in decode.

### E. Shared helpers

- `dequantize` (`quantized.h:557-663`), `qdot`/`qdot_safe` (`:221-450`),
  `load_vector`/`load_vector_safe` (`:28-219`): per-bit paths, bits=4 used.
- `QuantizedBlockLoader` (`:665-791`): dequantize-on-load into threadgroup
  memory; used by (B)/(C), not by (A).
- `adjust_matrix_offsets` gather variant (`:1636-1688`): per-threadgroup index
  lookup, negligible cost.
- `gemm_loop_aligned/unaligned/finalize` (`quantized_utils.h:6-90`): the
  K-loop for (B)/(C) only.

## Relations

- Feeds from: model (router `inds`, `x`), switch-layers (`_gather_sort` order).
- Feeds to: other-kernels (activation reads gate/up outputs), model (combine
  reads `y`).
- Shared expert uses non-gather quantized matmul: other-kernels territory.

## Inefficiencies

- `ineff-quantized-kernels-1` (open): per-pair qvm grids run expert traffic at
  ~1/3 peak; 0.5/2.5/8.6 ms ideal.
- `ineff-quantized-kernels-2` (superseded by `ineff-switch-layers-2`): launch
  fusion already covered there, broader; kernel-side numbers moved to a note.

## Files read

- `.venv/.../mlx/backend/metal/kernels/quantized.h` (all 2896 lines; gather
  kernels :2231-2718, impls :793-1595, helpers :17-791).
- `.venv/.../mlx/backend/metal/kernels/quantized_utils.h` (90 lines, K-loops).
- `.venv/.../mlx/backend/metal/kernels/utils.h:501-542` (`schedule_row_tile`).
- `mlx/backend/metal/quantized.cpp` @ `v0.32.4` via web fetch (dispatch
  :1280-1529, :1676-2094); installed MLX is 0.32.4.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:142-232`,
  `switch_layers.py:12-25,176-199` (call shapes, sort threshold).
- `results/runs/2026-11-01-breakdown/tables.md`,
  `results/runs/2026-11-02-gpu-counters/tables.md:8-18`,
  `results/runs/2026-10-30-micro/moe_parts.txt`,
  `claude/findings/step-breakdown.md:150-270` (costs).
- `muse-spark/exploration/ineff-runs-1.md`, `ineff-runs-3.md`,
  `ineff-model-1.md`, `ineff-model-2.md` (peer overlap check).

## From other agents

(none yet)
