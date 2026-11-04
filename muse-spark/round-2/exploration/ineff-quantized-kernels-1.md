# Inefficiency: `gather_qmm_rhs` BM=16 row tiles dispatch empty tiles and compute zero-padded rows at prefill

- **Owner:** explorer-quantized-kernels
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:25:27Z
- **Next:** verifier to confirm grid arithmetic and padded fractions

## Slot

`experts`

## What happens

MLX's prefill kernel `affine_gather_qmm_rhs`
(`quantized.h:2570-2718`, installed 0.32.4 headers) tiles sorted
pairs into `BM=16`-row tiles. The host over-provisions the grid:
`grid_y = min(B, ceil(B/16) + E - 1)` with `E = 60`
([quantized.cpp v0.32.4:1817-1819](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1817-L1819)),
i.e. up to 59 tiles beyond the all-in-one-expert minimum, because
any expert may contribute a partial tile. Each threadgroup then
runs `schedule_row_tile<16>` (`utils.h:501-542`) to find its
expert; threadgroups past the real tile count return false and
exit (`quantized.h:2637-2640`) — launched, scheduled, discarded.

Useful tiles are `T = Σ_e ceil(cnt_e/16)` over per-expert pair
counts `cnt_e`. Tiles that are real but partial
(`tgp_bm < 16`, `quantized.h:2636`) still run the full tile:
`BlockLoader::load_safe` zero-pads the missing `x` rows
(`steel/gemm/loader.h:83-128`), `BlockMMA::mma`
(`steel/gemm/mma.h:513-537`) multiplies them, and
`store_result_safe` discards them (`quantized.h:2687-2695`).
The weight tile is fully loaded and dequantized regardless
(`QuantizedBlockLoader`, `quantized.h:665-791`), so a partial
tile costs a full tile of load + MMA for a fraction of the rows.

This is distinct from the G2 failure itself
(`ineff-baseline-experts-1`): even a bit-exact replica of this
kernel inherits the waste; a replica that regroups rows
(different `BM`, or per-row accumulation with the same per-row
arithmetic — territory §C invariant) avoids it without
changing any valid output bit.

## Cost

Decode 1/8/32 rows: **0 kernels, 0 bytes** — this kernel never
runs there (`B/E ≤ 2.13 < 4`, territory §A table). All figures
below are per product per layer at the prefill sizes, balanced-expert
model (`cnt_e` near the uniform mean `B/60`; grader prompts are
random tokens, territory §C). `N` tiles: gate/up 12, down 24
(`BN=32`).

| prefill rows | B | dispatched `grid_y` | useful `T` | empty y-tiles | padded rows `T·16−B` |
|---|---|---|---|---|---|
| 127 | 508 | 91 | 60 | 31 (34.1%) | 452 / 960 (47.1%) |
| 1,016 | 4,064 | 313 | 300 | 13 (4.2%) | 736 / 4,800 (15.3%) |
| 2,048 | 8,192 | 571 | 540 | 31 (5.4%) | 448 / 8,640 (5.2%) |

Arithmetic: `grid_y` from the host formula; `T = 60·ceil((B/60)/16)`
in the balanced case (`ceil(8.47/16)=1`, `ceil(67.7/16)=5`,
`ceil(136.5/16)=9`). Empty threadgroups per gate/up call =
empty y-tiles × 12 (372 / 156 / 372), per down call × 24
(744 / 312 / 744), each 64 threads that only run the
`schedule_row_tile` scan over 60 experts (`utils.h:514-541`:
two 32-lane prefix iterations) and exit. Padded-row MMA work:
the kernel issues `2·T·16·K·N` flops for `2·B·K·N` useful
(`K·N = 294,912` for both products), so 47.1% / 15.3% / 5.2%
of issued row-work multiplies zero-padded rows. Weight bytes
are *not* wasted by padding (the tile's weights serve its valid
rows); the waste is issued MMA/load slots and threadgroup
occupancy. No measured prefill time exists: the grader times
decode only (`claude/adr/0010-exact-at-every-size.md:42`), so
no µs figure is claimed from measurement.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): at prefill the path is far
below both peaks, and the padded fraction is pure overhead —
issued work with no output. The empty-tile part is launch/
scheduling overhead; the padded-row part is wasted compute
inside otherwise useful threadgroups.

## Spans

`quantized-kernels` (host grid formula + `schedule_row_tile` +
`BlockMMA` partial-tile path) → slot `experts`. Neighbour:
`baseline-experts` (any grouped prefill branch of a candidate
chooses its own tiling; this entry prices MLX's).

## Predicted saving

0 µs per decode step at 1, 8 and 32 rows (kernel never runs
there). At prefill, removing both parts saves the padded
fraction of the main kernel's row work — up to ~47% at
127 rows, ~15% at 1,016, ~5% at 2,048 — plus the empty
threadgroups above, in time-to-first-token only. Prefill
speed is not gated (G3 times decode;
`claude/adr/0010-exact-at-every-size.md:42`), so this saving
is real but unbanked by the grader; it is worth taking only
if it comes free with the exact prefill branch (e.g. a
per-row `qmm_rhs`-arithmetic kernel has no tiles to pad).
In absolute terms the main kernel's useful MMA floors at
`2·B·K·N / 5.2 TFLOPS` ≈ 57.6 / 461.0 / 929.2 µs per
layer-product at peak (territory §E); issued MMA including
padding floors at 108.9 / 544.6 / 980.0 µs, so the padded
part alone floors at ≈51.3 / 83.5 / 50.8 µs per
layer-product *if* the kernel ran at peak compute — it
does not (two barriers per 32-K tile,
`quantized_utils.h:14-29`), so treat these as lower bounds
on the waste, not predictions.

## Verdict

verifier, 2026-11-04T03:25:27Z — **confirmed**. Checked: `affine_gather_qmm_rhs` BM=16 tiling ([quantized.h:2570-2718](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:2570), template at :2570, kernel at :2580, close at :2718) with host grid `grid_dims=((N+31)/32, min(B, ceil(B/16)+E-1), 1)`, E=60 (MLX v0.32.4 `quantized.cpp:1817-1819`, fetched; `bm=16,bn=32,bk=32` at :1748) ✓; `schedule_row_tile<16>` ([utils.h:501-542](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/utils.h:501)) returns false → whole threadgroup exits ([quantized.h:2637-2640](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:2637)) ✓; partial tiles (`tgp_bm<16` at :2636) zero-pad via `BlockLoader::load_safe` ([loader.h:83-128](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/loader.h:83)), multiply via `BlockMMA::mma` ([mma.h:513-537](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/mma.h:513)), discard via `store_result_safe` ([quantized.h:2687-2695](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:2687)); weight tile fully loaded via `QuantizedBlockLoader` ([quantized.h:665-791](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h:665)) ✓. Distinct from the G2 failure ([ineff-baseline-experts-1](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-experts-1.md)); regrouping is free per [territory-quantized-kernels.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-quantized-kernels.md) §C invariant ✓. Arithmetic (balanced model, B=508/4,064/8,192): grid_y 91/313/571, useful T=60·ceil((B/60)/16)=60/300/540, empty 31 (34.1%)/13 (4.2%)/31 (5.4%), padded T·16−B=452/960=47.1%, 736/4,800=15.3%, 448/8,640=5.2% ✓; empty threadgroups ×12 gate/up (372/156/372) and ×24 down (744/312/744), 64 threads each ✓; K·N=294,912 both products, padded MMA fractions as above ✓. Decode 0 kernels/0 bytes (B/E≤2.13<4) ✓; no prefill timing exists (decode only, [0010-exact-at-every-size.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md)) ✓. Floors 2·B·K·N/5.2 TFLOPS = 57.6/461.0/929.2 µs useful, 108.9/544.6/980.0 µs issued, padded ≈51.3/83.5/50.8 µs per layer-product at peak (lower bounds; two barriers per K tile, [quantized_utils.h](/Users/onur/Desktop/Github/Hackathon-muse/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized_utils.h)) ✓. Predicted saving 0 µs decode; prefill fractions in time-to-first-token only, unbanked, as written. Balanced-expert model is labelled expected case, not a bound — actual `cnt_e` on grader prompts will shift T by a few tiles without changing the mechanism or the decode-zero claim.

## From other agents

(none yet)
