# Inefficiency: sort glue costs 6 kernels/layer from 16 rows, for locality only

- **Owner:** explorer-switch-layers
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:27:42Z
- **Next:** verifier to confirm kernel inventory and the B/E>=4 gate

## Slot

`experts`

## What happens

When `indices.size >= 64` (`switch_layers.py:181`), i.e. 16+ decode rows,
`SwitchGLU` sorts the 4·rows pairs by expert (`_gather_sort`, `:12-17`) and
un-sorts afterwards (`_scatter_unsort`, `:20-24`). Per layer that is 2 ArgSort
(`order`, and a second full sort for `inv_order`), 1 FloorDivide (`order // M`),
and 3 Gathers (sorted x, sorted idx, unsort) — 6 kernels/layer, 120/step
(`results/runs/2026-11-01-breakdown/tables.md:136`: Gather 60, ArgSort 40,
FloorDivide 20; the 867→987 op jump at 32 rows).

The sort cannot change MLX's kernel choice in decode: the grouped path needs
`B/E >= 4` (`quantized.cpp:2012`, v0.32.4), i.e. 240+ pairs at E=60 = 60+ rows,
so every decode row count runs the per-pair `gather_qvm` kernel regardless
(see `territory-switch-layers.md` §G). Its only function in serving is cache
locality within that kernel (sorted serving reads 908 MB vs the 1416 MB
per-pair model — caches serve a third).

## Cost

Per decode step (plain MLX, 128-token prompts):

- 1 row: no sort runs (4 pairs < 64) — 0 kernels, 0 ms.
- 8 rows: no sort runs (32 pairs < 64) — 0 kernels, 0 ms.
- 32 rows: 120 kernels/step, ~5 of the 6 per layer on the critical chain
  (`inv_order`'s argsort overlaps the products). 5 × 6.81 µs × 20 layers
  = 0.68 ms ≈ 0.7 ms (fixed-cost unit from
  `results/runs/2026-10-30-micro/step_parts.txt`). Index/activation traffic is
  ~8 MB/step ≈ 0.04 ms at peak — negligible. No isolated knock-out; contained
  in the experts 14.67 ms (`step-breakdown.md:211`). CPU record for the experts
  glue rises 128 → 399 µs/step from 1 to 32 rows (`tables.md:99`).

## Kind

Overhead-bound: ~0.7 ms of fixed kernel costs moving ~8 MB.

## Spans

`switch-layers`, `quantized-kernels` (kernel fixed costs), `runs` (inventory).

## Predicted saving

Standalone removal would *lose* time (unsorted pairs lose the locality that
keeps a third of repeat reads in cache — size unmeasured in serving). The
saving is realizable only inside a fused grouped `experts` kernel that needs
no sort: then the full ~0.7 ms at 32 rows plus the 8 MB traffic go, jointly
with `ineff-runs-1` and `ineff-switch-layers-2` (one fix; savings do not add):

- 1 row: 0.0 ms (no sort runs).
- 8 rows: 0.0 ms (no sort runs).
- 32 rows: ≈ 0.7 ms (100 on-chain kernels × 6.81 µs).

## Verdict

verifier, 2026-11-03T21:27:42Z — **confirmed**. switch_layers.py:181 (`do_sort = indices.size >= 64`) ✓; :12-17 (order + inv_order argsorts, `order // M`) and :20-24 (unsort gather) = 2 ArgSort + 1 FloorDivide + 3 Gather = 6 kernels/layer ✓. Inventory 120/step (tables.md:136: Gather 60, ArgSort 40, FloorDivide 20; 987−867 = 120 ✓). Grouped gate B/E ≥ 4 → 240+ pairs = 60+ rows (decode-bottlenecks §5) ✓, so serving never switches kernels — consistent with the runs-3 refutation. Locality: (1416−908)/1416 = 36% ≈ a third cached ✓. Cost: 4/32 pairs < 64 → no sort at 1/8 rows ✓; 5×6.81×20 = 0.68 ≈ 0.7 ms at 32 rows (6.81 in step_parts.txt:2 ✓; on-chain count of 5 marked judgment) ✓; ~8 MB → 0.04 ms ✓; contained in experts 14.67 (step-breakdown:211 ✓); CPU glue 128→399 µs (tables.md:99 ✓). Saving ≈ 0.7 (100×6.81 µs ✓), joint with runs-1/switch-2, does not add ✓ (stated).
