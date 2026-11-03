- **Owner:** explorer-model
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:22Z
- **Next:** verifier to confirm against tables.md and the brief

## Slot

`combine`.

## What happens

Each layer's weighting (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:231-232`) launches
5 kernels: AsType (bf16 y→fp32 for the multiply), Multiply, Sum, AsType (back to bf16),
Add (`+ shared`) — 100 per step (`results/runs/2026-11-01-breakdown/tables.md:140`:
AsType 40, Multiply 20, Sum 20, Add 20). The two casts and the separate add exist only
because each op is its own kernel; the bf16↔fp32 round-trips are fused away by compiling
the block (`claude/findings/step-breakdown.md:426`).

## Cost

- Kernels × fixed cost: 100 × ~6.1 µs @1 row (0.61 ms ÷ 100, measured) ≈ 0.61 ms.
  @32 rows the knock-out (0.33 ms) is below 100 × 7 µs = 0.7 ms, so at 32 rows the chain
  partly overlaps other work — the knock-out, not the kernel count, caps the saving there.
- Bytes ÷ peak: activations only — per layer y [32,4,768] bf16 (196 KB) + shared [32,768]
  bf16 (49 KB) + out 49 KB ≈ 0.3 MB/layer, ~5.9 MB/step @32 rows → 0.03 ms at peak.
- Measured (knock-out, plain MLX, `tables.md:335`, `step-breakdown.md:171,212`):
  0.61 ms @1 row (13% of step), 0.33 ms @32 rows (1% of step).
  @8 rows not measured (`step-breakdown.md:91`); kernel inventory identical to 1 row
  (`tables.md:153`), cost estimated ~0.45 ms by interpolation.
- CPU record: 75 µs @1, 135 µs @32 per step (`tables.md:103`).

## Kind

Overhead-bound: pure chain of tiny elementwise kernels moving ~6 MB @32 rows.

## Spans

`model` (the chain) → slot `combine`. Neighbours: `switch-layers` (y producer),
shared expert (E in territory-model, computed outside slots, consumed by the Add).

## Predicted saving

One fused kernel per layer (20 × 7 µs ≈ 0.14 ms residual). Saving per decode step:
0.61 − 0.14 ≈ **470 µs @1 row**; ~0.45 − 0.14 ≈ **300 µs @8 rows** (estimate);
0.33 − 0.14 ≈ **190 µs @32 rows**. Together with ineff-model-1, routing+combine are
1.63 ms of the 4.75 ms step @1 row (34%).

## Verdict

(verifier writes)
