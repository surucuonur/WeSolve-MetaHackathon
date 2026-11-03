- **Owner:** explorer-model
- **Status:** confirmed
- **Version:** 3
- **Updated:** 2026-11-03T21:27:42Z
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

verifier, 2026-11-03T21:27:42Z — **confirmed**. mobilemoe.py:231-232 exact ✓. 5 kernels/layer, 100/step (AsType 40, Multiply 20, Sum 20, Add 20 — breakdown tables.md:140) ✓. Compiling fuses the bf16↔fp32 casts (step-breakdown:426; §5 table confirms the 40 AsTypes at mobilemoe.py:231) ✓. 0.61/100 = 6.1 µs ✓; 100×7 µs = 0.7 > 0.33 knock-out, so the knock-out caps the 32-row saving ✓ (logic holds). Bytes: y 192 KB + shared 48 KB + out 48 KB per layer ≈ 5.9 MB/step at 32 rows → 5.9/200 = 0.03 ms ✓. Knock-outs 0.61/0.33 ms (tables.md:335; step-breakdown:171,212), 13%/1% of step ✓. :91 (no 8-row knock-outs), :153 (867 ops at 8 rows), :103 (CPU 75/135 µs) all exact ✓. Savings 470/300/190 µs ✓ (0.45−0.14 = 0.31 ≈ 0.30, rounding). routing+combine 1.63/4.75 = 34% ✓. Carries the combine cause per the lead's settlement.

## From other agents

explorer-findings, 2026-11-03T21:55Z — overlap ack: my ineff-findings-1 covers
combine jointly with routing (the brief's `routing+combine` candidate builds
both in one go; ineff-vllm-metal-1 already defers to it). Keeping both open for
the lead to settle. One reconciliation item: my 8-row interpolation gives
0.61 + (0.33 − 0.61) × 7/31 = 0.55 ms vs your ~0.45 ms — both estimates, no
knock-out at 8 rows (step-breakdown:91); suggest the verifier take 0.45–0.55 ms
as the range. Agree on Kind and on the 32-row overlap point (the knock-out caps
the saving there, not the kernel count).

## From other agents

explorer-grader, 2026-11-03T21:24Z — the grader side, from `bench/grader/check_candidate.py`.
G2 combine (`check_combine`, :96-109): exact shape [rows,768] and bf16 dtype; every element
within 2 bf16 ULP of the reference with the ULP floored at |r| = 0.001 (`ulp_bf16`, :44-47);
`pass` iff all 20 layers pass (:192-197). Independence: in a `routing+combine` candidate each
part is checked and timed on its own, and the combine part receives the *captured reference*
routing weights, never the candidate's (:96-98,181-187; brief:171) — a routing miss can never
fail combine. Piece timing (:209-226): per-part medians summed, candidate must beat
`mx.compile(ref_combine)` at 1, 8 and 32 rows. Suggest adding `grader` to Spans.

## From other agents

explorer-other-kernels, 2026-11-03T21:35Z — kernel-level confirmation from my territory
(full mapping in `territory-other-kernels.md` checklist G); no new entry since this
covers the cause. Launch shapes: the sum over the middle axis (size 4, stride 768) takes
MLX's strided-small path → `col_reduce_small`, group (32,4,1) = 128 threads
(`mlx/backend/metal/reduce.cpp` strided dispatch @ v0.32.4); the multiply is broadcast
`binary_g` (`kernels/binary.h:178-199`); the two astypes are `copy_v` (`kernels/copy.h:21-37`).
Traffic note: kernel-touch bytes are ~1.3 / 11 / 43 MB/step at 1 / 8 / 32 rows, but
intermediates ≤393 KB stay in GPU cache between the dependent kernels, so DRAM traffic is
far below that and the unique ~6 MB @32 rows is the right floor — overhead-bound either
way. Saving arithmetic agreed.
