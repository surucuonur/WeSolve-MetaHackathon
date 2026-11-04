# Inefficiency: `w_router` is re-read once per row instead of once per layer

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:24:18Z
- **Next:** verifier to weigh the attempt-2 counter-evidence before confirming any saving

## Slot

`routing`

## What happens

Each row's threadgroup streams the full router weight for its own experts
straight from device memory (`muse-spark/kernels/routing/attempt-1/block-0.metal:30-34`):
60×768 float32 = 184 KB per row per layer, with no sharing across rows. The
reference's GEMM reads `w_router` once per layer and reuses each tile across
all rows in the tile (steel GEMM/GEMV dispatches, territory section 3.1).
Per layer the starting point therefore reads rows×184 KB where the reference
reads 184 KB: 0.18 MB at 1 row, 1.47 MB at 8 rows, 5.90 MB at 32 rows (and
377 MB at the 2,048-row prefill size, where only correctness is graded —
ADR 0010:42 — so that figure costs grader time, not step time).

## Cost

Traffic floor at the 200 GB/s peak (brief:9): candidate 0.9 / 7.4 / 29.5 µs
per layer at 1/8/32 rows vs the reference's 0.9 µs at every row count. At 32
rows the excess (5.90−0.18 MB ≈ 5.7 MB/layer ≈ 28.6 µs at peak, 0.57 ms/step
over 20 layers) is ~43% of the candidate's measured 68.9 µs/layer
(independent timing, `claude/findings/muse-round-1.md:47`) — the one place
in this slot where bytes, not launches, are a large share. At 1 and 8 rows
the excess is 0 and ~6.4 µs/layer respectively.

## Kind

Bandwidth-bound at 32 rows only; overhead-bound at 1/8 rows (the slot's
usual regime, `ineff-runs-2`).

## Spans

`baseline-routing`

## Predicted saving

Upper bound if each weight byte were read once per layer at peak bandwidth:
**~0 ms/step at 1 row, ~0.13 ms/step at 8 rows (6.4 µs/layer), ~0.57 ms/step
at 32 rows (28.6 µs/layer)**. Confidence is low at decode sizes, on measured
counter-evidence from round 1: attempt-2 tiled `w_router` through
threadgroup memory shared across 8 rows precisely to remove this re-read
and was *slower* at every row count (36.3/34.9/49.0 vs attempt-1's
28.5/23.6/41.8 µs/layer, `muse-spark/kernels/routing/attempt-2/notes.md`) —
barriers and sub-peak strided reads ate the traffic saving. A builder
claiming this saving needs a design that shares weights without
attempt-2's barrier structure (e.g. one threadgroup spanning several rows
in the steel-GEMM shape the exactness fix already pushes toward), and must
prove it on the chained timing, not the traffic arithmetic.

## Verdict

verifier, 2026-11-04T03:24:18Z — **confirmed** as a byte cost with an upper-bound-only saving. Checked: each row's threadgroup streams the full 60×768 f32 router (184,320 B) from device memory with no cross-row sharing ([block-0.metal:30-34](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-1/block-0.metal:30)) ✓; per layer candidate reads rows×184 KB = 0.18/1.47/5.90 MB at 1/8/32 rows (377 MB at 2,048-row prefill, untimed per [0010-exact-at-every-size.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md)) vs reference 184 KB once per layer (steel tile reuse, [territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-routing.md) §3.1) ✓. Floors at 200 GB/s: candidate 0.9/7.4/29.5 µs/layer vs reference 0.9; excess at 32 rows (5.90−0.18)=5.71 MB/layer = 28.6 µs/layer = 0.57 ms/step, at 8 rows ~6.4 µs/layer ✓. Upper bounds ~0 / 0.13 (6.4×20) / 0.57 ms/step (28.6×20) ✓. Counter-evidence correctly carried: attempt-2 tiled `w_router` across 8 rows to remove this re-read and was slower at every row count (36.3/34.9/49.0 vs 28.5/23.6/41.8 µs/layer, [attempt-2/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-2/notes.md)) — barriers/sub-peak reads ate the saving, so no saving is banked without a barrier-free sharing design proven on the chained timing, as written ✓. **Corrections:** excess share at 32 rows is 28.6/68.9≈41.5% (≈43% only on the full 29.5 µs floor), and in [GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md) terms the slot remains overhead-bound at 32 rows with a bandwidth-waste component (5.90 MB in 68.9 µs ≈86 GB/s, 43% of peak), not bandwidth-bound. Neither changes the floor arithmetic.

## From other agents

(none yet)
