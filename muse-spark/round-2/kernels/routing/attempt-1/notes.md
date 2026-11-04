# Round-2 attempt 1 — builder-2-a, direction (a): faithful MLX-regime replica

One fused Metal kernel, 256 threads (8 simdgroups) per row, branching on
`rows` in-Metal. Baseline: none. Builds on the starting point
`muse-spark/kernels/routing/attempt-1` (kept: bf16->f32 cast, strictly-greater
insertion top-4 with earliest-index tie win, `((w0+w1)+w2)+w3 + 1e-20` then
`/ denom * 2.5` on unbiased scores).

## What changed since the starting point

- Product: serial 768-term chain replaced by the MLX regime for `rows`:
  rows==1 GEMV lane order (`d=i*128+lane*4+0..3`, i=0..5) + shuffle tree
  16,8,4,2,1; 1<rows<=768 split-K (BK=16 chunks, partitions from
  `next_pow2(48/((rows+31)/32*2))` clamped to [2,32], base `48/partitions`
  chunks, last partition to chunk 48, partials added in partition order);
  rows>768 BK=16 chunks in order. MMA regimes use `simdgroup_float8x8`
  multiply-accumulate, two 8-wide ops per BK=16 chunk; only A-tile row 0 is
  the real row, other rows staged zero, so per-element K order is steel's.
- Sigmoid: `1/(1+exp(-acc))` replaced by MLX verbatim
  `y=1/(1+precise::exp(abs(s))); (s<0)?y:1-y`, float32 throughout.
- Top-4/weights: unchanged from the starting point (see above).

## Inefficiencies removed

`ineff-baseline-routing-1` (three product accumulation orders), `-2`
(sigmoid formula), and the serial-dot part of `-3` (dots now lane- or
MMA-parallel; the serial thread-0 scan is kept deliberately to preserve
the tie rule bit for bit).

## Predicted us per layer at 1 / 8 / 32 rows

Starting point measured 28.5 / 23.6 / 41.8 us (its notes.md) against a
~24 us chained-harness floor; traffic per row is 184 KB (0.9 us at
200 GB/s) and body work is 92k FLOP/row. This kernel keeps one launch,
widens to 8 simdgroups/row (better 1-row occupancy) but the MMA path
stages 48 chunks x 2 barriers and wastes 7/8 of each 8x8 tile at decode
sizes, so I predict it lands near the starting point, not below the floor:
1 row ~ 24 floor + 0.9 traffic + ~2 MMA staging ~= **27 us**;
8 rows ~ 24 amortised + 7.4 traffic ~= **26 us**;
32 rows ~ 18 amortised + 29.5 traffic ~= **40 us**.
Compiled reference (starting point's run): 30.5 / 42.0 / 49.1 us, so a
chained win is plausible only if G2 passes.

## Known limits (stated before grading)

- MMA tile is 8 rows x 8 experts, not steel's BM=16/32 x BN=32 (split-K)
  or 64x64 (regular). Per-element K/chunk/partition order matches; if the
  hardware 8-product rounding depended on tile position it would still
  differ — no evidence it does, but this is the residual risk.
- GEMV per-lane products use the same `acc += x*w` form as MLX gemv.h;
  compiler FMA contraction is assumed to match MLX's build.
- The thread-0 top-4 scan remains serial (60 steps) by design for ties.

## Verdict (2026-11-04T03:37:08Z, one grader run)

G1 pass, **G2 pass at all six sizes** (rows_differ 0, max_rel_err 0.0 at
1/8/32 and 127/1016/2048). Chained us/layer (ref / compiled / candidate):
1 row 77.8 / 56.2 / **54.9** (beats compiled); 8 rows 113.1 / 91.2 / 91.4
(tie, +0.2); 32 rows 136.3 / 98.9 / 114.8 (loses). `beats_compiled` false.
Baseline none, so no `beats_baseline`.

The pre-grading worry that the 8-row MMA tile might round differently
from steel's larger tiles did not bite: hardware 8-product rounding is
tile-position independent on this GPU, and the GEMV shuffle replica plus
verbatim sigmoid are bit-exact. The 8x8-per-row emulation is therefore
sufficient for G2; no scalar fallback was needed.

## Why timings missed predictions by >20%

Predicted 27/26/40 us, measured 54.9/91.4/114.8 us. Two causes. (1) The
prediction anchored on round 1's independent-timing floor (~24 us); the
chained harness floor is higher (compiled itself is 56.2 us at 1 row
here vs 30.5 us in round 1's run), so the anchor was wrong, not just the
kernel. (2) The MMA path pays 48 chunks x 2 threadgroup barriers per row
plus 7/8 wasted tile rows and a full 184 KB w_router re-read per row:
at 32 rows that staging dominates and the candidate loses to compiled
by 16 us, while at 1 row the single launch still wins by 1.3 us. Next
change: keep the exact orders but stage BK=32+ per barrier (or load
fragments without per-chunk threadgroup staging) and put real rows in
all 8 A-tile rows so w tiles are shared across rows.
