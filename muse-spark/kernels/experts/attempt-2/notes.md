# attempt-2: fused per-pair experts (direction b)

First in direction (b): fusion first, keeping per-pair reads. One Metal kernel,
one 128-thread threadgroup per (row, expert-slot) pair. Gate/up share one
i-loop over an on-chip x row; gate/up/hidden intermediates stay in threadgroup
memory (1920 bf16 = 3840 B, under the 32 KB limit); each expert's weights are
read once per pair with no cross-pair batching and no sort.

Removes `ineff-switch-layers-2` (unfused products + activation): 4 on-chain
kernels/layer become 1, and gate/up/hidden intermediates never touch DRAM. It
does NOT remove `ineff-quantized-kernels-1` (per-pair re-reads of weights) or
`ineff-runs-1` (465 MB excess) — those are directions (a)/(c).

Boundary kept: fp32 accumulation, bf16 rounding after each product and after
silu-mul, `y[r,j]` positional, any `inds` order, valid for all 1–32 rows
(grid is `rows*512` threads = `rows*4` pairs × 128, no rows-dependent indexing).

## Predicted µs per layer at 1 / 8 / 32 rows

Per-pair weight bytes (exact): each of gate/up/down is 147,456 B of `_w` +
18,432 B scales + 18,432 B biases = 184,320 B; ×3 = 552,960 B ≈ 553 KB
(agrees with ineff-quantized-kernels-1's 540 KB model). Pairs/layer: 4/32/128
→ 2.21 / 17.7 / 70.8 MB per layer. Rate basis: ~100 GB/s (below the 150 GB/s
streaming ideal — one thread's consecutive-i reads stride 48/96 words, so
coalescing is imperfect; but each word is read once per pair and x is on-chip,
unlike the reference's 12× x re-reads and strided down re-reads).

- 1 row: 2.21 MB / 100 GB/s = 22 µs + one launch ≈ 7 µs → **30 µs**
  (original 46 µs/layer = 0.92 ms ÷ 20, brief:92).
- 8 rows: 17.7 MB / 100 = 177 µs → **185 µs**
  (baseline ≈254 µs/layer = 5.08 ms ÷ 20, ineff-quantized-kernels-1 micro).
- 32 rows: 70.8 MB × ~0.65 (128 pairs share ~20–60 distinct experts/layer, so
  part of the re-reads hit cache) = 46 MB / 100 = 460 µs → **470 µs**
  (original 733 µs/layer = 14.67 ms ÷ 20, brief:92).

Plus the 3 saved launches/layer (3 × 6.81 µs = 20 µs, ineff-switch-layers-2)
are inside these numbers via the single-launch timing. beats_compiled:
predicted true at 1 and 8 rows; 32 rows a toss-up since per-pair bytes match
the reference.

## v2: bit-faithful numerics (same attempt folder)

v1 graded G1 pass / G2 fail: ~7% of elements over 2 ULP (max 11/32/68 ULP at
1/8/32 rows), all on near-zero down outputs. Cause: v1 accumulated each dot
serially, while the reference (`affine_gather_qvm`, quantized.h qvm_impl)
accumulates 32 strided per-lane partials and reduces with a `simd_sum` tree;
down outputs near cancellation amplify even 1-ULP hidden flips past the 2-ULP
floor, so gate/up/hidden must match bit-exactly. v2 replicates per output:
lane l sums k=l,l+32,... with qouter's exact op forms (`s/16` times the
unshifted high nibble for odd outputs), then a real `simd_sum` over the same
lane mapping; silu uses MLX's Sigmoid lowering
(`1/(1+precise::exp(abs(x)))`, branch on sign) in fp32, since the struct
cannot instantiate for bfloat. Design unchanged: same fusion, same per-pair
reads. Predictions unchanged: 30 / 185 / 470 µs per layer.

v2 graded G1 pass / G2 fail with IDENTICAL failures to v1 (same bad counts,
same worst values to 7 digits) — proving the error is systematic, not order.
CPU replication (/tmp/cpu_pair.py, safetensors read directly, no GPU) matched
the kernel bit-for-bit and showed all fp32 summation orders agree to ~1e-9
while the reference differs by ~7.7e-5: the down TERMS differ, i.e. hidden.
Gate/up proven order-insensitive (0 bf16 flips across serial/reverse/tree).
A search over all 64 bf16-rounding placements in silu (/tmp/cpu_silu.py) found
only full-bf16 (round after exp/add/recip/sub/mul/mul) matches: bad=0 on both
probe pairs. So MLX's compiled swiglu computes entirely in bf16.

## v3: bf16 silu (same attempt folder)

v2 + per-op bf16 rounding in silu (exp/add/recip/sub/mul/mul). Design,
predictions unchanged: 30 / 185 / 470 µs per layer.

## Verdict

G1 pass; G2 pass with max_ulp 0.0 at all row counts (bit-exact vs reference);
beats_compiled true. Piece timing µs/layer (candidate vs original vs
compiled): rows 1: 107.3 / 99.8 / 135.0; rows 8: 188.4 / 218.7 / 216.9;
rows 32: 647.6 / 768.6 / 761.7. Beats the compiled reference at all rows
(~20/13/15%) and the eager original at 8/32 rows (~14/16%), but REGRESSES
7.5% vs eager at 1 row.

## Miss analysis (prediction 30 / 185 / 470)

- Rows 1 (+257%): wrong regime. 4 pairs = 512 threads is occupancy-starved
  (2.2 MB in 107 µs = 20 GB/s effective), not streaming at 100 GB/s. Fusion
  saves 3 launches but one threadgroup per pair parallelizes worse than the
  reference's 96 small groups here. Lesson: direction (b) cannot win at 1
  row; needs the grouped designs' occupancy.
- Rows 8 (+2%): 17.7 MB in 188 µs = 94 GB/s. Model holds.
- Rows 32 (+38%): streams at 109 GB/s (better than assumed) but with NO
  cross-pair cache reuse — effective bytes are the full 70.8 MB/layer, not
  the assumed 0.65×46 MB. 128 pairs × 553 KB = 70 MB >> GPU cache, and the
  20 sequential layer evals evict; distinct experts/layer are too many for
  reuse. Lesson: per-pair re-reads cost full DRAM bandwidth; only batching
  same-expert pairs (directions a/c, `ineff-runs-1`) removes them.
