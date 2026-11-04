# Attempt 3: two-kernel grouped experts split (builder-1-c, direction c)

First in direction (c): kernel 1 fuses gate+up+silu-mul with grouped
same-expert reads; kernel 2 runs the down product with grouped reads. One
threadgroup per expert finds its pairs by scanning `inds`, streams each
weight tile once per chunk from DRAM, reuses it across the chunk. A `hidden`
[rows,4,384] bf16 buffer passes between the kernels in DRAM.

## What it removes, and how

- `ineff-quantized-kernels-1`: per-pair qvm grids re-read each expert's
  weights once per pair (908 MB/step at 32 rows for 443 MB distinct). Here
  each appearing expert's weights stream ~once per call (34 MB/layer at 32
  rows vs 71 MB per-pair); tile loads are coalesced.
- `ineff-switch-layers-2`: gate/up/activation fused (one hidden round-trip);
  no sort/unsort glue (`ineff-switch-layers-1`), valid at every 1-32 rows.
- `ineff-runs-1`: the same grouping removes the excess re-reads.

Final structure (v7): 192/384 threads per group, 2 outputs per thread (an
even pair shares one weight word + one scale/bias), chunks of 2 pairs with
scalar accumulators, branchless inner loop (short chunks pad with pair 0's x,
phantom stores skipped), K unrolled x2 with even/odd accumulators.

Threadgroup memory by hand (grader does not check 32 KB): k1 = 12 KB weight
tiles + 3 KB scales/biases + 128 B x tile + ~1 KB pair lists ≈ 16.1 KB; k2
likewise ≈ 16.1 KB. Threadgroups 192/384. All indexing valid for rows 1-32
(P = rows*4 ≤ 128; guarded plist reads); barriers uniform; each (pair,
output) written once, no atomics.

## Predicted us per layer at 1 / 8 / 32 rows

Weight bytes per expert (all three matrices): 552,960 B. Distinct experts
per layer: ~4 / ~20 / ~53 → traffic/layer ≈ 2.23 / 11.2 / 29.9 MB. At the
normal streaming rate 150 GB/s (ineff-quantized-kernels-1): 14.9 / 74.7 /
199 us + 2 launches (~10 us): **25 / 85 / 210 us per layer**.

## Version history (same folder)

- v1: chunk-8 with accumulator arrays, naive silu (round once). G1 pass, G2
  fail (~7% bad, max 11/32/68 ULP).
- v2: per-op bf16 silu rounds (S2). Barely moved (~6% bad) → silu rounding
  placement was not (only) the issue.
- CPU forensics (/tmp/probe_swiglu.py, /tmp/probe_ls.py: float64 replay from
  captured inputs + safetensors weights, zero GPU): reproduced the grader's
  S1/S2 bad counts EXACTLY (4594=4594 etc.), proving dequant/products correct
  and isolating the gap to swiglu internals. `strings` on libmlx.dylib gave
  MLX's Metal Sigmoid: `y = 1/(1+precise::exp(abs(x))); return (x<0)?y:1-y`
  in bfloat with per-op rounds. S17 (stable form + full bf16 rounds) matched
  the captured reference on all 2,519,040 elements, max 1 ULP. (Sibling 1-b
  independently found the same full-bf16 key, attempt-2/notes.md.)
- v3: S17 in Metal. G1+G2 PASS (bad 0, max 1/1/1.5), timing 150/659/2045.
- v4: pair loop outermost (scalar accs) after diagnosing v3's spill
  (dynamic-index accumulator arrays → stack). 74/330/1182.
- v5: chunk-8 explicitly unrolled, guarded. REGRESSED (226/842/1252):
  per-iteration guards + 8x loop bloat cost more than tile sharing saves;
  v4 was bandwidth-bound (71 MB re-reads, L2 does not save them — sibling
  1-b confirms: full 70.8 MB at 32 rows), v5 compute-bound.
- v6: skipped (design merged into v7).
- v7: chunk-2 padded branchless + 2-wide + split-K (final).

## Verdict (v7)

G1 pass; G2 pass with max_ulp 1.0 at all row counts (bad 0).
beats_compiled FALSE (2 of 3 rows). Piece timing us/layer (candidate vs
original vs compiled): rows 1: 68.1 / 70.0 / 64.9; rows 8: 238.4 / 214.7 /
213.1; rows 32: 654.1 / 769.8 / 764.0. Beats original+compiled decisively at
32 rows (~15%); misses compiled by 5% (3.2 us) at 1 row and 12% at 8 rows.

## Miss analysis (prediction 25 / 85 / 210 vs 68 / 238 / 654)

2.7-3.1x over prediction at all rows. Causes:

1. Rate basis wrong (all rows): assumed 150 GB/s streaming; achieved
   ~50-55 GB/s (32 rows: ~34 MB in 654 us). The tiled load→barrier→compute
   loop leaves DRAM idle during compute (duty cycle ~1/3); sibling 1-b's
   tighter per-pair loop reaches ~100 GB/s, still under the 150 ideal.
   Compute is not free either: the dequant+MAC stream costs ~150-200 us at
   32 rows even at full efficiency — the prediction modeled bandwidth only.
2. Fixed costs unmodeled (1 row): 2 launches + 60-group pair scans ≈ 15-20
   us floor; the 3.2 us gap at 1 row is within run-to-run baseline noise
   (compiled rows-1 measured 62-135 across runs).
3. Padding waste (8 rows): pairs/expert ≈ 1.6 but chunks pad to 2 → ~30%
   phantom MACs; small-n underfill. Grouping's DRAM edge needs large
   pairs/expert; at 8 rows it barely outweighs the padding.
4. L2 does not share re-reads across the 20-layer eval (sibling 1-b
   confirms full 70.8 MB/layer for per-pair at 32 rows) — this HELPS the
   grouped design relatively (its 34 MB is real DRAM saved) and is why the
   32-row win is the largest.

## Next change (for hardening, not this direction)

This direction is converged: 1/8-row gaps need a rows-specialized hybrid
(per-pair path at small rows — direction b's territory) or async-copy
double-buffering, both hardening-level changes. Recommended: harden
attempt-2 (per-pair, wins 8 rows at 188 us) with attempt-3's grouping at
large rows (wins 32 rows at 654 us; attempt-2 needs 648 us at 2x bytes).
