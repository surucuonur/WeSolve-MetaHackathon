# attempt-2: size-branched block-split experts (builder-1-a, direction a)

First attempt in direction (a) for `hyp-1-experts-exact`. Builds on the
starting point `muse-spark/kernels/experts/attempt-4` (baseline: none;
attempt-4 fails the current G2 at every prefill size). One spec, two
kernels in list order, each branching on `rows` read from
`inds_shape[0]` (the dispatch threshold `rows >= 60`, pairs >= 240,
from `territory-quantized-kernels` §A/§F, not the tested sizes):

- `experts_gate_up` (block-0): one 32-thread threadgroup per
  (pair, 32-output block) — grid `rows*1536` threads = `rows*4*12`
  groups. Computes gate and up for its block, applies the silu-mul,
  writes `hidden[pair, block*32 + lane]`.
- `experts_down` (block-1): one 32-thread threadgroup per
  (pair, 32-output block) — grid `rows*3072` threads = `rows*4*24`
  groups. Reads `hidden`, writes its 32 outputs of `y`.

Shared header (block-2): `qvm_block` copied byte-identically from
attempt-4, a new `qmm_block`, and attempt-4's full-bfloat16 silu.

## What changed since attempt-4, and which inefficiency it removes

1. **Prefill arithmetic (`ineff-baseline-experts-1`, the G2 gate).**
   Below 60 rows both kernels keep attempt-4's bit-exact `gather_qvm`
   arithmetic (float32 dequant through the product, strided lanes
   `k == lane (mod 32)` + `simd_sum`, full-bf16 silu). At 60 rows and
   above, `qmm_block` replicates `gather_qmm_rhs`: dequantized
   weights rounded to bfloat16 before the multiply
   (`dequantize` bits=4, `s` / `s/16` nibble form), then BK=32 tiles
   in increasing K order, each four sequential 8-wide
   `simdgroup_multiply_accumulate` steps on float fragments — the
   `BlockMMA` order. Row grouping is per-pair (fragment row 0, seven
   zero-padded rows in separate accumulators), which the boundary
   allows; per-row K order is unchanged. Gate/up/down each still
   round once from float32 to bfloat16, and the silu is shared.
2. **Chained 1-row latency (`ineff-baseline-experts-2`).** Attempt-4
   exposes 4 threadgroups per call at 1 row (one per pair) with
   barrier-serialised waves. Here a 1-row call exposes 48 (gate/up)
   + 96 (down) threadgroups, each computing exactly one 32-output
   block with the identical strided-lane + `simd_sum` order, so the
   GPU's 16 cores all have work while no block's bits change. Cost:
   a second launch per layer and one `hidden` round-trip (3 KB per
   pair); `x` is re-read from cache once per block (12x per pair,
   18 KB vs 369 KB of gate/up weights per pair).
3. `ineff-baseline-experts-3` (per-pair weight re-reads) is **not**
   removed: weight bytes per pair stay 552,960 B, as in attempt-4.
   The 8/32-row chained wins must therefore be kept by occupancy,
   not by fewer bytes.

Boundary kept: bit-exact accumulation order per size regime as
above; all four bfloat16 rounding points (gate, up, hidden, down);
positional `y[r,j]` for any `inds` order; valid at every `rows`
(grid and block counts scale with `rows*4` pairs, branch only on
the 60-row dispatch threshold); 32 threads/group <= 1024;
threadgroup memory by hand (grader does not check 32 KB): GU
1536 + 128 + (256+1024+256)*4 = 7,808 B; Down 768 + 6,144 =
6,912 B.

## Predicted µs per layer at 1 / 8 / 32 rows (chained, this grader)

Weight bytes per layer (552,960 B/pair): 2.21 / 17.7 / 70.8 MB at
4 / 32 / 128 pairs. Attempt-4 chained (round-2 exploration,
`console-chain.txt`): 132.9 / 175.9 / 587.6; reference chained
80.3 / 239.6 / 797.6.

- 1 row: 144 small groups fill 16 cores; gate/up 1.47 MB + down
  0.74 MB at an assumed ~55 GB/s effective for 32-thread groups
  gives ~40 µs including the extra launch → **40 µs**
  (attempt-4 chained 132.9; reference 80.3).
- 8 rows: 384 + 768 groups, streaming regime; 17.7 MB at
  ~110 GB/s ≈ 160 µs → **165 µs** (attempt-4 175.9).
- 32 rows: 1,536 + 3,072 groups; 70.8 MB at ~115 GB/s ≈ 615 µs
  → **610 µs** (attempt-4 587.6; the 4% block-split tax is the
  risk to the kept win).

G2: expect `differ == 0` at all six sizes. Decode reuses
attempt-4's proven per-block math unchanged; prefill depends on
the hardware's 8-wide fragment dot matching `BlockMMA`'s when fed
the same bfloat16-rounded operands in the same tile order.

## Verdict

G1 pass; **G2 pass — `differ` 0, `max_ulp` 0.0 at all six sizes**
(decode 1/8/32 and prefill 127/1,016/2,048). The prefill branch is
bit-exact: bfloat16-rounded dequantized weights plus BK=32 tiles of
four sequential 8-wide `simdgroup_multiply_accumulate` steps, fed a
single row per threadgroup, reproduce `gather_qmm_rhs` bit for bit
on every tested layer. `beats_compiled` false. Chained
µs/layer (candidate vs reference vs compiled, same run):
**103.6 / 326.8 / 887.6** vs 76.1 / 244.6 / 797.1 and
73.1 / 232.0 / 804.5. Baseline `none` — no
`wins_vs_baseline`/`beats_baseline`. The attempt is correct at
every size but slower than the compiled reference at all three
decode sizes, and slower than the starting point's chained times
(132.9 / 175.9 / 587.6, other harness) at 8/32 rows.

## Miss analysis (prediction 40 / 165 / 610; all misses > 20%)

Measured is 2.6x / 2.0x / 1.5x the prediction. Effective bandwidth
from 552,960 B/pair: 21 / 54 / 80 GB/s chained, against attempt-4's
chained 17 / 101 / 121 GB/s. The split helped exactly where group
count was the binding limit (1 row: 103.6 vs 132.9) and hurt
wherever streaming mattered. Two causes, both in the split shape,
not the arithmetic:

1. **One simdgroup per group has too little memory parallelism.**
   Each 32-thread group runs its block's 24-step strided K loop
   alone; attempt-4 runs 8 such blocks concurrently per group, so
   one stalled load hides behind seven others. More groups only
   partly compensate: at 8/32 rows the GPU was already saturated
   with groups, and the per-group latency chain became the limit.
2. **Threadgroup memory caps residency in the decode branch too.**
   The qmm tiles (6,144 B of floats) are allocated in both kernels
   at every size, so a decode gate/up group carries 7,808 B where
   attempt-4 carries 3,840 B. Resident groups per core fall, and
   with 32-thread groups the resident thread count falls with
   them — the opposite of the occupancy the split was meant to add.

Lesson for `ineff-baseline-experts-2`: exposing threadgroups is
not sufficient; a block-split kernel must keep several blocks in
flight per group (or per core) and keep decode threadgroup memory
near attempt-4's, e.g. 128-thread groups of 4 simdgroups each
owning one block (12 + 24 groups at 1 row), with the prefill tiles
overlaid on the decode buffers instead of added to them.

**Next change:** rebuild the decode branch as 4-simdgroup groups
(4 blocks per group, qvm_block unchanged), overlay the qmm tile
memory on the decode threadgroup buffers, keep `qmm_block` as is.
