# Attempt 1: full fused grouped experts kernel (builder-1-a, direction a)

First in direction (a) and first in the slot: ONE Metal kernel (60
threadgroups of 512 threads, one group per expert) that batches same-expert
pairs in chunks of 4, fuses gate/up/down plus silu-mul, and needs no
sort/unsort. Each group scans `inds` (<=128 uints) for its pairs, then per
chunk: gate/up accumulate in registers with weights dequantized on the fly,
hidden pairs sit in 3 KB of threadgroup memory, down accumulates in registers
and writes `y` positionally.

Removes, in one combined fix (savings do not add; joint ceiling per the
hypothesis is 443 MB at streaming rate):

- `ineff-quantized-kernels-1`: per-pair qvm grids re-read each expert's
  weights per pair with 192 B-strided lanes. Here each weight word is read
  once per 4-pair chunk with coalesced consecutive-word reads along the out
  dim (the transposed layout stores words consecutively along out), and x
  rows stay L1-resident instead of being re-read 12x per product.
- `ineff-runs-1`: the same chunking drops the 465 MB excess re-reads toward
  once-per-appearing-expert.
- `ineff-switch-layers-2`: 4 on-chain kernels/layer become 1; gate/up/hidden
  intermediates never touch DRAM.
- `ineff-switch-layers-1`: no sort/unsort glue at any row count (pairs are
  grouped on the fly by the scan, valid for every 1-32 rows).
- `ineff-findings-2`: grouped batching is built in (the measured 25% win).

Boundary kept: fp32 accumulation over the input dim in order, bf16 rounding
after each product and after silu-mul (`(gate*sigmoid(gate))*up` mirroring
`swiglu`), `y[r,j]` positional for any `inds` order, rows from `x_shape[0]`
with all indexing in bounds. Threadgroup memory by hand (grader does not
enforce 32 KB): hbuf 1536x2 + pcodes 128x2 + count 4 = 3,332 B. Threads 512
<= 1,024. Grid `E*512` is rows-independent.

## Predicted us per layer at 1 / 8 / 32 rows

Weight bytes per expert (all three matrices): 3x(147,456 + 2x18,432) =
552,960 B ~ 0.553 MB (hypothesis Boundary). Traffic per layer = distinct
experts x 0.553 MB x chunk factor (~1.0 at 1-8 rows, ~1.2 at 32 rows where
some experts hold >4 pairs) + x + y (<= 0.2 MB). Rate basis 150 GB/s
streaming (the output layer hits 158 GB/s, per ineff-quantized-kernels-1),
plus one ~7 us launch and a latency tail for the small-wave cases.

- 1 row: 4 distinct experts (runs-1: all 4 pairs distinct) -> 2.21 MB /
  150 GB/s = 15 us + 7 us launch + ~13 us tail (4 busy groups of 60,
  occupancy-starved) -> **35 us** (original 46 us = 0.92 ms / 20, brief:92).
- 8 rows: ~20 distinct/layer (qk-1 caveat) -> 11.1 MB / 150 = 74 us +
  launch/latency -> **100 us** (original ~255 us = 5.1 ms / 20, findings-2
  interpolation).
- 32 rows: ~40 distinct/layer (findings-2: 3.2 pairs/expert) x 1.2 ->
  26.5 MB / 150 = 177 us + overhead -> **220 us** (original 733 us =
  14.67 ms / 20, brief:92).

Main risk is the 1-row margin (35 vs 46 us, and the compiled reference is
still unmeasured): if the tail dominates, halve per-thread work with 1024
threads or split the scan. Compute check: ~9K MACs + ~9K dequants per thread
(~55K scalar ops); at 32 rows ~40 busy groups x 512 threads x 55K ~ 1.1G ops
~ 0.4 ms at 2.6 TFLOPS if compute-bound, overlapping the 0.18 ms of memory.

## v1 verdict: G1 pass, G2 fail (~7% bad, max 11/32/68 ULP at 1/8/32 rows)

v1 (serial fp32 accumulation + single-rounding fp32 silu) failed G2 with
byte-identical numbers to siblings 1-b v1 and 1-c v1 (max 11/32/68 ULP),
so the cause is a systematic modeling gap, not an indexing bug. Source
analysis (MLX v0.32.4, fetched from GitHub, paths below) shows two gaps:

1. The reference products run `affine_gather_qvm` (`quantized.h`
   `qvm_impl:1222-1330`): lane l accumulates k = l, l+32, ... sequentially
   with `qouter`'s bits=4 forms (`:503-509`: `s*(low nibble)+bias` for even
   outputs, `(s/16)*(unshifted high nibble)+bias` for odd), then a
   `simd_sum` tree over the 32 lanes. Serial accumulation flips gate/up by
   1 ULP too often, and down outputs near cancellation amplify hidden
   flips past the 2-ULP floor.
2. The reference activation is the *compiled* swiglu tape, and
   `compiled.cpp:206-208` rounds EVERY op to its bf16 dtype: Sigmoid runs
   with T=bfloat (`unary_ops.h:308-314`, `precise::exp` rounded through
   bfloat per `bf16_math.h`, preamble includes it via `utils.h:7-8`), then
   two bfloat multiplies - five bf16 rounding points, not one.

## v2 (this code): bit-exact replication, same structure

Same direction (a): one kernel, threadgroup per expert, 4-pair chunks, no
sort. Each product now replicates qvm lane-for-lane (k = lane+32t, exact
qouter op shapes, real `simd_sum`, lane 0 rounds to bfloat); the activation
replicates the compiled tape op-for-op in bfloat (explicit roundings where
the header overloads would apply, bfloat arithmetic written directly so the
JIT lowers it as in the reference). Work totals unchanged, so predictions
are unchanged: 35 / 100 / 220 us per layer. Open risk: FMA contraction of
`result += x*w` must match the precompiled metallib; the grade decides.

## v2 verdict: G1 pass, G2 pass bit-exact, timing 60x too slow

v2 passed G2 with max_ulp 0.0 at all row counts (FMA contraction matches the
metallib), but timed 4379.8 / 22507.8 / 47623.4 us per layer vs reference
66.9 / 219.1 / 766.7 and compiled 62.8 / 212.5 / 765.3. Diagnosis: v2 mapped
outputs strided across simdgroups (q = sg*96+it), so each weight cache line
was re-fetched ~32x with lines evicted between reuses (~200 MB DRAM per
busy expert-group vs 0.7 MB unique). Per-busy-group cost was a constant ~1 ms.

## v3 (this code): loop interchange, same math

Same direction (a), same bit-exact numerics (qvm lane structure, qouter op
forms, simd_sum, exact swiglu tape). Loops are now output-block outer (256
consecutive outputs), k middle, 16 consecutive outputs inner, so each weight
line is fetched once and fully used; gate/up share one k-sweep. Work totals
unchanged, predictions unchanged: 35 / 100 / 220 us per layer.

## v3 verdict: G1/G2 pass bit-exact, 1236.0 / 5060.2 / 10979.8 us

4.3x faster than v2 (loop interchange worked) but still 14x off the reference
(771 us at 32 rows). Diagnosis: v3 reloaded x/word/scale/bias per (output, k)
(~3.5 global loads per MAC) and sat at the L1 load limit (~2.6 loads/cycle).
The reference loads once per (lane, k) and fans out to 32 outputs.

## v4 (this code): load fan-out, split phases, same math

Same direction (a), same bit-exact numerics. Each 16-output block provably
never crosses a pair boundary, so one x value, 3 weight words, 2 scales and 2
biases are loaded per (block, k) into registers and fan out to 16 outputs
(~0.5 loads/MAC); gate/up/down run as three separate sweeps (16 accs, ~30
regs/thread) with gate/up/hidden in 9 KB of threadgroup memory. Predictions
unchanged: 35 / 100 / 220 us per layer.

## v4 verdict: G1/G2 pass bit-exact, 259.2 / 1218.9 / 2413.7 us

4.5x faster than v3 (fan-out worked) but still 3-5x off compiled
(68.2 / 219.6 / 765.2). Same op mix as the reference at ~1/4 the
throughput: suspect low occupancy from 512-thread groups (~35 regs/thread
= ~18K regs/group, few groups resident per core to hide L1 latency).

## v5 (this code): 256 threads, same math

Same direction (a), same bit-exact numerics and fan-out. 256 threads (8
simdgroups): 12/24 output blocks for gate-up/down, 6 silu outputs per
thread. The 16-block pair-boundary proof holds for q0 = 128*itb+16*sg
(q0%384 <= 368, q0%768 <= 752). Line sharing per (block, k) stays at 128
consecutive outputs = 16 words = one full line. Predictions unchanged:
35 / 100 / 220 us per layer.

## v5 verdict: G1/G2 pass bit-exact, 256.7 / 1148.7 / 2430.9 us

Identical to v4: not occupancy-bound. Same op mix as the reference at ~1/4
throughput with no sensitivity to thread count points at acc-array scratch
spilling (acc[m] with dynamic index) or pure ALU-issue bound.

## v6 (this code): forced full unroll of all m-loops, same math

Same direction (a), same bit-exact numerics. `#pragma clang loop
unroll(full)` (as the reference uses at quantized.h:1318) on all 9 m-loops
(3 inner, 3 simd_sum, 3 lane-0 stores) to force acc scalarization into
registers. Unrolling preserves accumulation order, so bit-exactness holds.
Predictions unchanged: 35 / 100 / 220 us per layer.

## v6 verdict: G1/G2 pass bit-exact, 248.5 / 1148.9 / 2435.1 us

Identical to v5: the compiler already scalarized; the kernel is issue-bound.
Accounting for the gap: my MACs = groups x 4 pairs-worth (dummy work for
<4-pair groups: 36% at 32 rows, ~75% at 1-8 rows) = 1.56x the reference's
128 pairs-worth, times ~14 ops/MAC vs the reference's ~6 = 2.3x: 1.56 x 2.3
= 3.6x explains the 2435/285 = 8.5x... no: 2435 vs reference 770 = 3.2x =
1.56 (dummy) x 2.05 (ops). Both must go.

## v7 (this code): exact itb-skip removes all dummy work, same math

Same direction (a), same bit-exact numerics. Each 128-q output block holds a
single pair (128*itb%384 <= 256 so +127 never crosses; same mod 768), so
blocks with p >= nloc are skipped with a fully uniform `continue`. Useful
work is now exactly the 128 pairs-worth, predicting ~65 / ~380 / ~1560 us
(v6 time x useful fraction 25% / 33% / 64%). Remaining gap is ops/MAC (~14
vs ~6), next.

## v7 verdict: G1/G2 pass bit-exact, 81.6 / 416.2 / 1573.8 us

Matches prediction (~65/~380/~1560): dummy work fully removed. Remaining gap
is ops/MAC (~21 with selects vs the reference's ~6).

## v8 (this code): fully static inner loops via 16-multiple alignment

Same direction (a), same bit-exact numerics and values/order. Each 16-output
block starts at a 16-multiple o0 (q0 and p*384 both are), so: o0%8==0 gives
two fixed word runs ([0,8): w0, [8,16): w1, no clamps since o0<=368); o0%32
in {0,16} gives one scale group (no split, no clamps); o0 even gives 8 direct
even/odd pairs sharing one byte each with bsh = 8*(j&3) constant
((o0/2)%4==0). Inner loop has no selects, branches, or clamps: ~5.5 ops/MAC
at reference parity. Predicts ~35 / ~170 / ~600 us per layer.

## v8 verdict: G1/G2 pass bit-exact, 86.8 / 462.3 / 1766.6 us (clean rerun)

10% SLOWER than v7 despite strictly fewer source-level ops: the compiler was
already folding v7's per-m computation, so both run ~6 folded ops/output -
yet 2.3x under the reference's throughput on the same mix. Either acc arrays
spill to scratch (unroll pragmas silently ignored) or something structural.

## v9 (this code): hand-unrolled scalar accs, same math

Same direction (a), same bit-exact values/order, generated (one-shot script,
283 lines): 16 explicit scalar accs per phase, 8 pairs expanded literally,
16 explicit simd_sums and stores. No acc arrays, no pragmas, nothing left to
heuristics. If timing collapses to ~600, spilling was the gap; if unchanged,
the gap is elsewhere. Predictions: ~600 us at 32 rows if scalarization was
the issue, else unchanged ~1750.

## v9 verdict: G1/G2 pass bit-exact, 87.4 / 461.1 / 1787.4 us

Identical to v8: not spilling either. qvm-lane replication runs the same
~6 ops/output as the reference at 1/2.3 throughput for reasons I could not
determine (not occupancy, spilling, traffic, or waves). Sibling evidence
redirects: attempt-3 passes G2 (max 1.0, bad 0) with even/odd-split
accumulation + tmem tiles (NO qvm lanes, NO simd_sum) at 654 us, and
attempt-2 passes bit-exact with per-pair fusion at 648 us. Bit-exact qvm
replication is unnecessary; the G2 key is the bfloat silu + near-exact
products, and the speed key is sibling-simple structure.

## v10 (this code): attempt-3 numerics fused single-kernel

Same direction (a) (grouped + fully fused), rewritten on attempt-3's
G2-proven forms: even/odd-split accumulation, plain s*(shifted nibble)+bias
dequant, tmem weight/scale/x tiles, MLX stable bfloat silu verbatim, 192
threads, chunks of 2 pairs. Bit-identical values to attempt-3 by
construction (same indices, order, op forms; hidden in on-chip hbuf instead
of DRAM). Down adapted to 192 threads x 4 outputs (one word + one scale
group per thread: 4o%8 in {0,4}, 4o%32+3 <= 31). Tmem ~17.1 KB. Predicts
~640 us at 32 rows (attempt-3 654 - hidden DRAM - 1 launch), ~225 at 8,
~60 at 1.

## v10 verdict: G1/G2 pass (max_ulp 1.0, bad 0), 93.7 / 278.3 / 801.6 us

2.2x faster than v9 (tmem tiles + even/odd work) but still short of compiled
(68.8 / 216.7 / 761.8) and 1.25x slower than attempt-3's 68/238/654 on
near-identical code. Root cause: tile loads use idx = o*K+k, i.e. consecutive
threads read K words apart (16x line waste in my down tiles, 8x in gate/up) -
every version so far has been traffic-bound on strided loads, never ALU-bound
(my "streaming" models were wrong; the lanes/threads always strided).

## v11 (this code): stride-1 tile loads, same math

Same direction (a), same values/order (same idx SETS, G2-safe by
construction). All four tile-load loops now use idx = k*192+o: consecutive
threads load consecutive words (1-2 lines per warp, ~100% util). Tile DRAM
traffic drops ~10x to ~unique (~30 MB/layer at 32 rows -> ~200 us at
150 GB/s + overlapped ALU). Predicts ~40 / ~110 / ~280 us per layer -
beats_compiled at all rows.

## v11 verdict: G1/G2 pass (max_ulp 1.0, bad 0), 100.8 / 272.1 / 783.4 us

Unchanged from v10: not traffic-bound either. Remaining structural gap is my
down adaptation (192 threads x 4 outputs, ~30 regs) vs attempt-3 k2's proven
384x2 (~18 regs): per-group analysis says my down runs ~2x slower on the same
work (occupancy/latency-hiding).

## v12 (this code): 384 threads, C=4, attempt-3 per-thread shapes

Same direction (a), same G2-proven values/order. 384 threads: gate/up keep
attempt-3 k1's exact per-thread shape (2 pairs x 2 outputs, 16 accs),
threads <192 take chunk pairs 0/1 and >=192 pairs 2/3 (C=4 halves weight
re-reads vs C=2); down is attempt-3 k2 verbatim (2 pairs x 2 outputs, 8
accs) with the same pair split. Stride-1 tile loads kept. Tmem ~19.25 KB.
Predicts ~600 us at 32 rows (attempt-3 parity + fusion + C=4), ~200 at 8,
~55 at 1 - beats_compiled at all rows.

## v12 verdict: G1 fail (name) then G2 fail (down undercoverage bug)

`half` is a reserved Metal type (renamed to hh); then G2 failed massively
because down covered 384x2x2 = 1536 of 3072 outputs (each half's threads
covered only half the D dim per pair). Not timed.

## v13 (this code): unified 4-pair threads, same proven math

Same direction (a), same G2-proven values/order, restructured cleanly: 384
threads, each handles all 4 chunk pairs (gate/up: 1 output x 4 pairs;
down: 2 outputs x 4 pairs), so dequantized weights amortize over 4 pairs
(~3.25 ops/MAC vs ~4.5). No pair-split halves anywhere: t0..t3 + n, phantom
slots always valid, y stores guarded per pair. Tmem ~17.7 KB. Predicts
~550-600 us at 32 rows, ~190 at 8, ~55 at 1 - beats_compiled at all rows.

## v13 verdict: G1/G2 pass (max_ulp 1.0, bad 0), 132.8 / 464.8 / 968.1 us

4-pair unification backfired (+21%): more xT/hT tmem reads per MAC (1.25
vs 1.0) dominate the dequant savings. Lesson: 2-pair threads with minimal
tmem reads are the sweet spot.

## v14 (this code): v10 gate/up + 8-acc down over 2 blocks

Same direction (a), same G2-proven values/order. 192 threads, C=2.
Gate/up identical to attempt-3 k1 (2 pairs x 2 outputs, 16 accs); down
matches k2's per-thread shape exactly (2 pairs x 2 outputs, 8 accs) over 2
output blocks of 384 (down weights re-read 2x - affordable, not
traffic-bound), restoring occupancy parity with k2. Tmem ~17.1 KB.
Predicts ~640 us at 32 rows, ~225 at 8, ~60 at 1.

## v14 verdict: G1/G2 pass (max_ulp 1.0, bad 0), 96.9 / 269.4 / 762.4 us

vs compiled 139.2 / 218.7 / 761.3 (1-row run noisy): 32-row ties compiled
+1 us; 8-row short 23%. Down blocks helped (+40) but down still pays either
re-reads (v14) or regs (v10).

## v15 (this code): 384 threads, C=4, att-3 shapes everywhere

Same direction (a), same G2-proven values/order. 384 threads, C=4 chunks:
gate/up split pairs across halves (2 pairs x 2 outputs, k1 shape, 16 accs);
down runs 2 pair-groups sequentially (2 pairs x 2 outputs, k2 verbatim, 8
accs). Stride-1 loads kept. Tmem ~19 KB. Predicts ~670 us at 32 rows
(k2-speed down + C=4 gate/up), ~230 at 8, ~65 at 1.

## v15 verdict: G1/G2 pass (max_ulp 1.0, bad 0), 161.0 / 556.8 / 1146.6 us

50% SLOWER than v14 across the board. The 384-thread C=4 restructure
backfired: more threads x more per-thread state raised register/tmem
pressure without adding useful parallelism (down pair-groups sequentialize
the same work; halves duplicate o-pair coverage logic). Direction (a)
lesson: 2-pair chunks at 192 threads are the sweet spot; wider costs more
than it shares.

## Final verdict (v14 restored as the attempt content)

Per the lead's wrap-up note, the folder holds v14 (best of v1-v15).
G1 pass; G2 pass with max_ulp 1.0, bad 0 at all rows; beats_compiled FALSE
(beats compiled only at 32 rows). Confirmation re-grade of the restored
files (clean run; first v14 grade was 96.9 / 269.4 / 762.4 vs a noisy
1-row baseline):

| rows | candidate | original | compiled |
| 1 | 79.3 | 64.1 | 62.0 |
| 8 | 260.3 | 214.4 | 214.1 |
| 32 | 743.7 | 767.1 | 761.0 |

## Miss vs prediction (>20%, explained)

Predicted 35 / 100 / 220 us per layer (distinct-expert bytes at 150 GB/s
streaming + one launch + small tail); measured 79.3 / 260.3 / 743.7 -
2.3x/2.6x/3.4x over. Three compounding causes, all learned the hard way:

1. The v1-v9 bit-exact-qvm path was a dead end for speed (47 ms -> 1.8 ms
   best). Replicating qvm's 32 strided lanes + simd_sum forces a 192 B
   strided access pattern per lane that no loop interchange, fan-out,
   unrolling, or thread-count change could make efficient (each fix
   verified neutral-or-negative by grade: v4=v5 threads, v5=v6 unroll,
   v8 slower than v7, v9=v8 scalars). Runs at ~1/2.3 reference throughput
   on the same op mix for undetermined microarchitectural reasons.
2. The att-3-style rewrite (v10+) reached ~800 us but fusion kept losing
   to att-3's 2-kernel split (654 us): my down adaptation paid either regs
   (v10 192x4, 16 accs) or re-reads (v14 8 accs over 2 blocks); C=4
   backfired twice (v13, v15) via tmem-read pressure (+21-50%).
3. The 150 GB/s streaming basis never materialized: achieved rates are
   ~55 GB/s (tiled load->barrier->compute duty cycle, per att-3's finding),
   and grouping's byte savings don't convert in the piece harness (cache
   absorbs per-pair re-reads across the chained 20-layer timing).

## Miss vs attempt-2 (per-pair fusion, 107.3 / 188.4 / 647.6)

Attempt-2 beats me at 8 rows (188 vs 260) and 32 rows (648 vs 744); I win
only at 1 row (79.3 vs 107.3). Why fusion-without-grouping wins in the
piece harness: the dominant prize is removing kernels + intermediates
(switch-layers-2) and the sort (switch-layers-1), which per-pair fusion
captures fully with the simplest possible structure (128-thread groups,
one pair each, no scan/chunk/barrier/phantom overhead, minimal tmem).
Grouping's extra byte savings (~2x at 32 rows) don't convert because the
harness reuses weights through cache across layers/repeats, while my
grouped kernel pays real overhead: pair scan, chunk loop, barriers (76+),
phantom-slot dummy work (up to 50% at n=1), 17 KB tmem, occupancy loss.
For G3 end-to-end (larger working set, less cache residency) grouping may
pay more than the piece timing suggests - but measured here, fusion pays,
grouping costs. Hardening should start from attempt-2's structure and add
grouping only if G3 shows cache pressure.

## Learnings for hardening (direction a)

- G2 numerics (reusable): bit-exact qvm replication is NOT required.
  attempt-3's forms pass: even/odd-K split accumulation, plain
  s*(shifted nibble)+bias dequant (no s/16 trick needed), (bfloat)(e+o)
  rounds, MLX stable bfloat silu verbatim (per-op bf16 rounds). My
  independent source analysis (quantized.h qvm_impl:1222-1330,
  qouter bits=4 :503-509, unary_ops.h Sigmoid :308-314, compiled.cpp
  :206-208 rounding every op, bf16_math.h) confirms WHY v1-style
  fp32-silu fails (~7% bad, max 11/32/68 ULP, identical across all
  three builders' v1s): five bf16 rounding points, not one.
- What made the grouped single-kernel slow, in order: (1) qvm-lane
  structure (strided, unfixable ~2.3x); (2) tmem-read pressure grows
  with pairs/thread (C=4: 1.25 vs C=2: 1.0 reads/MAC); (3) down
  adaptation dilemma at fixed thread count (regs vs re-reads);
  (4) 384-thread configs slower than 192 (pressure > parallelism).
- Neutral (verified by grade, don't repeat): thread count 512/256
  (v4=v5), forced unroll (v5=v6), hand-unrolled scalars (v8=v9),
  stride-1 vs strided tile loads (v10=v11), 16-multiple static loops
  (v8 slower than v7 - compiler folds anyway).
- Boundary hygiene kept throughout: y positional, any inds order, all
  rows 1-32 (grid rows-independent), no unwritten-tmem reads (phantom
  slots always valid), 17.1 KB tmem by hand.
