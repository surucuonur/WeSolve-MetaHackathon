# Territory: quantized-kernels (round 3 — prefill SPEED)

- **Owner:** explorer-quantized-kernels
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T17:41:57Z
- **Next:** none — rhs speed anatomy mapped (§S1–§S7), 2 inefficiencies posted, note left on `territory-baseline-experts`

## What it does

This territory is MLX 0.32.4's `mx.gather_qmm` Metal kernels behind the
`experts` slot (`quantized.h`, `quantized_utils.h`). Round 1 mapped the
decode kernel (`affine_gather_qvm`) and round 2 mapped the prefill
kernel's ROUNDING (`affine_gather_qmm_rhs`: bfloat-rounded dequant +
8-wide frag accumulation, and why a size branch at rows ≥ 60 is
required) —
[`muse-spark/exploration/territory-quantized-kernels.md`](../../exploration/territory-quantized-kernels.md)
and
[`muse-spark/round-2/exploration/territory-quantized-kernels.md`](../round-2/exploration/territory-quantized-kernels.md)
(both frozen; cited, not rewritten). Round 3 needs its SPEED: the
compiled prefill runs at 960.8 / 4306.0 / 11998.9 µs per layer at
127 / 1,016 / 2,048 rows
(`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`),
15–28× faster than the starting point's scalar prefill branch, and any
fast exact prefill branch — or `max_rows` hand-off that skips one —
must match or avoid what is mapped here. This entry writes only what is
new: tiling, occupancy, dequant/x/MMA dataflow, rate analysis, and the
match-or-avoid list. Rounding facts are round 2's (§D/§E/§F there).

## The checklist

### S1. Tiling and grid — what `gather_qmm_rhs` launches (speed view)

- **Callers:** same dispatch as round 2 §A (frozen): `M==1 && B>=16 &&
  right_sorted_ && B/E>=4`, i.e. rows ≥ 60. Legacy tiles `bm=16, bn=32,
  bk=32`, `wm=1, wn=2` (round 2 §B; NAX never runs on this GPU).
- **Callees:** `affine_gather_qmm_rhs<bfloat16,32,4,BM=16,BN=32,BK=32,
  WM=1,WN=2,transpose=false>` (`quantized.h:2570-2593`) with
  `group_dims=(32,2,1)=64` threads and `grid_dims=((N+31)/32,
  min(B,ceil(B/16)+E-1),1)` (round 2 §B; host formula at
  `quantized.cpp:1817-1819`). Per call:

  | prefill rows | B | grid_y | gate/up groups (12×gy) | down groups (24×gy) | per layer (3 prods) |
  |---|---|---|---|---|---|
  | 127 | 508 | 91 | 1,092 | 2,184 | 4,368 dispatched (2,880 useful + 1,488 empty) |
  | 1,016 | 4,064 | 313 | 3,756 | 7,512 | 15,024 dispatched (14,400 + 624) |
  | 2,048 | 8,192 | 571 | 6,852 | 13,704 | 27,408 dispatched (25,920 + 1,488) |

  Useful counts use balanced-expert `T=60/300/540` row tiles (round 2
  §C); empty counts are round-2 `ineff-quantized-kernels-1`'s table ×
  48 N-tiles per layer. Each group runs `K_it=K/32` loop iterations
  (24 gate/up, 12 down) over §S2–§S4's dataflow, then one
  `store_result(_safe)`.
- **Data:** threadgroup `Xs[16×40]+Ws[32×40]` bfloat16 = 1,280+2,560 =
  3,840 B per group (round 2 §C; `BK_padded=BN_padded=40`,
  `quantized.h:2596-2597`). 64 threads × 3.84 KB is a small, cheap
  group — the opposite of the decode `qvm` grid's occupancy problem.
- **Neighbours:** 3 offsets kernels + sort/unsort + `swiglu` share the
  chained slot time (§S5); the mains dominate by byte floors (§S5).
- **Cost:** no per-kernel prefill timing exists (grader times the
  chained slot only); §S5 prices the mains from the chained totals.
- **Invariants:** regrouping rows is free (round 2 §C invariant):
  padded rows never mix into valid rows (`store_result_safe`,
  `:2694`), so a replacement may use any BM, variable tiles, or no
  tiles, as long as each row's own accumulation matches round 2
  §D+§E. Grid over-provisioning (≤59 y-tiles) is pure waste (§S6).

### S2. Dequant dataflow — `QuantizedBlockLoader`, per-K-tile (new)

- **Callers:** the rhs kernel instantiates the loader with
  `BROWS=BK=32, BCOLS=BN=32, tgp_size=64` (`quantized.h:2613-2621`).
- **Callees:** `load_unsafe` (`quantized.h:736-747`) → `dequantize`
  bits=4 (`:617-623`): per packed byte, `w[2i]=(bfloat)(s·(b&0x0f)+b)`,
  `w[2i+1]=(bfloat)((s/16)·(b&0xf0)+b)`, with `s,b` the float-upcast
  scale/bias. The `s/16` high-nibble form equals `s·(b>>4)+b` as a
  float (power-of-two scaling is exact), so both forms round
  identically — attempt-7's `qmm_accum` relies on this
  (`muse-spark/round-2/kernels/experts/attempt-7/block-1.metal:203-205`).
- **Data:** `n_reads=(16·32)/64=8`: each of the 64 threads loads 1
  scale + 1 bias (threads pair up on 32 distinct `bi` rows, so
  scales are fetched 2× — negligible), reads 8 consecutive packed
  bytes, and writes 16 dequantized bfloat into `Ws`
  (`:686-734`). Per K-iteration per group: a full 32×32 tile,
  1,024 elements from 512 B packed + 128 B scales/biases.
  `next()` advances scales by `group_stride=BROWS·src_ld/group_size`
  per K tile (`:772-790`) — exactly one scale group per (K tile,
  N tile), matching the `[E,K,N/32]` layout.
- **Neighbours:** weights stream per (row tile, N tile, K tile)
  visit; NOTHING is reused across visits — each extra row tile of
  an expert re-reads and re-dequantizes its full 184,320 B
  (147,456 weights + 36,864 scales/biases; round 2 §C). Dequant
  element-ops per product per layer = `T·K·N` = 17.7M / 88.5M /
  159.2M (round 2 §D). At ~10 ALU ops each (and/shift, mul, add,
  round, tg store) that is ≈0.18/0.89/1.59 GOP per product,
  ≈0.5/2.7/4.8 GOP per layer — under 10% of the measured time even
  at scalar rates, so dequant ALU is a real but minor share; the
  design win is AMORTIZATION (one dequant serves ≤16 rows) plus
  coalescing, not ALU savings.
- **Cost:** weight bytes per layer = `3·T·184,320` = 33.2 / 165.9 /
  298.6 MB → floors at 200 GB/s (`claude/GLOSSARY.md:10`) of 166 /
  829 / 1,493 µs (§S5). Coalescing: `BlockLoader`-style row runs —
  thread pairs `(2i,2i+1)` share K-row `bi` and read bytes 0–7 and
  8–15 of it (consecutive, one 16 B line), pairs stride by
  `N/2=192/384 B` per K-row (`:727-734`) — the standard tiled-GEMM
  pattern, full lines, no `qvm`-style 192 B-stride scalar loads
  (contrast round-1 §A / round-2 §F).
- **Invariants:** the bfloat16 rounding into `Ws` is rounding point
  1 (round 2 §D) — a fast branch must reproduce these exact rounded
  weights (same `s·q+b` float value, same cast), whatever its
  grouping.

### S3. Activation (x) dataflow — re-read per N-tile (new)

- **Callers:** `BlockLoader<bfloat16,BM=16,BK=32,dst_ld=40,
  reduction_dim=1,tgp_size=64>` (`quantized.h:2611-2612`).
- **Callees:** `load_unsafe` (`steel/gemm/loader.h:74-80`) copies
  verbatim bfloat16→bfloat16 (exact, no rounding); partial row
  tiles (`tgp_bm<16`) take `load_safe` (`loader.h:83-128`), which
  zero-pads via a per-element `tmp_idx/tmp_val` bound-check dance —
  strictly slower loads on every partial tile.
- **Data:** `n_reads=512/64=8, TCOLS=4, TROWS=16`: threads `(4i..4i+3)`
  own x-row `bi=i`, each reading 8 consecutive bfloat (one 16 B
  vector) — a full coalesced 16×32 tile per K-iteration, 1 KB per
  group per iteration. But EVERY (row tile, N tile) group loads its
  own copy of the same x rows: issued x bytes per product =
  `T·(N/32)·K_it·1KB` = T·12·24 KB (gate/up) = 17.3 / 86.4 /
  155.5 MB per product per layer — the x row-tile (24 KB) is
  re-read 12× (gate/up) or 24× (down), once per N-tile. Total
  issued x per layer ≈ 52 / 259 / 467 MB — EXCEEDS issued weight
  bytes (33/166/299 MB). Counterpart: the 12–24 sharing groups run
  concurrently on 24 KB of shared addresses, so all but the first
  read per row tile should hit in cache; distinct x per layer is
  only `B·(768+768+384)·2B` = 2.0 / 15.6 / 31.4 MB. DRAM x-traffic
  is therefore between 2/16/31 MB (all hits) and 52/259/467 MB
  (all miss) — unmeasured, and a replacement that loads each x
  row-tile once per N-slab (or once per row, qvm-style into
  registers) banks the difference for free.
- **Neighbours:** at 127 rows EVERY useful tile is partial
  (`cnt_e≈8.5<16`, all 60 tiles): 100% of useful work takes
  `gemm_loop_unaligned` + `load_safe` + `store_result_safe`. At
  1,016 ≈20% of tiles are partial (5th of 5 per expert), at 2,048
  ≈11% (9th of 9). The 127-row size thus pays the safe-path tax
  everywhere — a regrouped branch with full tiles (e.g. smaller or
  variable BM) avoids it.
- **Cost:** folded into §S5's mains range; the safe-path tax has no
  separate measured figure (no per-tile timing exists).
- **Invariants:** x values are exact through the loader (no rounding
  point) — regrouping never changes them.

### S4. MMA structure — what each K-iteration does (new)

- **Callers:** `gemm_loop_aligned` (full tiles) /
  `gemm_loop_unaligned` (partial) (`quantized_utils.h:6-75`): per
  iteration, barrier → load x + dequant w → barrier → `mma()`.
  TWO threadgroup barriers per 32-K step, 48 per gate/up group, 24
  per down group — the kernel's latency skeleton.
- **Callees:** `BlockMMA<...,BM=16,BN=32,BK=32,WM=1,WN=2>`
  (`steel/gemm/mma.h:439-537`), `AccumType=float` default
  (`mma.h:451`). Per simdgroup: `TM=16/8=2`, `TN=32/16=2` frags;
  `mma()` steps `kk` over BK in 4 8-wide steps, each
  `tile_matmad` issuing 2·2·1=4 `simdgroup_matrix` 8×8×8
  multiply-accumulates (`mma.h:406-427,513-537`) with 3
  `simdgroup_barrier`s per step. Per group per K-iteration:
  2 simdgroups × 4 steps × 4 MMAs = 32 MMAs = 32,768 flops;
  per gate group total: 24 × 32,768 = 786K flops.
- **Data:** operands reload from threadgroup (`Ws` bfloat→float
  widen on load, `mma.h:57-67`); accumulators stay in float frag
  registers across the whole K loop; one float→bfloat16 store per
  output (`store_result`, `mma.h:540-592`) — rounding point 2's
  container (round 2 §E). Useful MMA per layer = `3·2·B·K·N`
  (K·N=294,912) = 0.90 / 7.19 / 14.50 GFLOP → floors at 5.2
  TFLOPS (`claude/GLOSSARY.md:14`) of 173 / 1,383 / 2,788 µs.
- **Neighbours:** BM=16 means only TM=2 M-frags per simdgroup —
  half the M-reuse of a BM=32 tile; the NAX variant MLX uses on
  M3-class GPUs runs `bm=32/64, bn=bk=64` (round 2 §H), i.e. MLX
  itself buys 4–8× bigger tiles where the hardware allows. A
  replacement is free to tile bigger (regrouping is free, §S1)
  but must keep the 8-wide frag accumulation order per row
  (round 2 §E invariant) — bigger tiles change WHICH rows share
  loads, never a row's own add order, as long as BK=32 frag
  stepping is kept.
- **Cost:** §S5.
- **Invariants:** per-output accumulation grouping (contiguous
  8-wide frag dots, sequential over BK=32 tiles, increasing K)
  is what the scalar `qmm_accum` mimics with serial FMAs
  (attempt-7 `block-1.metal:197-230`) — bit-exact today
  (differ 0 at all prefill sizes,
  `grade-phases-experts-attempt-7.json`), because serial FMAs
  over the same contiguous order round identically to the
  hardware tree on this data. A fast branch keeps that order
  and swaps the scalar engine for MMA.

### S5. Rate analysis — where the 961/4306/11999 µs go (new)

Basis: chained compiled prefill per layer 960.8 / 4306.0 / 11998.9 µs
(`grade-phases-experts-attempt-7.json`; round-3 README Starting
point). The chained slot = `_gather_sort` + 3×(offsets + rhs main) +
compiled `swiglu` + `_scatter_unsort` (`bench/grader/slots.py:55-82`).
Glue bounds: sort+unsort move x twice (`2·B·768·2B` = 1.6/12.5/25.2
MB → 8/62/126 µs at 200 GB/s); swiglu is elementwise on `B·384·2`
(+exp/sigmoid ALU, compiled); offsets ≤3×6.6 µs (ineff entry 2).
Mains ≈ measured − glue ≈ 930 / 4,150 / 11,700 µs per layer.

| prefill rows | measured/layer | useful-MMA floor (§S4) | weight-byte floor (§S2) | mains vs MMA peak | mains vs BW peak |
|---|---|---|---|---|---|
| 127 | 960.8 | 173 (18%) | 166 (17%) | 5.4× (19%) | 5.6× (18%) |
| 1,016 | 4,306.0 | 1,383 (32%) | 829 (19%) | 3.0× (33%) | 5.0× (20%) |
| 2,048 | 11,998.9 | 2,788 (23%) | 1,493 (12%) | 4.2× (24%) | 7.8× (13%) |

Ratios use mains≈930/4150/11700. IMPLIED ISSUED intensity:
14.50 GFLOP / (298.6+466.6+25.1 MB issued) ≈ 18 FLOP/B at 2,048 —
below the ridge point 26 (`claude/GLOSSARY.md:26`), bandwidth-side,
yet the kernel reaches only ~13–20% of peak bandwidth on issued
bytes (less on DRAM bytes after cache hits, §S3). **Verdict:
overhead/latency-bound** (`claude/GLOSSARY.md:40`): the time goes to
the 2-barriers-per-K-tile skeleton (48/24 per group), MMA running
small BM=16 tiles, dequant recompute per row tile (§S2), the
safe-path tax on partial tiles (§S3 — 100% of tiles at 127 rows),
schedule scans and empty tiles (§S6) — not to bytes or flops. Same
family as decode `qvm`'s verdict (62 GB/s = 31% of peak, round-1
§A), reached the opposite way: prefill has ABUNDANT parallelism
(S1: 4.4K/15.0K/27.4K groups per layer, 64 threads each — no
occupancy starvation; whole-step decode occupancy was 9%, round-1
§A) but low per-group efficiency, while decode `qvm` is
occupancy-starved with efficient-enough groups.

Caveat: a second run timed compiled prefill at 966.6 / 3953.2 /
7347.5 µs (`results/runs/2026-11-04-handoff-check/
grade-experts-a7-max59.json`) — the 2,048-row figure is 39% lower
than the 11,998.9 basis above while the reference matches across
runs (7359.8 vs 7386.8). The 127/1,016 bases agree within 8%; the
2,048-row basis — and any saving scaled from it — carries that
run-to-run variance. Flagged for `territory-grader` (what the
two-phase timing still can't see).

### S6. What makes compiled prefill fast (vs the scalar branch) (new)

Attempt-7's `rows>=60` branch (`qmm_accum`, `block-1.metal:206-230`,
called per output per pair, `block-0.metal:59-72,103-112`) is bit-exact
and 15.8/27.7/20.1× slower than compiled (15,220/119,154/240,681 vs
961/4306/11999 µs/layer). Three compounding causes, all in the gap
between §S1–§S4 and one-thread-per-output scalar code (full arithmetic
left as a note on `territory-baseline-experts`, which owns that
kernel):

1. **No cross-row grouping: ~15× the distinct weight traffic.**
  `qmm_accum` loads each pair's full expert weights through its own
  512-thread group (each lane re-loads its block's words; lanes
  share only via cache): distinct ≈186 KB per pair per product × B
  pairs × 3 = 283 MB / 2.27 GB / 4.57 GB per layer vs MLX's grouped
  33/166/299 MB — ratios 8.5×/13.7×/15.3×, matching the measured
  15.8/27.7/20.1× once rates are folded in (attempt-7 runs its bytes
  at ~9–10% of peak, MLX at ~13–20%, §S5). MLX's T-vs-B grouping
  (round 2 §C) is the single biggest speed fact.
2. **Scalar serial K-chains with dependent loads, no MMA.** Each
  output accumulates K=768/384 terms one FMA at a time behind a
  dependent device-load chain (weight word → shift → scale/bias →
  round → FMA, 768 iterations unhideable in one thread), where MLX
  issues 8×8 hardware MMAs over threadgroup-resident tiles (§S4).
3. **Per-pair grids at prefill scale:** `B` groups of 512 threads
  (8,192 groups at 2,048 rows) each re-loading x, re-running silu
  serially, and re-paying barriers — where MLX's 64-thread,
  3.84 KB groups tile the same work with 12–48× fewer threads per
  row and shared x in threadgroup memory.

### S7. What a fast exact branch — or a `max_rows` hand-off — must match or avoid (new)

For `baseline-experts` builders (this territory states the reference
side; the starting-point kernel is theirs):

- **Must match (exactness, from round 2 §D–§F — cited, not
  re-derived):** bfloat-rounded dequant weights (same float
  `s·q+b`, same cast, §S2); contiguous-BK accumulation order per
  row (§S4); the four bfloat16 rounding points (gate, up, hidden,
  down); the `rows>=60` dispatch threshold (brief:
  `claude/muse-agent-brief.md:187`), not the grader's 32/127
  bracket. The scalar `qmm_accum` proves the order can be met
  without MMA — a fast branch keeps its order, changes its engine.
- **Must match (speed, from §S1–§S5):** cross-row weight sharing
  (≈22/14/12 KB distinct weights per pair — MLX's `T·184KB/B`
  amortization — or fewer bytes another way); hardware MMA, not
  scalar FMAs; coalesced full-line loads (no strided scalars);
  enough groups to fill the GPU without 512-thread-per-pair
  overprovisioning.
- **Must avoid (free savings, this round's entries):** tile
  quantization — padded rows + empty tiles + all-partial safe
  paths (`ineff-quantized-kernels-1`, this round); recomputing
  `gather_mm_offsets` (`ineff-quantized-kernels-2`, this round);
  x row-tile re-reads per N-tile where the design can share them
  (§S3, unpriced — cache-mitigated, banks only what misses).
- **Or skip all of it with `max_rows`.** The brief's hand-off
  (`claude/muse-agent-brief.md:173`): inputs above the named row
  count skip the candidate and run the compiled reference, graded
  bit-exact and timed at prefill sizes. Claude verified it on a
  copy of attempt-7 with `max_rows` 59: bit-exact at all six
  sizes, `same` as compiled at every prefill size
  (`results/runs/2026-11-04-handoff-check/
  grade-experts-a7-max59.json`;
  `claude/findings/muse-round-2.md:174`). The threshold semantics
  align with MLX's own rows≥60 switch (round 2 §A): rows ≤ 59
  (all real-model `qvm` sizes, 33–59 included) stay on the
  candidate's decode branch, prefill sizes ride the reference.
  Price: prefill parity with compiled (the §S6 gap — up to
  ~229 ms/layer at 2,048 rows vs the scalar branch — banked in
  full), decode unchanged (keeps 8/32-row wins AND the 1-row
  loss: the hand-off alone does not fix `beats_compiled` for
  experts, whose decode phase still needs the 1-row fix; for
  routing/combine, whose decode already wins everywhere, it banks
  the slot outright — their explorers' call).

## Relations

- **Feeds from:** switch-layers (sorted `idx`, gathered x — the
  `right_sorted_` flag that selects this kernel; round 2 §A).
- **Feeds to:** other-kernels (compiled `swiglu` between products),
  then combine via `_scatter_unsort`.
- **Slot view:** `baseline-experts` owns the starting-point kernel
  that must branch or hand off (§S7; note left there with the
  scalar-gap arithmetic); `ineff-baseline-experts-1` (round 2)
  owns the G2-failure statement this round's branch question
  builds on. `grader` owns the two-phase timing and the §S5
  2,048-row variance caveat (flagged; their entry is `done` —
  lead to forward if needed).

## Inefficiencies

- `ineff-quantized-kernels-1` (this round, open): BM=16 tile
  quantization repriced in prompt µs — padded rows, empty tiles,
  all-partial safe paths; up to ~440/640/610 µs per layer at
  127/1,016/2,048 rows, 0 at decode.
- `ineff-quantized-kernels-2` (this round, open): redundant
  `gather_mm_offsets` repriced — ≤13 µs per layer per prefill
  size, 0 at decode; now bankable prompt time.
- Decode per-pair waste stays round 1's confirmed
  `ineff-quantized-kernels-1` (round-2 restatement
  `ineff-baseline-experts-3`); the scalar prefill branch gap is
  `baseline-experts` territory (note left there), not duplicated
  here.

## Files read

- `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h`
  (installed 0.32.4): `dequantize` bits=4 :557-663,
  `QuantizedBlockLoader` :665-791, `affine_gather_qmm_rhs`
  :2570-2718 — §S1/§S2/§S4.
- `.venv/.../kernels/quantized_utils.h` (90 lines): `gemm_loop_*`
  barrier skeleton — §S4.
- `.venv/.../kernels/utils.h:497-542` (`schedule_row_tile`) —
  §S1/§S6 (cited from round 2; re-read for the empty-tile scan).
- `.venv/.../kernels/steel/gemm/mma.h:406-592` (`BlockMMA`,
  `tile_matmad`, frag stepping, stores) — §S4.
- `.venv/.../kernels/steel/gemm/loader.h:14-134` (`BlockLoader`
  safe/unsafe, `n_reads`/`TCOLS`/`TROWS` mapping) — §S3.
- `bench/grader/slots.py:55-82` (`ref_experts`: sort, 3×`qmm`,
  `swiglu`, unsort — the chained prefill slot) — §S5.
- `results/runs/2026-11-04-muse-round-2/
  grade-phases-experts-attempt-7.json` (chained two-phase timing:
  compiled 960.8/4306.0/11998.9, candidate 15220/119154/240681) —
  §S5/§S6 basis.
- `results/runs/2026-11-04-handoff-check/
  grade-experts-a7-max59.json` (hand-off: G2 pass, `same` at all
  prefill sizes; alternate compiled basis 966.6/3953.2/7347.5) —
  §S5 caveat, §S7.
- `muse-spark/round-2/kernels/experts/attempt-7/block-0.metal`
  (112 lines: prefill dispatch `:22`, per-output `qmm_accum` calls)
  and `block-1.metal:197-230` (`qmm_accum` scalar loop) —
  §S2 invariant note, §S6.
- `muse-spark/round-2/kernels/experts/attempt-7/notes.md`
  (miss analysis, 1-row remainder) — §S7 context.
- `claude/findings/muse-round-2.md:164-174` (starting-point table,
  hand-off verification) — §S7.
- `claude/muse-agent-brief.md` (*The task*, *The slots*,
  launch-spec `max_rows :173`, *How each attempt is judged*) and
  `claude/GLOSSARY.md` (peaks, ridge point, overhead-bound,
  occupancy) — throughout.
- Round-3 `exploration/README.md` (Starting point, items 1+8 of
  *What is left*, territory table) — task framing.
- Frozen (cited, not rewritten):
  `muse-spark/exploration/territory-quantized-kernels.md` §A–§E,
  `muse-spark/round-2/exploration/territory-quantized-kernels.md`
  §A–§H, `muse-spark/round-2/exploration/
  ineff-quantized-kernels-1.md` and `-2.md` (both confirmed).

## From other agents

(none yet)
