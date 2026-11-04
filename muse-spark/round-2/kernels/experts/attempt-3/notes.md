# attempt-3 (round 2): hardening — 4-block groups, scalar prefill (builder-1-c)

Hardening attempt for `hyp-1-experts-exact` (slot `experts`, baseline
`none`). Builds on the best attempt
`muse-spark/round-2/kernels/experts/attempt-1` (G2 pass all six,
104.3/193.3/535.7 µs/layer chained vs compiled 77.6/235.7/791.6 —
loses only at 1 row). Starting point for the decode arithmetic is
attempt-1's proven helpers; attempt-2's failure modes are the
constraint.

## The one main change

Split attempt-1's single fused per-pair kernel (4 threadgroups per
call at 1 row, 512 threads each) into two kernels with 4-block
threadgroups, exposing more threadgroups per call without
attempt-2's 1-simdgroup-per-group collapse:

- `experts_gate_up` (block-0): one 128-thread threadgroup (4
  simdgroups) owns 4 distinct 32-output blocks — simdgroup `s` of
  chunk `c = tg % 3` computes block `blk = c*4 + s` of pair
  `tg / 3` (12 blocks/pair ÷ 4 = 3 groups/pair). Grid
  `rows*1536` threads = `rows*12` groups: **12 groups at 1 row**
  (96 at 8, 384 at 32). Each simdgroup runs gate then up for its
  block, lane 0 (decode) or every lane (prefill) stores the
  bfloat16-rounded results to `tg_gate`/`tg_up`, one barrier, then
  every lane applies the shared full-bfloat16 silu-mul and writes
  `hidden[pair, blk*32 + lane]`.
- `experts_down` (block-1): same shape for the 24 down blocks —
  `blk = (tg % 6)*4 + simd`, pair `tg / 6` (24 ÷ 4 = 6 groups/pair).
  Grid `rows*3072` threads = `rows*24` groups: **24 groups at
  1 row** (192 at 8, 768 at 32).

Everything else is copied, not redesigned:

- **Decode (`rows < 60`):** `qvm_block` in block-2 is byte-identical
  to attempt-1's (verified by diff; itself attempt-4's proven
  helper) — strided lanes `k ≡ lane (mod 32)` serial in float32,
  float32 dequant `x·(scale·q+bias)`, `simd_sum` tree. Splitting
  blocks across threadgroups does not change any block's
  accumulation order, so decode bits are attempt-1's.
- **Prefill (`rows >= 60`):** `qmm_accum` in block-2 is
  byte-identical to attempt-1's scalar path (verified by diff) —
  weight rounded to bfloat16 first
  (`float(bfloat(scale·q+bias))`), then K accumulated contiguously
  in `BK=32` tile order by one thread per output, one output per
  lane. Attempt-2's MMA tiles (`qmm_block`,
  `simdgroup_multiply_accumulate`, 6,144 B of float tile buffers)
  are **not** used, so no tile memory is allocated in either branch
  and decode residency is not taxed for prefill.
- Same branch threshold (`rows >= 60` from `inds_shape[0]`, the
  `gather_qmm_rhs` dispatch threshold, not the tested sizes), same
  four bfloat16 rounding points (gate, up, hidden via silu-mul,
  down — silu helper byte-identical to attempt-2's copy of
  attempt-4's full-bfloat16 lowering), fp32 accumulation, and
  positional `y[r,j]` (`pair = r*4 + j` throughout; `hidden` is
  indexed by the same flat pair).

## Which inefficiency it removes, and how

- `ineff-baseline-experts-2` (chained 1-row latency): attempt-1
  leaves 12 of 16 GPU cores idle per call (4 threadgroups). Here a
  1-row layer call exposes 12 (gate/up) then 24 (down)
  threadgroups, so up to 12/16 cores hold gate/up work and all 16
  hold down work, while each group keeps 4 blocks in flight —
  the in-group load overlap whose loss (1 block/group) cost
  attempt-2 its 8/32-row wins (326.8/887.6). Cost of the split:
  one extra launch per layer on the chain and a `hidden`
  round-trip (1.5 KB/pair written + re-read vs 552,960 B/pair of
  weights, 0.5%); `x` is re-read per chunk-group (3× per pair,
  4.5 KB vs 368 KB of gate/up weights).
- `ineff-baseline-experts-1` (prefill G2 gate): already removed by
  attempt-1's branch; kept by reusing its scalar `qmm_accum`
  verbatim instead of attempt-2's MMA construction.
- `ineff-baseline-experts-3` (per-pair re-reads): deliberately
  kept, as in attempt-1 — weight bytes per pair stay 552,960 B;
  round-1 grouped evidence says removing them loses at this duty
  cycle.

## Limits (by hand; the grader does not check the 32 KB cap)

128 threads/group ≤ 1,024. Threadgroup memory: gate/up
`tg_x[768] + tg_gate[128] + tg_up[128]` bfloat16 = **2,048 B**;
down `tg_h[384]` = **768 B** — both below attempt-1's 3,840 B and
far below the 32,768 B limit, at every row count (allocations are
rows-independent; no prefill-only buffers exist). Grids scale as
`rows*12` / `rows*24` groups, so the kernel is valid at every
`rows ≥ 1`, including untested counts on both sides of 60.

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Weight bytes per layer (552,960 B/pair): 2.21 / 17.69 / 70.78 MB
at 4 / 32 / 128 pairs. References: attempt-1 chained (this round's
grader) 104.3 / 193.3 / 535.7 (effective 21 / 92 / 132 GB/s);
attempt-2 chained 103.6 / 326.8 / 887.6 (21 / 54 / 80 GB/s);
compiled (attempt-1's run) 77.6 / 235.7 / 791.6.

- **1 row: 74.** Latency-bound, not byte-bound (floor 11.1 µs at
  peak). Attempt-1's 104.3 with 4 groups ≈ fixed cost (launch/link,
  `tg_x` load, barriers, lane-0 stores) + a wave-serial part only
  4 cores can work on. Gate/up now spreads over 12 groups and down
  over 24 (1.5 waves of 16), each group with 4 concurrent block
  K-chains instead of attempt-2's 1: assuming the serial part
  scales with cores busy (4 → 12) and adding ~6 µs for the second
  launch + `hidden` round-trip, 2.21 MB at ~33 GB/s effective
  (between attempt-2's 21 with 1 block/group and the 8-row
  streaming rate) ≈ 67 µs + ~7 µs fixed ≈ **74** — just under the
  compiled 77.6, with the launch overhead as the risk.
  Per step this is the ineff-2 headroom: (104.3 − 74) × 20 ≈
  0.61 ms/step recovered of the 1.05 ms gap.
- **8 rows: 162.** Streaming with 96 + 192 groups: resident groups
  per core are capped by 2,048/768 B threadgroup memory (vs
  attempt-2's 7,808/6,912 B), and each group hides loads behind
  4 blocks. 17.69 MB at ~115 GB/s (attempt-1's 92 GB/s at 32
  groups, raised by 3× the groups at lower memory) ≈ 154 µs +
  ~8 µs launch/hidden ≈ **162**, keeping attempt-1's win over
  compiled 235.7.
- **32 rows: 530.** 384 + 768 groups, already saturated:
  70.78 MB at ~135 GB/s (attempt-1 reached 132 GB/s with 128
  fused groups; the split's extra `x`/`hidden` traffic is <1% of
  weight bytes) ≈ 524 µs + ~6 µs ≈ **530**, keeping the win over
  compiled 791.6. Risk: if the second launch's serialization on
  the chain costs more than ~10 µs/layer at streaming sizes, this
  lands nearer attempt-1's 535.7 — still a win.

G2 prediction: pass at all six sizes with `differ == 0` — decode
reuses attempt-1's per-block math unchanged (only the block→group
mapping moved), and prefill reuses attempt-1's `qmm_accum`
verbatim, which already matched `gather_qmm_rhs` bit for bit.

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** (decode 1/8/32 and prefill 127/1,016/2,048) — the
split kept attempt-1's bits in both regimes: per-block
`qvm_block` order unchanged by the block→group remapping, and the
scalar `qmm_accum` prefill path matched `gather_qmm_rhs` again.
Chained timing µs/layer (this run; candidate vs reference vs
compiled): **1 row 109.5 / 93.1 / 86.9 (loses both)**; 8 rows
195.0 / 248.8 / 243.6 (wins); 32 rows 637.7 / 846.0 / 798.9
(wins). `beats_compiled` false — the 1-row size still misses, as
in attempt-1. Baseline `none` — no
`wins_vs_baseline`/`beats_baseline`.

Against attempt-1 (different run; 1-row baselines shifted
77.6 → 86.9 compiled between runs, so treat 1 row as indicative):
109.5 vs 104.3 at 1 row (no gain), 195.0 vs 193.3 at 8 rows
(flat), 637.7 vs 535.7 at 32 rows (**+19% regression**). The
hardening did not beat the best attempt at any size.

## Miss analysis (prediction 74 / 162 / 530; all misses > 20%)

Measured 109.5 / 195.0 / 637.7 is **+48% / +20% / +20%** over
prediction. Effective bandwidth on pair bytes: 20 / 91 / 111 GB/s
— the 1- and 8-row rates are attempt-1's (21 / 92 GB/s), and the
32-row rate falls between attempt-2's 80 and attempt-1's 132 GB/s.
Three facts together locate the error in the prediction's model,
not in the arithmetic:

1. **Group count is not the 1-row binding limit past 4 groups.**
   Attempt-1 (4 groups × 512 threads), attempt-2 (48+96 groups ×
   32 threads) and this attempt (12+24 groups × 128 threads, 4
   blocks in flight per group, 2,048/768 B threadgroup memory) all
   land at 103.6–109.5 µs/layer chained at 1 row, across a 36×
   range of group counts. The prediction assumed attempt-1's
   serial part scales with cores busy (4 → 12); instead the
   ~100 µs plateau is invariant to how the blocks are packed. What
   does not shrink under any packing is each block's own serial
   K chain (24 strided load-accumulate steps per lane for gate,
   24 for up, 12 for down, gate→up→down ordered by the hidden
   dependency) plus the chained launch/link fixed cost — and this
   attempt *adds* a second launch and a `hidden` round-trip on
   that chain. Splitting exposes cores the critical path cannot
   use.
2. **At 32 rows the split tax is real and the fused kernel's
   in-group overlap was load-bearing.** 637.7 vs attempt-1's
   535.7: the two-kernel form pays 1,152 group dispatches (vs
   128), 3× duplicated `tg_x` loads and 6× duplicated `tg_h`
   loads, two extra barrier rounds per pair, and the `hidden`
   write+read — small bytes (<1% of weight traffic) but pure
   latency/occupancy overhead at streaming sizes, while a fused
   512-thread group keeps 16 blocks' loads in flight behind one
   barrier set. Restoring 4 blocks/group and low threadgroup
   memory recovered most of attempt-2's loss (80 → 111 GB/s) but
   not attempt-1's 132 GB/s.
3. **The 8-row rate confirms the mechanism:** 91 GB/s equals
   attempt-1's 92 GB/s — once enough groups exist to stream, the
   packing makes no difference, and the prediction's assumed
   residency gain (115 GB/s from lower threadgroup memory) did
   not appear; residency was not the cap.

Lesson for `ineff-baseline-experts-2`: the 1-row lever is not
threadgroups per call (three packings, one plateau). The next
lever must shorten or overlap the per-block serial chain itself
— e.g. interleaving the independent gate and up K-loops so one's
loads hide the other's latency — or cut the chained fixed cost,
without touching any block's `simd_sum` order.

**Next change (one):** return to attempt-1's fused single kernel
and interleave gate/up: issue both blocks' K-loop loads per wave
before accumulating, keeping `qvm_block`'s per-block order and
all rounding points identical.
