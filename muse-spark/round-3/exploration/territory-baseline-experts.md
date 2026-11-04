# Territory: baseline-experts

- **Owner:** explorer-baseline-experts
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T17:39:32Z
- **Next:** none — checklist filled, 2 inefficiencies posted, no notes to answer

## What it does

The `experts` slot of one MoE layer: for each row `r` and slot `j`,
`y[r, j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]`
with `e = inds[r, j]`. The code in the slot is now the round-2
starting point `muse-spark/round-2/kernels/experts/attempt-7`: one
fused per-pair kernel (512 threads per pair, `rows*4` threadgroups)
with interleaved gate/up and interleaved down behind a `rows >= 60`
branch — a `qvm` replica below 60 rows, a scalar `qmm_accum`
(`bfloat16`-rounded dequant, contiguous `BK=32` order) at/above.
It passes G2 bit-exact at all six sizes, beats compiled at 8/32
decode rows, and loses at 1 row and at every prefill size
(15–28× slower than compiled).

Round 2's `muse-spark/round-2/exploration/territory-baseline-experts.md`
(§A–§F, frozen) mapped the attempt-4 starting point, the reference,
and both MLX kernels; its three inefficiency entries are confirmed
and frozen. This entry cites them where they still hold and writes
only what is new: the attempt-7 code as slot code, the two-phase
timing that did not exist in round 2, and why the starting point
is no baseline.

## Checklist

### A. `block-0.metal` — fused per-pair kernel body (attempt-7)

Delta on round-2 §A (attempt-4 body).

- **Callers:** `spec.json` kernel `fused_experts_branch` (see §C),
  via `bench/grader/slots.py:303-345` (`build`) and
  `slots.make_moe_call` (`slots.py:97-119`), as in round-2 §A.
  Shapes new: prefill 127/1,016/2,048 rows → 508/4,064/8,192 pairs
  (`slots.py:214-254`); decode 1/8/32 → 4/32/128 pairs. Grader
  times all six sizes chained (`check_candidate.py:345-368`).
- **Callees:** decode (`x_shape[0] < 60`): `qvm_gate_up_pair`
  (`block-0.metal:53-55`, one wave, simd 0–11, 4 idle) then
  `qvm_down_pair` (`:98-99`, 12 pairs in one wave, 4 idle).
  Prefill (`>= 60`): scalar `qmm_accum` — gate and up as two
  sequential `simd < 12` phases (`:61-71`), down in 2 waves
  (`:105-111`, wave 1 uses simdgroups 8–11 only). Silu-mul inline
  in both branches (`:77-85`), 384 of 512 threads active.
- **Data:** inputs/outputs unchanged from round-2 §A (11 inputs,
  `y` `[rows, 4, 768]` bfloat16 positional, `pair >> 2`, `e =
  inds[pair]`, `block-0.metal:16-21`). Threadgroup memory
  unchanged: `tg_x[768] + tg_gate/up/h[384]` = 3,840 B
  (`:24-27`). Thread registers new: two kernel-scope 32-float
  accumulators `p`, `q` (`:42-43`) shared across phases —
  attempt-6 proved three distinct arrays fail G1 at 512 threads
  (pipeline max 448; `attempt-7/notes.md:10-15`).
- **Neighbours:** unchanged from round-2 §A (routing `inds`
  before, combine after, any order/duplicates accepted).
- **Cost:** 1 kernel per layer call at every size. Grid
  `rows*2048` threads ÷ 512 = `rows*4` groups (4 at 1 row,
  8,192 at 2,048 rows). Weight bytes per pair unchanged at
  552,960 B (`attempt-7/notes.md:119-121`; round-2 §A
  arithmetic). Measured chained, this run
  (`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`,
  candidate vs compiled): 92.1 vs 78.6 (1 row), 162.8 vs 235.5
  (8), 515.3 vs 797.8 (32), 15,220.0 vs 960.8 (127), 119,154.4
  vs 4,306.0 (1,016), 240,681.1 vs 11,998.9 (2,048) µs/layer.
  Effective on pair bytes: 24.0 / 108.7 / 137.3 GB/s decode,
  18.5 / 18.9 / 18.8 GB/s prefill (200 GB/s peak,
  `claude/GLOSSARY.md:10`).
- **Invariants:** G2 passes with `differ == 0`, `max_ulp 0.0` at
  all six sizes (grade-phases file, `correctness`). The four
  bfloat16 rounding points kept (`:56-57,64,70,84,100-101,109`).
  Branch threshold `x_shape[0] >= 60` (`:22`) is the dispatch
  source value (round-2 §E/§F, frozen).

### B. `block-1.metal` — pair helpers and `qmm_accum`

Delta on round-2 §B (`qvm_block` replica).

- **Callers:** `block-0.metal` as in §A above. `qvm_block`
  (`block-1.metal:22-58`) is kept verbatim but now unused; its
  arithmetic is copied into the pair helpers
  (`block-1.metal:8-10`).
- **Callees:** `simd_sum` trees only (`:120-127,187-194`), one
  per accumulator (two per pair call). `qmm_accum` (`:206-230`)
  calls nothing: one thread per output, `BK=32` tiles in
  contiguous K order, `wq` rounded to bfloat16 before the
  multiply (`:225-226`).
- **Data:** per pair helper call two 32-output blocks; per
  `qmm_accum` call one output (`word = wb + (lane >> 3)`,
  `shift = (lane & 7) * 4`, `:217-218`), so 8 lanes share each
  weight word and all 32 share `S`/`B`/`xv[k]` per iteration.
- **Neighbours:** helpers read `tg_x` (gate/up) or `tg_h`
  (down) after barriers, as in round-2 §B. No barrier between
  the gate and up prefill phases (disjoint outputs,
  `block-0.metal:61-72`); barriers after `tg_x` (`:32`),
  after gate/up (`:73`), after silu (`:86`).
- **Cost:** decode per thread: gate/up 24 strided steps
  (`K=768`), down 12 merged steps (`K=384`), each step ~2×(2
  scale/bias + 4 weight words) + 64 dependent accumulates.
  Prefill per pair: 884,736 single-thread iterations
  (gate/up 384×768 each, down 768×384), each 1 W + 1 S + 1 B
  device load + 1 `tg` load on one serial accumulator —
  ~7.1 MB of scalar load issues per pair for 553 KB of
  unique weight bytes (8× W, 32× S/B redundancy; see
  ineff-baseline-experts-1). Serial depth per thread
  (simd < 8): 768 + 768 + 768 = 2,304 dependent iterations.
- **Invariants:** per-block decode order is `qvm_block`'s
  order (strided `k = lane, lane+32, …`, same term sequence,
  same `simd_sum`), so decode bits equal attempt-5's by
  construction; prefill order is contiguous per-output K,
  which G2 confirms equals `gather_qmm_rhs` bits (`differ
  0`) — the round-3 README *Refuted claims* records that
  scalar sums reproduce this MMA rounding.

### C. `spec.json` — launch spec

Delta on round-2 §C: one kernel entry renamed
`fused_experts_branch`, grid `["rows*2048", 1, 1]`,
threadgroup `[512, 1, 1]` — still `rows*4` threadgroups at
every row count. All 11 slot inputs, output `y`
`[rows, 4, 768]` bfloat16, no atomics. Limits hold by hand:
512 ≤ 1,024 threads, 3,840 B ≤ 32 KB
(`attempt-7/notes.md:125-131`).

### D. `notes.md` + `hyp-1-experts-exact.md` — what round 2 built and learned

New (no round-2 counterpart in this territory; the entries
lived in implementation).

- Six fused exact packings graded chained (own runs):
  attempt-1 104.3 (512-wide), attempt-2 103.6 (split
  per-block groups — 8/32 rows regressed hard), attempt-3
  109.5 (two kernels, 4 blocks/group — recovered streaming,
  1 row unmoved), attempt-4 94.0 (gate/up interleave broke
  the plateau at latency sizes), attempt-5 92.1 (parallel
  stores; restored 32-row rate), attempt-7 93.8 (down
  interleave via shared `q`; latency-neutral at 1 row, new
  best at 32 rows). Attempt-6 failed G1 (512 > 448). Full
  per-attempt numbers in the notes recorded as builder notes
  on round-2 `ineff-baseline-experts-2.md:120-251` (frozen)
  and `attempt-7/notes.md` (frozen).
- Two load-bearing lessons: (1) merging blocks helps only
  where they were sequential phases sharing an activation
  read (gate/up); down's blocks already overlapped across
  the wave, so the merge trades cross-simdgroup for
  in-step parallelism and is neutral at 4 groups
  (`attempt-7/notes.md:267-276`). (2) Register headroom at
  512 threads is exactly two 32-float accumulators.
- Round 2's verdict: hypothesis dropped, slot keeps the
  reference; "what I'd try next" is a reference-shaped
  unfused structure at 1 row — "every fused packing
  plateaued at 92–110 µs/layer chained, so stop fusing"
  (`muse-spark/round-2/summary.md:9`).

### E. `grade-phases-experts-attempt-7.json` — two-phase timing

New: round 2 timed decode only
(`claude/adr/0010-exact-at-every-size.md:42`, frozen);
Claude's re-grade on main times all six sizes chained
(measured, piece). Candidate vs compiled µs/layer: 1 row
92.1 vs 78.6 (`slower`, 1/30 wins); 8 rows 162.8 vs 235.5
(`faster`); 32 rows 515.3 vs 797.8 (`faster`); 127 rows
15,220.0 vs 960.8 (`slower`, 15.8×); 1,016 rows 119,154.4
vs 4,306.0 (27.7×); 2,048 rows 240,681.1 vs 11,998.9
(20.1×). `beats_compiled` false: decode `slower` (the
1-row size), prefill `slower` (every size). G2 `pass`
everywhere with `differ 0`. Candidate cost per pair per
layer is constant at prefill: 29.96 / 29.32 / 29.38 µs
(15,220/508, 119,154/4,064, 240,681/8,192) — the branch
scales linearly with pairs at ~29.5 µs/pair.

### F. Engine comparison — what the starting point does to whole requests

New: Claude timed attempt-7 anyway though round 2 dropped
it (round-3 README *Starting point*, per slot `experts`).
Per decode step in vLLM vs the unchanged model: +0.29 ms
(+5.4%) with 1 request, −1.22 ms (−8.4%) with 8, −5.35 ms
(−17.5%) with 32 (`console-vllm.txt`). Over whole requests
(with routing attempt-4): throughput −10.0% / −37.5% /
−48.0% / −82.5%, prompt processing 6.63× / 14.50× / 12.50×
/ 13.09× slower
(`results/runs/2026-11-04-engine-comparison/`). The 1-row
piece gap (+13.5 µs/layer × 20 = 0.27 ms/step this run;
+19.0 × 20 = 0.38 ms own-run) reproduces the +0.29 ms G3
effect to within run shift — no second in-step cause is
needed at 1 request. The prompt multiples are the prefill
branch (§B/E): every prompt chunk here has ≥ 128 rows, so
it always runs `qmm_accum`.

## Relations

- **Fed by / feeds:** unchanged from round-2 Relations
  (routing `inds`, post-attention `x`; combine consumes `y`
  positionally; shared expert independent).
- **Reference internals:** round-2 §E/§F (frozen) still own
  the reference and both MLX kernels; round-2
  `territory-quantized-kernels.md` §D/§E (frozen) own the
  prefill rounding points. This round's `quantized-kernels`
  territory owns prefill *speed* (tiling, occupancy); the
  `max_rows` hand-off alternative and its bit-exact check
  (`results/runs/2026-11-04-handoff-check/`) belong to the
  `grader` territory's `check_ready.py` / hand-off grading
  map — this territory prices the gap the hand-off would
  bank, not the hand-off mechanism.
- **Prior entries reused:** round-2 `ineff-baseline-experts-1`
  (G2 gate) is removed by the branch — kept, not re-posted;
  `ineff-baseline-experts-3` (per-pair re-reads, confirmed)
  persists unchanged (attempt-7 keeps 552,960 B/pair) and
  is carried by round-3 README item 6, not duplicated here;
  `ineff-quantized-kernels-1/2` (confirmed) price reference
  prefill waste a fast branch takes for free (round-3
  README item 8).

## Inefficiencies

- `ineff-baseline-experts-1` (open): scalar prefill
  branch 15–28× slower than compiled — serial dependent
  chain, scalar redundant loads, no cross-row reuse.
- `ineff-baseline-experts-2` (open): 1-row decode chain
  gap — serial K-chain + structural cost behind 4 groups
  at the register ceiling.

## Files read

- `muse-spark/round-2/kernels/experts/attempt-7/block-0.metal`
  (112 lines), `block-1.metal` (230 lines), `spec.json`,
  `notes.md` — §A–§D.
- `results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`
  — §E (all chained numbers in this entry).
- `muse-spark/round-2/implementation/hyp-1-experts-exact.md:1-120`
  (Inefficiency, Boundary) — thresholds, two arithmetics.
- `muse-spark/round-2/kernels/experts/attempt-1/notes.md:10-49`
  — `qmm_accum` rationale, ~20 µs fixed-cost model.
- `muse-spark/round-2/summary.md:6-10` — dropped verdict,
  unfused-structure suggestion.
- `bench/grader/check_candidate.py:186-236,248-368` (`link`,
  `chained`, `timed`, two-phase judgment) and
  `bench/grader/slots.py:211-254` (prefill capture) — §A, §E.
- `claude/GLOSSARY.md` (peak limits, overhead-bound),
  `claude/muse-agent-brief.md` (slot contract, per the
  round-2 entries), round-3 `exploration/README.md`
  (starting point, leftovers, refuted claims).
- Round 2 (cited, frozen, not rewritten):
  `territory-baseline-experts.md` §A–§F,
  `ineff-baseline-experts-1/2/3.md`,
  `territory-quantized-kernels.md` §D/§E,
  `ineff-quantized-kernels-1/2.md`.

## From other agents

explorer-quantized-kernels, 2026-11-04T17:41:57Z — the scalar-prefill
gap from the reference side (for your ineff-baseline-experts-1; full
speed anatomy in my `territory-quantized-kernels.md` §S1–§S7, this
round). Attempt-7's `rows>=60` branch is bit-exact and 15.8/27.7/20.1×
slower than compiled (15,220/119,154/240,681 vs 960.8/4306.0/11998.9
µs/layer, `grade-phases-experts-attempt-7.json`). Three compounding
causes: (1) NO CROSS-ROW GROUPING — `qmm_accum`
(`block-1.metal:206-230`) loads each pair's full expert weights
through its own 512-thread group (each lane re-loads its block's
words; shared only via cache): distinct ≈186 KB/pair/product × B × 3
= 283 MB / 2.27 GB / 4.57 GB per layer vs MLX's grouped T·184,320·3 =
33/166/299 MB — ratios 8.5×/13.7×/15.3×, matching measured once rates
fold in (attempt-7 ≈9–10% of peak BW on distinct bytes: 283MB/15.2ms
= 18.6 GB/s at 127 rows; MLX ≈13–20% — my §S5). MLX's T-vs-B grouping
(round-2 territory §C) is the single biggest speed fact. (2) SCALAR
SERIAL K-CHAINS — each output accumulates K=768/384 terms one FMA at a
time behind a dependent device-load chain (word → shift → S/B → round
→ FMA), vs MLX's 8×8 `simdgroup_matrix` MMAs over threadgroup tiles
(my §S4: 32 MMAs = 32,768 flops per group per K-iteration). (3)
PER-PAIR 512-THREAD GRIDS at prefill scale (8,192 groups at 2,048
rows) re-loading x and re-paying barriers per pair, vs MLX's 64-thread
3.84 KB groups tiling the same work. A fast exact branch must match:
cross-row weight sharing (≈22/14/12 KB distinct/pair, MLX's T·184KB/B), HW
MMA, coalesced lines; your `qmm_accum` order (contiguous BK=32 + bfloat
dequant) is already the right arithmetic — keep the order, change the
engine. Or skip with `max_rows` 59 (Claude-verified bit-exact, `same`
at all prefill sizes, `results/runs/2026-11-04-handoff-check/`) —
banks the full §S6 gap at prefill, keeps the 1-row decode loss.
- explorer-g3, 2026-11-04T17:41:19Z — measurement-side companion to your ineff-baseline-experts-1: ineff-g3-3 shows the prefill piece excess composes additively to prompt time (chunks x 20 layers within 2-15% at all four workloads; routing+experts prompt excess +282ms/+2.7s/+9.2s/+78.9s). A max_rows hand-off banks exactly that excess; decode wins kept, not added.
