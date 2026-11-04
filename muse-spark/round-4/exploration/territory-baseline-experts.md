# Territory: baseline-experts

- **Owner:** explorer-baseline-experts
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T21:43:42Z
- **Next:** none — checklist filled for all five pieces, 2 inefficiencies posted, no notes to answer

## What it does

The `experts` slot of one MoE layer: for each row `r` and slot
`j`, `y[r, j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]`
with `e = inds[r, j]`. The code in the slot is round 3's
`muse-spark/round-3/kernels/experts/attempt-1`: one fused per-pair
kernel (512 threads per pair, `rows*4` threadgroups) plus
`"max_rows": 59` in `spec.json`, so rows 1–59 run the fused kernel
and rows above 59 run the compiled reference. It passes G2
bit-exact at all six sizes, beats compiled at 8/32 decode rows,
and loses at 1 row — the only reason `beats_compiled` is false.

Round 3's `muse-spark/round-3/exploration/territory-baseline-experts.md`
and `ineff-baseline-experts-1/2.md` (both confirmed, frozen) and
`muse-spark/round-3/implementation/hyp-2-experts-prefill.md`
(frozen) mapped this code when it was attempt-7 plus the hand-off.
This entry cites them where they still hold and writes only what
is new for round 4: the `min_rows: 2` spec change ADR 0015 adds,
and where the fused kernel still spends its time now that 1 row
and prefill can both leave the kernel.

## What `min_rows: 2` with `max_rows: 59` changes (new)

The grader now supports a lower hand-off bound. `build()` reads
`min_rows` like `max_rows` (int ≥ 1, ≤ `max_rows` when both are
given, bools rejected) and the returned function hands off when
`rows > max_rows or rows < min_rows`
(`bench/grader/slots.py:330-343`); the brief states the same
contract (`claude/muse-agent-brief.md:174`). Adding
`"min_rows": 2` to the starting point's `spec.json` — no kernel
change — moves the kernel-visible window from 1–59 to **2–59**:

- **1 row: handed off.** The kernels no longer run there at all.
  G2 still checks the size, but through the attempt's own
  dispatch it runs the compiled reference, which was bit-exact
  with the reference at all six sizes in round 3
  (`muse-spark/round-3/kernels/experts/attempt-1/notes.md:75-78`),
  so G2 is unaffected. The piece timing reports the size
  `handed off` instead of timing compiled code against itself
  (`bench/grader/check_candidate.py:389-390`;
  `claude/muse-agent-brief.md:216`). The 1-row loss priced by
  round 3's `ineff-baseline-experts-2` (candidate 96.9 vs
  compiled 86.6 µs/layer, `grade.json` — +10.3 µs/layer =
  +0.21 ms per 20-layer decode step) disappears from the
  grader, not from the machine: with 1 request every decode
  step has 1 row, so this slot then runs MLX's own code for the
  whole 1-request workload and is exactly at its compiled bar
  there (`claude/muse-agent-brief.md:174`). It can pass G3
  only in a set whose other kernels are faster at 1 request.
  That composition question belongs to the `g3` territory; this
  territory only records that the experts contribution at
  1 request becomes 0 by construction.
- **Decode phase verdict flips (prediction).** `judge()` drops
  `handed off` sizes, calls a phase `faster` when every
  remaining size is and at least one remains, and sets
  `beats_compiled` when one phase is `faster` and none is
  `slower` (`bench/grader/check_candidate.py:253-273`). With
  1 row handed off, decode rests on 8 and 32 rows alone — both
  `faster` (29 wins each) in the starting point's run — and
  prefill rests on nothing (all three sizes already hand off
  above 59), so the predicted verdicts are decode `faster`,
  prefill `handed off`, **`beats_compiled` true**. This is a
  prediction from the grader's stated rules and the frozen
  `grade.json`; no `min_rows` attempt has been graded yet.
  It is the only ungraded direction in any slot (round-4
  exploration README, *What is left* item 1).
- **2–59 rows: kernels unchanged, byte for byte.** Rows 2–7 and
  9–31 were never timed, but they run the same `qvm` decode
  path graded at 1/8/32 (`block-0.metal:22,45-58`), per-pair
  with no cross-row interaction, so the G2 verdict extends to
  them by construction (round 3's boundary check on attempt-1,
  `hyp-2-experts-prefill.md` *Boundary checks*). Their speed
  is unmeasured; nothing about `min_rows: 2` changes it.
- **The 33–59 band: unchanged, still unmeasured.** Decode steps
  run 1–32 rows (`claude/muse-agent-brief.md:26`), so this band
  never occurs in decode. It can occur as the remainder of a
  prefill chunk (a chunk is at most 2,048 rows; a final chunk
  of 33–59 rows would run the candidate, not the compiled
  reference). Both sides take the `qvm` arithmetic there — the
  kernels branch on `x_shape[0] >= 60` (`block-0.metal:22`) and
  MLX switches its own kernel at the same 60 — so correctness
  is exact by the same argument as 1/8/32. Speed has never
  been timed at any size in 33–59. Extrapolation only
  (prediction, not measurement): at 32 rows the kernel runs
  4.33 µs per pair (554.7 µs ÷ 128 pairs, `grade.json`), so a
  33-row call (132 pairs) would land near 570 µs/layer if the
  per-pair rate held, still ahead of compiled's 32-row rate —
  but occupancy and cache behaviour at 132–236 threadgroups
  are unmeasured, and a slow remainder chunk would tax prompt
  time in G3 in a way the piece timing cannot see. No
  inefficiency is posted: no time loss is measured there.
  Builders who keep `max_rows: 59` should note the alternative
  `max_rows: 32` would hand the whole band off and forfeit
  nothing graded — the boundary window round 3 established was
  32–59 (`hyp-2-experts-prefill.md:79-91`), and 59 was chosen
  to keep the fast path where it wins, on unmeasured sizes.
- **The scalar branch stays dead — `min_rows` does not change
  it.** The `prefill` flag is `x_shape[0] >= 60`
  (`block-0.metal:22`). With `max_rows: 59` the kernels already
  never saw 60+ rows in round 3, so the `else` branches
  (`block-0.metal:59-72,103-112`) and `qmm_accum`
  (`block-1.metal:206-230`) were already unreachable at every
  kernel-visible size; narrowing the window to 2–59 leaves
  them exactly as dead. Round 3 kept them as a G2-exact
  fallback if `max_rows` were ever raised
  (`attempt-1/notes.md:19-24`). No inefficiency is posted for
  them: the branch is never taken, its runtime cost is one
  predictable test on `x_shape[0]`, and no measurement prices
  its compiled presence (register allocation, code size) at
  8/32 rows — round 2's binding ceiling was the two 32-float
  thread arrays `p`, `q` (`block-0.metal:42-43`), which the
  scalar branch does not widen (`qmm_accum` uses one local
  `acc`, `block-1.metal:219`). Deleting the branch and the
  also-unused `qvm_block` (`block-1.metal:22-58`, kept
  verbatim, its arithmetic copied into the pair helpers) is
  safe cleanup a builder may fold into a `min_rows` attempt —
  predicted runtime saving 0, to be confirmed by re-grading —
  but it is not time lost, so it gets no entry.

## Checklist

### A. `block-0.metal` — fused per-pair kernel body (112 lines)

Unchanged from round 3 §A; the delta is only which row counts
can reach it.

- **Callers:** `spec.json` kernel `fused_experts_branch` via
  `bench/grader/slots.py` `build()` / `make_moe_call`, as in
  round-3 territory §A. Shapes reaching the kernel today:
  decode 1/8/32 rows (4/32/128 pairs) and the ungraded 2–59
  band; prefill 127/1,016/2,048 never reach it (handed off
  above 59). Under `min_rows: 2`, 1 row stops reaching it too;
  graded kernel shapes shrink to 8/32 rows only.
- **Callees:** decode path (`x_shape[0] < 60`, the only path
  reachable at 2–59 rows): `qvm_gate_up_pair`
  (`block-0.metal:51-58`, simdgroups 0–11, 4 idle), then
  `qvm_down_pair` (`:95-102`, 12 simdgroups in one wave, 4
  idle). Prefill path (`>= 60`): scalar `qmm_accum` phases
  (`:59-72,103-112`) — unreachable under `max_rows: 59`, with
  or without `min_rows` (see above). Silu-mul inline in both
  branches (`:77-85`), 384 of 512 threads active.
- **Data:** as round-3 §A — 11 inputs, `y` `[rows, 4, 768]`
  bfloat16, `pair >> 2` → row, `e = inds[pair]`
  (`block-0.metal:16-21`); threadgroup `tg_x[768] +
  tg_gate/up/h[384]` = 3,840 B (`:24-27`); two kernel-scope
  32-float accumulators `p`, `q` (`:42-43`) shared across
  phases, at the 512-thread pipeline ceiling round 2 proved
  (a third array failed G1; round-3 territory §A).
- **Neighbours:** routing's `inds` before, combine after,
  `y[r,j]` positional in `inds` order — unchanged. A
  `min_rows` attempt changes no neighbour contract: handed-off
  rows return the compiled reference's `y`, bit-identical by
  the hand-off check.
- **Cost:** 1 kernel per layer call whenever the kernel runs.
  Measured, piece (chained µs/layer, starting point's own
  `grade.json`, candidate vs compiled): 1 row 96.9 vs 86.6
  (`slower`, 5 wins); 8 rows 168.9 vs 238.2 (`faster`, 29);
  32 rows 554.7 vs 883.4 (`faster`, 29); prefill `same` ×3 by
  construction (both sides run compiled code). Weight bytes
  per pair 552,960 B (arithmetic in §B below). Effective rate
  on pair bytes at 200 GB/s peak (`claude/GLOSSARY.md`): 1 row
  22.8 GB/s, 8 rows 104.8 GB/s, 32 rows 127.6 GB/s — see
  ineff entries for what the 8/32 remainder is.
- **Invariants:** G2 `differ == 0` at all six sizes
  (`grade.json` correctness). Four bfloat16 rounding points
  kept (`block-0.metal:56-57,84,100-101` and helpers). Branch
  threshold `x_shape[0] >= 60` (`:22`) equals MLX's own switch,
  so every kernel-visible count (now 1–59, proposed 2–59)
  agrees with the reference's `gather_qvm` arithmetic.

### B. `block-1.metal` — pair helpers, `qvm_block`, `qmm_accum` (230 lines)

- **Callers:** `block-0.metal` only. Live helpers:
  `qvm_gate_up_pair` (`block-1.metal:69-128`) and
  `qvm_down_pair` (`:137-195`). Dead under the hand-offs:
  `qvm_block` (`:22-58`, unused replica kept verbatim) and
  `qmm_accum` (`:206-230`, scalar prefill mimic, reachable
  only at rows ≥ 60 the dispatcher never delivers).
- **Callees:** `simd_sum` trees only — two per pair-helper
  call, four per pair in total (`:120-127,187-194`).
  `qmm_accum` calls nothing.
- **Data:** per pair, the expert's full weights are streamed
  independently: gate/up/down words at per-pair offsets
  `eW = e * 36864`, `eS = e * 9216` (`block-0.metal:34-35`).
  Weight bytes per pair: each product reads
  `K × Ww × 4 B` words + scales/biases — gate/up
  768×48×4 = 147,456 B words + 768×12×2×2 = 36,864 B
  scales+biases = 184,320 B; down 384×96×4 = 147,456 B +
  384×24×2×2 = 36,864 B = 184,320 B; total
  **3 × 184,320 = 552,960 B per pair**, as round 2 §A.
  Two pairs selecting the same expert read those bytes twice;
  nothing in the kernel sorts, groups or shares across pairs.
  Costed as `ineff-baseline-experts-1`.
- **Neighbours:** helpers read `tg_x` (gate/up) or `tg_h`
  (down) after the barriers at `block-0.metal:32,73,86`;
  stores to disjoint `y` slices per pair. No atomics, no
  cross-pair communication — which is exactly why re-reads
  persist.
- **Cost:** per thread on the decode path: gate/up 24 strided
  steps (`K=768`, `k = lane; k < K; k += 32`), down 12 merged
  steps (`K=384`); each step loads 2×(2 scale/bias + 4 weight
  words) across the pair helpers and runs 64 dependent
  accumulates. Per pair: 884,736 MACs (gate 384×768 + up
  384×768 + down 768×384) = 1.77 MFLOP. At 32 rows the kernel
  sustains 128 pairs × 1.77 MFLOP ÷ 554.7 µs ≈ 0.41 TFLOPS,
  ~8% of the 5.2 TFLOPS peak (`claude/GLOSSARY.md`) —
  overhead-bound, not compute-bound. The serial structure
  behind that rate is `ineff-baseline-experts-2`.
- **Invariants:** per-block accumulation order is `qvm_block`'s
  (strided `k`, same term sequence, same `simd_sum` tree), so
  decode bits are exact by construction and G2 confirms it.
  `qmm_accum`'s contiguous `BK=32` order with bfloat16-rounded
  dequant (`:220-227`) remains the correct prefill arithmetic
  — merely unreachable while `max_rows: 59` stands.

### C. `spec.json` — launch spec (1 line)

- **Callers:** `bench/grader/slots.py:330-343` reads
  `min_rows`/`max_rows`; `check_candidate.py` times or hands
  off per size (`:389-390`) and judges phases with handed-off
  sizes dropped (`:253-273`).
- **Callees:** one kernel, `fused_experts_branch`
  (`source_block: 0`, `header_block: 1`), grid
  `["rows*2048", 1, 1]` = `rows*4` threadgroups of
  `[512, 1, 1]`; `slot_outputs: {"y": "y"}`.
- **Data:** 11 slot inputs in order (`x`, `inds`, `gate_w`,
  `up_w`, `down_w`, `gate_scales`, `gate_biases`, `up_scales`,
  `up_biases`, `down_scales`, `down_biases`); output `y`
  `[rows, 4, 768]` bfloat16; `atomic_outputs: false`.
  Currently `"max_rows": 59` and no `min_rows`. The round-4
  change is one added key, `"min_rows": 2` (2 ≤ 59, valid);
  kernels, grid and threadgroup are untouched, so G1's
  512 ≤ 1,024 threads and 3,840 B ≤ 32 KB hold as before.
- **Neighbours:** the spec is the slot contract — a builder
  attempt that adds `min_rows` must keep `max_rows: 59`: the
  32 ≤ `max_rows` ≤ 59 window round 3's boundary established
  (`hyp-2-experts-prefill.md:79-91`) still applies, now read
  as: kernels run only where they replicate the reference's
  arithmetic *and* win — 2–32 graded, 33–59 exact-but-untimed.
- **Cost:** the spec itself launches nothing; it routes.
  Today rows 1–59 cost the kernel's time (§A) and rows > 59
  cost compiled time. With `min_rows: 2`, row 1 also costs
  compiled time (86.6 µs/layer in this run's bar).
- **Invariants:** hand-off bounds are ints ≥ 1, `min_rows ≤
  max_rows`, bools rejected (`slots.py:331-335`) — `2` and
  `59` pass. G2 checks handed-off sizes through the dispatch,
  so bit-exactness at 1 row and prefill is inherited from the
  compiled reference, not waived
  (`claude/muse-agent-brief.md:175`).

### D. `notes.md` — round-3 attempt-1 notes (142 lines)

- **Callers/callees:** documentation only; records that the
  kernels are byte copies of round-2 attempt-7 and the single
  change is `"max_rows": 59` (`notes.md:11-24`).
- **Data:** the hand-off window argument (32 ≤ `max_rows` ≤
  59, `notes.md:37-52`), predictions (93.8/160.1/510.1 decode,
  = compiled prefill, `:54-85`) and the verdict against them
  (`:97-135`): decode misses +3.3%/+5.5%/+8.7%, all inside the
  20% bar and attributed to run-to-run noise since the kernels
  are byte-identical at rows ≤ 32 and the compiled bar itself
  moved between runs.
- **Neighbours:** names hyp-4 (1-row decode on this attempt)
  as the next change (`:137-142`). Round 4 supersedes that
  direction: the spec surface now hands 1 row off instead of
  closing its gap in-kernel, and round 3's two in-kernel fixes
  (attempt-2 vectorised scalar, bought nothing; attempt-5
  unfused, 2× worse) are on the round-4 README's refuted list.
  No builder should retry them.
- **Cost:** none itself. Its unmeasured list (`:87-95`) —
  kernel speed at 33–59 rows, sizes above 2,048, other
  distributions — carries into round 4 unchanged.
- **Invariants:** records the scalar branch as deliberately
  kept dead-code fallback (`:19-24`); see §B/§What-changes
  for its round-4 status (still dead, still costless as far
  as anyone has measured).

### E. `grade.json` — starting point's grader verdict

- **Callers:** produced by `bench/grader/check_candidate.py`
  on this attempt; the round-4 README's starting-point
  numbers quote it.
- **Data:** `g1: true`, `g2: true`, `beats_compiled: false`;
  correctness `differ 0 / max_ulp 0.0` at all six sizes;
  chained µs/layer candidate/reference/compiled with wins and
  per-size verdicts; `phases_vs_compiled`: decode `slower`
  (the 1-row size alone), prefill `not slower`.
- **Cost (the numbers every round-4 claim stands on):**
  1 row 96.9 vs compiled 86.6 (`slower`, 5 wins); 8 rows
  168.9 vs 238.2 (`faster`, 29 wins); 32 rows 554.7 vs 883.4
  (`faster`, 29 wins); prefill 127/1,016/2,048 `same`
  (19/15/16 wins — identical code both sides). Margins:
  −10.3 / +69.3 / +328.7 µs/layer = −0.21 / +1.39 / +6.57 ms
  per 20-layer decode step at 1/8/32 requests (prediction
  arithmetic, ×20).
- **Neighbours:** absolute chained times drift up to ~35%
  between runs on this machine (round-3 lesson), so only
  same-run paired verdicts and margins transfer. A graded
  `min_rows: 2` attempt must be compared in its own run.
- **Invariants:** this file is why the starting point is no
  baseline — `beats_compiled` false on the 1-row size alone,
  and no G3 pass. The `min_rows` prediction in *What changes*
  is derived from this file plus `judge()`'s rules, and stays
  a prediction until a builder grades it.

## Where the fused kernel still spends its time at 8/32 rows (new costing)

With 1 row handed off and prefill already handed off, 8 and 32
rows are the only sizes the kernel is still graded on — and
the only decode sizes where a set still needs its margin
(round-4 README *What is left* items 1–2). On pair bytes:

| Rows | Pairs | Pair bytes/layer | Candidate | Byte floor (200 GB/s) | Effective | Above floor |
|---|---|---|---|---|---|---|
| 8 | 32 | 17.69 MB | 168.9 µs | 88.5 µs | 104.8 GB/s | 80.4 µs (48%) |
| 32 | 128 | 70.78 MB | 554.7 µs | 353.9 µs | 127.6 GB/s | 200.8 µs (36%) |

Candidate, compiled and floors are from `grade.json` and
§B's 552,960 B/pair; floors are arithmetic at peak bandwidth
(`claude/GLOSSARY.md`). Two distinct losses fill the time:

1. **Bytes read more than once** — the pair traffic itself
   counts every duplicate expert selection again. Posted as
   `ineff-baseline-experts-1`.
2. **Time above even that inflated byte floor** — the serial
   K-chain, 4-of-16 idle simdgroups, phase barriers and the
   silu chain leave the kernel at 52–64% of peak bandwidth
   and ~8% of peak compute. Posted as
   `ineff-baseline-experts-2`. Round 3's `ineff-…-2` priced
   this chain only at 1 row and explicitly costed 8/32 at 0
   (already ahead); what is new here is pricing the same
   structure as the margin limiter at the two surviving
   sizes, where widening the win is now the whole game.

Per-pair time falls from 24.2 µs at 1 row to 5.28 µs at 8 and
4.33 µs at 32 (`grade.json` ÷ pairs): more threadgroups hide
the chain's latency, which is why the kernel wins at 8/32
despite it. Neither loss is removable by the spec surface —
both need a kernel change that keeps every accumulation order
bit-exact — and grouped designs that remove loss 1 have
historically paid it back on loss 2 (round-1 grouped attempts
ran at a ~55 GB/s duty-cycle ceiling and lost at 8 rows;
round-2 `ineff-baseline-experts-3`, confirmed, frozen). The
entries keep the two savings separate and un-added.

## Relations

- **Fed by / feeds:** routing's `inds` and post-attention `x`
  in; `y` to combine, positionally. Unchanged from round 3.
- **Grader territory** owns the `min_rows` mechanism
  (`slots.py` `hands_off`, `check_candidate.py` `judge`) —
  this territory prices what flipping it buys this slot.
- **G3 territory** owns whether the flipped slot composes:
  at 1 request this slot becomes neutral, so the set's
  1-request win must come from routing/combine; the untested
  three-slot interaction (round-4 README item 3) is theirs.
- **Prior entries reused, not re-posted:** round 3
  `ineff-baseline-experts-1` (scalar prefill, confirmed) is
  removed in full by the existing `max_rows: 59` hand-off —
  closed by construction, nothing new to say. Round 3
  `ineff-baseline-experts-2` (1-row chain, confirmed) is
  removed from the grader by `min_rows: 2` and from the
  kernel's graded life entirely; its chain structure survives
  at 8/32 in the new costing of this round's
  `ineff-baseline-experts-2`. Round 2
  `ineff-baseline-experts-3` (per-pair re-reads, confirmed)
  persists byte-for-byte; this round's
  `ineff-baseline-experts-1` restates it on the starting
  point's chained numbers with the 1-row/prefill sizes now
  handed off.

## Inefficiencies

- `ineff-baseline-experts-1` (open): per-pair weight re-reads
  at 8/32 rows — 552,960 B streamed per pair with no sharing
  across pairs that select the same expert.
- `ineff-baseline-experts-2` (open): the serial K-chain and
  idle simdgroups hold 8/32 rows 80–201 µs/layer above their
  own byte floor — the remaining margin limiter once both
  hand-offs stand.

## Files read

- `muse-spark/round-3/kernels/experts/attempt-1/block-0.metal`,
  `block-1.metal`, `spec.json`, `notes.md`, `grade.json` —
  §A–§E, all kernel/spec/verdict numbers.
- `muse-spark/round-3/exploration/territory-baseline-experts.md`,
  `ineff-baseline-experts-1.md`, `ineff-baseline-experts-2.md`
  (frozen) — prior checklist and both confirmed entries.
- `muse-spark/round-3/implementation/hyp-2-experts-prefill.md`
  (frozen) — hand-off window 32–59, attempt-1 boundary check,
  failed 1-row/prefill directions.
- `muse-spark/round-2/exploration/ineff-baseline-experts-3.md`
  (frozen, confirmed) — per-pair re-read counters and the
  grouped-design bar this round's entry 1 stands on.
- `bench/grader/slots.py:307-367` (`build`, `hands_off`),
  `bench/grader/check_candidate.py:253-273,389-390`
  (`judge`, handed-off timing) — the `min_rows` mechanics.
- `claude/adr/0015-sets-and-hand-off-below.md`,
  `claude/muse-agent-brief.md:173-176,211-217` (`min_rows`
  contract, handed-off verdicts), `claude/GLOSSARY.md`
  (peak limits, overhead-bound).
- `muse-spark/round-4/exploration/README.md` — starting
  point, *What is left* items 1–3, refuted claims.

## From other agents

- explorer-grader, 2026-11-04T21:44:11Z — from [territory-grader.md](territory-grader.md) *Worked example*: with `min_rows: 2, max_rows: 59` the piece verdict rests entirely on 8 and 32 rows (1 handed off, 127/1,016/2,048 handed off); decode is `faster` only if both 8 and 32 are `faster`, and rows 2–7 plus 33–59 run the kernels ungraded — carried as **Unverified**. The hand-off's side of 60 is proved by `check_sizes.py` (59/60/61), never by the piece verdict.
