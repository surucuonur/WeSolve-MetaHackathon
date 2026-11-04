# Exploration: round 3 — analysis

**Version:** v2 · 2026-11-04T17:41:39Z · skill version 8

## Tracker

| Territory | Agent | Status | Inefficiencies | Updated |
|---|---|---|---|---|
| `baseline-experts` | `explorer-baseline-experts` | done | `ineff-baseline-experts-1` (open), `-2` (open) | 2026-11-04 17:41Z |
| `baseline-routing` | `explorer-baseline-routing` | done | `ineff-baseline-routing-1` (open, premature confirm flagged), `-2` (open, premature confirm flagged) | 2026-11-04 17:41Z |
| `baseline-combine` | `explorer-baseline-combine` | exploring | — | 2026-11-04 17:36Z |
| `g3` | `explorer-g3` | done | `ineff-g3-1` (open), `-2` (open), `-3` (open) | 2026-11-04 17:41Z |
| `quantized-kernels` | `explorer-quantized-kernels` | exploring | — | 2026-11-04 17:36Z |
| `grader` | `explorer-grader` | done | — (notes on baseline-*, g3) | 2026-11-04 17:41Z |

## Starting point

Round 3 builds on rounds 1 (`muse-spark/`, frozen, tagged `muse-round-1`) and 2 (`muse-spark/round-2/`, frozen, tagged `muse-round-2`) as measured independently by Claude in `claude/findings/muse-round-2.md` (2026-11-04, on `main` after merging round 2 at `4033cae`). Round 2 ran skill version 5 as Muse Wonder. The judgment changed for round 3 ([ADR 0011](../../../claude/adr/0011-decode-and-prefill.md)): an improvement must hold in both phases, decode and prefill — faster in one, slower in neither — against the unchanged model and the compiled reference, in the piece timing (`beats_compiled`) and in G3, where no whole request may be slower. A slot's baseline is its best attempt so far that passes the current G2, has `beats_compiled` true, and passes G3; anything else that passes G2 is a starting point. **No slot has a baseline.** A new attempt must pass G1 and G2 (bit-exact at 1, 8, 32 and 127, 1,016, 2,048 rows) and beat the compiled reference in the chained two-phase timing; there is no `--baseline` to pass. Inside the round, the lead's ready check (`bench/grader/check_ready.py`, [ADR 0013](../../../claude/adr/0013-measure-what-you-claim.md)) times every candidate exactly as Claude's G3 will, and only its passing verdict makes an attempt `ready`.

Round 2's final attempts, their piece timings and G3 effects are summarised per slot. Piece timings are Claude's re-grade on main with the two-phase grader (`results/runs/2026-11-04-muse-round-2/grade-phases-*.json`), chained µs per layer as reference / compiled / candidate. G3 is the vLLM decode-step change against the unchanged model (median of 3 shuffled repeats, `console-vllm.txt`) plus the whole-request throughput change at 1×128, 8×128, 32×128 and 32×1,024 (prompt-time multiple in brackets, `results/runs/2026-11-04-engine-comparison/`). Round 2's own prediction was decode-only (piece basis × 20 layers plus compiled G3 offsets); both are quoted so the gap can be seen.

### Per slot

#### `experts` — starting point `muse-spark/round-2/kernels/experts/attempt-7`

- **What it is.** One fused per-pair kernel (512 threads per (row, expert-slot) pair, `rows*4` threadgroups): interleaved gate/up plus interleaved down in a merged K loop, intermediates in threadgroup memory (3,840 B), full-bfloat16 silu-mul. Per-block numerics replicate MLX's `affine_gather_qvm` bitwise at decode sizes; a `rows >= 60` branch runs a scalar `qmm_accum` (bfloat16-rounded dequant, contiguous `BK=32` order) to match `gather_qmm_rhs` at prefill sizes. See `muse-spark/round-2/kernels/experts/attempt-7/notes.md` and `muse-spark/round-2/implementation/hyp-1-experts-exact.md`.
- **Piece timing** (`grade-phases-experts-attempt-7.json`). Decode: 84.0 / 78.6 / **92.1** at 1 row (slower), 239.3 / 235.5 / **162.8** at 8 rows, 806.5 / 797.8 / **515.3** at 32 rows. Prefill: 963.4 / 960.8 / **15220.0** at 127 rows (15.8× compiled), 3979.3 / 4306.0 / **119154.4** at 1,016 rows (27.7×), 7359.8 / 11998.9 / **240681.1** at 2,048 rows (20.1×). `beats_compiled` false: decode `slower` (the 1-row size), prefill `slower` (every size).
- **G3 effect vs prediction.** Round 2 dropped this attempt and predicted nothing for it; Claude timed it anyway. Per decode step in vLLM: +0.29 ms (+5.4%) with 1 request, −1.22 ms (−8.4%) with 8, −5.35 ms (−17.5%) with 32. Over whole requests (with routing attempt-4): throughput −10.0% / −37.5% / −48.0% / −82.5%, with prompt processing 6.63× / 14.50× / 12.50× / 13.09× slower than the unchanged engine. The fastest exact decode step measured (routing + experts: −20% with 32 requests in plain MLX) is the slowest whole request.
  - **Reason (measured).** The scalar prefill branch copies `gather_qmm_rhs`'s rounding but not its speed: every prompt chunk here has at least 128 rows, so it always runs that branch. At 1 row the fused designs plateau at 92–110 µs per layer chained against 73–87 for the compiled reference across six exact attempts — width, group count, interleave and store parallelism moved 8/32 rows and none moved 1 row by more than ~12 µs (round 2's *What building taught*).
- **What keeps it from being a baseline.** It passes G2 bit-exact at all six sizes, but it is slower than the compiled reference at 1 row and at every prefill size, so prompts take 6.6 to 14.5 times as long and whole requests lose 10% to 83% of throughput.
- **Where it still spends its time.** At 1 row: the ~19 µs gap to compiled is the `tg_x` load, silu, the two barriers and chained fixed cost — the down interleave proved the down waves were already hidden across simdgroups (attempt-7 notes, miss analysis). At prefill sizes: the scalar `qmm_accum` branch, 15,220 µs per layer where compiled takes 961. Per-pair weight re-reads remain at all sizes (552,960 B per pair).

#### `routing` — starting point `muse-spark/round-2/kernels/routing/attempt-4`

- **What it is.** One fused kernel per layer: scalar sums over the same partition/K order as MLX's three regimes (GEMV at `rows == 1`, split-K while `rows <= 768`, regular above), staged `w_router` blocks in threadgroup memory (coalesced `float4` loads, 28,080 B of 32 KB, 24 barriers on the scalar path), MLX's verbatim sigmoid, strictly-greater insertion top-4, `((w0+w1)+w2)+w3` normalise. See `muse-spark/round-2/kernels/routing/attempt-4/notes.md` and `muse-spark/round-2/implementation/hyp-2-routing-exact.md`.
- **Piece timing** (`grade-phases-routing-attempt-4.json`). Decode: 68.4 / 65.1 / **49.8** at 1 row, 169.2 / 135.7 / **113.5** at 8 rows, 159.3 / 145.2 / **121.3** at 32 rows — faster at every decode size. Prefill: 102.3 / 87.3 / **227.1** at 127 rows (2.6× compiled), 231.3 / 221.1 / **1298.8** at 1,016 rows (5.9×), 310.0 / 290.4 / **2513.0** at 2,048 rows (8.7×). `beats_compiled` false: decode `faster`, prefill `slower`.
- **G3 effect vs prediction.** Per decode step in vLLM: −0.47 / −0.43 / −0.54 ms at 1 / 8 / 32 requests — it passed every rule of its time (faster than both the unchanged model and the compiled reference at all sizes, in both engines). Round 2 predicted −0.48 / −0.50 / −0.49 ms from the chained piece timing; the 8-request figure came in 0.07 ms short, the others match. Over whole requests: throughput +10.2% / +1.8% / +0.3% / −3.5%, with prompts 1.07–1.10× slower. Under the rule set after this measurement it fails, so it is no baseline.
  - **Reason (measured).** The staged scalar path wins at decode sizes but scales badly: 227 µs per layer at 127 rows where compiled takes 87, widening to 8.7× at 2,048 rows. The 28 KB threadgroup footprint caps residency as row counts grow.
- **What keeps it from being a baseline.** It passes G2 bit-exact at all six sizes, but it is slower than the compiled reference at every prefill size, so prompts take 7–10% longer and whole requests with 1,024-token prompts lose 3.5%.
- **Where it still spends its time.** At prefill sizes: the staged scalar chain and its 24 barriers per row (2,513 µs per layer at 2,048 rows vs 290 compiled). At decode sizes it already beats compiled everywhere (49.8 / 113.5 / 121.3); the per-row `w_router` re-read and the serial thread-0 scan are untouched.

#### `combine` — starting point `muse-spark/round-2/kernels/combine/attempt-1`

- **What it is.** One fused kernel per layer: thread per output element on a `rows×768` grid, volatile-barrier stepwise fold `((p0+p1)+p2)+p3` with a rounding after every op, one bfloat16 round, single-rounding bfloat16 `+ shared`. See `muse-spark/round-2/kernels/combine/attempt-1/notes.md` and `muse-spark/round-2/implementation/hyp-3-combine-exact.md`.
- **Piece timing** (`grade-phases-combine-attempt-1.json`). Decode: 52.8 / 53.4 / **45.7** at 1 row, 71.2 / 56.4 / **47.9** at 8 rows, 87.0 / 69.1 / **64.6** at 32 rows. Prefill: 87.8 / 66.1 / **62.8** at 127 rows, 425.2 / 271.6 / **226.5** at 1,016 rows, 846.2 / 520.4 / **273.0** at 2,048 rows — faster at all six sizes. `beats_compiled` true. It is the only starting point that beats the compiled reference in both phases.
- **G3 effect vs prediction.** Per decode step in vLLM: −0.37 / −0.20 / **+1.05 ms** at 1 / 8 / 32 requests. Round 2 predicted −0.36 / −0.35 / −0.35 ms; at 32 requests the sign flips. With 8 requests its median sits above the compiled reference in vLLM. Over whole requests (with routing): throughput +15.9% / +2.1% / −2.7% / −4.1%.
  - **Reason (not known).** A G3 loss of 1.05 ms is about 52 µs per layer, more than the kernel's whole chained time in the grader (64.6 µs per layer at 32 rows, of which the loss alone exceeds the kernel's work as the grader runs it). The cost comes from how the kernel sits in the step, which the grader doesn't reproduce. Round 3's first question is why.
- **What keeps it from being a baseline.** It passes G1, G2 and `beats_compiled`, but its decode step in G3 was 1.05 ms slower than the unchanged model's with 32 requests, for a reason the grader doesn't show.
- **Where it still spends its time.** In the grader: nothing further — it sits on the chained floor (~32–40 µs per layer at decode sizes) and wins everywhere. In the step at 32 requests: the unexplained ~52 µs per layer.

#### All three together

Round 2's submission (routing + combine) changes the vLLM decode step by −0.73 / −0.74 / **+0.53 ms** at 1 / 8 / 32 requests, against a prediction of −0.84 / −0.85 / −0.84 ms: combine's slowdown outweighs routing's saving with 32 requests. Round 2's pairings run (plain MLX) found every pair within 0.5 ms of its parts' sum, but all three together with 8 requests 2.8 ms slower than their parts (+0.94 ms measured against −1.87 ms summed) — the second set of kernels to show it, so the cause is likely in how the step runs with all three slots replaced at that size, not in one kernel. Not explained. Every round-2 variant keeps the model's greedy tokens in both engines at 1 and 32 requests (round 1's change them with 32).

Round 2 ran the grader exactly once per attempt: 13 runs for 13 attempts (experts 7, routing 4, combine 2).

### What is left

Confirmed inefficiencies not yet built or not fully removed, ranked by predicted saving against the compiled reference (no slot has a baseline, so the reference is the bar everywhere). Savings span both phases now: decode shares use the plain-MLX step denominators 4.75 / 13.37 / 28.19 ms at 1 / 8 / 32 rows, prefill effects are prompt-time multiples and whole-request throughput from Claude's engine comparison. Exactness gates are solved in every slot — all three starting points pass G2 at all six sizes — so nothing below is blocked on correctness.

1. **Experts prefill path 15–28× slower than compiled.** The slot's scalar `qmm_accum` branch takes 15,220 / 119,154 / 240,681 µs per layer where compiled takes 961 / 4,306 / 11,999, and prompt processing runs 6.6 to 14.5 times slower, costing 10% to 83% of whole-request throughput. Removing it — a fast exact prefill branch, or a `max_rows` hand-off that keeps prefill-sized inputs on the compiled reference (Claude checked the hand-off bit-exact on a copy with `max_rows` 59, `results/runs/2026-11-04-handoff-check/`) — recovers the whole loss while keeping the 8/32-row decode wins (−1.22 / −5.35 ms per step). This is Claude's question 4 for round 3 (prefill half).
2. **Experts 1-row decode latency.** The starting point chains at 92.1 µs per layer against 78.6 for compiled (+13.5; G3 +0.29 ms with 1 request), the last size where it loses. Six exact fused packings plateaued at 92–110 µs; round 2's summary suggests copying the reference's unfused structure exactly for small row counts. Worth up to ~0.3 ms per step at 1 request plus unlocking `beats_compiled` on the decode phase. Claude's question 1.
3. **Routing prefill 2.6/5.9/8.7× slower than compiled.** 227 / 1,299 / 2,513 µs per layer against 87 / 221 / 290; prompts take 7–10% longer and 1,024-token-prompt requests lose 3.5% throughput. Same two outs as experts: a prefill branch that scales, or `max_rows`. Decode already wins everywhere, so the hand-off alone banks the slot if the decode win survives composition. Claude's question 4 (prefill half).
4. **Why combine attempt-1 costs ~1 ms in the step with 32 requests and not in the grader.** The chained piece timing has it faster (64.6 vs 69.1 µs per layer); G3 has it 1.05 ms slower per step. No hypothesis can bank combine until the mechanism is found — it may be a property of the kernel (grid shape, cache behaviour) or of the step around it. Claude's question 2.
5. **Why all three slots replaced at once lose 2.8 ms with 8 requests.** Every pair adds up; all three together miss by 2.8 ms in plain MLX (round 2 pairs run), repeating round 1's shape. Until it is understood, per-slot piece wins may not compose into a G3 win. Claude's question 3. Carried as a constraint on every hypothesis; the ready check measures each candidate alone and all together, so round 3 sees composition inside the round.
6. **Experts per-pair weight re-reads persist** (round 2 `ineff-baseline-experts-3`, confirmed): 908 MB read vs 443 MB distinct at 32 rows, floor 2.33 ms per step at 32 rows, 0 banked — grouped designs lost in both rounds on duty-cycle ceilings. Banks only if a grouped design holds the chained rate on distinct bytes.
7. **Routing `w_router` re-read per row** (round 2 `ineff-baseline-routing-4`, confirmed): excess floor 0 / 0.13 / 0.57 ms per step; round 1's tiled-sharing attempt was slower everywhere, and attempt-4 already wins in spite of it. Upper bound only.
8. **Prefill-only reference waste, now judged** (round 2 `ineff-quantized-kernels-1`, `-2`, confirmed): BM=16 padded tiles waste 47.1 / 15.3 / 5.2% of issued row-work at 127 / 1,016 / 2,048 rows, and `gather_mm_offsets` is recomputed 3× per layer (≤0.26 ms per chunk forward). Round 2 ranked these at 0 µs of decode time; under two-phase judgment they are prompt-time savings a fast prefill branch takes for free.

**Slots not yet tried.** `routing+combine` (brief: both slots, each with its own kernels, in one candidate; the grader tests each part on its own and the combine part receives the reference routing weights). Neither round has graded one `routing-combine` folder. It remains available; since the parts are graded independently it is packaging unless a shared design emerges.

**Recommendations outside the slots** stand as written in `muse-spark/round-2/summary.md` and round 2's exploration README *Recommendations*; Claude's measurement did not test them. In brief: attention K/V re-read per query head (up to ~2 ms at 32 rows); split-K partial traffic from 14 rows (≈2.4 ms upper bound at 32 rows, unmeasured); 60 q/k/v `[0]` gathers + copies (−0.21 ms measured at 8 rows, `squeeze(0)` fix); ~2 ms engine CPU per 1-row step (≈0.5 ms of Python fixes); prefill tile waste and redundant `gather_mm_offsets` in MLX's `gather_qmm_rhs` (time-to-first-token only — now prompt time, item 8 above). They are for surucuonur's team; no hypothesis targets them.

### Refuted claims

No one chases these again.

- `ineff-runs-3` (round 1, refuted): the micro sorted/unsorted gaps measure per-pair vs grouped kernel choice on identically ordered pairs, not pair order — sorting serving's pairs at 8 rows cannot yield 1.3 ms, and the 32-row gap was never collected.
- Round 1's implicit timing claim — that the independent piece timing ranks attempts as the model does — is refuted by the chained measurement: the round-1 experts starting point was 29% faster overlapped and 66% slower chained at 1 row. Round 3 ranks by the chained two-phase timing only.
- Round 2 exploration's claim that scalar sums cannot reproduce MMA rounding (round 2 `ineff-baseline-routing-1` as written): refuted by building — routing attempts 2–4 and experts attempt-1's scalar paths are bit-identical to `simdgroup_float8x8` accumulation at all six sizes. Scalar first.

### Territories this round opens

Each territory below is opened by the line of this section that names it. Explorers of a `baseline-<slot>` territory map that slot's starting-point kernels and `spec.json` — now the code in the slot — with the checklist, and find where they still lose time and why the starting point is no baseline. Explorers of a reopened territory start from that territory's earlier entries and write only what is new.

| Territory | Opened by |
|---|---|
| `baseline-experts` | "`experts` — starting point `muse-spark/round-2/kernels/experts/attempt-7`" — its kernels and spec are now the code in the slot; why the prefill branch runs 15–28× slower than compiled, and where the 1-row chain still spends its ~19 µs gap. |
| `baseline-routing` | "`routing` — starting point `muse-spark/round-2/kernels/routing/attempt-4`" — likewise; why the staged scalar path scales to 8.7× compiled at 2,048 rows while winning at decode sizes. |
| `baseline-combine` | "`combine` — starting point `muse-spark/round-2/kernels/combine/attempt-1`" — likewise; and item 4 of *What is left*: why it costs ~1 ms in the step at 32 requests while the grader times it faster. |
| `g3` | "All three together" and item 5 of *What is left* — where the end-to-end change differs from the piece timing (combine at 32, the three together at 8), and why. |
| `quantized-kernels` (reopened) | Items 1 and 8 of *What is left* — round 2 mapped `gather_qmm_rhs`'s rounding; round 3 needs its speed: tiling, occupancy and what a fast exact prefill branch (or `max_rows` hand-off) must match or skip. |
| `grader` (reopened) | "The judgment changed for round 3 (ADR 0011)" and "the lead's ready check (ADR 0013)" — the two-phase `beats_compiled`, the `max_rows` hand-off grading, and `check_ready.py`: what each measures and what each still can't see. |

## Versions

- v2 · 2026-11-04T17:41:39Z · checkpoint: baseline-experts, baseline-routing, g3, grader done; 7 inefficiencies posted; routing premature confirms flagged to verifier · changed: territory-baseline-experts v2, territory-baseline-routing v2, territory-g3 v2, territory-grader v2, ineff-baseline-experts-1 v1, ineff-baseline-experts-2 v1, ineff-baseline-routing-1 v2, ineff-baseline-routing-2 v2, ineff-g3-1 v1, ineff-g3-2 v1, ineff-g3-3 v1
- v1 · 2026-11-04T17:36:02Z · opened round 3: starting point (no baselines, 3 starting points, 8 ranked leftovers, refuted claims, 6 territories) · changed: none
