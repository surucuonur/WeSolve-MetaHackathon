# Exploration: round 4 — analysis

**Version:** v3 · 2026-11-04T23:01:20Z · skill version 10

## Tracker

| Territory | Agent | Status | Inefficiencies | Updated |
|---|---|---|---|---|
| `baseline-experts` | `explorer-baseline-experts` | done | `ineff-baseline-experts-1` (refuted), `-2` (confirmed) | 2026-11-04 21:48Z |
| `baseline-routing` | `explorer-baseline-routing` | done | `ineff-baseline-routing-1` (confirmed), `-2` (confirmed) | 2026-11-04 21:48Z |
| `baseline-combine` | `explorer-baseline-combine` | done | `ineff-baseline-combine-1` (confirmed), `-2` (confirmed) | 2026-11-04 21:48Z |
| `g3` | `explorer-g3` | done | `ineff-g3-1` (confirmed), `-2` (confirmed) | 2026-11-04 21:48Z |
| `grader` | `explorer-grader` | done | — (notes on baseline-experts, g3, baseline-routing) | 2026-11-04 21:48Z |

Verdicts: 7 confirmed, 1 refuted, 0 superseded, 0 open. All 5 territory entries `done`.

## Starting point

Round 4 is the last round ([ADR 0014](../../../claude/adr/0014-muse-wonder-last-round.md)). It builds on rounds 1–3 as measured independently by Claude in [`claude/findings/muse-round-3.md`](../../../claude/findings/muse-round-3.md). Round 3 ran as Muse Spark (model `muse-spark-1.3-contributor`, skill version 8) and submitted the reference in every slot: no candidate passed the ready check alone, and the skill then counted only candidates that pass alone. Round 4 changes three rules ([ADR 0015](../../../claude/adr/0015-sets-and-hand-off-below.md)): candidates that pass together in `muse-all` are `ready` as a **set**; an attempt may hand off inputs below `min_rows` as well as above `max_rows`; and a handed-off size is reported `handed off` instead of being timed. A slot's **baseline** is still its best attempt so far that passes the current G2, has `beats_compiled` true in the grader, and passes G3 ([ADR 0011](../../../claude/adr/0011-decode-and-prefill.md)). **No slot has a baseline.** The bar in every slot is the compiled reference, and a new attempt must beat it in the same grader run (no `--baseline` argument). Round 4 runs as Muse Wonder (muse-wonder-1.0).

### Per slot

#### `experts` — starting point `muse-spark/round-3/kernels/experts/attempt-1`

- **What it is.** Round 2's fused per-pair kernel, byte-identical, plus `max_rows: 59` in `spec.json`: rows 1–59 run the fused kernel, rows above 59 run the compiled reference. See `muse-spark/round-3/kernels/experts/attempt-1/notes.md` and its `grade.json` (G1 pass, G2 differ 0 at all six sizes).
- **Piece timing** (chained µs per layer, candidate vs compiled, its own grade run): 1 row 96.9 vs 86.6 (`slower`, 5 wins); 8 rows 168.9 vs 238.2 (`faster`, 29 wins); 32 rows 554.7 vs 883.4 (`faster`, 29 wins); prefill 127 / 1,016 / 2,048 `same` (19 / 15 / 16 wins — the hand-off runs compiled code). `beats_compiled` false: decode `slower` on the 1-row size alone.
- **G3 effect.** Never timed end to end (round 3, section 4). Its ancestor (round 2 attempt-7, same kernels without the hand-off) shortened the decode step 17.5% with 32 requests but made prompt processing 6.6–14.5× slower; the hand-off removes the prefill half of that, so no G3 prediction from round 3 stands — Claude's reading is that only the 1-row gap remains.
- **What keeps it from being a baseline.** It is not a baseline because `beats_compiled` is false — the 1-row decode loss (96.9 vs 86.6 µs/layer) — and it has no G3 pass. Round 3 tried a vectorised scalar path (attempt-2, bought nothing) and an unfused path copying the reference's structure (attempt-5, 2× worse at 1 row) and concluded no fix existed in the spec surface. From round 4 the surface has `min_rows`: `min_rows: 2` with `max_rows: 59` hands 1-row inputs to the compiled reference while keeping the 8/32-row wins — a direction not yet graded. It can then pass G3 only in a set whose other kernels are faster with 1 request, because with 1 request every decode step runs MLX's own code for this slot.
- **Where it still spends its time.** At 8/32 rows it already wins by 69.3 / 328.7 µs per layer; what remains is the 1-row chain (the `tg_x` load, silu and barrier chain named in round 3) — now hand-off-able — and per-pair weight re-reads at all sizes (552,960 B per pair, round 3's entry).

#### `routing` — starting point `muse-spark/round-3/kernels/routing/attempt-6`

- **What it is.** Round 2's staged scalar-exact kernel (attempt-4 kernels) with `max_rows: 32`: every decode step (1–32 rows) runs the staged scalar kernels, all prefill sizes run the compiled reference. Attempt-6 is a byte-identical re-grade of round 3's attempt-1 (empty `diff` on both blocks and `spec.json`), run to test whether attempt-1's 2,048-row `slower` on identical code reproduced; it did not (19/30 wins, `same`). See `muse-spark/round-3/kernels/routing/attempt-6/notes.md`.
- **Piece timing.** Round 3's grade: decode `faster` at all three sizes (candidate 51.7 / 90.9 / 132.2 vs compiled 66.3 / 104.0 / 152.0 µs per layer; 28 / 27 / 25 wins), prefill `same` at all three sizes. Claude's re-grade with the round-4 grader ([`results/runs/2026-11-04-grader-hand-off/`](../../../results/runs/2026-11-04-grader-hand-off/)): decode `faster` (29 / 26 / 27 wins; candidate 0.72 / 0.90 / 0.88 of the compiled reference at 1 / 8 / 32 rows), the three prefill sizes `handed off`, `beats_compiled` true.
- **G3 effect.** In round 3's swarm ready check it failed alone against `compiled-routing` at 8×128 on overlapping ranges, with no phase slower anywhere (decode medians −0.38 / −0.31 / −0.59 / −0.48 ms vs the unchanged model at 1×128 / 8×128 / 32×128 / 32×1024). Claude's G3 agreed: `same` as `compiled-routing` at 8×128. Together with combine attempt-2 (`muse-all`) it passed the swarm's ready check but failed Claude's G3 against `compiled-all` at 32×1024, where the medians are 0.9% apart with overlapping ranges.
- **What keeps it from being a baseline.** G3 alone: it passes G2 and has `beats_compiled` true, but it has never passed G3, alone or (in Claude's run) as a pair. Under round 4's set rule the pair would have been `ready` in round 3, but a set still has to pass G3 as `muse-all`, so the pair is no baseline.
- **Where it still spends its time.** Decode wins at every size; the open question is margin, above all at 32×1024 where the pair gains only 0.25–0.30 ms over the compiled reference (~1%), inside the 5-repeat spread. The per-row `w_router` re-read is untouched (upper bound only, round 2's `ineff-baseline-routing-4`).

#### `combine` — starting point `muse-spark/round-3/kernels/combine/attempt-2`

- **What it is.** Round 2's attempt-1 kernel with its seven `volatile` rounding barriers replaced by seven plain `float` locals and a header block (`block-1.metal`) carrying `#pragma STDC FP_CONTRACT OFF` / `#pragma clang fp contract(off)`: same 7-op stepwise fold `((p0+p1)+p2)+p3`, same grid (`rows*768`) and threadgroup [256,1,1], no `max_rows`. See `muse-spark/round-3/kernels/combine/attempt-2/notes.md`.
- **Piece timing.** Round 3's grade: `faster` at all six sizes (decode 31.8 / 34.7 / 32.7 vs compiled 36.9 / 41.1 / 45.7 µs per layer, 24 / 27 / 30 wins; prefill 38.2 / 100.9 / 198.0 vs 64.0 / 281.8 / 556.2, 30 / 30 / 29 wins); `beats_compiled` true in both phases. Absolute chained times drift up to ~35% between runs on this machine (round 3's lesson), so only same-run paired verdicts transfer.
- **G3 effect.** In round 3's ready check it failed alone against the unchanged model at 1×128 on a single outlier repeat (5.892 ms, above the unchanged model's lowest 5.430) and, in Claude's G3, against `compiled-combine` at 32×128 and 32×1024 — again range overlaps with no phase slower anywhere (swarm decode medians −0.53 / −0.59 / −0.44 / −0.26 ms vs the unchanged model). Its ancestor's in-step cost (+1.05 ms/step at 32 requests from the volatile barriers) is what the pragma removes; round 3's measured decode medians favour the kernel everywhere.
- **What keeps it from being a baseline.** G3 alone, on spread: every median favours it, but the 5-repeat ranges overlap the bars'. No kernel change is named that tightens spread on 7 plain ops (round 3's implementation README); as part of a set it already passes `muse-all` in the swarm's run.
- **Where it still spends its time.** Nothing further in the grader — it sits on the chained floor and wins at all six sizes. What remains is the same margin question as routing: more separation from the compiled reference at 32×128 and 32×1024, or a set that clears the bar together.

#### All three together

Round 3's pair (routing attempt-6 + combine attempt-2) is the best result of any round: against the unchanged model it shortens the decode step 16.5% / 5.4% / 2.8% / 1.6% and whole requests 14.1% / 5.6% / 2.1% / 2.4% at 1×128 / 8×128 / 32×128 / 32×1024, with prompts 1.6–3.1% faster and nothing slower — but it fails G3 against `compiled-all` at 32×1024 (0.9% median gap, overlapping ranges). About half the gain over the unchanged model is `mx.compile` of the two slots' reference code alone. The three-slot composition is untested: in rounds 1 and 2, replacing all three slots at once lost about 2.8 ms with 8 requests for an unknown reason, and round 3 installed no experts candidate, so no run since has met it.

### What is left

Ranked by predicted saving against the compiled reference (no slot has a baseline, so the reference is the bar everywhere). Claude's measurement names three things round 4 has to find out; the ranking follows them.

1. **Experts with `min_rows`.** Whether attempt-1's 8/32-row speed (168.9 vs 238.2 and 554.7 vs 883.4 µs per layer) carries into G3 once 1 row runs MLX's own code (`min_rows: 2`, `max_rows: 59`). This is the only ungraded direction in any slot and the largest piece saving left: the 8/32-row wins are 69.3 / 328.7 µs per layer, worth (PREDICTION, from those same-run margins × 20 layers) about 1.4 ms per step at 8 requests and 6.6 ms at 32 requests against the compiled reference, with prefill `handed off` above 59 rows and 1-row decode `handed off` — so the gain exists only at 8 and 32 requests, and G3 can pass only in a set whose other kernels win with 1 request. The risk is the three-slot interaction (item 3).
2. **A margin at 32×1024.** The pair gains 0.25–0.30 ms over the compiled reference there (~1%), which 5 repeats cannot tell from noise; a set needs more than that with 32 long prompts. This is a margin problem in routing and/or combine decode kernels, not a correctness one: both starting points are bit-exact and already `faster` at every graded size, so any new attempt must widen the same-run margin at decode sizes without touching prefill (`max_rows: 32` for routing, faster-everywhere for combine). A saving here is worth the difference between a passing and a failing set — the largest whole-request prize left, and unquantified until a builder measures it.
3. **All three slots at once.** In rounds 1 and 2 the three-slot replacement lost about 2.8 ms with 8 requests for an unknown reason; a set with an experts attempt meets that again. No single-slot kernel change is predicted to remove it, so it is carried as a constraint: per-slot wins are not assumed to compose, and the ready check (which times candidates alone and all together) is where composition is measured. If the three together lose, the fallback set is the routing+combine pair.
4. **Slots not yet tried.** `routing+combine` (both slots, each with its own kernels, in one candidate) has never been graded as one folder. The parts are graded independently and the combine part receives the reference routing weights, so it is packaging unless a shared design emerges; the set rule already submits the two slots together without it. No hypothesis is planned on it unless exploration finds a shared-design reason.
5. **Recommendations outside the slots** stand as written in round 1's and round 2's exploration READMEs (attention K/V re-read per query head; split-K partial traffic from 14 rows; the 60 q/k/v `[0]` gathers; engine CPU per 1-row step). Claude's round-3 measurement did not test them. They are for surucuonur's team; no hypothesis targets them.

### Refuted claims

No one chases these again.

- Round 3's experts 1-row fixes: a vectorised scalar path (attempt-2) bought nothing and an unfused path copying the reference's structure (attempt-5) was 2× worse at 1 row — the 1-row gap is not closed by either approach, and round 4 does not retry them; it hands 1 row off instead.
- Routing attempt-1's 2,048-row `slower` verdict on byte-identical code: attempt-6's re-grade (19/30, `same`) confirmed it as paired noise, not a hand-off cost. Handed-off sizes are no longer timed at all.
- `ineff-runs-3` (round 1, refuted): the micro sorted/unsorted gaps measure kernel choice, not pair order.
- Round 1's implicit claim that the independent (overlapped) piece timing ranks attempts as the model does: refuted by the chained measurement. Round 4 ranks by the chained timing only.
- The claim that scalar sums cannot reproduce MMA rounding: refuted by building in round 2 (routing attempts 2–4, experts scalar paths bit-identical at all six sizes).

### Territories this round opens

Each territory is opened by the line of this section that names it. A `baseline-<slot>` explorer maps that slot's starting-point kernels and `spec.json` — now the code in the slot — with the checklist, and finds where they still lose time and, for a starting point, why it is no baseline. Explorers start from the earlier rounds' entries for their territory and write only what is new.

| Territory | Opened by |
|---|---|
| `baseline-experts` | "`experts` — starting point `muse-spark/round-3/kernels/experts/attempt-1`" — its kernels and `spec.json` are now the code in the slot; what `min_rows: 2` changes (the 1-row hand-off, the 33–59 band, the dead scalar branch), and where the fused kernel still spends its time at 8/32 rows. |
| `baseline-routing` | "`routing` — starting point `muse-spark/round-3/kernels/routing/attempt-6`" — likewise; where decode margin can still be widened, above all for the 32×1024 workload (item 2 of *What is left*). |
| `baseline-combine` | "`combine` — starting point `muse-spark/round-3/kernels/combine/attempt-2`" — likewise; whether any exact change widens its decode margin, and what the pragma leaves in the compiled binary. |
| `g3` | "All three together" and items 1–3 of *What is left* — where the end-to-end change differs from the piece timing (the pair at 32×1024, the untested three-slot interaction), and why. |
| `grader` | Round 4's three rule changes (ADR 0015) — how `min_rows`, `handed off` and the set rule are implemented in `bench/grader/slots.py`, `check_candidate.py` and `check_ready.py`, and what each still can't see. |

*Corrections exploration made to this Starting point.* None to the per-slot facts: every starting-point number was re-derived from its cited file by the explorers and the verifier. Two additions: (1) the experts per-pair re-read entry (`ineff-baseline-experts-1`) was refuted as written for mixing two byte bases and must be restated on the counter basis (floor 2.33 ms/step at 32 rows) before anyone prices it; (2) the grader territory confirms the exact grading of a `min_rows: 2, max_rows: 59` experts attempt — sizes 1, 127, 1,016 and 2,048 `handed off`, decode `faster` requires both 8 and 32 rows faster — and that the ready check's set verdict is read from its `muse-all` entry, not a top-level pass flag.

## Context

A decode step produces one token per request; in vLLM the unchanged step takes 5.457 / 14.337 / 30.061 / 39.410 ms at 1×128 / 8×128 / 32×128 / 32×1024 (Claude's round-3 G3 medians). The MoE block runs in all 20 layers of every step, in decode and prefill alike; the three slots mirror its split, and each starting point already fuses its slot to 1 kernel per layer and passes G2 bit-exact at all six graded sizes (1, 8, 32 decode rows; 127, 1,016, 2,048 prefill rows). What remains is not launch count or exactness: it is the experts 1-row hand-off the new `min_rows` unlocks, the routing+combine pair's ~1% margin over the compiled reference at 32×1024, and whether three replaced slots compose at 8 requests. No slot has a baseline, so every saving is against the compiled reference.

## Territory map

| Territory | Receives | Passes on | Measured cost |
|---|---|---|---|
| `baseline-experts` | `x` [rows,768] bf16 and `inds` [rows,4] from routing; expert weights [60,…] | `y` [rows,4,768] bf16 in `inds` order to combine | Starting point 1 kernel/layer; 1 row `slower` (96.9 vs 86.6 µs/layer), wins at 8/32 (168.9/554.7 vs 238.2/883.4); per-pair re-reads unpriced after refutation |
| `baseline-routing` | `x` from post-attention norm; `w_router` [60,768] f32, `bias` [60] f32 | `inds` to experts, `weights` to combine (order is part of G2) | Starting point 1 kernel/layer; decode margins 17.7/15.8/19.7 µs/layer over compiled; per-row `w_router` excess 0/6.5/28.6 µs/layer |
| `baseline-combine` | `y` from experts, `weights` from routing, `shared` [rows,768] from shared expert | `out` [rows,768] bf16 to residual/next layer | Starting point 1 kernel/layer; `faster` at all six sizes; chained floor ~32 µs/layer at decode, bytes+compute ≤1.5 µs of it |
| `g3` | The three starting points' piece timings and Claude's G3 | Composition gaps and set predictions to implementation | Pair keeps −0.36 of −0.66 ms/step piece win at 32×1024; three-slot term +2.13–2.86 ms/step at 8 requests, untested |
| `grader` | References from model code; captured decode inputs; prefill chunks | The round-4 contract (`min_rows`, `handed off`, set rule) to builders and boundary | The contract, not a cost: what counts as identical, and how each size and set is priced |

How they feed each other: layernorm → `baseline-routing` (`inds`/`weights`) → `baseline-experts` (`y`) → `baseline-combine` (`out`) → residual. `grader` sits across all three (what counts as identical, and how time is priced). `g3` sits across the whole step and the whole request (whether per-slot prices compose).

## Inefficiencies, ranked

Confirmed inefficiencies inside the slots, ranked by predicted saving against the compiled reference (no slot has a baseline). Decode shares are of the decode step; prefill effects are stated per chunk. Each item says what removing it does to the other phase and whether a hand-off keeps that phase unchanged.

1. **The experts 1-row block is a hand-off, not a kernel problem** (starting point, `territory-baseline-experts.md`). With `min_rows: 2, max_rows: 59` the 1-row size is `handed off` and decode needs only 8 and 32 rows `faster` — both already `faster` by 69.3 / 328.7 µs per layer. Predicted (PREDICTION, same-run margins × 20 layers): `beats_compiled` true with decode `faster`, prefill `handed off`; end-to-end (additive, from `ineff-g3-2.md`) −0.44 / −1.76 / −7.12 / −6.94 ms per step vs `compiled-all` at 1×128 / 8×128 / 32×128 / 32×1024 in a three-slot set — at 1 request the experts slot contributes nothing (handed off), so the set passes there only on the pair. Other phase: prefill unchanged by construction (handed off above 59 rows). Cost: the 1-row loss 96.9 vs 86.6 µs/layer disappears from the verdict, not from the machine — with 1 request the slot runs MLX's own code, exactly as the bar does.
2. **The pair's 32×1024 margin** (`ineff-g3-1.md`, confirmed, Claude's). The routing+combine pair keeps only −0.36 of its −0.66 ms/step piece win vs the compiled reference at 32×1024, and the 0.9% median gap sits inside the 5-repeat spread: ~0.17 ms/step more separation would have passed Claude's run. No mechanism is confirmed and no single-slot change is priced to recover it; the pools any widening must come from are items 3–4. A decode saving bought with prefill time is not one: routing keeps `max_rows: 32`, combine stays faster at prefill sizes.
3. **`w_router` re-read once per row** (`ineff-baseline-routing-1.md`, confirmed). Traffic excess 0 / 6.5 / 28.6 µs per layer at 1 / 8 / 32 rows against decode margins of 17.7 / 15.8 / 19.7 — the excess exceeds the whole 32-row margin. Upper bound 0 / 0.13 / 0.57 ms per step (overlap-discounted, prediction). Prefill: 0 at every size by hand-off construction (`max_rows: 32` kept). Round 1's tiled-sharing attempt was slower everywhere, so this banks only if a sharing design holds the chained rate.
4. **The staged scalar path's shape cost** (`ineff-baseline-routing-2.md`, confirmed). 27 barriers, a 23%-duty tail third and serial tails at decode sizes; bounded single-digit µs per layer (prediction). Overlaps item 3 (do not add). Prefill: 0 by hand-off.
5. **The experts serial K-chain headroom** (`ineff-baseline-experts-2.md`, confirmed). 8/32 rows sit 80.4 / 200.8 µs per layer above their own byte floor (headroom 1.61 / 4.02 ms per step, 0 banked): grouped designs lost in both earlier rounds on duty-cycle ceilings, so this is headroom, not a plan. It matters only if the `min_rows` set fails and a faster 8/32-row kernel is needed. Prefill: handed off, unchanged.
6. **Combine is at its floor** (`ineff-baseline-combine-1.md` confirmed with achievable saving 0; `ineff-baseline-combine-2.md` confirmed at ≤0.5 µs per layer, below paired noise). No bit-exact change inside the slot widens its decode margin; the ~32 µs/layer chained floor is dispatch, not bytes or compute. Do not chase either. Round 4 carries the starting point into the set with a byte-identical re-grade (the round-3 attempt-6 precedent), not a new kernel.

Carried as a constraint, not a ranked build: **the three-slot interaction** (`ineff-g3-2.md`, confirmed) — +2.13 to +2.86 ms/step at 8 requests in rounds 1–2 against a −2.16 ms/step additive prediction. No single-slot change removes it; the ready check's `muse-all` is its first test with these kernels, and the fallback if the triple loses is the measured routing+combine pair.

## Recommendations

Confirmed inefficiencies outside the slots stand as written in round 1's and round 2's exploration READMEs (attention K/V re-read per query head; split-K partial traffic from 14 rows; the 60 q/k/v `[0]` gathers; engine CPU per 1-row step). Claude's round-3 measurement did not test them and this round raises no new outside-slot item. They are for surucuonur's team.

## Refuted claims

- `ineff-baseline-experts-1.md` (this round, refuted): the 32-row per-pair re-read floor mixed the pair-model byte count (70.78 MB/layer) with counter distinct bytes (22.15 MB/layer); restate on the counter basis (floor 2.33 ms/step at 32 rows) before pricing. The re-read itself is real; its price as written is not.
- Round 3's experts 1-row kernel fixes (vectorised scalar, attempt-2; unfused path, attempt-5): bought nothing and 2× worse. Round 4 hands 1 row off instead.
- Routing attempt-1's 2,048-row `slower` on byte-identical code: paired noise (attempt-6 re-grade, 19/30 `same`); handed-off sizes are no longer timed.
- `ineff-runs-3` (round 1): micro sorted/unsorted gaps measure kernel choice, not pair order.
- The independent (overlapped) piece timing as a ranking: refuted by the chained measurement. Round 4 ranks chained only.
- Scalar sums cannot reproduce MMA rounding: refuted by building in round 2.

## Files read

Each territory entry's *Files read* list: [territory-baseline-experts.md](territory-baseline-experts.md), [territory-baseline-routing.md](territory-baseline-routing.md), [territory-baseline-combine.md](territory-baseline-combine.md), [territory-g3.md](territory-g3.md), [territory-grader.md](territory-grader.md).

## What building taught

The analysis survives building with three corrections. (1) The `min_rows` hand-off works exactly as the grader territory mapped it: sizes 1, 127, 1,016 and 2,048 grade `handed off`, decode needs only 8 and 32 rows `faster`, and the experts attempt's measured whole-request decode (−1.57 ms/step at 8×128, −4.02 / −5.09 at the 32-request workloads vs base) matches its piece-margin prediction (−1.49 / −5.23 vs compiled) — piece timing at 8/32 rows transfers; the slot's problem was never speed. (2) The routing margin pools were smaller in the step than on paper: sharing `w_router` (item 3) narrowed the 32-row margin instead of widening it, and neither shape trim (item 4) widened attempt-1's margins, so the pair's 32×1024 gap (item 1 of *What is left*, `ineff-g3-1`) was not closed by any single-slot change — in ready-2 the pair's 32×1024 decode median is +0.86 ms vs base where +0.17 ms/step of extra separation was wanted. (3) The three-slot interaction is no longer historical: with these kernels the three-slot set measures +0.32 ms/step vs base at 8×128 against an additive ≈−2.16, i.e. ≈+2.5 ms/step of interaction, while the same slots at 32 requests compose fine (−4.31 / −4.00). Any future set should be designed at 8 requests first. The verdict-level lesson stands from round 3 and is now three times measured: at gains of ~1%, 5 shuffled repeats judge by spread, and every variant in both ready checks failed `not slower` with favourable medians in most workloads.

## Versions

- v3 · 2026-11-04T23:01:20Z · What building taught added (hand-off transfers, margin pools small, interaction live at 8 requests) · changed: none
- v2 · 2026-11-04T21:48:45Z · final analysis: 5 territories done, 7 confirmed 1 refuted, 6 ranked items · changed: territory-baseline-experts v2, territory-baseline-routing v2, territory-baseline-combine v2, territory-g3 v2, territory-grader v4, ineff-baseline-experts-1 v2, ineff-baseline-experts-2 v2, ineff-baseline-routing-1 v2, ineff-baseline-routing-2 v2, ineff-baseline-combine-1 v2, ineff-baseline-combine-2 v2, ineff-g3-1 v2, ineff-g3-2 v2
- v1 · 2026-11-04T21:40:34Z · opened round 4 starting point, 5 territories · changed: none
