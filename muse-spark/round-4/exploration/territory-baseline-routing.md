# Territory: baseline-routing — round 3 attempt-6 (staged scalar, `max_rows: 32`) under the set rule

- **Owner:** explorer-baseline-routing
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T21:44:34Z
- **Next:** none — checklist filled for all pieces, ineff-baseline-routing-1..2 posted, grader note folded

## What it does

Same slot contract as rounds 2–3 (`muse-spark/round-3/exploration/territory-baseline-routing.md:11-20`, frozen): per layer, `scores = sigmoid(float32(x) @ w_router.T)`, top-4 on `scores + bias` (selection only), weights from the unbiased scores (`((s0+s1)+s2)+s3 + 1e-20`, divide, ×2.5). Outputs `inds` [rows,4] uint32 (feeds experts) and `weights` [rows,4] float32 (feeds combine, in emitted order; G2 is ordered bitwise equality at all six sizes, brief:190-193).

The code in the slot is `muse-spark/round-3/kernels/routing/attempt-6/` — round 2 attempt-4's kernels byte-identical (`diff` of `block-0.metal` and `spec.json` against attempt-1 empty, verified this round; attempt-1 = attempt-6, `notes.md:13-20`) plus `max_rows: 32` in `spec.json:1`. One fused kernel per layer: 256-thread threadgroup per row, `rows == 1` GEMV shuffle branch, staged scalar split-K above 1 row, verbatim sigmoid, thread-0 insertion top-4. Rows 1–32 run the kernels; every prefill size runs the compiled reference. Round 3's entries (`territory-baseline-routing.md`, `ineff-baseline-routing-1..2`, all frozen) mapped this code's *prefill* loss; that loss is now handed off, so this entry writes only what is new: the decode side, priced as margin.

## The headline finding (details in ineff-baseline-routing-1 and -2)

The starting point already wins everywhere it runs. Two grader runs of the identical bytes:

| rows | round-3 grade (`attempt-6/grade.json`) cand / compiled | Claude re-grade, round-4 grader (`results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`) cand / compiled | margin (re-grade) |
|---|---|---|---|
| 1 | 51.7 / 66.3 (28 wins, `faster`) | 45.1 / 62.8 (29 wins, `faster`) | 17.7 µs/layer (0.72×) |
| 8 | 90.9 / 104.0 (27, `faster`) | 144.2 / 160.0 (26, `faster`) | 15.8 µs/layer (0.90×) |
| 32 | 132.2 / 152.0 (25, `faster`) | 151.3 / 171.0 (27, `faster`) | 19.7 µs/layer (0.88×) |
| 127 / 1,016 / 2,048 | `same` × 3 (timed, identical code) | `handed off` × 3 | — (not timed; ADR 0015) |

Chained absolutes drift up to ~2× between runs on this machine (round-3 README; hyp-3 failure ledger) — the 8-row absolute moves 90.9 → 144.2 while the relative win holds — so only same-run margins and ratios transfer. In per-step terms (×20 layers, prediction from the same-run piece margins, not a whole-request measurement): the routing margin is worth ≈0.35 / 0.32 / 0.39 ms per decode step at 1 / 8 / 32 rows against the compiled reference.

Why it is no baseline: G3 alone. It passes G1+G2 (0 rows differing, all six sizes) and has `beats_compiled` true in both grading runs, but it has never passed G3 — alone it is `same` as `compiled-routing` at 8×128 in both the swarm's ready check (`muse-spark/round-3/checks/ready-1/ready-verdict.json`: `muse-routing` fail vs compiled, pass vs base, no phase slower anywhere) and Claude's G3 (`results/runs/2026-11-04-muse-round-3/g3-vllm.json`: `same` at 8×128 vs compiled-routing, `same` at 32×1024 vs base), and as the routing+combine pair it fails Claude's G3 against `compiled-all` at 32×1024 (medians 0.9% apart, ranges 38.59–39.08 vs 38.91–39.40 ms overlapping; `claude/findings/muse-round-3.md` §1). Under round 4's set rule the pair would have been `ready` in round 3, but a set still has to pass G3 as `muse-all` (round-4 README:34). The bar is the compiled reference; there is no `--baseline`.

The margin question, priced: at 32×1024 the pair gains only 0.25–0.30 ms/step over the compiled reference (~1% of the 39.4 ms step; round-4 README:35,49-55 item 2), inside the 5-repeat spread (range widths ≈0.49 ms on both sides; `ineff-g3-1.md`: the pair keeps −0.362 of its −0.656 piece win in Claude's draw). Non-overlap in Claude's draw needed roughly a further 0.17 ms/step (`ineff-g3-1.md`), i.e. +7.5–12.5 µs/layer across routing+combine at 32 rows (arithmetic: 0.15–0.25 ms ÷ 20 layers). Routing's whole current 32-row margin is 19.7 µs/layer, so routing alone would have to widen by ~40–60% to clear it; any µs/layer routing adds at 32 rows widens the set's margin 1:1 (0.02 ms/step per µs/layer). Every candidate change must keep `max_rows: 32` (or otherwise leave prefill on the compiled reference) — a change that lets the kernels run at prefill sizes re-opens round 3's confirmed prefill entries and is not a saving under ADR 0011.

Where the remaining decode time goes (both entries below): the candidate's chained time at 1/8/32 rows (re-grade 45.1/144.2/151.3 µs/layer) sits far above its traffic floor (0.9/7.4/29.7 µs/layer at 200 GB/s, brief:9) — the slot is overhead-bound at decode sizes (`claude/GLOSSARY.md:39-42`). Two named pools remain inside it: the per-row `w_router` re-read (traffic floor 29.7 µs/layer at 32 rows, larger than the whole 19.7 margin, but overlapped rather than serial — entry 1) and the staged scalar path's shape cost at decode sizes (27 barriers/row, 4 of 12 groups at 23% duty, 28,080 B footprint, serial tails — entry 2).

## Checklist

### 1. The starting-point kernels — `muse-spark/round-3/kernels/routing/attempt-6/block-0.metal` (174 lines) + `block-1.metal` (3 lines)

One threadgroup of 256 threads per row at every kernel-visible size (`spec.json:1`: grid `rows*256`, threadgroup [256,1,1]). Threadgroup memory `xpad[816]` + `scores[60]` + `parts[2048]` + `wsh[4096]` floats = 28,080 B of 32 KB (85.7%, headroom 4,688 B; block-0:15,22-25). Phases per row:

- **(a) `x` staging + `parts` zeroing (block-0:27-33, 1 barrier).** `x[row]` (768 bf16 → f32) into padded `xpad[(d>>4)*17+(d&15)]` (one pad float per BK=16 chunk); `parts[2048]` zeroed (8 floats/thread). On the `rows == 1` path both `parts` and `wsh` are then never read — the zeroing is dead work at 1 row (entry 2).
- **(b) GEMV branch, `rows == 1` only (block-0:35-55).** 8 simdgroups; simdgroup `sg` takes experts `sg, sg+8, …`; each lane serially sums 24 products from padded `xpad` and `w_router` read straight from device (uncoalesced by expert), shuffle tree 16,8,4,2,1, lane 0 writes `scores[e]`. This branch carries the largest relative win (0.72× compiled).
- **(c) Staged scalar path, `rows > 1` (block-0:56-128).** Partition geometry (block-0:57-75): at every decode size 2–32, `tm = 1`, `ratio = 24`, so `partitions = 32`, `base_iters = 1`, last partition absorbing chunks 31–47 (the graded 8/32 and every ungraded 2–7/9–31 run this exact configuration — hyp-3 boundary check, frozen). 12 groups of 4 K-chunks (block-0:76-120): coalesced `float4` staging of `w_router` into transposed `wsh[slot][k][expert]` (stride 64), barrier, scalar FMAs resuming `parts[p*64+e]` in increasing `d`, barrier — 24 barriers in the loop, 27 per row total (block-0:33,94,119,129,137). Scalar duty: groups at `gbase` 0–28 run 240/256 threads (94%); the four groups at `gbase` 32–44 touch only partition 31 and run 60/256 (23%) — one third of all K-chunks (16 of 48) issue through a quarter of the lanes at every decode size (recomputed from block-0:75,96-101). Final sequential partial sum 60/256 threads (block-0:121-127). Per-row device traffic: the full 184,320 B `w_router` re-read, private to each row's threadgroup (entry 1).
- **(d) Verbatim sigmoid (block-0:131-137).** `1/(1+precise::exp(abs(s)))`, `(s<0)?y:1-y`, 60 of 256 threads. MLX's exact form; not a cost item.
- **(e) Serial thread-0 top-4 + weights (block-0:139-174).** Strictly-greater insertion over 60 biased scores (`scores[e] + bias[e]`, `bias` read from device by thread 0 alone), earliest index wins ties; `((s0+s1)+s2)+s3 + 1e-20f`, divide, ×2.5. 255 threads idle through it, after the last barrier, on every row at every size.

- **Callers:** the grader builds it via `slots.build` into one `mx.fast.metal_kernel` (`bench/grader/slots.py:303-345`) and calls it once per layer per size in G2 and in the chained two-phase piece timing; above `max_rows` (and below `min_rows` when set) `run()` dispatches to `mx.compile(ref_routing)` instead (`slots.py:338-345`). In the model it replaces the routing part of `make_moe_call` (`slots.py:97-119`). Kernel-visible shapes: rows 1–32 only (grid `rows*256`).
- **Callees:** none inside the kernels — 1 kernel/layer (20/step) where the reference launches ~12/layer (brief:49). Above 32 rows the callee is the compiled reference itself.
- **Data:** in: `x` [rows,768] bfloat16, `w_router` [60,768] float32, `bias` [60] float32; out: `inds` [rows,4] uint32, `weights` [rows,4] float32 (`spec.json:1`; brief:32-38). All row-contiguous (brief:26). Threadgroup 28,080 B, row-count independent. `w_router` is the same 184,320 B buffer for all 20 layers' calls in a step, re-read per row per layer by construction.
- **Neighbours:** before: post-attention layernorm output reshaped to [rows,768] (`mobilemoe.py` `MoE.__call__`, `slots.py:107`); after: experts gather by `inds`, combine sums `(y * weights[...,None])` in emitted order (`slots.py:51-52`). A replacement must keep shapes, dtypes, the unbiased-score rule and the reference's exact per-row order.
- **Cost:** chained µs/layer in the table above (measured, piece: `attempt-6/grade.json` and `results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`). Traffic per layer rows×184,320 B (`w_router`) + rows×1,536 B (`x`): floors 0.9/7.4/29.7 µs at 1/8/32 rows (200 GB/s). End to end (measured, whole request: round-3 ready check + Claude's G3): routing-alone decode medians −0.38/−0.31/−0.59/−0.48 ms vs the unchanged model at 1×128/8×128/32×128/32×1024 (swarm) and −0.41/−0.31/−0.61/−0.62 (Claude); vs the compiled reference it separates only where ranges miss (see headline).
- **Invariants:** G1+G2 pass at all six sizes in both runs (`rows_differ` 0, `max_rel_err` 0.0). Bit-exactness rests on: exact bf16→f32 cast of `x`; the `rows`-branched regime matching MLX's dispatch (GEMV at `rows == 1`, split-K while `rows <= 768`) with partition counts from the `rows`-only formula; scalar≡MMA rounding at every grouping (proven rounds 2–3); verbatim sigmoid; strictly-greater tie rule; `((s0+s1)+s2)+s3 + 1e-20` fold. Feeding changes (staging, barriers, thread mapping, cross-row sharing) that keep each row's accumulation order keep G2 — attempt-4 proved it against attempt-3, and round 3's sharing kernel (attempt-5) proved it again for a different feeding. Anything that reorders one expert's serial chain (e.g. sharding one expert's K across threads) fails G2 (hyp-3, builder-3-c, frozen).

### 2. `spec.json` (1 line) and the hand-off

- **Callers/callees/data/neighbours:** declares the single kernel `fused_routing_exact` (inputs `x`, `w_router`, `bias`; outputs `inds`, `weights`; header block 1 = `#include <metal_simdgroup>`), `"max_rows": 32`, no `min_rows`. `slots.build` validates both hand-off keys (int ≥ 1, `min_rows <= max_rows`; `slots.py:330-337`).
- **Cost:** the hand-off is the whole prefill story: at 127/1,016/2,048 rows the candidate *is* the compiled reference, reported `handed off` under the round-4 grader (no paired timing, no identical-code verdict risk — the round-3 attempt-1 8/30 `slower` noise event is retired by construction). The choice of 32 is the bottom of the boundary's 32–126 window (hyp-3 boundary, frozen): every decode step (1–32 rows, brief:26) stays on the kernels, and no graded prefill size runs them. Kernel-visible band 33–126 in real serving (chunked prefill) does not exist for this attempt — counts above 32 hand off in serving too.
- **Invariants:** `min_rows` is available in round 4 (brief:174) but this slot does not need it: the kernel wins at 1 row in both runs (0.72–0.78× compiled, 28–29 wins), the strongest relative margin of the three decode sizes. Handing 1 row off would surrender that win and, in a set with an experts `min_rows: 2` attempt, leave 1-request decode with no routing contribution. `atomic_outputs` false; grid counts threads (brief:180).

### 3. The measurement tables

- `muse-spark/round-3/kernels/routing/attempt-6/grade.json` (frozen): G2 pass ×6; decode `faster` ×3 (28/27/25 wins), prefill `same` ×3 timed against identical code; `beats_compiled` true.
- `results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`: Claude's re-grade with the round-4 grader — decode `faster` ×3 (29/26/27), prefill `handed off` ×3, `beats_compiled` true. The current piece-timing source for this territory.
- `muse-spark/round-3/checks/ready-1/ready-verdict.json`: the swarm's G3 — `muse-routing` fail vs compiled-routing (8×128 overlap), `muse-all` pass in that run.
- `results/runs/2026-11-04-muse-round-3/g3-vllm.json` + `claude/findings/muse-round-3.md` §§1–2: Claude's G3 — `muse-routing` fail vs both bars somewhere, pair fail vs `compiled-all` at 32×1024 only; decode-step medians and the 0.9% figure.
- Round-3 frozen analysis: `territory-baseline-routing.md` (the code's anatomy), `ineff-baseline-routing-1.md` (confirmed: per-row re-read priced at prefill), `ineff-baseline-routing-2.md` (confirmed: staged shape priced at prefill), `implementation/hyp-3-routing-prefill.md` (six attempts: hand-off attempts 1/6 the only `beats_compiled`; GEMM-sharing attempts 2/4 G1-fail on the per-pipeline thread cap 640→576, attempt-5 exact but 1.44/2.52/3.50× off compiled at prefill with decode `same` at 8/32; narrower-footprint attempt-3 flipped decode to `same` on a ~2–3 µs copy launch).

## Relations

- **Fed by:** the model territory (layernorm output `x`; router weight and `expert_bias`); the grader territory (`max_rows`/`min_rows` dispatch in `slots.build`, the `handed off` verdict, chained timing).
- **Feeds:** `baseline-experts` (`inds` chooses the (row, expert) pairs and their order); `baseline-combine` (`weights` multiply the experts' outputs in emitted order — the reason order is part of G2); `g3` (the pair's 32×1024 margin is the binding set constraint; routing's 32-row piece margin is the largest single routing contribution to it).
- **Shared constraint:** per-slot wins are not assumed to compose (round-4 README item 3, the three-slot interaction); both entries below price routing alone, and the set's margin is measured only by the ready check timing `muse-all`.

## Inefficiencies

- `ineff-baseline-routing-1` — the per-row `w_router` re-read at decode sizes: rows×184 KB per layer where the compiled reference reads it once; traffic excess 0/6.5/28.6 µs/layer at 1/8/32 rows against a current margin of 17.7/15.8/19.7 µs/layer. Predicted saving (upper bound, overlap-discounted in the entry): 0/0.13/0.57 ms per decode step, 0 at prefill by hand-off construction.
- `ineff-baseline-routing-2` — the staged scalar path's shape cost at decode sizes: 27 barriers/row, the last third of K-chunks at 23% thread duty, 28,080 B footprint, dead `parts` zeroing at 1 row, serial thread-0 tail. Predicted saving bounded in the entry; every µs/layer removed widens the 32-row margin 1:1.

## Files read

- `muse-spark/round-3/kernels/routing/attempt-6/{block-0.metal,block-1.metal,spec.json,notes.md,grade.json}` — the code in the slot: staged scalar decode path, GEMV branch, `max_rows: 32`, its own chained timings; byte-identity with attempt-1 re-verified by `diff` this round.
- `results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json` — Claude's re-grade with the round-4 grader: decode margins 17.7/15.8/19.7 µs/layer, prefill `handed off`.
- `muse-spark/round-3/exploration/territory-baseline-routing.md` (frozen) — the slot anatomy and prefill pricing this entry supersedes on the decode side.
- `muse-spark/round-3/exploration/ineff-baseline-routing-1.md` and `-2.md` (frozen, confirmed) — the re-read and shape costs priced at prefill; entry 1's decode slice and entry 2's decode residual are repriced in this round's entries.
- `muse-spark/round-2/exploration/ineff-baseline-routing-4.md` (frozen, confirmed) — the original decode-side re-read entry (upper bound 0/0.13/0.57 ms/step, round-1 attempt-2 counter-evidence).
- `muse-spark/round-3/implementation/hyp-3-routing-prefill.md` (frozen) — the six graded directions: what sharing, narrowing and splitting actually measured, the thread-cap failures, the boundary (32–126 window, feeding-vs-order rule).
- `muse-spark/round-2/kernels/routing/attempt-4/notes.md` (frozen) — the staged design, the cross-threadgroup overlap account (marginal ≈0.16 µs/row from 8→32 rows) that discounts traffic-floor savings at decode.
- `muse-spark/round-3/checks/ready-1/ready-verdict.json`; `results/runs/2026-11-04-muse-round-3/g3-vllm.json`; `claude/findings/muse-round-3.md` — the G3 record: where routing separates from the compiled reference and where it does not.
- `muse-spark/round-4/exploration/README.md` — the starting point, the 32×1024 margin item, this territory's opening line.
- `bench/grader/slots.py:19,44-52,97-119,303-345` — `D/E/K`, `ref_routing`, `make_moe_call`, `slots.build` with `min_rows`/`max_rows` dispatch.
- `claude/muse-agent-brief.md:26-49,173-174,184-218`; `claude/GLOSSARY.md:33-42` — slot shapes, the hand-off keys, the judgment, overhead/bandwidth terms.

## From other agents

- explorer-grader, 2026-11-04T21:44:11Z — from [territory-grader.md](territory-grader.md) pieces 2–3: routing attempt-6 (`max_rows: 32`) under the round-4 grader reports its three prefill sizes `handed off` (not timed, no `wins_vs_compiled`) with decode `faster` and `beats_compiled` true (ADR 0015 re-grade); for the set reading, use `ready-verdict.json`'s `variants."muse-all".pass` — the top-level `pass`/exit code still fail a set that passes only together.
  - **Folded 2026-11-04T21:44:34Z:** agreed and consistent with §2 above — the re-grade cited throughout this entry (`results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`) is exactly that `handed off` verdict, and the headline's "never passed G3 as a set in Claude's run" is the `muse-all` reading, not the top-level exit code. No number changes; the set-pass criterion for builders is `variants."muse-all".pass` in the round-4 ready check.
