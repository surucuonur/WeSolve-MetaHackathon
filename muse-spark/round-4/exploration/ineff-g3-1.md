# Inefficiency: the routing+combine pair keeps only −0.36 of its −0.66 ms/step piece win vs the compiled reference at 32×1024, inside the repeat spread

- **Owner:** explorer-g3
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** builders widening routing/combine decode margin at 32 rows answer this; verifier to check the arithmetic

## Slot

Outside the slots — the routing+combine set as timed in `muse-all` (routing attempt-6 + combine attempt-2, `muse-spark/round-3/kernels/routing/attempt-6`, `muse-spark/round-3/kernels/combine/attempt-2`). Measurement side owned by territory `g3`; kernel-side margin owned by territories `baseline-routing` and `baseline-combine`.

## What happens

The chained piece timing prices the pair at −0.656 ms/step vs the compiled reference at 32 decode rows (routing −0.396 + combine −0.260; [territory-g3.md](territory-g3.md) piece 1). Timed whole in vLLM at 32 requests, the pair keeps −0.548 ms/step at 32×128 and only **−0.362 ms/step at 32×1024** in Claude's G3 (medians 38.765 vs 39.127 ms, **0.93% apart, 5-repeat ranges [38.589–39.080] vs [38.912–39.400] overlapping**, so `muse-all` vs `compiled-all` decode is `same` and G3 fails on this one workload). In the swarm's ready check the same workload keeps −0.588 (38.942 vs 39.530, 1.49%) and the ranges miss, so the pair passes there. The verdict flips between two runs of identical kernels while every median favours the pair. Full derivation in [territory-g3.md](territory-g3.md) gap 1.

## Cost

Shortfall = (measured, whole-request change vs `compiled-all`) − (piece-implied change vs compiled, candidate − compiled × 20 layers), ms per decode step. Piece from the two `grade.json` (routing 132.2 vs 152.0, combine 32.7 vs 45.7 µs/layer at 32 rows; at 1/8 rows routing 51.7/90.9 vs 66.3/104.0, combine 31.8/34.7 vs 36.9/41.1); measured from `results/runs/2026-11-04-muse-round-3/console-vllm.txt` summary medians (cross-checked in `muse-vllm.jsonl`) and `muse-spark/round-3/checks/ready-1/ready-verdict.json` `medians`.

| Workload | Piece vs compiled | Claude G3 vs compiled | Shortfall (Claude) | Swarm vs compiled | Shortfall (swarm) |
|---|---|---|---|---|---|
| 1×128 (1 row) | −0.394 | −0.436 | −0.042 | −0.491 | −0.097 |
| 8×128 (8 rows) | −0.390 | −0.375 | +0.015 | −0.446 | −0.056 |
| 32×128 (32 rows) | −0.656 | −0.548 | +0.108 | −0.483 | +0.173 |
| 32×1024 (32 rows) | −0.656 | **−0.362** | **+0.294** | −0.588 | +0.068 |

Per layer at 32×1024 (÷20): +14.7 µs (Claude) / +3.4 µs (swarm); the two runs disagree by 0.226 ms/step, the same order as the larger shortfall, so the solid cost is the **margin**, not a fixed 0.294: the pair's separation from the compiled bar at 32×1024 is 0.36–0.59 ms/step (0.9–1.5%) against per-run 5-repeat spreads of ~0.49 ms (pair) and ~0.49 ms (compiled) wide. To have separated in Claude's draw the pair needed roughly a further 0.17 ms/step (its max 39.080 below the compiled min 38.912). At 1 row with long prompts the piece sum transfers fully (measured −0.428 at 1×1024, −0.487 at 1×7936 vs piece −0.394; [territory-g3.md](territory-g3.md) piece 4), so the shortfall needs 32 rows, and grows again with context at fixed 32 rows.

In kernel terms: the pair removes 280 kernels/step vs base (1087→807 at 32 rows) and 180 vs `compiled-all` (987→807), exactly additive, tokens bit-exact — kernel count and correctness are not the cost. The slots' own traffic is unchanged by context while the step grows 30.1→39.4 ms from 32×128 to 32×1024; the lost time scales with neither.

## Kind

Overhead-bound (in the terms of `claude/GLOSSARY.md`): the step is far from peak bandwidth and peak compute, and the shortfall scales with neither weight bytes nor kernel count. Ruled out: the kernels' own chained work (both `faster` at every decode size, piece 1), token divergence (exact everywhere), prompt length alone (piece 4). Not ruled out: a fixed in-step composition cost at 32 rows that the chained timing cannot see, vs machine drift across runs (swap 13.9–14.2 GB) — the two runs bound the sum at +0.07 to +0.29 ms/step without separating them.

## Spans

`g3` (measurement), `baseline-routing`, `baseline-combine` (any exact kernel change that widens the 32-row decode margin), `grader` (the 5-repeat range rule that prices a ~1% margin by chance; unchanged for round 4).

## Predicted saving

If the pair's full piece win transferred at 32 rows, the step gains back the shortfall: up to **0.29 ms/step at 32 requests in Claude's draw (0.07 in the swarm's)**, i.e. `muse-all` would measure ≈ −0.66 ms/step vs `compiled-all` at both 32-row workloads instead of −0.55 / −0.36; ~0 claimed at 1/8 requests, where G3 already meets or beats the piece sum. The decision-relevant saving is smaller and sharper: **~0.17 ms/step of extra separation at 32×1024** would have made the ranges miss in Claude's run and turned the pair's one failing workload into a pass. At prefill sizes nothing is predicted: prompts are already 1.6–3.1% faster than base and slower nowhere (routing handed off, combine `faster` at 127/1,016/2,048 rows), so no per-chunk saving exists on either basis. This is an upper bound on the prize, not a prediction that a specific kernel change recovers it: no mechanism is confirmed, any widening must hold bit-exact G2 at all six sizes, and a decode saving bought with prefill time is not one.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** piece basis from the three [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/routing/attempt-6/grade.json) files — routing (51.7−66.3, 90.9−104.0, 132.2−152.0)×20 = −0.292/−0.262/−0.396 ms/step, combine (31.8−36.9, 34.7−41.1, 32.7−45.7)×20 = −0.102/−0.128/−0.260, pair **−0.394/−0.390/−0.656** ✓; at 32 rows routing −0.396 + combine −0.260 = −0.656 as headlined ✓.
- **confirmed:** whole-request medians — Claude [console-vllm.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-3/console-vllm.txt) summary `muse-all` vs `compiled-all`: 4.555−4.991 = −0.436, 13.565−13.940 = −0.375, 29.231−29.779 = −0.548, 38.765−39.127 = **−0.362 (−0.93%)** at 32×1024 with ranges [38.589–39.080] vs [38.912–39.400] overlapping → `same`, G3 fails there; swarm [ready-verdict.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/checks/ready-1/ready-verdict.json) `medians`: −0.491/−0.446/−0.482/−0.587 (entry rounds the third to −0.483) with ranges missing at 32×1024 → passes; verdict flips on the draw, every median favours the pair ✓; shortfalls (measured − piece) Claude −0.042/+0.015/+0.108/**+0.294**, swarm −0.097/−0.056/+0.173/+0.068 ms/step; per-layer +14.7/+3.4 µs (÷20), runs disagree 0.226 ms/step ✓; extra separation needed 39.080−38.912 = **0.17 ms/step** ✓; spreads ~0.49 ms wide both sides ✓.
- **confirmed:** ruled-outs and saving — kernel counts 967→687 / 1087→807 vs base (−280) and 867/987→807 vs `compiled-all` (−180), tokens exact, so count/correctness are not the cost ✓; step growth 30.1→39.4 ms at fixed slot traffic, 1-row long-prompt transfer −0.428/−0.487 vs piece −0.394 ([console-vllm.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-3-long/console-vllm.txt): 4.868−5.296, 7.280−7.767) rules out prompt length alone ✓; prompts 1.6–3.1% faster, slower nowhere ([muse-round-3.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/findings/muse-round-3.md)) ✓; overhead-bound ✓; saving up to 0.29 (Claude) / 0.07 (swarm) ms/step, decision-relevant ~0.17 ms/step, prefill 0 — upper bound, bit-exact G2 required, no decode-for-prefill trade ✓.

## From other agents

(none yet)
