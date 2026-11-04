# Inefficiency: all three slots replaced at once gain 2.1–2.9 ms/step at 8 requests while every pair adds up

- **Owner:** explorer-g3
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:41:19Z
- **Next:** verifier to check; pairs measured, cause still open

## Slot

`routing` + `combine` + `experts` — the cause, whatever it is, only appears when all three slot replacements are installed together; each pair behaves as its parts predict. Spans territory `g3` and, kernel-side, `baseline-experts`, `baseline-routing`, `baseline-combine`. New instance for round-2 kernels of frozen round-2 [ineff-g3-1](../../round-2/exploration/ineff-g3-1.md) (round-1 kernels, confirmed); the pairs measurement it asked for is now run.

## What happens

The G3 runs install each round-2 kernel alone and all three at once and time the decode step (median of 3 shuffled repeats; [territory-g3.md](territory-g3.md) pieces 4–6). At 8 requests the combined variant is slower than the sum of the three single-slot changes by **+2.1 to +2.9 ms/step in all four runs**, while in the pairs run every pair comes within ±0.35 ms of its parts' sum. At 1 request the three-way term is within the spread (not claimed); at 32 it is +0.19 ms — real but 15× smaller than at 8, and round 1's +2.8/+3.3 ms 32-request interaction is gone. Full derivation in [territory-g3.md](territory-g3.md) gap 2.

## Cost

Three-way interaction = (change of all-three) − (sum of the three alone changes), ms per decode step, against the unchanged model.

| Run (engine) | Parts at 8 requests (r / c / e) | Sum | All three | **Interaction** |
|---|---|---|---|---|
| pairs (plain) | −0.383 / −0.265 / −1.219 | −1.867 | +0.937 | **+2.804** |
| main (plain) | −0.482 / −0.112 / −1.263 | −1.857 | +1.001 | **+2.858** |
| main (vLLM) | −0.434 / −0.197 / −1.217 | −1.848 | +0.280 | **+2.128** |
| engine (vLLM) | combos only (see below) | −1.541 | +0.894 | **+2.435** |

Sources: `results/runs/2026-11-04-muse-round-2-pairs/console-plain.txt`, `results/runs/2026-11-04-muse-round-2/console-{plain,vllm}.txt`, `results/runs/2026-11-04-engine-comparison/console-vllm.txt` summary blocks. Engine-comparison sum from combos (no alone combine/experts there): (routing+experts −1.511) + (submitted −0.339) − (routing −0.309) = −1.541.

Per layer (÷20): +140.2 / +142.9 / +106.4 / +121.8 µs at 8 requests. All-three range never overlaps base (pairs: [14.358–14.476] vs [13.520–13.833]) while every part sits below base — not noise.

Pair interactions at 8 (pairs run): r+c −0.128, r+e +0.257, c+e +0.354 — all within ±0.5 ms ([muse-round-2.md §4](../../../claude/findings/muse-round-2.md)). At 1: three-way +0.262 vs base spread 0.275 wide — not claimed. At 32: +0.192 vs spread 0.057.

In kernel terms: 466 graph ops/step run slower than 866 unchanged (+0.94 ms, pairs at 8) although each removed kernel was, alone, part of a faster variant — and the count reductions are exactly additive (−200 −80 −120 = −400). Kernel count is therefore not the cost being paid.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): the step is far from peak bandwidth and peak compute, and the lost time scales with neither weight bytes (unchanged per slot, shape-fixed grids) nor kernel count (reduced, additively). Two exact kernel sets (rounds 1 and 2) show the 8-request three-way term, so the cause is likely in how the step runs with all three slots replaced at that size, not in one kernel — while the 32-request term vanished with the round-1 kernels, so that one was a round-1-kernel property. Ruled out: kernel count, token divergence (exact everywhere this round), repeat noise (no overlap, four runs, both engines), any single pair (additive — new since round 2's entry).

## Spans

`g3` (measurement), `baseline-experts`, `baseline-routing`, `baseline-combine` (the three kernels as installed), `grader` (the ready check must time the 8-request all-together step or round 3 re-learns this from Claude).

## Predicted saving

If the three-way interaction were removed while keeping each slot's alone change, the step gains back the interaction term: up to **2.86 ms/step at 8 requests in plain MLX** (2.13 vLLM), i.e. all-three would measure ≈ −1.86 ms instead of +0.94/+0.28 ms. Nothing claimed at 1 request (within spread) or 32 (0.19 ms). No per-prefill-chunk figure: no run times pairs or triples on the prefill piece basis; the prompt-side composition of the three kernels together is unverified except via whole-request prompt times (engine comparison), where all-three vs routing+experts differs only by combine ([ineff-g3-1](ineff-g3-1.md)). This is an upper bound on the prize, not a prediction that a specific change recovers it: the mechanism (serial-latency composition vs a three-kernel dispatch/occupancy interaction at 8 rows) is still open, and any fix must hold bit-exact G2 in every slot it touches.

## Verdict

(none — open)
