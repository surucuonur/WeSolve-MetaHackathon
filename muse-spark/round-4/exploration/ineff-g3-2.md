# Inefficiency: a set with an experts min_rows: 2 attempt meets the untested three-slot interaction, +2.13 to +2.86 ms/step at 8 requests in rounds 1–2

- **Owner:** explorer-g3
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** the round-4 ready check (candidates alone and all together) is the first test; verifier to check the arithmetic

## Slot

Outside the slots — the three-slot set (routing + combine + experts) as timed in `muse-all`. New instance for round 4's kernel versions of frozen round-3 [ineff-g3-2.md](../../round-3/exploration/ineff-g3-2.md) (confirmed for round-2 kernels, whose pair measurement this entry reuses); no run has timed round 4's three versions together.

## What happens

Round 4's largest direction is a set containing experts attempt-1 with `min_rows: 2, max_rows: 59` beside the routing+combine pair. Its additive prediction at 8×128 is −2.16 ms/step vs base (pair measured −0.772 + experts piece −1.386; [territory-g3.md](territory-g3.md), *What a set containing…*). In rounds 1–2, replacing all three slots at once at 8 requests measured **+2.13 to +2.86 ms/step worse than the sum of the three alone changes, while every pair added up** — with round-2 kernels in four runs across both engines, ranges never overlapping the base. Round 3 installed no experts candidate, so the term has never been measured against kernels with hand-offs (routing `max_rows: 32`, experts `max_rows: 59`) or without combine's volatile barriers. If it recurs at its historical size it erases the set's 8×128 gain entirely; the ready check is the first instrument that will see it.

## Cost

Three-way interaction = (change of all three) − (sum of the three alone changes), ms per decode step vs the unchanged model, all measured, whole request (frozen derivation in round 3's [territory-g3.md](../../round-3/exploration/territory-g3.md) piece 5 / [ineff-g3-2.md](../../round-3/exploration/ineff-g3-2.md) Cost; sources `results/runs/2026-11-04-muse-round-2-pairs/console-plain.txt`, `results/runs/2026-11-04-muse-round-2/console-{plain,vllm}.txt`, `results/runs/2026-11-04-engine-comparison/console-vllm.txt`):

| Run (engine) | Interaction at 8 requests | Per layer (÷20) |
|---|---|---|
| pairs (plain MLX) | **+2.804** | +140.2 µs |
| main (plain MLX) | **+2.858** | +142.9 µs |
| main (vLLM) | **+2.128** | +106.4 µs |
| engine (vLLM, from combos) | **+2.435** | +121.8 µs |

Pair interactions at 8 in the same run: routing+combine −0.128, routing+experts +0.257, combine+experts +0.354 — all within ±0.5 ms. At 1 request the three-way term is +0.262 against a 0.275-wide base spread (not claimed); at 32 requests +0.192 against a 0.057 spread (real, ~15× smaller than at 8). Round 1's 32-request term (+2.8/+3.3) did not recur with round-2 kernels, so that size was a kernel property; the 8-request term recurred across two exact kernel sets.

Applied to round 4's prediction (prediction arithmetic, [territory-g3.md](territory-g3.md)): at 8×128, −2.158 + 2.128 = **−0.030 ms/step** vs base at the term's smallest historical size, and −2.158 + 2.858 = **+0.700 ms/step** (slower than base) at its largest; whole-request effect ≈ 248 steps × the term, +528 to +709 ms against a predicted −559 ms gain. At 32 rows the same arithmetic costs only ~0.19 ms/step against a −6.57 experts effect. In kernel terms every run kept counts exactly additive (−200 −80 −120 = −400 kernels at 1 row) while time was not: 466 graph ops ran slower than 866 unchanged at 8 requests although each removed kernel belonged to a faster-alone variant.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): the lost time scales with neither weight bytes (shape-fixed grids, unchanged per slot) nor kernel count (reduced, additively). Ruled out in the frozen verdict: kernel count, token divergence (bit-exact in every combination), repeat noise (no range overlap, four runs, both engines), any single pair (additive). Open: serial-latency composition across the three replacements vs a three-kernel dispatch/occupancy interaction specific to 8 rows; and whether hand-offs or the de-volatilised combine change it — both kernel differences postdate the measurement.

## Spans

`g3` (measurement), `baseline-experts`, `baseline-routing`, `baseline-combine` (the three kernels as they would be installed together), `grader` (the ready check must time the 8-request all-together step, alone-vs-together, or round 4 re-learns this from Claude's G3).

## Predicted saving

If the three-way interaction were removed — or shown absent for round 4's versions — while keeping each slot's own effect, the set gains back the term: up to **2.86 ms/step at 8 requests in plain MLX (2.13 in vLLM)**, i.e. the set would measure its additive ≈ −2.16 ms/step vs base at 8×128 instead of −0.03 to +0.70. Nothing claimed at 1 request (term inside the spread) and ~0.19 ms/step at 32. Per prefill chunk nothing is predicted: no run has timed pairs or triples on the prefill piece basis, the set's prompts are handed off kernel-by-kernel to the pair's measured prompt times, and any prompt-side composition is unverified, never assumed zero. This is an upper bound on the prize, not a prediction that a specific change recovers it: the mechanism is unknown, the fallback if the triple loses is the measured routing+combine pair, and any fix must hold bit-exact G2 in every slot it touches.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** historical interaction carried from frozen round-3 [ineff-g3-2.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/exploration/ineff-g3-2.md) (confirmed) and its consoles — pairs parts −0.383/−0.265/−1.219 sum −1.867, all +0.937 → **+2.804**; main plain −1.857 → +1.001 → **+2.858**; main vLLM −1.848 → +0.280 → **+2.128**; engine combos −1.541 → +0.894 → **+2.435** ms/step at 8 requests; per-layer +140.2/+142.9/+106.4/+121.8 µs (÷20) ✓; pairs r+c −0.128, r+e +0.257, c+e +0.354 within ±0.5 ms; 1-request +0.262 inside 0.275 spread (not claimed), 32-request +0.192 vs 0.057 spread (~15× smaller); round-1 32-request +2.8/+3.3 gone with round-2 kernels ✓; no run has timed round-4's three versions together (round 3 installed no experts candidate) ✓.
- **confirmed:** round-4 application (prediction, labelled) — pair measured vs base at 8×128 −0.772 (13.565−14.337, Claude [console-vllm.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-3/console-vllm.txt)) + experts piece (168.9−238.2)×20 = −1.386 → additive **−2.158 ms/step**; −2.158+2.128 = **−0.030**, −2.158+2.858 = **+0.700** ms/step; whole-request 248 steps ([muse-vllm.jsonl](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-3/muse-vllm.jsonl): pair wall change −215.4 ms at 8×128) → term +528 to +709 ms against −559 ms predicted gain (−215.4+248×−1.386) ✓; 32-row term ~0.19 against −6.574 experts effect ✓; counts additive (−200−80−120 = −400) while time is not, tokens exact, ranges non-overlapping in four runs/both engines — overhead-bound, pairs ruled out (the “466 vs 866 at 8 requests” counts are rows-1 check records applied at 8 by the same unsorted-graph inference the round-3 verdict noted; additivity is exact at both measured row counts, so the ruled-out stands) ✓.
- **confirmed:** saving up to 2.86 ms/step plain (2.13 vLLM) at 8 requests if the term is absent/removed, ~0 at 1, ~0.19 at 32, prefill unverified never assumed zero; upper bound only, mechanism unknown, fallback is the measured routing+combine pair, ready check is the first test ✓.

## From other agents

(none yet)
