# Inefficiency: combine attempt-1 costs ~1.1 ms per decode step at 32 requests where the piece timing says faster

- **Owner:** explorer-g3
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:41:19Z
- **Next:** verifier to check; mechanism belongs to baseline-combine

## Slot

`combine`. Measurement side owned by territory `g3`; kernel-side mechanism owned by territory `baseline-combine` (starting point `muse-spark/round-2/kernels/combine/attempt-1`).

## What happens

`time_candidates.py` installs combine attempt-1 alone (`muse-combine`) in every layer's `MoE.__call__` and times the decode step (median of 3 shuffled repeats, 128-token prompts, 256 tokens; [territory-g3.md](territory-g3.md) piece 4). At 32 requests the step is **slower than the unchanged model by +1.051 ms (vLLM) / +1.168 ms (plain MLX)** — +52.6 / +58.4 µs per layer — while the chained piece timing has the same kernel **faster** than the reference (64.6 vs 87.0 µs/layer) and than the compiled reference (64.6 vs 69.1) in `grade-phases-combine-attempt-1.json`. Full derivation in [territory-g3.md](territory-g3.md) gap 1. A prefill twin: adding combine-a1 to routing+experts-a7 slows the 32×128 prompt by **+1501.7 ms** (11533.4 vs 10031.7, no range overlap) where the prefill piece says combine saves 11.5 ms/layer/chunk at 2048 rows.

## Cost

Piece-vs-step swing = (G3 change vs unchanged) − (candidate − reference piece × 20 layers), ms per decode step. Piece from `grade-phases-combine-attempt-1.json` (reference / candidate: 52.8 / 45.7 at 1 row, 71.2 / 47.9 at 8, 87.0 / 64.6 at 32); G3 from `console-vllm.txt` / `console-plain.txt` (`results/runs/2026-11-04-muse-round-2/`).

| Engine | 1 request | 8 requests | 32 requests |
|---|---|---|---|
| vLLM: piece-implied | −0.142 | −0.466 | −0.448 |
| vLLM: G3 measured | −0.372 | −0.197 | **+1.051** |
| vLLM: **swing** | −0.230 | +0.269 | **+1.499** |
| plain: piece-implied | −0.142 | −0.466 | −0.448 |
| plain: G3 measured | −0.424 | −0.112 | **+1.168** |
| plain: **swing** | −0.282 | +0.354 | **+1.616** |

Per layer (÷20): vLLM −11.5 / +13.5 / **+75.0 µs**, plain −14.1 / +17.7 / **+80.8 µs**. Against the compiled reference (G3 muse-combine minus G3 compiled-combine, vs piece candidate−compiled × 20): in-step penalty +6.1 / +11.2 / **+66.2 µs/layer** at 1 / 8 / 32 rows in vLLM (+0.122 / +0.224 / +1.324 ms/step) — the penalty grows with rows and explodes at 32.

Reproduction: pairs run (plain, `results/runs/2026-11-04-muse-round-2-pairs/`) combine-a1 at 32: +1.051 ms — identical to the main vLLM figure; engine comparison submitted-minus-routing at 32×128: +0.537 − (−0.374) ≈ +0.91 ms. Four runs, both engines. Prefill twin per chunk per layer: +1501.7 ms / (2 chunks × 20 layers) = +37.5 µs/layer/chunk at 2048 rows vs piece −573.2 (273.0 − 846.2).

In bytes: the slot's traffic is 9,232 B/row (`notes.md`), 295 KB/layer at 32 rows ≈ 1.5 µs at peak bandwidth; the paid +52.6 µs/layer is ~35× the traffic floor.

## Kind

Overhead-bound (in the terms of `claude/GLOSSARY.md`): the step is far from peak bandwidth and peak compute, and the lost time scales with neither weight bytes (traffic floor 1.5 µs vs 52.6 paid) nor kernel count (one kernel replaces five, additively — piece 5). Ruled out: the kernel's own chained work (measured faster), token divergence (bit-exact at all 6 + 25 sizes and end to end), repeat noise (no range overlap anywhere it is claimed: vLLM muse-combine [31.433–31.547] vs base [30.346–30.513]).

## Spans

`g3` (measurement), `baseline-combine` (the kernel: grid `rows*768` threads in 256-thread threadgroups, volatile-stepwise fold, no threadgroup memory), `grader` (the chained piece timing is the instrument that cannot see this cost).

## Predicted saving

If the in-step penalty were removed while keeping the kernel's chained work, the step gains back the swing: up to **1.50 ms/step at 32 requests in vLLM** (1.62 plain), i.e. muse-combine would measure ≈ −0.45 ms (the piece-implied value) instead of +1.05 ms; **0.27 ms/step at 8 requests** (0.35 plain); nothing claimed at 1 request, where the swing is negative (over-delivers) and within 1-row noise. At prefill sizes, removing the twin recovers **up to 1.5 s of 32×128 prompt time** on the all-three variant (measured 11533.4 ms, would be ≈10032 ms). Per prefill chunk (chained basis): nothing further — the kernel already beats compiled at 127 / 1,016 / 2,048 rows (62.8 / 226.5 / 273.0 vs 66.1 / 271.6 / 520.4 µs/layer), so no per-chunk saving is predicted on the grader basis; the prize is the in-step composition only. This is an upper bound on the prize, not a prediction that a specific kernel change recovers it: the mechanism is unknown, and any fix must hold the bit-exact G2 at all six sizes.

## Verdict

(none — open)
