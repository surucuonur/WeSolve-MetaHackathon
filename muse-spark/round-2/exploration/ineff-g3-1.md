# Inefficiency: the three fused attempts together lose the time each saves alone

- **Owner:** explorer-g3
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:23:45Z
- **Next:** pairwise G3 and chained routing/combine piece timings to separate serial-latency composition from a three-kernel interaction

## Slot

`experts` + `routing` + `combine` — the cause, whatever it is, only appears when the three slot replacements are installed together; each slot's attempt alone behaves as its piece timing (chained, for experts) predicts. Spans territory `g3` and, kernel-side, `baseline-experts`, `baseline-routing`, `baseline-combine`.

## What happens

`time_candidates.py` installs each attempt alone (`muse-<slot>`) and all three at once (`muse-all`) in every layer's `MoE.__call__` ([time_candidates.py:72-96](../../../bench/grader/time_candidates.py)) and times the decode step (median of 3 shuffled repeats, 128-token prompts, 256 tokens). The kernel-count reductions add exactly — at 1 request in vLLM 967 → 847 / 767 / 887 alone and 567 together (−120 −200 −80 = −400; `muse-vllm.jsonl` `check` records) — but the step-time changes do not add: the combined variant is slower than the sum of the three single-slot changes in both engines, at every row count, and slower than the unchanged model outright at 1 and 8 requests. The compiled references installed the same way stay within ±0.29 ms of additive, so composing three replacements is not by itself the cost. Full derivation in [territory-g3.md](territory-g3.md) pieces 5/6 and gap 3.

## Cost

Interaction = (change of `muse-all`) − (sum of the three `muse-<slot>` changes), ms per decode step, against the unchanged model. Sources: `console-vllm.txt` / `console-plain.txt` summary blocks (`results/runs/2026-11-04-muse-round-1/`).

| Engine | 1 request | 8 requests | 32 requests |
|---|---|---|---|
| vLLM: sum of alone | +0.833 | −1.919 | −5.034 |
| vLLM: all three | +0.980 | +0.163 | −1.756 |
| vLLM: **interaction** | **+0.147** | **+2.082** | **+3.278** |
| plain MLX: sum of alone | +0.604 | −1.946 | −4.385 |
| plain MLX: all three | +0.927 | +0.881 | −1.560 |
| plain MLX: **interaction** | **+0.323** | **+2.827** | **+2.825** |

Per layer (÷20): +7.4 / +104.1 / +163.9 µs (vLLM), +16.1 / +141.3 / +141.2 µs (plain). At 8 and 32 requests the interaction is 15–25× the repeats' spread (vLLM base ranges 14.327–14.475 and 29.657–29.952 ms) and appears in two independent engine runs; at 1 request it is the size of the spread and may be nil. Control, compiled references in vLLM: alone sum −0.442 / −0.419 / −0.532 ms vs `compiled-all` −0.498 / −0.558 / −0.242 ms, interaction −0.056 / −0.139 / +0.290 ms.

In kernel terms: 567 kernels/step at 1 request run slower than 967 unchanged (+0.980 ms, vLLM) although each removed kernel was, alone, part of a faster variant. Kernel count is therefore not the cost being paid.

## Kind

Overhead-bound (in the terms of `claude/GLOSSARY.md`): the step is far from peak bandwidth and peak compute, and the lost time scales with neither weight bytes (unchanged per slot, shape-fixed grids) nor kernel count (reduced, additively). The two candidate mechanisms are both overhead-side: serial per-call latency of fused kernels composing on one layer chain, or a dispatch/occupancy interaction between the three custom kernels. Token divergence is excluded as a cause in [territory-g3.md](territory-g3.md) piece 7: every slot's grid and bytes are fixed by shape, whichever tokens are generated.

## Spans

`g3` (measurement), `baseline-experts`, `baseline-routing`, `baseline-combine` (the three kernels as installed), `grader` (the chained piece timing is the instrument that has so far only been run on experts).

## Predicted saving

If the interaction were removed, the step would gain back the interaction term while keeping each slot's alone change: up to **2.08 ms/step at 8 requests and 3.28 ms/step at 32 requests in vLLM** (2.83 / 2.83 ms in plain MLX), i.e. `muse-all` would measure ≈ −1.92 / −5.03 ms (vLLM) instead of +0.16 / −1.76 ms. At 1 request the predicted saving is 0.15–0.32 ms and is within the spread, so it is not claimed. This is an upper bound on the prize, not a prediction that a specific kernel change recovers it: which mechanism (serial-latency composition vs three-kernel interaction) decides where the fix lives, and the separating measurements — pairwise G3 for the three slot pairs and chained piece timings for routing/combine — are named in [territory-g3.md](territory-g3.md) gap 3 and have not been run.

## Verdict

verifier, 2026-11-04T03:23:45Z — **confirmed**, with the spread multiple corrected. Checked: harness installs `muse-<slot>` / `muse-all` via `MoE.__call__` ([time_candidates.py:72-96](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/time_candidates.py:72)), median of 3 shuffled repeats, 128-token prompts, 256 tokens ✓. Kernel reductions additive at 1 request vLLM 967→847/767/887 alone and 567 together (−120−200−80=−400, `muse-vllm.jsonl` `check` records via [territory-g3.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-g3.md) pieces 5/6) ✓. Interaction = all − sum(alone) from [console-vllm.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/console-vllm.txt) / [console-plain.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/console-plain.txt): vLLM sum +0.833/−1.919/−5.034 (1.162+0.185−0.514 etc.) vs all +0.980/+0.163/−1.756 → +0.147/+2.082/+3.278 ms/step; plain sum +0.604/−1.946/−4.385 vs all +0.927/+0.881/−1.560 → +0.323/+2.827/+2.825 ✓. Per layer ÷20: +7.4/+104.1/+163.9 µs vLLM, +16.1/+141.3/+141.2 µs plain ✓. Compiled control: alone sum −0.442/−0.419/−0.532 vs `compiled-all` −0.498/−0.558/−0.242 → interaction −0.056/−0.139/+0.290 ms, within ±0.29 ✓. 567 vs 967 kernels slower (+0.980 ms) — kernel count is not the cost ✓. Token divergence excluded (grids fixed by shape, [territory-g3.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-g3.md) piece 7) ✓. Predicted saving as upper bound (muse-all would be ≈−1.92/−5.03 ms vLLM) with separating measurements unrun, as written ✓. **Wrong (significance wording only):** “15–25× the repeats' spread” — on the cited vLLM base ranges (widths 0.148 ms at 8 requests, 0.295 ms at 32) the ratios are 2.082/0.148≈14× and 3.278/0.295≈11×. Still ≫ spread in both engines, so the interaction stands; the multiple is 11–14×, not 15–25×.

