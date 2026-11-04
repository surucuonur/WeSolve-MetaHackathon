# Round 2 of the Muse agent, measured

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

In round 2 (2026-11-04, 03:13 to 04:16 UTC), the Muse agent (Muse Wonder, `muse-wonder-1.0`, at `max` effort) ran skill version 5. It started from [Claude's measurement of round 1](muse-round-1.md), under the stricter grader of [ADR 0010](../adr/0010-exact-at-every-size.md):
- G2 asks for outputs bit-exact with MLX's reference, at decode and prefill sizes;
- the piece timing chains the 20 layers.

Its summary, [muse-spark/round-2/summary.md](../../muse-spark/round-2/summary.md), submits two kernels for G3:
- **routing:** `muse-spark/round-2/kernels/routing/attempt-4`;
- **combine:** `muse-spark/round-2/kernels/combine/attempt-1`.

For experts it keeps MLX's reference. Its best experts kernel, attempt-7, was bit-exact and faster with 8 and 32 rows, but it was dropped for being slower with 1 row. The summary predicted the submission would shorten the decode step by 0.84 ms at 1, 8 and 32 requests.

This is Claude's independent measurement, on main after merging round 2 (`4033cae`):
- **re-grading** the two submitted kernels and experts attempt-7;
- **checking all three at 25 row counts** that G2 doesn't test;
- **checking tokens end to end and timing the decode step (G3),** in plain MLX and vLLM.

G3 timed each kernel alone, the submission, experts attempt-7 added to it, round 1's three kernels and the compiled references, all in the same run. A last run timed every pairing of round 2's three kernels, to find where they interfere.

**Where everything is.** The raw data is in [results/runs/2026-11-04-muse-round-2/](../../results/runs/2026-11-04-muse-round-2/), with the pairings in [results/runs/2026-11-04-muse-round-2-pairs/](../../results/runs/2026-11-04-muse-round-2-pairs/) and the whole-request comparison in [results/runs/2026-11-04-engine-comparison/](../../results/runs/2026-11-04-engine-comparison/). The scripts are:
- `bench/grader/check_candidate.py`, the grader;
- `bench/grader/check_sizes.py`, G2's checks at the extra row counts;
- `bench/grader/time_candidates.py`, the end-to-end check and G3.

**The machine.** The same M1 Pro, with Docker quit. 19.5 to 20.6 GB of 20.5 to 21.5 GB of swap were in use (`machine-*.txt`). vLLM ran with `VLLM_METAL_KV_COMMIT_PROBE=0`.

## Summary

- **Every kernel round 2 graded is correct.** The two submitted kernels and experts attempt-7 are bit-exact with MLX's reference:
  - at the grader's six sizes;
  - at 25 more row counts, from 2 to 1,500 rows, on either side of MLX's switches at 16 and 60 rows, and inside each of routing's row-count regimes;
  - end to end, where the model's greedy tokens are identical in plain MLX and in vLLM, at 1 and 32 requests, for every round-2 variant.

  Round 1's kernels, timed in the same runs as a control, still change the tokens with 32 requests.
- **routing attempt-4 passed G3 as it stood, on decode steps.** It is faster than both the unchanged model and the compiled reference at 1, 8 and 32 requests, in both engines. In vLLM it saves 0.47, 0.43 and 0.54 ms per step (8.7%, 3.0% and 1.8%). It is the project's first Muse kernel that passed every rule of its time. Over whole requests its gain is smaller, since it processes prompts 7–10% slower: throughput rises 10.2% with 1 request and 1.8% with 8, is flat with 32 and falls 3.5% with 32 requests of 1,024-token prompts ([section 5](#5-whole-requests-prompts-and-throughput)). Under the rule set after this measurement, that an improvement must hold in both phases ([ADR 0011](../adr/0011-decode-and-prefill.md)), it fails, so it is no baseline for round 3 ([For round 3](#for-round-3-baselines-and-starting-points)).
- **combine attempt-1 fails G3.** The grader timed it faster than the compiled reference at every row count. In the model it is slower than the unchanged step with 32 requests (+1.05 ms in vLLM, +1.17 ms in plain MLX), and no faster than the compiled reference with 8.
- **The submission (routing + combine), in vLLM:** −0.73 ms (−13%) with 1 request, −0.74 ms (−5%) with 8, **+0.53 ms (+2%) with 32**, against a prediction of −0.84 ms at each. combine's slowdown outweighs routing's saving with 32 requests.
- **experts attempt-7, not submitted, makes the decode step faster and whole requests much slower.** Per decode step in vLLM: −5.35 ms (−17.5%) with 32 requests, −1.22 ms (−8.4%) with 8, and +0.29 ms (+5.4%) with 1, which is why round 2 dropped it. But it processes prompts 6.6 to 14.5 times slower than the unchanged engine, so whole requests lose 10% to 83% of their throughput ([section 5](#5-whole-requests-prompts-and-throughput)). G3 times only decode steps, and this went unseen until the comparison in section 5.
- **All three slots replaced at once lose time with 8 requests,** as in round 1, although every pair of kernels adds up as expected. With 8 requests in plain MLX, the three alone save 1.87 ms together, and all three installed cost 0.94 ms ([section 4](#4-where-the-kernels-interfere)).
- **Against round 1,** in the same runs, round 2's submission is faster at 1 and 8 requests (−0.73 against +0.92 ms, and −0.74 against +0.13 ms) and slower at 32 (+0.53 against −1.85 ms), with identical tokens where round 1's change them.
- **The process held.** Round 2 ran the grader exactly once per attempt: 13 runs for 13 attempts (experts 7, routing 4, combine 2). Round 1 ran it 29 times for experts' 4.

## 1. The grader, on main

`grade-<slot>.json`; µs per layer, chained, as reference / compiled reference / attempt.

| Slot, attempt | G2 | 1 row | 8 rows | 32 rows |
|---|---|---|---|---|
| routing, attempt-4 | bit-exact at all 6 sizes | 82.0 / 63.4 / **47.7** | 196.6 / 128.6 / **117.0** | 160.9 / 129.4 / **126.2** |
| combine, attempt-1 | bit-exact at all 6 sizes | 62.9 / 55.8 / **44.3** | 65.7 / 60.1 / **46.3** | 62.3 / 47.3 / **42.2** |
| experts, attempt-7 (not submitted) | bit-exact at all 6 sizes | 151.4 / 143.5 / **140.7** | 235.3 / 233.6 / **168.9** | 801.6 / 800.6 / **512.4** |

All three beat the compiled reference here, even experts attempt-7 at 1 row, where round 2's own run had it slower (93.8 against 74.8 µs). The absolute times also differ from round 2's runs by up to 2 times at 1 row. Only the comparison within one run means anything, and at 1 row even that is close to the noise. G3 decides.

## 2. Row counts G2 doesn't test

`check_sizes.py` applies the grader's bit-exact checks to every layer's slot inputs in the first N tokens of a one-request prefill (`sizes-check.json`). The 25 values of N are:
- 2, 3, 5, 15, 16, 17, 31, 33, 40, 59, 60, 61, 64, 65;
- 100, 128, 129, 200, 256, 257, 500, 768, 769, 1,024 and 1,500.

They cover:
- either side of 16 rows, where the experts' sort starts;
- either side of 60 rows, where MLX's experts kernel changes to `gather_qmm_rhs`;
- either side of each boundary in routing attempt-4's notes, which copies MLX's matrix product in seven row-count regimes (1; 2 to 32; 33 to 64; 65 to 128; 129 to 256; 257 to 768; 769 and up).

**All three kernels are bit-exact at every one of the 25.**

## 3. End to end: tokens and G3

**Tokens.** 24 greedy tokens after 128-token prompts, at 1 and 32 requests (`muse-plain.jsonl`, `muse-vllm.jsonl`, records of kind `check`). Every round-2 variant, alone or combined, gives the unchanged model's tokens in both engines. Round 1's three kernels together change them with 32 requests in both, as before.

**G3, vLLM**, the change in decode step period against the unchanged model, in ms (median of 3 repeats, with their range in `console-vllm.txt`):

| Variant | 1 request | 8 | 32 |
|---|---|---|---|
| unchanged | 5.442 | 14.524 | 30.470 |
| routing attempt-4 | **−0.474** (−8.7%) | **−0.434** (−3.0%) | **−0.537** (−1.8%) |
| routing, compiled reference | −0.272 | −0.225 | −0.133 |
| combine attempt-1 | −0.372 | −0.197 | **+1.051** (+3.4%) |
| combine, compiled reference | −0.340 | −0.251 | −0.183 |
| **submission: routing + combine** | **−0.726** (−13.3%) | **−0.735** (−5.1%) | **+0.527** (+1.7%) |
| both compiled references | −0.404 | −0.505 | −0.179 |
| experts attempt-7, not submitted | +0.292 (+5.4%) | **−1.217** (−8.4%) | **−5.345** (−17.5%) |
| experts, compiled reference | −0.066 | +0.032 | −0.220 |
| submission + experts attempt-7 | −0.406 | +0.280 | −4.510 (−14.8%) |
| all three compiled references | −0.571 | −0.534 | −0.197 |
| round 1's three kernels | +0.923 | +0.127 | −1.849 |
| round 2's prediction for the submission | −0.84 | −0.85 | −0.84 |

**Plain MLX** (`console-plain.txt`), the change in ms against an unchanged step of 4.740, 13.623 and 28.485 ms:
- routing attempt-4: −0.486, −0.482, −0.404;
- combine attempt-1: −0.424, −0.112, **+1.168**;
- submission: −0.898, −0.772, **+0.486**;
- experts attempt-7: +0.529, −1.263, −4.880;
- submission + experts attempt-7: −0.310, **+1.001**, −4.159;
- round 1's three kernels: +1.054, +0.774, −1.895.

**Against the brief's G3,** an attempt must be faster than both the unchanged model and the compiled reference, beyond the spread of the repeats. Here, that means all three of its repeats are below all three of the others'.
- **routing attempt-4** passes at 1, 8 and 32 requests, in both engines.
- **combine attempt-1** fails:
  - with 32 requests it is slower than the unchanged model in both engines;
  - with 8, its range overlaps the compiled reference's in plain MLX, and its median is above it in vLLM;
  - with 1 request, its vLLM range touches the compiled reference's.
- **experts attempt-7** fails with 1 request, being slower than the unchanged model, and passes with 8 and 32.

**Why combine is slow in the model** is not known. Its chained piece timing at 32 rows is 42.2 µs per layer against the compiled reference's 47.3. A G3 loss of 1.05 ms is about 52 µs per layer, more than the kernel's whole time in the grader. So the cost doesn't come from the kernel's own work as the grader runs it. It comes from how it sits in the step, which the grader doesn't reproduce.

## 4. Where the kernels interfere

A separate plain MLX run timed round 2's three kernels in every combination, each kernel alone again, and routing's compiled reference (`results/runs/2026-11-04-muse-round-2-pairs/console-plain.txt`). Every combination kept the tokens identical. Change in ms against an unchanged step of 4.731, 13.528 and 28.442 ms:

| Variant | 1 request | 8 | 32 |
|---|---|---|---|
| routing attempt-4 | −0.469 | −0.383 | −0.392 |
| combine attempt-1 | −0.418 | −0.265 | +1.051 |
| experts attempt-7 | +0.323 | −1.219 | −5.048 |
| routing + combine | −0.938 | −0.776 | +0.413 |
| routing + experts | −0.266 | −1.345 | **−5.664** (−19.9%) |
| combine + experts | +0.205 | −1.130 | −4.491 |
| all three | −0.302 | **+0.937** | −4.197 |
| the three alone, added up | −0.564 | −1.867 | −4.389 |

- **Every pair comes within 0.5 ms of its parts' sum.**
- **All three together, with 8 requests, are 2.8 ms slower than their parts.** It is the only case that misses by much.
- **Round 1 did the same.** Its three kernels, with 8 requests, came out at +0.88 ms in plain MLX against −1.95 ms for the parts ([round 1](muse-round-1.md#3-g3-the-decode-step)). Two different sets of kernels show it, so the cause is likely in how the step runs with all three slots replaced at that size, not in one kernel. It is not explained.
- **The fastest decode step with exact output** is routing attempt-4 with experts attempt-7: −20% with 32 requests, −10% with 8 and −6% with 1 in plain MLX, and −18.8% with 32 requests in vLLM (section 5). Over whole requests it is the slowest version measured, because of experts attempt-7's prompt processing.

## 5. Whole requests: prompts and throughput

A last vLLM run (`results/runs/2026-11-04-engine-comparison/`, tables built by `bench/grader/engine_table.py`) timed eight versions of the engine at four workloads: 1, 8 and 32 requests with 128-token prompts, and 32 requests with 1,024-token prompts. Besides the decode step it recorded the time until every prompt was processed and each run's whole time, as generated tokens per second. Every version but round 1's kept the tokens identical. Throughput change against the unchanged engine (prompt time as a multiple of the unchanged engine's in brackets):

| Version | 1 × 128 | 8 × 128 | 32 × 128 | 32 × 1,024 |
|---|---|---|---|---|
| compiled references, all three slots | +11.9% (1.04×) | +3.1% (0.99×) | +0.6% (0.98×) | +0.9% (0.98×) |
| round 2: routing attempt-4 | +10.2% (1.07×) | +1.8% (1.09×) | +0.3% (1.10×) | −3.5% (1.10×) |
| round 2 submission: routing + combine | +15.9% (1.11×) | +2.1% (1.09×) | −2.7% (1.08×) | −4.1% (1.09×) |
| round 2: routing + experts attempt-7 | −10.0% (6.63×) | −37.5% (14.50×) | −48.0% (12.50×) | −82.5% (13.09×) |
| round 2: all three with experts attempt-7 | −10.7% (6.75×) | −43.0% (14.33×) | −53.7% (14.37×) | −83.3% (13.66×) |
| round 1: all three (changes the output) | −13.3% (1.60×) | −9.6% (2.30×) | −8.1% (2.30×) | −33.6% (2.32×) |

- **Compiling MLX's own code for the three slots,** with no new kernel, gains 0.6% to 3.1% with 8 and 32 requests, where its repeats and the unchanged engine's don't overlap. With 1 request its median gain is 11.9%, but one of its three repeats falls inside the unchanged engine's range (1,374 to 1,559 ms against 1,486 to 1,580), so that gain is not certain. No Muse version gains in every workload.
- **The unchanged engine is vLLM's own.** It ran with Claude's step hooks and vLLM's engine in the script's process. Alternating runs on the same afternoon found it within −0.2% to +1.9% of vLLM with neither, inside the spread between repeats in every workload (`bench/micro/clean_vllm.py`, [results/runs/2026-11-04-clean-vllm/](../../results/runs/2026-11-04-clean-vllm/)).
- **Round 2's kernels are slower than MLX's when processing prompts,** in every slot measured: routing alone by 7–10%, routing with combine by 8–11%, and experts attempt-7 by 6.6 to 14.5 times. Every prompt chunk here has at least 128 rows, so it runs experts attempt-7's path for 60 rows and more, which copies `gather_qmm_rhs`'s rounding but not its speed.
- **The grader checks the kernels' output at prompt sizes, and G3 times only decode steps,** so neither saw this. Its decode-step figures in sections 1 to 4 stand; they describe one part of a request.

## 6. How round 2 worked

- **Time:** 63 minutes; the lead and 34 helpers; 983 model calls. 981 were `muse-wonder-1.0`, and 2 were Muse Code's reminder agents on `muse-spark-1.3` (`muse-spark/transcripts/01a1a48b-e027-7462-898f-0b01290c8c9a.summary.json`).
- **Tokens:** 125.6M input, of which 121.1M were cached, and 0.77M output. At Muse Wonder's prices that is about $58: $18 uncached input, $24 cached input and $15 output. It would be about $65 if the 0.36M reasoning tokens are billed on top of the output.
- **Attempts:** one grader run per attempt, 13 in total, as the skill asks. Every graded attempt is committed.
- **What it got right:** it read round 1's measurement, made exactness its first goal, and met it in every slot. Its explorers found that routing's reference changes its matrix-product strategy in row-count regimes, and its builder copied each regime exactly. Those regimes are what section 2 tests.
- **What it missed:** its prediction for combine (−0.35 ms) and for the submission at 32 requests. The chained piece timing still doesn't show combine's cost in the step.

## For round 3: baselines and starting points

**The rule,** from round 3 ([ADR 0011](../adr/0011-decode-and-prefill.md)): a slot's baseline is its best attempt so far that passes the current G2, has `beats_compiled` true in the grader, and passes G3. Both of those now judge two phases, decode and prefill: faster in one, slower in neither, and in G3 no whole request slower, against the unchanged model and the compiled reference. An attempt that passes G2 but not the rest is a starting point, not a bar: making it the bar would ask a fix to beat it where it already wins.

The current grader re-graded round 2's three kernels on 2026-11-04 (`results/runs/2026-11-04-muse-round-2/grade-phases-*.json`). Times are multiples of the compiled reference's, per layer:

| Slot | Baseline | Starting point | Decode, 1 / 8 / 32 rows | Prefill, 127 / 1,016 / 2,048 rows | Why it is no baseline |
|---|---|---|---|---|---|
| `routing` | none | `muse-spark/round-2/kernels/routing/attempt-4` | 0.76 / 0.84 / 0.84, faster | 2.6 / 5.9 / 8.7, slower | Slower at every prefill size, so prompts take 7–10% longer and whole requests with 1,024-token prompts lose 3.5% (section 5). |
| `combine` | none | `muse-spark/round-2/kernels/combine/attempt-1` | 0.86 / 0.85 / 0.93, faster | 0.95 / 0.83 / 0.52, faster | Its decode step in G3 was 1.05 ms slower than the unchanged model's with 32 requests, for a reason the grader doesn't show (section 3). |
| `experts` | none | `muse-spark/round-2/kernels/experts/attempt-7` | 1.17 / 0.69 / 0.65, slower at 1 row | 15.8 / 27.7 / 20.1, slower | Slower at 1 row and at every prefill size, so prompts take 6.6 to 14.5 times as long (section 5). |

Round 3 has no baseline in any slot, so its bar is the compiled reference everywhere.

**What round 3 has to find out,** in Claude's reading of these results:
1. **experts with 1 row.** attempt-7 keeps its 8- and 32-row speed and loses at 1 row, as every fused design in round 2 did (92 to 110 µs per layer chained, against 73 to 87 for the compiled reference, in round 2's runs). Round 2's summary suggests copying the reference's unfused structure exactly for small row counts.
2. **Why combine attempt-1 costs 1 ms in the step with 32 requests** and not in the grader.
3. **Why all three slots replaced at once lose 2.8 ms with 8 requests,** when every pair adds up (section 4).
4. **Prefill speed.** From round 3 the grader times the prefill sizes and G3 times prompts and whole requests (ADR 0011). routing attempt-4 and experts attempt-7 are slower than the compiled reference at every prefill size; combine attempt-1 is faster. An attempt may keep inputs above a row count it names on the compiled reference (`max_rows`, in the brief). Claude checked that hand-off on a copy of experts attempt-7 with `max_rows` 59: bit-exact at all six sizes, and the `same` as the compiled reference at every prefill size (`results/runs/2026-11-04-handoff-check/`).
