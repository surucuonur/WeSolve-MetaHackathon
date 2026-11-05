# Round 4 of the Muse agent, measured

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

In round 4, the last (2026-11-04, 21:38 to 23:07 UTC), the Muse agent ran skill version 10.
- **The model:** Muse Wonder, `muse-wonder-1.0`, at `max` effort, with tool approvals off ([ADR 0014](../adr/0014-muse-wonder-last-round.md)).
- **The starting point:** [Claude's measurement of round 3](muse-round-3.md).
- **Three new rules** ([ADR 0015](../adr/0015-sets-and-hand-off-below.md)):
  - candidates that pass the ready check together are ready as a set;
  - an attempt may also hand inputs below `min_rows` to the compiled reference;
  - the grader reports a handed-off size as `handed off` instead of timing it.

The round made 26 commits, the last `22f1ac8`, and graded 6 attempts: 2 for experts, 3 for routing, 1 for combine. By the session logs' per-call token counts it cost about $36 at Muse Wonder's API prices (3.7M uncached and 52.6M cached input tokens, 0.53M output); that is an estimate, not the bill.

**Every attempt passed the grader and beat the compiled reference**, experts for the first time in four rounds. The experts attempt is round 3's attempt-1 unchanged, with `min_rows` 2.

**Both ready checks failed every variant** on overlapping ranges, so the summary, [muse-spark/round-4/summary.md](../../muse-spark/round-4/summary.md), submits the reference in every slot.

Claude measured the three attempts alone, in all three pairs and all together:
- **experts:** `muse-spark/round-4/kernels/experts/attempt-1`, round 3's fused kernel with `min_rows` 2 and `max_rows` 59;
- **routing:** `muse-spark/round-4/kernels/routing/attempt-1`, round 3's attempt-6 without the loop that zeroed 2,048 partial sums per call;
- **combine:** `muse-spark/round-4/kernels/combine/attempt-1`, round 3's attempt-2, byte for byte.

**Where everything is.**
- G3: [results/runs/2026-11-04-muse-round-4/](../../results/runs/2026-11-04-muse-round-4/), the verdict in `g3-vllm.json`.
- MLX alone: [results/runs/2026-11-04-muse-round-4-mlx/](../../results/runs/2026-11-04-muse-round-4-mlx/).
- The swarm's ready checks: [muse-spark/round-4/checks/](../../muse-spark/round-4/checks/).
- The pairs are `time_candidates.py --variant` runs: `muse-experts-routing` against `compiled-experts-routing`, and so on.

**The machine.** The same M1 Pro, with Docker quit and the round's Muse Code session idle. 16.6 to 17.5 GB of swap were in use, more than in any earlier measurement, and the repeats were noisier than round 3's: the unchanged model's own decode step at 32×1024 ranged from 42.0 to 48.7 ms.

## Summary

- **Every kernel is bit-exact, and the tokens are unchanged,** alone, in pairs and all together.
- **The best version is experts with routing.** Against the unchanged model in vLLM (medians of 5 repeats):
  - the decode step is 5.5%, 10.2%, 18.6% and 21.1% shorter with 1, 8 and 32 requests of 128 tokens and 32 of 1,024;
  - whole requests are 4.9% to 17.1% shorter, and throughput is 6% to 21% higher;
  - prompts are 2% to 11% faster with 8 and 32 requests.
- **Round 3's best, the routing + combine pair, shortened the decode step by 1.6% to 5.4%** with 8 and 32 requests ([round 3](muse-round-3.md)). With 1 request a set without experts is still as good, since experts then runs MLX's own code.
- **G3 fails every variant,** because every range overlaps a bar's in at least one workload. No variant is slower than either bar anywhere.
- **All three slots together lose with 8 requests** (+8.4% decode), while every pair gains there (−5.7% to −11.1%). The interaction rounds 1 and 2 found needs all three kernels at once.

## 1. The grader

All six attempts are bit-exact at the six sizes and have `beats_compiled` true. Chained µs per layer, attempt against compiled reference, from each attempt's own run:

| Attempt | 1 row | 8 rows | 32 rows | Prefill |
|---|---|---|---|---|
| experts attempt-1 (`min_rows` 2, `max_rows` 59) | handed off | 157.6 / 232.1 | 571.1 / 832.6 | handed off |
| experts attempt-2 (`max_rows` 32) | handed off | 170.5 / 236.8 | 538.1 / 823.2 | handed off |
| routing attempt-1 (no zeroing loop) | 58.5 / 86.1 | 136.9 / 157.3 | 74.7 / 92.1 | handed off |
| routing attempt-2 (router weights shared by 4 rows) | 44.4 / 59.2 | 71.6 / 87.6 | 83.8 / 92.1 | handed off |
| routing attempt-3 (busier last groups) | 49.3 / 66.2 | 74.5 / 86.1 | 81.8 / 94.8 | handed off |
| combine attempt-1 (round 3's attempt-2) | 39.3 / 47.1 | 45.5 / 60.5 | 43.2 / 69.6 | 0.66 / 0.33 / 0.32 of compiled |

- **Experts' `min_rows` worked as intended.** The fused kernel from round 2, which lost only at 1 row, is 31% faster than the compiled reference at 8 and 32 rows, on 30 wins of 30.
- **Fixing routing's re-read didn't pay.** Attempt-2 cut the router-weight traffic at 32 rows from 5.90 to 1.47 MB per layer, yet ran at 0.91 of the compiled reference there, against attempt-1's 0.81. Bytes were not what limited it.
- **The three routing attempts can't be ranked by the grader.** Their ratios differ by about as much as the same code drifts between runs, about ±0.05.

## 2. The ready checks

The swarm ran both of its two ready checks, 5 repeats each:
- `ready-1`: experts, routing and combine, alone and all three together;
- `ready-2`: the routing + combine pair.

Every variant failed on overlapping ranges, with no phase slower anywhere. In `ready-1` experts alone shortened the decode step by 10.3%, 13.2% and 12.6% at 8×128, 32×128 and 32×1024, and all three together lost 2.1% at 8×128. The swarm took no pair containing experts to a ready check. The skill allowed only two, and the lead spent the second on the routing + combine fallback.

The ready check and Claude's G3 agree in direction and in every verdict: every variant fails, nothing is slower.
- **Experts alone:** the ready check measured −10.3%, −13.2% and −12.6% at 8 and 32 requests, Claude's G3 −7.4%, −16.7% and −16.9%.
- **All three together:** both lose with 8 requests, by 2.1% in the ready check and 8.4% in G3.

## 3. G3: alone, in pairs and together

Claude's G3, the change of medians against the unchanged model, decode step / whole request, at 1×128 / 8×128 / 32×128 / 32×1024 (5 repeats):

| Variant | Decode step | Whole request | Throughput, tokens/s |
|---|---|---|---|
| unchanged model | 6.11 / 15.51 / 31.87 / 45.07 ms | 1.64 / 4.17 / 8.99 / 19.03 s | 156 / 491 / 911 / 430 |
| compiled reference, all three slots | −5.0 / −2.7 / −2.1 / −7.4% | −3.8 / −2.1 / −2.5 / −9.6% | 163 / 501 / 935 / 476 |
| experts alone | +3.6 / −7.4 / −16.7 / −16.9% | +5.9 / −4.8 / −15.4 / −13.3% | 148 / 516 / 1,077 / 496 |
| **experts + routing** | **−5.5 / −10.2 / −18.6 / −21.1%** | **−4.9 / −7.9 / −16.9 / −17.1%** | **165 / 533 / 1,097 / 519** |
| experts + combine | −2.9 / −11.1 / −16.8 / −19.0% | −1.1 / −10.2 / −15.6 / −16.0% | 158 / 547 / 1,079 / 513 |
| routing + combine | −8.6 / −5.7 / −2.2 / −9.9% | −7.5 / −4.4 / −2.0 / −11.7% | — |
| all three | −10.1 / **+8.4** / −14.0 / −18.3% | −9.6 / **+9.7** / −9.0 / −15.9% | 173 / 448 / 1,001 / 512 |

Against each variant's own compiled counterpart, the decode step of experts + routing is 2.2%, 10.7%, 18.5% and 15.8% shorter; of experts + combine 4.3%, 9.3%, 16.4% and 15.6%.

**How to read it.**
- **The unchanged model's 32×1024 time is high in this run:** 45.07 ms, against 39.4 to 40.9 in earlier runs, with repeats from 42.0 to 48.7 ms. The changes against it at 32×1024 are overstated by a few points. The comparison with the compiled counterparts, timed in the same run, is the safer one: about 16%.
- **With 1 request, experts adds nothing.** Every decode step has 1 row, so it runs MLX's own code by design. The 1×128 gains come from routing and combine, and they are within the spread.
- **Prompts** are 2% to 11% faster for both pairs with experts with 8 and 32 requests, and nothing is slower beyond the spread.

**Why G3 fails it.** Every range of experts + routing overlaps a bar's in at least one workload, most of them at 1×128 and 8×128. With ranges of 1 to 7 ms on a machine this loaded, 5 repeats can't show a 10% gain at 8 requests. Its medians beat both the unchanged model's and its compiled counterpart's in 11 of the 12 measurements; the exception is the prompt at 1×128, 6.8% slower than the unchanged model's by median, within ranges of 51 to 85 ms.

## 4. The three-slot interaction

With 8 requests, each pair gains: experts + routing −10.2%, experts + combine −11.1%, routing + combine −5.7%. All three together lose 8.4%. In `ready-1` the same set lost 2.1% with 8 requests while experts alone gained 10.3%. The loss needs all three Muse kernels at once, and it appears only with 8 requests: with 32, all three together gain 14% to 18%. Rounds 1 and 2 saw the same with their own kernels, about 2.8 ms with 8 requests. Its mechanism is still unknown.

## 5. How round 4 worked

- **Fast and aimed.** Exploration took 8 minutes. The analysis ranked the `min_rows` hand-off first, and the first experts attempt passed with 30 of 30 wins at 8 and 32 rows, 15 minutes into the round.
- **The checkers did their job.** Each hypothesis failed its first check, and experts and routing their second too, on claims that an unmeasured cost was "zero" or "costs nothing", and on Unverified lists missing items the boundary named. Every claim in the final *Ready to test* cites a measurement.
- **Predictions were labelled and checked.** The lead predicted the three-slot set from piece timings, adding up the slots: −7.1 ms per step at 32×128 against the compiled reference, labelled a prediction. G3 measured −3.8 ms for that set, and −5.9 ms for experts + routing.
- **What it missed:** pairs with experts. With two ready checks allowed and the three-slot set failing at 8 requests, the second check went to the routing + combine pair, round 3's result, which had already missed G3 once. A pair with experts was the likelier pass, and is the best version measured.
- **The ready check's rule can't see these gains on this machine.** Every candidate in the round was faster by median in most workloads, and none could prove it with 5 repeats. A paired test on medians was considered and declined for round 4 ([ADR 0015](../adr/0015-sets-and-hand-off-below.md)).

## 6. Against MLX on its own

vLLM is one way to serve the model. mlx_lm can also run it directly, with no server: one batch of prompts, processed in chunks, then decode steps. Claude timed that too, with the unchanged model and with the best version, experts + routing ([results/runs/2026-11-04-muse-round-4-mlx/](../../results/runs/2026-11-04-muse-round-4-mlx/), 3 repeats, every variant's tokens identical). The two engines ran one after the other, not interleaved, so compare across them with care.

Decode step in ms and throughput in generated tokens per second, medians:

| Workload | MLX alone | MLX + Muse | vLLM original | vLLM + compiled MLX code | vLLM + Muse |
|---|---|---|---|---|---|
| 1×128 | 4.92 · 174 | 4.52 · 169 | 6.11 · 156 | 5.90 · 160 | 5.77 · 165 |
| 8×128 | 15.33 · 479 | 13.62 · 526 | 15.51 · 491 | 15.61 · 487 | 13.93 · 533 |
| 32×128 | 29.48 · 994 | 23.98 · 1,169 | 31.87 · 911 | 31.83 · 914 | 25.95 · 1,097 |
| 32×1024 | 38.23 · 504 | 31.56 · 554 | 45.07 · 430 | 42.27 · 468 | 35.58 · 519 |

*Muse* is experts + routing; *compiled MLX code* is the same two slots' reference under `mx.compile`.

- **The kernels help in both engines.** In MLX alone they shorten the decode step by 8%, 11%, 19% and 17%, close to vLLM's 6%, 10%, 19% and 21%.
- **vLLM with the kernels outserves MLX alone with 8 and 32 requests:** 533 against 479 tokens per second with 8, and 1,097 against 994 with 32.
- **With 1 request, MLX alone is fastest.** Its decode step is 4.9 ms against vLLM's 6.1. vLLM spends about 2 ms of engine CPU time per 1-row step, which round 1 measured and its recommendations name.
