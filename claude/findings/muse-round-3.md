# Round 3 of the Muse agent, measured

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

In round 3 (2026-11-04, 17:35 to 19:00 UTC), the Muse agent ran skill version 8 ([ADR 0013](../adr/0013-measure-what-you-claim.md)). The model was Muse Spark's contributor model, `muse-spark-1.3-contributor`, at `max` effort, with tool approvals off ([ADR 0012](../adr/0012-muse-spark-contributor-unattended.md)). It started from [Claude's measurement of round 2](muse-round-2.md). There was no baseline in any slot, so the compiled reference was the bar everywhere. The round made 24 commits, the last `46679e7`, and graded 13 attempts: 5 for experts, 6 for routing and 2 for combine.

For the first time, the swarm ran G3 itself before calling anything ready: the **ready check**, `bench/grader/check_ready.py` ([ADR 0013](../adr/0013-measure-what-you-claim.md)). Its two candidates each failed it alone, on overlapping ranges, and passed it together. The skill counted only candidates that pass alone, so the summary, [muse-spark/round-3/summary.md](../../muse-spark/round-3/summary.md), submits the reference in every slot. It reports the pair as measured but not submitted:
- **routing:** `muse-spark/round-3/kernels/routing/attempt-6`, round 2's attempt-4 kernels with `max_rows` 32;
- **combine:** `muse-spark/round-3/kernels/combine/attempt-2`, round 2's attempt-1 with its float contractions turned off.

Claude measured the pair afterwards, alone and together, as G3 does.

**Where everything is.**
- G3: [results/runs/2026-11-04-muse-round-3/](../../results/runs/2026-11-04-muse-round-3/), the verdict in `g3-vllm.json`.
- One request with long prompts: [results/runs/2026-11-04-muse-round-3-long/](../../results/runs/2026-11-04-muse-round-3-long/).
- The swarm's ready check: [muse-spark/round-3/checks/ready-1/](../../muse-spark/round-3/checks/ready-1/).
- The script: `bench/grader/time_candidates.py vllm <out> routing=… combine=…`, with `--settings 1x1024,1x7936 --max-model-len 8192` for the long prompts.

**The machine.** The same M1 Pro, with Docker quit. 13.9 to 14.2 GB of 15.4 GB of swap were in use (`machine-*.txt`). vLLM ran with `VLLM_METAL_KV_COMMIT_PROBE=0`.

## Summary

- **Every kernel is bit-exact, and the tokens are unchanged.** Every variant, alone or together, gives the unchanged model's greedy tokens.
- **The pair is the best result of any round.** Against the unchanged model in vLLM, routing attempt-6 with combine attempt-2:
  - shortens the decode step by 16.5%, 5.4%, 2.8% and 1.6% with 1, 8 and 32 requests of 128 tokens and 32 of 1,024;
  - shortens whole requests by 14.1%, 5.6%, 2.1% and 2.4%;
  - processes prompts 1.6 to 3.1% faster, and is slower nowhere.
- **G3 still fails the pair, by one workload.** It passes against the unchanged model, but against the compiled reference (`compiled-all`) its decode step with 32 requests of 1,024 tokens is only 0.9% shorter, and the ranges of the 5 repeats overlap.
- **With one request and long prompts, the pair passes both bars.** With 1,024- and 7,936-token prompts its decode step is 14.8% and 11.3% shorter than the unchanged model's, and 8.1% and 6.3% shorter than the compiled reference's.
- **The ready check measured what G3 measures.** The swarm's decode changes match Claude's within 0.15 ms in every workload, and 2 of its 3 verdicts are the same. The third, the pair against `compiled-all`, passed in the swarm's run and failed in Claude's.
- **Experts never reached the ready check.** Its best attempt lost to the compiled reference only at 1 row (section 4).

## 1. The ready check against Claude's G3

Both runs time the same variants in the same four workloads, 5 shuffled repeats each. The decode-step change against the unchanged model, in ms, at 1×128 / 8×128 / 32×128 / 32×1024 (requests × prompt tokens; medians):

| Variant | The swarm's ready check | Claude's G3 | Verdicts, swarm / Claude |
|---|---|---|---|
| `muse-routing` | −0.38 / −0.31 / −0.59 / −0.48 | −0.41 / −0.31 / −0.61 / −0.62 | fail / fail |
| `muse-combine` | −0.53 / −0.59 / −0.44 / −0.26 | −0.56 / −0.51 / −0.45 / −0.48 | fail / fail |
| `muse-all`, both | −0.92 / −0.89 / −0.73 / −0.78 | −0.90 / −0.77 / −0.83 / −0.65 | pass / **fail** |

**Why the verdicts differ.** Every failure in both runs is an overlap of ranges: no variant is slower than either bar anywhere.
- `muse-routing` fails against `compiled-routing` at 8×128 in both runs, and in Claude's also against the unchanged model at 32×1024.
- `muse-combine` fails in the swarm's run on one 1×128 repeat (5.892 ms, above the unchanged model's lowest, 5.430). In Claude's it fails against `compiled-combine` at 32×128 and 32×1024.
- `muse-all` against `compiled-all` at 32×1024: in Claude's run the 5 decode times are 38.59 to 39.08 ms and the compiled reference's 38.91 to 39.40, medians 0.9% apart. In the swarm's run the medians were 1.5% apart, and the ranges just missed each other.

With gains of about 1%, 5 repeats decide by chance. Every median favours the kernels, but the rule asks for ranges that miss each other. That rule stays for round 4 ([ADR 0015](../adr/0015-sets-and-hand-off-below.md), *Considered options*).

## 2. G3: whole requests and prompts

Claude's G3, the change against the unchanged model (medians of 5 repeats; the unchanged step was 5.457, 14.337, 30.061 and 39.410 ms):

| Workload | Variant | Decode step | Prompt time | Whole request |
|---|---|---|---|---|
| 1×128 | routing | −7.5% | −2.7% | −6.9% |
| | combine | −10.2% | −1.0% | −7.7% |
| | both | **−16.5%** | −3.1% | **−14.1%** |
| | compiled reference, both slots | −8.5% | −4.1% | −8.0% |
| 8×128 | both | **−5.4%** | −1.6% | **−5.6%** |
| | compiled reference | −2.8% | +0.1% | −2.7% |
| 32×128 | both | **−2.8%** | −2.7% | **−2.1%** |
| | compiled reference | −0.9% | −1.4% | −0.8% |
| 32×1024 | both | **−1.6%** | −2.8% | **−2.4%** |
| | compiled reference | −0.7% | −0.3% | −0.6% |

Bold: faster than the unchanged model beyond the spread of the repeats.

- **The gain is about the same in milliseconds everywhere:** 0.65 to 0.9 ms per decode step. The share falls as more requests make each step longer.
- **Prompts gain a little, and lose nothing.** Routing hands every prompt-sized input to the compiled reference (`max_rows` 32), and combine attempt-2 is faster than the reference at every prefill size. Round 2's routing kernel made prompts 7 to 10% slower; this one doesn't.
- **Half the gain is compiling MLX's own code.** `mx.compile` of the two slots' reference code alone gains 8.5% at 1×128. The kernels add the other 8.8%.

## 3. One request with long prompts

G3's workloads stop at 1,024-token prompts with 32 requests. Claude also timed one request with 1,024- and 7,936-token prompts, the longest that fits the model's 8,192-token context with 256 generated tokens ([results/runs/2026-11-04-muse-round-3-long/](../../results/runs/2026-11-04-muse-round-3-long/)). `time_candidates.py` gained `--max-model-len` for this; G3's default is unchanged.

| Prompt | Decode step | Prompt time | Whole request | Throughput |
|---|---|---|---|---|
| 1,024 tokens | 5.7 → 4.9 ms, **−14.8%** | **−2.9%** | **−13.8%** | 152 → 176 tokens/s |
| 7,936 tokens | 8.2 → 7.3 ms, **−11.3%** | −2.1% | **−6.4%** | 56 → 60 tokens/s |

Against the compiled reference, the decode step is 8.1% and 6.3% shorter. In these two workloads the pair passes G3's rule against both bars, and so does combine attempt-2 alone. The decode step saves about the same 0.85 to 0.93 ms however long the prompt is. A 7,936-token prompt takes about half the request, though, so the whole request gains less.

## 4. Experts: lost at 1 row only

hyp-2 graded 5 attempts and dropped the slot. The best, attempt-1, is round 2's fused kernel with `max_rows` 59. It is bit-exact, `same` as the compiled reference at every prefill size, and much faster at 8 and 32 rows, but slower at 1. Chained µs per layer, attempt against compiled reference:

| Rows | 1 | 8 | 32 |
|---|---|---|---|
| attempt-1 | 96.9 | 168.9 | 554.7 |
| compiled reference | 86.6 | 238.2 | 883.4 |

`beats_compiled` needs one phase faster at every size, so the 1-row loss sank it. The round then tried to win at 1 row, in two ways:
- **a vectorised scalar path** (attempt-2) bought nothing;
- **an unfused path copying the reference's structure** (attempt-5) was twice as slow at 1 row.

The round's lead concluded that "no third exists in the spec surface". The spec could not keep 1 row on MLX's own code while running the kernel at 8 and 32 rows; from round 4 it can (`min_rows`, ADR 0015). No experts attempt has been timed end to end since round 2's attempt-7, which shortened the decode step by 17.5% with 32 requests and processed prompts 6.6 to 14.5 times slower.

## 5. How round 3 worked

The new rules held:
- every end-to-end number in *Ready to test* comes from the ready check, and predictions are labelled;
- `checker-final` found 7 rounding slips in the last digit and had them fixed before the round ended;
- when the fused experts kernel kept losing at 1 row, attempt-5 changed approach to an unfused path, and its ledger line says so.

What went wrong:
- **Builders were spent on directions exploration had already ruled out:** experts attempt-2 and routing attempt-3.
- **Extra kernel launches cost more than they saved.** Routing attempt-3's added copy launch, about 2 to 3 µs per layer, flipped its decode from `faster` to `same`. Experts attempt-5's five launches instead of two cost about 18 µs per layer, part of its 2× loss at 1 row (`hyp-2-experts-prefill.md`, `hyp-3-routing-prefill.md`).
- **The thread limit cost two attempts.** Metal's limit per pipeline depends on the kernel's registers, 640 and then 576 threads here, while the grader checks only for 1,024. Routing attempts 2 and 4 failed G1 on it.
- **The grader timed identical code against itself.** Above `max_rows` an attempt runs the compiled reference, and the grader still timed it against the compiled reference. That judged routing attempt-1 `slower` at 2,048 rows on 8 wins of 30, and cost a re-grade of a byte-identical copy (attempt-6). From round 4 such a size is `handed off` (ADR 0015).
- **The set had no rule.** The skill said submitting together needs `muse-all` to pass, but made only single candidates `ready`. The lead read it strictly and submitted nothing. From round 4 a passing `muse-all` makes its candidates `ready` as a set (ADR 0015).
- **Absolute grader times drift up to twice between runs** on this machine, so comparing attempts across runs misleads; the round learned to compare ratios against the compiled reference from the same run.

## For round 4: baselines and starting points

**The rule** is unchanged ([ADR 0011](../adr/0011-decode-and-prefill.md)): a slot's baseline is its best attempt so far that passes the current G2, has `beats_compiled` true in the grader, and passes G3. Round 4 adds sets and `min_rows` ([ADR 0015](../adr/0015-sets-and-hand-off-below.md)).

| Slot | Baseline | Starting point | Grader | G3 | Why it is no baseline |
|---|---|---|---|---|---|
| `routing` | none | `muse-spark/round-3/kernels/routing/attempt-6` | `beats_compiled` true; decode 0.72 / 0.90 / 0.88 of the compiled reference at 1 / 8 / 32 rows, prefill `handed off` (Claude's re-grade with the round-4 grader, [results/runs/2026-11-04-grader-hand-off/](../../results/runs/2026-11-04-grader-hand-off/)) | fails alone: `same` as `compiled-routing` at 8×128 | G3 alone |
| `combine` | none | `muse-spark/round-3/kernels/combine/attempt-2` | `beats_compiled` true in both phases (round 3's grade) | fails alone: `same` as `compiled-combine` at 32×128 and 32×1024 | G3 alone |
| `experts` | none | `muse-spark/round-3/kernels/experts/attempt-1` | slower at 1 row, faster at 8 and 32, prefill `same` (section 4) | not timed | the 1-row loss |

**The pair** routing attempt-6 with combine attempt-2 passes G3 against the unchanged model, and against the compiled reference everywhere but 32×1024, where it is 0.9% shorter with overlapping ranges (section 1). Under round 4's set rule it would be a `ready` set, but it is no baseline. Round 4's bar is the compiled reference in every slot.

**What round 4 has to find out,** in Claude's reading of these results:
1. **experts with `min_rows`.** Whether attempt-1's 8- and 32-row speed carries into G3 once 1 row runs MLX's own code. With 1 request it then adds nothing, so it can pass G3 only in a set whose other kernels are faster with 1 request, such as the pair.
2. **All three slots at once.** In rounds 1 and 2, replacing all three slots lost about 2.8 ms with 8 requests, for an unknown reason ([round 2](muse-round-2.md), section 4). A set with experts meets that again.
3. **A margin at 32×1024.** The pair gains 0.25 to 0.3 ms over the compiled reference there, about 1%, which 5 repeats can't tell from noise. A set needs more than that with 32 long prompts.
