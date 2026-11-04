# Round 1 of the Muse agent, measured

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

In round 1 (2026-11-03), the Muse agent (Muse Spark, `muse-spark-1.3-contributor`) explored the decode step as a swarm, then wrote fused Metal kernels for three slots of the mixture-of-experts block. Its summary, [muse-spark/summary.md](../../muse-spark/summary.md), names one final attempt per slot. Each one passed the grader, and the summary predicted, from the per-slot timing, that together they would shorten the decode step by 0.54, 1.81 and 5.72 ms at 1, 8 and 32 requests.

This is Claude's independent measurement of those three attempts, done on main after merging round 1 (`a4995e4`), on 2026-11-04:
- **re-grading** them with the grader;
- **checking them end to end** in the model, in plain MLX and in vLLM;
- **timing the decode step (G3)** with each slot alone and with all three together.

It is the measurement that round 2's skill reads first: the last section names what round 2 starts from in each slot.

**Where everything is.** The raw data is in [results/runs/2026-11-04-muse-round-1/](../../results/runs/2026-11-04-muse-round-1/). The scripts are:
- `bench/grader/check_candidate.py`, the grader;
- `bench/grader/time_candidates.py`, the end-to-end check and G3;
- `bench/micro/chain_timing.py`, the chained-timing experiment.

**The machine.** The M1 Pro measured the baseline ([method](../benchmark/method.md)). Docker was quit for these runs, with surucuonur's permission. Another job of surucuonur's was running, and 21.3 to 22.7 GB of the 22.5 to 23.5 GB of swap were in use (`machine-*.txt`). vLLM ran with `VLLM_METAL_KV_COMMIT_PROBE=0`, as in earlier runs.

## Summary

- **All three attempts pass the grader as round 1 ran it.** On main, each one compiles, matches the reference within G2's tolerances at 1, 8 and 32 rows, and beats the compiled reference in the per-slot timing.
- **In the model, all three together make the decode step slower with 1 and 8 requests.** In vLLM:
  - +0.98 ms (+18%) with 1 request;
  - +0.16 ms (+1%) with 8;
  - −1.76 ms (−6%) with 32.

  Round 1's per-slot predictions (its piece basis, `muse-spark/implementation/README.md`), added up by Claude, were −0.54, −1.81 and −5.72 ms; round 1 itself stated no total.
- **The experts kernel alone slows the step by 1.2 ms (22%) with 1 request,** although the grader timed it 36% faster than the reference.
  - The grader ran the 20 layers' calls with nothing between them, so the GPU overlapped them.
  - In the model each layer waits for the one before. Chained like that, the kernel takes 133 µs per layer against the reference's 80.
- **Each attempt changes the model's tokens,** breaking the brief's end-to-end rule. With 32 requests in vLLM, each one alone changes 7 to 12 of the 32 sequences from the first generated token.
  - **experts:** the prefill runs the same slot at 127 to 2,048 rows. There, MLX's own experts kernel changes, and the attempt differs from the reference in 62% of its outputs. The grader tested decode sizes only.
  - **routing and combine:** their small rounding differences, within G2's tolerances, are enough to flip greedy tokens.
- **The grader changes for round 2** ([ADR 0010](../adr/0010-exact-at-every-size.md)). G2 is now bit-exact and also covers the prefill's sizes, and the timing chains the layers. None of round 1's attempts passes the new G2.
- **Round 2 has no baseline in any slot.** Its bar is the compiled reference. Round 1's three final attempts are its starting points.
- **Round 1 ran the grader 29 times on the experts slot** for the 4 attempts its summary counts, 18 of those on attempt-1.

## 1. The grader, as round 1 ran it

Re-run on main with the grader of `a4995e4` (`grade-<slot>.json`). µs per layer: reference / compiled reference / attempt.

| Slot, attempt | 1 row | 8 rows | 32 rows | G1, G2 |
|---|---|---|---|---|
| experts, attempt-4 | 143.8 / 139.5 / **92.1** | 222.5 / 221.4 / **165.4** | 770.1 / 762.4 / **529.9** | pass, pass (0 units in the last place) |
| routing, attempt-1 | 39.9 / 31.5 / **25.3** | 83.5 / 42.7 / **24.3** | 101.2 / 87.6 / **68.9** | pass, pass (weights within 2.6×10⁻⁶) |
| combine, attempt-1 | 23.1 / 23.8 / **18.6** | 22.5 / 26.1 / **17.0** | 39.5 / 32.0 / **18.9** | pass, pass (up to 2 units) |

Round 1's own runs gave the attempts 50.0 / 157.8 / 515.4 µs (experts), 28.5 / 23.6 / 41.8 (routing) and 17.2 / 17.0 / 18.5 (combine). The experts kernel at 1 row is the outlier, at 92 µs here against 50 then. Round 1's summary already warned that 1-row timings shift between runs.

## 2. End to end: the tokens

`time_candidates.py` installs each attempt in every layer's `MoE.__call__`, and generates 24 greedy tokens after 128-token prompts. It does this for each slot alone, for all three together, and for each slot's compiled reference. The brief's G2 asks for the same tokens as the unchanged model. Sequences that differ, and the first generated token where one does (`muse-plain.jsonl`, `muse-vllm.jsonl`, records of kind `check`):

| Variant | Plain MLX, 1 request | Plain MLX, 32 | vLLM, 1 | vLLM, 32 |
|---|---|---|---|---|
| experts attempt-4 | 1, from token 2 | 19, from token 0 | 0 | 12, from token 0 |
| routing attempt-1 | 1, from token 2 | 7, from token 1 | 0 | 9, from token 0 |
| combine attempt-1 | 1, from token 2 | 5, from token 2 | 0 | 7, from token 0 |
| all three | 0 | 19, from token 0 | 0 | 13, from token 0 |
| every compiled reference | 0 | 0 | 0 | 0 |

The compiled references give the same tokens in every case, so the unchanged model is deterministic, and every difference comes from the attempts.

**Why the tokens change.** Greedy decoding picks the most likely next token. On these random-token prompts, the top two candidates are often very close, so any change to the numbers can flip the choice, and every later token follows from it. That gives two causes:

- **The prefill (measured).** The model's prefill runs `MoE.__call__` on the prompt in chunks. With 128-token prompts and 2,048-token chunks, that is 127, 1,016 and 2,048 rows at 1, 8 and 32 requests.
  - **From the code:** from 60 rows on, MLX switches its experts kernel from `gather_qvm` to `gather_qmm_rhs`, which rounds differently. Round 1's own analysis found this threshold (`muse-spark/exploration/territory-switch-layers.md`, section G).
  - **The attempt:** experts attempt-4 copies `gather_qvm`'s arithmetic exactly, which is why it is bit-exact at decode sizes. With the prefill's inputs, it differs from the reference in 62% of its outputs, by up to 110 units in the last place (`grade-exact-experts.json`).
  - **The grader:** it never tested more than 32 rows.
- **Small rounding differences (measured).** routing attempt-1 picks the same experts in the same order, but every row's weights differ from the reference's, by up to 2.6×10⁻⁶ at decode sizes and 8.7×10⁻⁶ in the prefill. combine attempt-1 differs in 2 to 130 elements per size. Both are inside G2's old tolerances, and both still change 5 to 9 of 32 sequences.

## 3. G3: the decode step

The decode step's period: the median over 3 repeats, in shuffled order, at 1, 8 and 32 requests, with 128-token prompts and 256 generated tokens. The change is against the unchanged model, and the range is over the 3 repeats (`console-plain.txt`, `console-vllm.txt`).

**vLLM, ms per step:**

| Variant | 1 request | 8 | 32 |
|---|---|---|---|
| unchanged | 5.370 (5.320–5.452) | 14.363 (14.327–14.475) | 29.851 (29.657–29.952) |
| experts attempt-4 | **+1.162** (+21.6%) | −0.817 (−5.7%) | −4.154 (−13.9%) |
| experts, compiled reference | +0.066 | +0.024 | −0.103 |
| routing attempt-1 | **+0.185** (+3.4%) | −0.397 (−2.8%) | −0.490 (−1.6%) |
| routing, compiled reference | −0.220 | −0.183 | −0.157 |
| combine attempt-1 | −0.514 (−9.6%) | −0.705 (−4.9%) | −0.390 (−1.3%) |
| combine, compiled reference | −0.288 | −0.260 | −0.272 |
| all three attempts | **+0.980** (+18.3%) | **+0.163** (+1.1%) | −1.756 (−5.9%) |
| all three compiled references | −0.498 (−9.3%) | −0.558 (−3.9%) | −0.242 (−0.8%) |
| round 1's per-slot predictions, added up (piece basis) | −0.54 | −1.81 | −5.72 |

**Plain MLX**, change in ms against an unchanged step of 4.998, 13.541 and 28.496 ms:
- experts: +1.228, −0.832, −3.661;
- routing: +0.233, −0.361, −0.318;
- combine: −0.857, −0.753, −0.406;
- all three: +0.927, +0.881, −1.560;
- all three compiled references: −0.617, −0.524, −0.255.

What it shows:

- **Against the brief's G3,** an attempt must beat both the unchanged model and the compiled reference, by more than the spread between repeats. In vLLM:
  - **experts attempt-4** does so with 8 and 32 requests, and is slower than both with 1.
  - **routing attempt-1** does so with 32 requests. With 8, its range ends 0.05 ms below the compiled reference's; with 1, it is slower than both.
  - **combine attempt-1** does so with 8 requests. With 1, its range overlaps the compiled reference's; with 32, it is 0.12 ms ahead of it, within the spread.
- **All three together are slower than the parts suggest.**
  - With 8 requests in vLLM, the three alone save 0.82, 0.40 and 0.71 ms, but together they cost 0.16 ms.
  - With 32 requests, the experts kernel alone saves 4.15 ms, and all three together save 1.76 ms.
  - **Measured but not explained:** together the three launch 567 kernels per step with 1 request, against 967 unchanged, so a shorter list of kernels runs slower. The cause is an open question for round 2.
- **The compiled references alone save 0.5 to 0.6 ms** with 1 and 8 requests in both engines, without any new kernel.

## 4. Why the grader's timing missed the 1-request slowdown

**Measured.** `bench/micro/chain_timing.py` times the experts slot on the 20 layers' captured inputs in two ways (`console-chain.txt`, µs per layer):

| | 1 row | 8 rows | 32 rows |
|---|---|---|---|
| independent calls, as the grader ran them: reference / attempt-4 | 86.0 / **61.0** | 219.1 / 165.1 | 767.3 / 535.0 |
| chained calls, each waiting for the one before: reference / attempt-4 | 80.3 / **132.9** | 239.6 / 175.9 | 797.6 / 587.6 |

- **Chained, at 1 row,** attempt-4 is 66% slower than the reference: 52.6 µs × 20 layers is 1.05 ms per step, close to G3's +1.16 ms.
- **Independent,** the GPU runs the 20 calls side by side, and the attempt looks 29% faster.
- **Estimate, from the code:** attempt-4 gives each (row, expert) pair one threadgroup of 256 threads. With 1 row that is 4 threadgroups, too few to fill the GPU, so one call's time is long even when many calls together go fast. Round 1 called this "occupancy starvation", and attempt-4 doubled the threads per pair to fight it (`muse-spark/kernels/experts/attempt-4/block-0.metal`, opening comment).

## 5. How round 1 counted its attempts

Round 1's summary counts "experts 4, routing 2, combine 2" graded attempts. In its transcripts (`muse-spark/transcripts/`), the shell commands that ran the grader number:
- **experts: 29.** 18 on `attempt-1`, 3 on `attempt-2`, 6 on `attempt-3`, 1 on `attempt-4` and 1 on `probe-f64`;
- **routing: 2;**
- **combine: 3.** 1 on `attempt-1` and 2 on `attempt-2`.

`builder-1-a` edited attempt-1's files before nearly every one of its 18 runs, so that folder's kernel is the last of many versions rather than one try. Skill version 4 made "one grader run is one attempt" a rule, and version 5 keeps it.

## 6. What changes for round 2

surucuonur chose these rules on 2026-11-04 ([ADR 0010](../adr/0010-exact-at-every-size.md)). The [brief](../muse-agent-brief.md) and the grader now apply them:

- **Every call.** A slot's kernels run wherever `MoE.__call__` runs, the prefill included.
- **Bit-exact.** G2 asks for outputs identical to the reference's at 1, 8 and 32 rows of a decode step, and at 127, 1,016 and 2,048 rows of a prefill. For routing that means the same experts and weights, in the same order.
- **Chained timing.** Each layer's call waits for the one before, through one small link kernel that every function timed gets alike.

Round 1's final attempts under the new grader (`grade-exact-<slot>.json`):

| Slot, attempt | Decode: 1 / 8 / 32 rows | Prefill: 127 / 1,016 / 2,048 rows |
|---|---|---|
| experts, attempt-4 | exact / exact / exact | 4,868,667 / 39,078,912 / 78,741,557 elements differ (up to 43, 65 and 110.5 units) |
| routing, attempt-1 | 20 / 159 / 638 rows differ, of 20 / 160 / 640 | 2,518 / 16,187 / 32,743 rows differ |
| combine, attempt-1 | exact / 2 / 2 elements differ | 6 / 50 / 130 elements differ (up to 1, 16 and 32 units) |

Each count is summed over the 20 layers. None of the three passes.

## For round 2: baselines and starting points

**No slot has a baseline.** No earlier attempt passes the current G2. A new attempt must pass G1 and the new G2 and beat the compiled reference at 1, 8 and 32 rows, in the chained timing; there is no `--baseline` to pass. Each slot's starting point, the attempt to build from:

| Slot | Starting point | What keeps it from passing G2 | Its G3 alone, vLLM, 1 / 8 / 32 requests (ms) |
|---|---|---|---|
| `experts` | `muse-spark/kernels/experts/attempt-4` | Exact at decode sizes only. From 60 rows on it doesn't match `gather_qmm_rhs`'s rounding. | +1.16 / −0.82 / −4.15 |
| `routing` | `muse-spark/kernels/routing/attempt-1` | Every row's weights differ from the reference's in the last bits; the experts and their order match. | +0.19 / −0.40 / −0.49 |
| `combine` | `muse-spark/kernels/combine/attempt-1` | 2 to 130 elements differ per size, from 8 rows on. | −0.51 / −0.71 / −0.39 |

**What round 2 has to find out,** in Claude's reading of these results:
1. **The experts kernel at prefill sizes.** It must match MLX's `gather_qmm_rhs` from 60 rows on, bit for bit, or follow a design whose rounding equals both of MLX's kernels.
2. **The experts kernel at 1 row,** where it is slower than the reference when chained: 133 against 80 µs per layer.
3. **routing's and combine's exact arithmetic.** Each needs the reference's order of operations and rounding points, for example the float32 sum order of the router's product, and of combine's sum over the 4 pairs.
4. **Why the three together lose time** that each saves alone (section 3).

**Recommendations outside the slots** from round 1 stand as written in its [summary](../../muse-spark/summary.md). This measurement didn't test them.
