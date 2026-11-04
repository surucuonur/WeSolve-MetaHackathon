# Territory: g3 — Claude's round-2 measurement, end to end vs piece by piece

- **Owner:** explorer-g3
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T17:41:19Z
- **Next:** verifier to check ineff-g3-1..3; mechanisms belong to baseline-combine (g3-1), open (g3-2), baseline-experts/routing (g3-3)

## What it does

This territory is not code in the decode step. It is the measurement that decides whether round 2's attempts actually made the model faster: Claude's re-grading of the two submitted kernels (routing attempt-4, combine attempt-1) and experts attempt-7, the 25 extra row-count checks, the end-to-end token + decode-step (G3) runs, the pairings run, and the whole-request engine comparison, all written up in [claude/findings/muse-round-2.md](../../../claude/findings/muse-round-2.md). Raw data in `results/runs/2026-11-04-muse-round-2/`, `results/runs/2026-11-04-muse-round-2-pairs/` and `results/runs/2026-11-04-engine-comparison/`. Starting point: round 2's [territory-g3.md](../../round-2/exploration/territory-g3.md) and [ineff-g3-1.md](../../round-2/exploration/ineff-g3-1.md) / [ineff-g3-2.md](../../round-2/exploration/ineff-g3-2.md) (frozen; cited, not repeated). Its question for round 3 is where the end-to-end change differs from the piece timing, and why.

## Checklist

### 1. Re-grading under round 2's rules (`grade-<slot>.json`)

- **Callers:** Claude's measurement on `main` after merging round 2 at `4033cae` ([muse-round-2.md:15](../../../claude/findings/muse-round-2.md)), re-running `bench/grader/check_candidate.py` as round 2 ran it (decode-only chained timing).
- **Callees:** G1 (compiles/runs), G2 (bit-exact at all six sizes), chained piece timing (20 layers, median).
- **Data:** µs per layer, reference / compiled / candidate. Routing attempt-4: 82.0 / 63.4 / **47.7** at 1 row, 196.6 / 128.6 / **117.0** at 8, 160.9 / 129.4 / **126.2** at 32. Combine attempt-1: 62.9 / 55.8 / **44.3**, 65.7 / 60.1 / **46.3**, 62.3 / 47.3 / **42.2**. Experts attempt-7: 151.4 / 143.5 / **140.7**, 235.3 / 233.6 / **168.9**, 801.6 / 800.6 / **512.4**. All three G2 pass, `beats_compiled: true` here — including experts at 1 row (140.7 vs 143.5), where round 2's own run had it slower (93.8 vs 74.8).
- **Neighbours:** absolute times differ from round 2's runs by up to 2× at 1 row; only the within-run comparison means anything ([muse-round-2.md:55](../../../claude/findings/muse-round-2.md)).
- **Cost:** piece pricing only. Against the compiled reference × 20 layers: routing −0.314 / −0.232 / −0.064, combine −0.230 / −0.276 / −0.102, experts −0.056 / −1.294 / −5.764 ms/step at 1 / 8 / 32 rows.
- **Invariants:** decode-only basis; says nothing about prefill (piece 2 does that).

### 2. Re-grading with the two-phase grader (`grade-phases-<slot>-*.json`)

- **Callers:** same measurement, current grader (ADR 0011): `beats_compiled` needs faster-in-one-phase, slower-in-neither, at decode 1 / 8 / 32 and prefill 127 / 1,016 / 2,048 rows.
- **Data:** µs per layer, reference / compiled / candidate. Routing: decode 68.4 / 65.1 / **49.8**, 169.2 / 135.7 / **113.5**, 159.3 / 145.2 / **121.3** (faster); prefill 102.3 / 87.3 / **227.1** (2.6×), 231.3 / 221.1 / **1298.8** (5.9×), 310.0 / 290.4 / **2513.0** (8.7×). Combine: faster at all six (decode 52.8 / 53.4 / **45.7**, 71.2 / 56.4 / **47.9**, 87.0 / 69.1 / **64.6**; prefill 87.8 / 66.1 / **62.8**, 425.2 / 271.6 / **226.5**, 846.2 / 520.4 / **273.0**). Experts: decode 84.0 / 78.6 / **92.1** (slower at 1), 239.3 / 235.5 / **162.8**, 806.5 / 797.8 / **515.3**; prefill 963.4 / 960.8 / **15220.0** (15.8×), 3979.3 / 4306.0 / **119154.4** (27.7×), 7359.8 / 11998.9 / **240681.1** (20.1×).
- **Cost:** this is the piece basis every gap below is measured against. Piece-implied step changes vs the unchanged model (candidate − reference × 20): combine −0.142 / −0.466 / −0.448 ms/step at 1 / 8 / 32 rows; routing −0.372 / −1.114 / −0.760; experts +0.162 / −1.530 / −5.824.
- **Invariants:** `beats_compiled` true for combine only. Experts 1-row flips sign between runs (piece 1 vs piece 2: 140.7 vs 143.5 here-faster, 92.1 vs 78.6 here-slower) — 1-row piece comparisons sit near the noise.

### 3. Row counts G2 doesn't test (`sizes-check.json`, `console-sizes.txt`)

- **Callers:** `bench/grader/check_sizes.py` on every layer's slot inputs in the first N tokens of a one-request prefill, 25 values of N covering either side of 16 rows (experts sort starts), 60 rows (`gather_qvm` → `gather_qmm_rhs`), and each routing regime boundary.
- **Data:** all three kernels bit-exact at all 25. Correctness only, no timing.
- **Cost:** none — this piece removes inexactness as an explanation for every gap below. Round 1's gaps could hide behind tolerance-level differences; round 2's cannot.

### 4. G3 main run: tokens and decode step (`console-plain.txt`, `console-vllm.txt`, `muse-*.jsonl`)

- **Callers:** `time_candidates.py plain|vllm` (harness mechanics as in round 2's territory-g3 piece 4: `muse-<slot>` / `muse-all` / `compiled-*` variants, median of 3 shuffled repeats, 128-token prompts, 256 tokens; kernel counts from `count_ops`).
- **Data:** every round-2 variant keeps the greedy tokens in both engines at 1 and 32 requests; round 1's change them with 32 (control, reproduced). Decode-step change vs unchanged (vLLM, base 5.442 / 14.524 / 30.470 ms): routing −0.474 / −0.434 / −0.537; combine −0.372 / −0.197 / **+1.051**; submission (r+c) −0.726 / −0.735 / **+0.527**; experts-a7 +0.292 / −1.217 / −5.345; all+experts −0.406 / +0.280 / −4.510. Plain MLX (base 4.740 / 13.623 / 28.485): routing −0.486 / −0.482 / −0.404; combine −0.424 / −0.112 / **+1.168**; submission −0.898 / −0.772 / **+0.486**; experts +0.529 / −1.263 / −4.880; all+experts −0.310 / **+1.001** / −4.159. Round 2's prediction for the submission: −0.84 / −0.85 / −0.84.
- **Cost:** the submission gap vs prediction is +0.11 / +0.12 / **+1.37 ms** (vLLM). At 32 it is entirely combine: combine measured +1.051 vs predicted −0.35 (gap +1.40), routing measured −0.537 vs predicted −0.49 (beats by 0.05).
- **Invariants:** G3 rule — faster than both unchanged and compiled, beyond the repeat spread (all three repeats below all three of the others'). Combine at 32 fails outright (slower than unchanged, both engines, no range overlap); at 8 its median sits above compiled in vLLM ([14.254–14.330] vs compiled [14.205–14.417], overlapping) and overlaps compiled in plain; at 1 its vLLM range touches compiled's at 5.092.

### 5. Pairs run (`results/runs/2026-11-04-muse-round-2-pairs/console-plain.txt`, `muse-plain.jsonl`)

- **Callers:** plain-MLX run timing each kernel alone, all three pairs, and all three (base 4.731 / 13.528 / 28.442 ms). Every combination keeps the tokens identical.
- **Data:** 1 request: r −0.469, c −0.418, e +0.323, r+c −0.938, r+e −0.266, c+e +0.205, all −0.302. 8: r −0.383, c −0.265, e −1.219, r+c −0.776, r+e −1.345, c+e −1.130, all **+0.937**. 32: r −0.392, c +1.051, e −5.048, r+c +0.413, r+e −5.664, c+e −4.491, all −4.197.
- **Cost / additivity:** pair interaction (measured − sum of parts) at 8 requests: r+c −0.128, r+e +0.257, c+e +0.354 — all within ±0.5 ms. Three-way at 8: parts sum −1.867, measured +0.937, interaction **+2.804 ms** (all-three range [14.358–14.476] vs base [13.520–13.833], no overlap). At 1: three-way +0.262 vs base spread 0.275 wide — not claimed. At 32: +0.192 vs spread 0.057 — real but 15× smaller than at 8.
- **Invariants:** kernel counts exactly additive again (rows 1: −200 −80 −120 = −400 = 866−466; rows 32: −200 −80 −240 = −520 = 986−466), time is not — same fact round 2's territory-g3 pieces 5/6 met.

### 6. Engine comparison: prompts and whole requests (`console-vllm.txt`, `versions.csv`, `muse-vllm.jsonl`)

- **Callers:** vLLM run over 1×128, 8×128, 32×128, 32×1024 (requests × prompt tokens), recording decode-step period, prompt time (until every prompt processed) and whole-run tokens/s; tables built by `bench/grader/engine_table.py` into `versions.csv`.
- **Data:** throughput change vs unchanged (prompt multiple in brackets). Compiled-3: +11.9% (1.04×) / +3.1% (0.99×) / +0.6% (0.98×) / +0.9% (0.98×). Routing: +10.2% (1.07×) / +1.8% (1.09×) / +0.3% (1.10×) / −3.5% (1.10×). Submitted (r+c): +15.9% (1.11×) / +2.1% (1.09×) / −2.7% (1.08×) / −4.1% (1.09×). Routing+experts-a7: −10.0% (6.63×) / −37.5% (14.50×) / −48.0% (12.50×) / −82.5% (13.09×). All-three: −10.7% (6.75×) / −43.0% (14.33×) / −53.7% (14.37×) / −83.3% (13.66×). No Muse version gains in every workload; compiled references gain 0.6–3.1% at 8/32 (1-request gain uncertain: one repeat inside base range).
- **Cost:** absolute prompt times (ms, `versions.csv`): base 50.1 / 201.7 / 802.7 / 6521.3; routing 53.4 / 220.8 / 880.4 / 7183.8 (excess +3.3 / +19.1 / +77.7 / +662.5); routing+experts 332.5 / 2923.6 / 10031.7 / 85377.2 (excess +282.4 / +2721.9 / +9229.0 / +78855.9). The prefill piece timing (piece 2) predicts these: chunks × 20 layers × (candidate − reference), with 1 / 1 / 2 / 16 chunks of ≤2048 rows — routing 2.5 / 21.4 / 88.1 / 705.0 (within 15% everywhere); experts+routing 287.6 / 2324.9 / 9421.0 / 75368.4 (within 2% at 1×128 and 32×128, 5% at 32×1024, 15% low at 8×128). Submitted (r+c): predicted 2.0 / 17.4 / 65.1 / 521.0 vs measured +5.7 / +18.2 / +63.0 / +555.1 — within 7% except 1×128, where prompt reps overlap across variants (base [47.1–54.2], submitted [49.5–58.3]) so a 4 ms miss is spread.
- **Invariants:** the unchanged engine is vLLM's own within −0.2% to +1.9% (clean-vLLM control). Prompt multiples at 1×128 carry the repeat spread above; the large multiples do not (e.g. all-three 32×128 prompt [11511.6–11566.0] vs routing+experts [10026.1–10068.7], no overlap).

## Where the end-to-end change differs from the piece timing, and why

Three gaps, all new measurements (round 2's gaps 1–3 were round-1 kernels; gap 3's separating measurement — pairwise G3 — is piece 5 here):

1. **Combine at 32 requests: piece says −0.45 ms/step, step says +1.05/+1.17 ms — not explained.** Piece-implied (piece 2, vs unchanged): (64.6−87.0)×20 = −0.448 ms/step. G3 combine-alone: +1.051 (vLLM) / +1.168 (plain). Swing: +1.499 / +1.616 ms/step = **+75.0 / +80.8 µs per layer** — the loss alone (≈52–59 µs/layer vs unchanged) exceeds the kernel's whole chained time in the grader (64.6 µs/layer). Against the compiled reference the in-step penalty grows with rows: +6.1 / +11.2 / +66.2 µs/layer at 1 / 8 / 32 in vLLM (piece −0.154 / −0.170 / −0.090 vs in-step −0.032 / +0.054 / +1.234). Same sign in plain (+0.207 at 8, +1.123 at 32 vs compiled). Reproduced in 4 runs (both engines, main + pairs + engine-comparison submitted-minus-routing ≈ +0.91 at 32×128). A prefill twin: adding combine-a1 to routing+experts slows the 32×128 prompt by +1501.7 ms (no range overlap) where the prefill piece says combine *saves* 11.5 ms/layer/chunk at 2048 rows. Posted as [ineff-g3-1](ineff-g3-1.md). What the data rules out: the kernel's own chained work (measured faster), token divergence (exact everywhere, piece 3/4), repeat noise (no overlap, 4 runs). Candidate mechanisms belong to `baseline-combine` (grid `rows*768` threads, volatile-spill traffic scaling with rows — untested).
2. **All three together at 8 requests: parts sum −1.87 ms, measured +0.94 ms — measured, narrowed to three-way, not explained.** Three-way interaction +2.804 ms (pairs, plain) / +2.858 (main, plain) / +2.128 (main, vLLM) / +2.435 (engine, vLLM, from combos) — four runs, both engines, +2.1 to +2.9 ms. Every pair adds up (±0.35 at 8, piece 5), and the 32-request three-way interaction round 1 showed (+2.8/+3.3 ms) is gone (+0.19, piece 5). So: the 32-request interaction was a round-1-kernel property; the 8-request three-way interaction reproduces across two exact kernel sets and is likely in how the step runs with all three slots replaced at that size, not in one kernel. Posted as [ineff-g3-2](ineff-g3-2.md), citing frozen round-2 [ineff-g3-1](../../round-2/exploration/ineff-g3-1.md). Rules out (as before): kernel count (additive, piece 5), token divergence (exact), noise (no overlap), pairs (additive — new).
3. **Prompt time and whole requests: decode-only piece predictions overstate every whole-request gain — measured, and the prefill piece timing predicts the prompt side.** Round 2's decode-only prediction (−0.84/step) implied whole-request wins; measured throughput: submitted +15.9% / +2.1% / −2.7% / −4.1%, routing-only −3.5% at 32×1024, experts variants −10% to −83%. The new positive result: prefill piece timing composes additively to prompt time (chunks × layers, piece 6: routing within 15%, experts within 2–15%), so the two-phase grader prices prompts while decode keeps the two anomalies above. Posted as [ineff-g3-3](ineff-g3-3.md) (measurement side; kernel side owned by the `baseline-*` territories).

## Relations

- **Fed by:** the three starting-point attempts (territories `baseline-experts`, `baseline-routing`, `baseline-combine` — their piece timings are pieces 1/2 here), and the grader (territory `grader` — pieces 2/4/6 define the two-phase judgment and the ready check that must reproduce pieces 4–6 inside round 3).
- **Feeds:** every round-3 hypothesis. G3 decides (brief + ADR 0011), the chained two-phase piece timing guides; a per-slot win that does not survive composition (gaps 1–2) or the prefill phase (gap 3) is not a step win. The ready check measures each candidate alone and all together, so round 3 sees composition inside the round.

## Inefficiencies

- [ineff-g3-1](ineff-g3-1.md) — combine attempt-1 costs +1.05/+1.17 ms per decode step at 32 requests (and +1.5 s of 32×128 prompt with experts-a7) where the piece timing says faster (combine slot).
- [ineff-g3-2](ineff-g3-2.md) — all three slots replaced at once gain +2.1 to +2.9 ms/step at 8 requests while every pair adds up (three-way interaction).
- [ineff-g3-3](ineff-g3-3.md) — decode-only piece wins do not predict whole requests; prefill piece timing predicts prompt time within ~15% (measurement side of the prompt cost).

## Notes for other territories (outside-slot observations — no inefficiency posted)

- **For `baseline-combine`:** gap 1 is your in-step cost at 32 rows, measured end to end: +75.0 µs/layer swing from the piece figure in vLLM (+80.8 plain), penalty vs compiled growing +6.1 / +11.2 / +66.2 µs/layer at 1 / 8 / 32 rows, plus a +1501.7 ms prompt excess at 32×128 when added to routing+experts (piece 6). [ineff-g3-1](ineff-g3-1.md) carries the measurement side; the mechanism (grid shape, volatile-spill traffic, cache behaviour vs the live `y`) is yours.
- **For `baseline-experts` / `baseline-routing`:** gap 3's prompt arithmetic is yours kernel-side: prefill piece excess composes to prompt time (piece 6), so a `max_rows` hand-off banks exactly the predicted prompt excess while keeping decode wins. Experts 1-row piece sign flips between runs (pieces 1–2) — treat 1-row piece margins as noise.
- **For `grader`:** gap 1's mechanism is the thing the chained piece timing cannot see; whatever `check_ready.py` reproduces must include the 32-request step with combine alone (gap 1) and the 8-request all-together step (gap 2), or round 3 re-learns them from Claude.

## Files read

- [claude/findings/muse-round-2.md](../../../claude/findings/muse-round-2.md) — the measurement write-up; every headline number above is traced to its raw file below.
- `results/runs/2026-11-04-muse-round-2/console-plain.txt`, `console-vllm.txt` — G3 repeat lines and median summary blocks (piece 4).
- `results/runs/2026-11-04-muse-round-2/grade-routing.json`, `grade-combine.json`, `grade-experts.json` — decode-only re-grade verdicts (piece 1).
- `results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`, `grade-phases-combine-attempt-1.json`, `grade-phases-experts-attempt-7.json` — two-phase verdicts (piece 2).
- `results/runs/2026-11-04-muse-round-2/console-sizes.txt`, `sizes-check.json` (head) — 25 row-count exactness (piece 3).
- `results/runs/2026-11-04-muse-round-2/muse-plain.jsonl`, `muse-vllm.jsonl` — check records (kernel counts, tokens) and run records (piece 4).
- `results/runs/2026-11-04-muse-round-2-pairs/console-plain.txt`, `muse-plain.jsonl` — pairs additivity (piece 5).
- `results/runs/2026-11-04-engine-comparison/console-vllm.txt`, `versions.csv`, `muse-vllm.jsonl` — prompts and throughput (piece 6).
- `muse-spark/round-2/kernels/combine/attempt-1/notes.md`, `spec.json` (+ routing/experts `spec.json`) — the three attempts' grids behind the mechanism notes.
- Round 2's [territory-g3.md](../../round-2/exploration/territory-g3.md), [ineff-g3-1.md](../../round-2/exploration/ineff-g3-1.md), [ineff-g3-2.md](../../round-2/exploration/ineff-g3-2.md) — frozen starting point.

## From other agents

(none yet)
