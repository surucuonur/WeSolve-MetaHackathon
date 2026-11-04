# Inefficiency: decode-only piece wins do not predict whole requests; the prompt side is priced by the prefill piece timing

- **Owner:** explorer-g3
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:41:19Z
- **Next:** verifier to check; kernel side owned by baseline-experts / baseline-routing

## Slot

`experts` + `routing` (`combine` saves prefill piece time and is the exception that bounds the claim). Measurement side owned by territory `g3`; kernel side (fast prefill branch or `max_rows` hand-off) owned by territories `baseline-experts` and `baseline-routing`.

## What happens

Round 2 predicted whole-step gains from the decode-only chained piece timing (−0.84 ms/step for the submission at each size). The engine comparison (`results/runs/2026-11-04-engine-comparison/versions.csv`, [territory-g3.md](territory-g3.md) piece 6) times whole requests at 1×128, 8×128, 32×128 and 32×1024 and finds the decode wins partly or wholly consumed by slower prompt processing: routing alone processes prompts 1.07–1.10× slower (whole-request throughput +10.2% / +1.8% / +0.3% / −3.5%), the submission 1.08–1.11× slower (+15.9% / +2.1% / −2.7% / −4.1%), experts-a7 variants 6.63–14.50× slower (−10.0% to −83.3%). The new positive result: the two-phase grader's **prefill** piece timing composes additively to prompt time — chunks × 20 layers × (candidate − reference) predicts the measured prompt excess within ~15% everywhere (often <10%) — so prompts are priced, while decode keeps the two composition anomalies ([ineff-g3-1](ineff-g3-1.md), [ineff-g3-2](ineff-g3-2.md)).

## Cost

Prompt excess vs the unchanged engine, ms (base prompt: 50.1 / 201.7 / 802.7 / 6521.3 at 1×128 / 8×128 / 32×128 / 32×1024; all from `versions.csv`):

| Version | 1×128 | 8×128 | 32×128 | 32×1024 |
|---|---|---|---|---|
| routing: measured | +3.3 | +19.1 | +77.7 | +662.5 |
| routing: piece-predicted | +2.5 | +21.4 | +88.1 | +705.0 |
| routing+experts-a7: measured | +282.4 | +2721.9 | +9229.0 | +78855.9 |
| routing+experts-a7: piece-predicted | +287.6 | +2324.9 | +9421.0 | +75368.4 |
| submitted (r+c): measured | +5.7 | +18.2 | +63.0 | +555.1 |
| submitted (r+c): piece-predicted | +2.0 | +17.4 | +65.1 | +521.0 |

Prediction = chunks × 20 × (candidate − reference): 1 / 1 / 2 / 16 chunks of ≤2048 rows against the 127 / 1,016 / 2,048-row piece sizes (routing excess 139.8 / 1077.7 / 2203.0 µs/layer; experts 14256.6 / 115175.1 / 233321.3; combine −25.0 / −198.7 / −573.2). Worst miss is 15% (routing+experts at 8×128, low by 397 ms of 2722; 1×128 submitted +3.7 ms is within the prompt repeat spread — base [47.1–54.2], submitted [49.5–58.3]).

Whole-request throughput paid (tokens/s change, `versions.csv`): routing −3.5% at 32×1024 (its −0.37 ms/step decode win kept, its +663 ms prompt excess decisive); submitted −2.7% / −4.1% at 32×128 / 32×1024; routing+experts −10.0% / −37.5% / −48.0% / −82.5% — the fastest exact decode step measured (−18.8% at 32×128) is the slowest whole request. Neither the old G2 (correct at prompt sizes but untimed) nor decode-only G3 saw any of this.

## Kind

Overhead-bound for routing (its staged scalar path: 227 / 1299 / 2513 µs/layer where compiled takes 87 / 221 / 290 — per-row re-reads and barriers, not bytes); experts-a7's prefill branch is algorithmic (scalar `qmm_accum`: 15220 / 119154 / 240681 µs/layer vs 961 / 4306 / 11999 compiled, 15.8 / 27.7 / 20.1×). Either way the prompt phase, not the decode step, is where the request's time goes once installed.

## Spans

`g3` (this measurement), `baseline-experts`, `baseline-routing` (the prefill-slow kernels), `quantized-kernels` (what a fast exact prefill branch must match — `gather_qmm_rhs` speed), `grader` (the two-phase `beats_compiled` and `max_rows` hand-off grading that now gate this).

## Predicted saving

Recovering the prompt excess while keeping each variant's decode-step change (a fast exact prefill branch, or a `max_rows` hand-off — Claude checked the hand-off bit-exact on an experts copy with `max_rows` 59, `results/runs/2026-11-04-handoff-check/`): prompt time returns to the unchanged engine's 50.1 / 201.7 / 802.7 / 6521.3 ms, i.e. up to **+663 ms of prompt time at 32×1024 for routing** (+3.3 / +19.1 / +77.7 at the smaller workloads) and up to **+78.9 s at 32×1024 for routing+experts-a7** (+282.4 ms / +2.72 s / +9.23 s at 1×128 / 8×128 / 32×128). Per prefill chunk (chained basis, µs per layer): routing saves 139.8 / 1077.7 / 2203.0 at 127 / 1,016 / 2,048 rows; experts saves 14256.6 / 115175.1 / 233321.3; combine already saves and predicts nothing further here. Decode-step savings are the measured G3 figures (routing −0.47 / −0.43 / −0.54, experts-a7 +0.29 / −1.22 / −5.35 ms/step in vLLM), kept, not added to: a saving in one phase that costs the other is not one. Whole-request throughput after the fix is unverified until the ready check times it — predicted direction is up everywhere the decode step already wins (routing all workloads, experts at 8/32 requests), with the arithmetic above as the prediction to stand beside the measured value.

## Verdict

(none — open)
