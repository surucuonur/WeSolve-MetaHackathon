# Inefficiency: `w_router` re-read once per row at decode sizes — traffic excess larger than the whole 32-row margin

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check the decode repricing against both grade runs; a builder needs a barrier-light sharing design proven on the chained timing

## Slot

`routing`

## What happens

Every row's threadgroup streams its own full copy of the router weight from device memory. On the staged scalar path (`rows > 1`) all 256 threads cooperatively load all 48 K-chunks × 60 experts as `float4`s into `wsh` across the 12 groups (`muse-spark/round-3/kernels/routing/attempt-6/block-0.metal:81-93`, the code in the slot); in the GEMV branch (`rows == 1`) each lane reads `w_router` straight from device (block-0:41-45). Nothing is shared across rows: per layer the kernel reads rows × 60×768×4 B = rows × 184,320 B of `w_router`, plus rows × 1,536 B of `x`. The compiled reference's GEMM/GEMV reads `w_router` once per layer and reuses each tile across rows (round-2 `territory-baseline-routing.md` §3.1, frozen).

This is the decode slice of round 2's `ineff-baseline-routing-4` (frozen, confirmed, upper-bound-only) and round 3's `ineff-baseline-routing-1` (frozen, confirmed), repriced because the context changed twice over: (1) `max_rows: 32` in `spec.json` now hands every prefill size to the compiled reference, so the prefill side of those entries — 116.1/935.4/1,886.5 µs/layer at 127/1,016/2,048 rows — is banked and costs nothing; what remains is exactly the decode side they ranked as not worth chasing. (2) The round-4 question is no longer beating the compiled reference (the starting point does, at every decode size) but *widening the margin*, above all at 32 rows for the 32×1024 set workload (round-4 exploration README item 2). At 32 rows the re-read excess is larger than the entire margin there is to widen.

## Cost

Traffic floors at the 200 GB/s peak (brief:9), candidate per layer; measured = chained µs/layer, candidate vs compiled reference, from the two runs of the identical bytes (`muse-spark/round-3/kernels/routing/attempt-6/grade.json`; `results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json`):

| rows | `w_router` read | floor (w+x) | measured cand / compiled (re-grade) | margin | excess over read-once |
|---|---|---|---|---|---|
| 1 | 0.18 MB | 0.9 µs | 45.1 / 62.8 | 17.7 µs | 0 µs |
| 8 | 1.47 MB | 7.4 µs | 144.2 / 160.0 | 15.8 µs | 6.5 µs |
| 32 | 5.90 MB | 29.7 µs | 151.3 / 171.0 | 19.7 µs | 28.6 µs |
| 127 / 1,016 / 2,048 | — (handed off) | — | `handed off` | — | 0 (compiled reference runs) |

The excess is (rows−1) × 184,320 B ÷ 200 GB/s. At 32 rows the excess floor (28.6 µs/layer) is 1.45× the whole measured margin over the compiled reference (19.7 µs/layer, re-grade; 19.8 in the round-3 grade) — the only pool inside this slot big enough, on its own, to supply the +7.5–12.5 µs/layer (routing+combine combined) that non-overlap at 32×1024 needs (territory entry, headline). At 8 rows the excess (6.5) is 41% of the margin (15.8); at 1 row there is no excess (a single row reads the weight once by definition).

The floor is an upper bound, not time on the clock. The kernel is overhead-bound at decode sizes, and the staging loads overlap other resident threadgroups' scalar compute: the starting point's own measurements put the marginal cost of rows 9–32 at ≈0.16 µs/row (`muse-spark/round-2/kernels/routing/attempt-4/notes.md`, flat 69.1→72.9 µs from 8 to 32 rows), against a 0.92 µs/row traffic floor — at most ~17% of the floor appears as marginal time, and a sharing design still pays its own staging and barriers out of that. The measured counter-evidence stands: round 1's attempt-2 tiled `w_router` across 8 rows to remove this re-read and was slower at every decode size (36.3/34.9/49.0 vs 28.5/23.6/41.8 µs/layer, `muse-spark/kernels/routing/attempt-2/notes.md`, frozen), and round 3's sharing kernel (attempt-5) proved cross-row sharing bit-exact but ran it only above 32 rows — no sharing design has ever been timed at decode sizes 2–32.

## Kind

Overhead-bound at decode sizes with a bandwidth-waste component (`claude/GLOSSARY.md:39-42`; at 32 rows 5.90 MB in ~132–151 µs is ≈40 GB/s, a fifth of peak — the waste is real bytes but the time is not byte-limited). At prefill sizes: not applicable — the kernels do not run there.

## Spans

`baseline-routing`. The comparison point (read-once tile reuse) is MLX's GEMM dispatch behind the compiled reference; the prize lands in `g3` territory (the set's 32×1024 margin), and any design must keep the `max_rows: 32` hand-off in `grader` territory's dispatch terms.

## Predicted saving

Upper bound if each weight byte were read once per layer at peak bandwidth, per layer and per decode step (×20 layers); prefill 0 at every size by hand-off construction (the design runs only where the kernels already run, `rows <= 32`, and changes no prefill path — so it cannot cost the other phase):

| rows | per layer | per decode step |
|---|---|---|
| 1 | 0 µs | 0 ms |
| 8 | 6.5 µs | 0.13 ms |
| 32 | 28.6 µs | 0.57 ms |
| 127 / 1,016 / 2,048 | 0 µs | 0 ms per prefill chunk |

Arithmetic: (rows−1) × 184,320 B ÷ 200 GB/s; at 32 rows 31 × 184,320 B = 5.71 MB ÷ 200 GB/s = 28.6 µs/layer × 20 = 0.57 ms/step. Confidence is low that more than a fraction is realised — the overlap account above caps the plausibly recoverable slice at roughly the 0.16 µs/row marginal (≈3.8 µs/layer ≈ 0.08 ms/step at 32 rows) unless the design also removes staging latency the current kernel hides by overlap. Label: prediction, counting only this slot's own effect; it stands beside a builder's same-run chained measurement before anyone banks it.

What a builder needs: cross-row `w_router` sharing that keeps every row's exact accumulation order (sharing bytes is free; reordering any row's serial chain fails G2 — hyp-3 boundary, frozen) without round-1 attempt-2's barrier structure. Hard constraints already measured: threadgroups of 960 and 600 threads fail G1 on a per-pipeline cap that moved 640→576 between recompiles (round-3 attempts 2/4), while 480 threads (8 rows × 60 experts) is the proven-passing band (attempt-5); attempt-5's own named next probe — stage each `w` chunk once per group while streaming 2–4 sequential row-blocks through it, cutting traffic and barriers per row 2–4× with no more threads — is untested, and untested at decode sizes in any form. A rows-conditional design (shared path only at, say, 16–32 rows where the excess exceeds ~14 µs/layer, staged path kept below) keeps the proven 1/8-row paths untouched; the branch must be on `rows` regimes, never on the graded values.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** per-row full `w_router` streaming — staged path loads 48 chunks × 60 experts as `float4`s into `wsh` at [block-0.metal:81-93](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/routing/attempt-6/block-0.metal:81), GEMV lanes read device direct at [block-0.metal:41-45](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/routing/attempt-6/block-0.metal:41), no cross-row sharing; `rows × 60×768×4 = rows × 184,320` B + `rows × 1,536` B `x` ✓; reference reads once per layer with tile reuse (frozen [territory-baseline-routing.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/exploration/territory-baseline-routing.md:54), round-2 §3.1) ✓; prefill banked by `max_rows: 32`, decode-only remainder ✓.
- **confirmed:** floors at 200 GB/s — `w_router` 0.18/1.47/5.90 MB, (w+x) floors 0.9/7.4/29.7 µs/layer; re-grade [grade-routing-attempt-6.json](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-grader-hand-off/grade-routing-attempt-6.json) 45.1/62.8, 144.2/160.0, 151.3/171.0 → margins **17.7/15.8/19.7** µs/layer (round-3 [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/routing/attempt-6/grade.json) 19.8 at 32 rows) ✓; excess (rows−1)×184,320 ÷ 200 GB/s = **0/6.5/28.6** µs/layer; 28.6÷19.7 = 1.45× the 32-row margin, 6.5÷15.8 = 41% at 8 rows ✓; ~40 GB/s at 32 rows, overhead-bound with bandwidth waste ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) ✓.
- **confirmed:** overlap discount and counter-evidence — marginal 0.16 µs/row vs 0.92 µs/row floor ≈ 17% ([attempt-4/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/kernels/routing/attempt-4/notes.md:96), flat 69.1→72.9 µs 8→32 rows) ✓; round-1 tiled attempt-2 slower everywhere 36.3/34.9/49.0 vs 28.5/23.6/41.8 ([attempt-2/notes.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/routing/attempt-2/notes.md)) ✓; attempt-5 sharing computes only above 32 rows (`rows <= 32` copy branch in attempt-5 `block-2.metal`), so no sharing timed at decode 2–32 ✓; thread caps 960-vs-640, 600-vs-576, 480 passing ([hyp-3-routing-prefill.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/implementation/hyp-3-routing-prefill.md)) ✓; upper bounds 0/0.13/0.57 ms/step (×20), prefill 0, low confidence labelled prediction ✓.

## From other agents

- builder-2-a, 2026-11-04T21:56:24Z: round-4 attempt-2 (`muse-spark/round-4/kernels/routing/attempt-2`) is the first sharing design timed at decode sizes: at rows > 16 one 256-thread group stages `w_router` once per 4 rows (5.90 → 1.47 MB/layer at 32 rows), register-folded in the exact partition order, G2 pass at all six sizes. Margins over compiled 14.8 / 16.0 / 8.3 µs/layer at 1 / 8 / 32 rows vs the starting point's 17.7 / 15.8 / 19.7 — the 32-row margin narrowed despite the traffic cut, because each row dropped from 256 partition-parallel threads to 60 serial `(row, expert)` threads (8 compute groups). Confirms this entry's overlap account from the other side: the realised traffic slice (~3.8 µs/layer priced) is smaller than the parallelism the per-row layout was buying.
