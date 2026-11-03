# Inefficiency: experts re-reads expert weights per (token, expert) pair

- **Owner:** explorer-runs
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to confirm against the knock-out table

## Slot

`experts`

## What happens

Each (token, expert) pair's `gather_qmm` reads that expert's gate/up/down weights afresh, so popular experts are re-read once per pair instead of once per step. The knock-out table (`results/runs/2026-11-02-gpu-counters/tables.md`, "Each piece" rows) shows knocking out experts saves 908 MB of reads per step at 32 rows while the model must read only 443 MB for the distinct experts (per-pair model: 1416 MB). At 1 row the two agree (51 vs 44 MB) because all 4 pairs use distinct experts. Mechanism belongs to the switch-layers / quantized-kernels territories; this entry is the measurement.

## Cost

At 1, 8 and 32 rows, per decode step (plain MLX, 128-token prompts):

- 1 row: experts costs 0.92 ms (knock-out, unrecorded) and reads 51 MB vs 44 MB must-read — excess 7 MB.
- 8 rows: no knock-out row; the whole step reads 685 MB vs 393 MB must-read (whole-step table), so the step-wide excess is 292 MB, of which experts is the largest piece by the r1/r32 pattern.
- 32 rows: experts costs 14.67 ms of a 28.19 ms step and reads 908 MB vs 443 MB must-read — excess 465 MB. Micro corroboration (`results/runs/2026-10-30-micro/moe_parts.txt`): experts 16.60 ms per 20 layers at R=32 vs 2.64 ms at peak bandwidth for the same bytes.

## Kind

Bandwidth-bound: the excess is bytes re-read, ~2× the necessary expert traffic at 32 rows. (The remaining ~12 ms of the 14.67 is kernel overhead and the necessary traffic, owned by the kernel territories.)

## Spans

`runs`, `switch-layers`, `quantized-kernels`

## Predicted saving

Excess bytes removed ÷ peak bandwidth (200 GB/s), per decode step:

- 1 row: 7 MB ÷ 200 GB/s = 0.04 ms ≈ 0.0 ms (nothing lost to re-read at 1 row).
- 8 rows: scaled estimate — pairs scale 32/128 of r32, excess ≈ 465 × 32/128 = 116 MB → 0.6 ms.
- 32 rows: 465 MB ÷ 200 GB/s = 2.3 ms.

Arithmetic is a floor (values the excess at peak bandwidth); the measured experts time is 14.67 ms, so the rest must come from overhead removal, not this entry.

## Verdict

(verifier to write)

## From other agents

explorer-findings, 2026-11-03T21:55Z — scope proposal + two corrections. Scope:
this entry (excess-bytes floor: 465 MB / 200 GB/s = 2.3 ms @32 rows) and my
ineff-findings-2 (full expert cost + grouped-kernel measurement: 25% × 14.67 =
3.67 ms @32 rows) are compatible — floor vs measured-kernel basis. Propose
keeping both, linked. Corrections: (1) Kind `Bandwidth-bound` contradicts the
glossary (bandwidth-bound = achieved bandwidth CLOSE TO peak) — the experts move
62 GB/s = 31% of peak at 6% of peak compute, and bottleneck-explained §1
verdicts every step overhead-bound. Suggest overhead-bound with a
bandwidth-waste component. (2) The 8-row excess scaling (465 × 32/128 = 116 MB)
assumes excess ∝ pairs, but excess comes from collisions: at 8 rows 32 pairs
spread over ~20 experts (1.6 pairs/expert, breakdown tables.md:401) vs 3.2 at 32
rows, so the 8-row excess is likely much smaller than 116 MB; the 292 MB
whole-step excess cannot be attributed to experts (cache-disturbance caveat,
step-breakdown:375-379). Also: the grouped kernel is 25–36% faster on equal
balanced data with the pairs in expert order in BOTH arms (decode-bottlenecks
§5:183-195) — the prize is the kernel, not the order; see my note on
ineff-runs-3.

## From other agents

explorer-grader, 2026-11-03T21:24Z — the grader side, from `bench/grader/`. G2 experts
(`check_candidate.py:112-130`): exact shape [rows,4,768] and bf16 dtype; every element
within 2 bf16 ULP of the reference with the ULP floored at |r| = 0.001 (`ulp_bf16`,
:44-47). The rounding points a read-once kernel must reproduce (brief:90): each product
accumulates in f32 and rounds to bf16; silu(gate)*up is computed on those bf16 values and
rounded to bf16 before the down product. Piece timing (:209-226) compares against the
reference *including* its ≥16-row sort (`do_sort = inds.size >= 64`, `slots.py:62-82`),
so at 32 rows the baseline already collects the sorted layout — the re-read excess is the
remaining prize there. Suggest adding `grader` to Spans (owner decides).

## From other agents

explorer-switch-layers, 2026-11-03T21:31Z — code-side mechanism (no duplicate entry posted):
- Every decode `gather_qmm` lands in MLX's per-pair vector kernel `gather_qvm`
  (`quantized.cpp:2076-2093`, v0.32.4), whose grid is (M, N/64, B) threadgroups of
  64 threads (`:1462-1500`): each pair's tiles read their expert's weights
  independently. That is the per-pair read.
- The grouped path `gather_qmm_rhs` needs `M == 1 && B >= 16 && right_sorted_ &&
  B/E >= 4` (`quantized.cpp:2012`). The port's call gives M=1, B=4·rows, E=60,
  right_sorted_=True (`ops.cpp:5824-5825` + `primitives.h:1703-1710`), so the
  gate is B/E>=4 = 240+ pairs = 60+ rows — never in decode (max 32). Below 60
  rows the flag the port sets changes nothing about kernel choice.
- Detail for the entry's 1-row line: the 51-vs-44 MB agreement is not (only)
  because the 4 pairs use distinct experts — with 4 pairs they always do — but
  because at 4 pairs there is almost nothing to re-read. Full analysis in
  `territory-switch-layers.md` §G.
