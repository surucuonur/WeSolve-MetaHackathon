# Inefficiency: experts re-reads expert weights per (token, expert) pair

- **Owner:** explorer-runs
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:25:00Z
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
