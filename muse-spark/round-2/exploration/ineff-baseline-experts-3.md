# Inefficiency: per-pair weight re-reads persist in the starting point (465 MB excess at 32 rows)

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:23:32Z
- **Next:** verifier to confirm byte arithmetic against GPU counters

## Slot

`experts`

## What happens

Attempt-4 fuses gate/up/down + silu-mul into one kernel (removing round
1 `muse-spark/exploration/ineff-switch-layers-2`, confirmed: 4 on-chain
kernels/layer + bf16 intermediates written and re-read), but it keeps
per-pair reads: each (row, expert-slot) pair's threadgroup streams its
expert's full gate/up/down weights independently
(`muse-spark/kernels/experts/attempt-4/block-0.metal:23,31,43,70`;
offsets `eW = e * 36864`, `eS = e * 9216` per pair). A popular expert is
read once per pair that selects it, not once per layer. This is round
1's `ineff-runs-1` (confirmed) and the traffic half of
`ineff-quantized-kernels-1` (confirmed), restated for the starting
point because attempt-4 does not remove it — it only hides part of it
with fusion and 8 in-flight blocks per group
(`attempt-4/notes.md:66-75`: 32-row effective 109 → ~137 GB/s
independent). The `x` re-read half of round 1's kernel entry *is*
removed: `tg_x` is loaded once per pair (`block-0.metal:18-20`) where
`qvm_impl` re-streams the row per tile (round 1
`territory-quantized-kernels.md` §A: ~55 KB x-traffic per pair vs
3.75 KB needed).

What is new this round: the same per-pair grid also runs at prefill
sizes, where fan-in is far larger (508/4,064/8,192 pairs per layer over
at most 60 distinct experts), and the chained timings now price the
remaining waste at decode sizes against the reference the slot must
beat.

## Cost

Per decode step (GPU counters `results/runs/2026-11-02-gpu-counters/tables.md:26,32`
via round 1 `ineff-runs-1.md`; per-pair bytes 552,960 B, territory §A):

- **1 row:** 4 pairs, all distinct (4.0 distinct experts/layer): reads
  51 MB vs 44 MB must-read, excess 7 MB. Chained candidate 132.9
  µs/layer (`console-chain.txt`) — but at 1 row the loss is latency
  (ineff-2), not these bytes; excess floor 7 MB ÷ 200 GB/s = 0.04 ms.
- **8 rows:** 32 pairs/layer; no serving knock-out at 8 rows. Modelled
  pair traffic 32 × 552,960 × 20 = 353.9 MB/step (weights only;
  round 1 modelled 383 MB with x-traffic). Distinct experts/layer
  unmeasured at 8 rows; bounded by 32 pairs and 60 experts.
- **32 rows:** 128 pairs/layer read 908 MB/step vs 443 MB for the
  distinct experts — excess 465 MB (51% of bytes are re-reads).
  Per layer: 70.8 MB pair traffic vs 22.2 MB distinct. Candidate
  chained 587.6 µs/layer = 120.5 GB/s on pair traffic (60% of the
  200 GB/s peak, `claude/GLOSSARY.md:10`); the same time on distinct
  bytes alone would be 177 GB/s-equivalent — there is no bandwidth
  headroom left to absorb the re-reads at this rate.

Fixed costs: candidate 1 kernel/layer vs reference 7 (1 row) / 13
(32 rows, incl. 6 sort-glue kernels from 16 rows, round 1
`ineff-switch-layers-1` confirmed: ~0.7 ms/step at 32 rows, removable
only inside a grouped kernel). Attempt-4 already banks the launch and
intermediate-traffic savings; this entry is only the weight re-reads.

## Kind

Overhead-bound with a bandwidth-waste component (round 1 wording for
the same cause): 120.5 GB/s chained at 32 rows is 60% of peak, and the
step as a whole sits far below both limits; the wasted bytes are real
but removing them at the current rate is worth less than the floor
arithmetic suggests (see Predicted saving).

## Spans

`baseline-experts` (`block-0.metal` per-pair offsets), `quantized-kernels`
(per-pair `qvm` grids this mirrors), `switch-layers` (grouped path the
reference only takes from 60 rows), `runs` (counter tables).

## Predicted saving

Excess bytes at peak bandwidth (floor, per decode step):

- 1 row: 7 MB ÷ 200 GB/s = **0.04 ms ≈ 0.0 ms**.
- 8 rows: excess unmeasured; if distinct experts/layer ≈ 25 of 32 pairs
  (round 1 micro pattern), excess ≈ (32−25) × 552,960 × 20 = 77 MB →
  **0.39 ms** floor. Marked estimate — no 8-row counter exists.
- 32 rows: 465 MB ÷ 200 GB/s = **2.33 ms** floor.

Against the compiled reference this floor is **not** the predicted
saving, and round 1's building is the reason: both grouped attempts
that read each expert ~once (attempt-1 full fused grouped,
attempt-3 two-kernel grouped, ~34 MB/layer at 32 rows) ran at a
~55 GB/s duty-cycle ceiling (load→barrier→compute leaves DRAM idle)
and *lost* to per-pair attempt-4 in the piece harness at 8 rows and
tied/won only at 32 (round 1 `muse-spark/exploration/README.md` *What
building taught* #1; `ineff-quantized-kernels-1.md` builder note;
`hyp-1-fused-experts.md` Decision). The starting point already beats
the chained reference by 63.7 µs/layer at 8 rows and 210.0 µs/layer at
32 rows (1.27 and 4.20 ms/step, `console-chain.txt`) *with* the
re-reads, because the reference pays them too plus 7–13 launches.
A grouped fix banks the 2.33 ms floor only if it holds ≥120 GB/s
effective on distinct bytes — i.e. ≤185 µs/layer at 32 rows
(22.2 MB ÷ 120.5 GB/s) before its scan/chunk/barrier overhead, a bar
neither round-1 grouped attempt met. Predicted saving vs the compiled
reference (itself within 3% of the reference in the old harness,
`grade-experts.json`): **0 ms banked today; up to ~2.3 ms/step at
32 rows (0.12 ms/layer × 20) if a grouped design holds the starting
point's rate on distinct bytes; negative at 8 rows on round 1's
evidence.** Do not add to ineff-2 (disjoint: latency at 1 row, bytes
at 32) or to round 1's joint ceiling (443 MB at streaming rate ≈
11.5 ms at 32 rows, verifier reconciliation in
`ineff-quantized-kernels-1.md`).

## Verdict

verifier, 2026-11-04T03:23:32Z — **confirmed**, with two corrections that do not change the floor. Checked: per-pair offsets `eW=e·36864, eS=e·9216` at [block-0.metal:23](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/experts/attempt-4/block-0.metal:23) with gate/up/down at :31/:43/:70 — each pair streams its expert independently ✓; `x` half removed (`tg_x` once per pair at :18-20 vs `qvm_impl` re-streaming, [territory-quantized-kernels.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/territory-quantized-kernels.md) §A ~55 KB vs 3.75 KB) ✓; fusion banks launch/intermediate savings ([ineff-switch-layers-2](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/ineff-switch-layers-2.md) confirmed) while re-reads persist, restating [ineff-runs-1](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/ineff-runs-1.md) / [ineff-quantized-kernels-1](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/ineff-quantized-kernels-1.md) for the starting point ✓. Counters ([tables.md](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-02-gpu-counters/tables.md) :26 1-row experts 51 vs 44 MB, :32 32-row 908 vs 443 MB): 1-row excess 7 MB → 0.04 ms; 8-row modelled 32×552,960×20=353.9 MB, distinct unmeasured ✓; 32-row excess 908−443=465 MB (51.2% re-reads), per-layer 70.8 MB pair-model vs 22.2 MB distinct (443/20), chained 587.6 µs/layer = 120.5 GB/s on pair traffic (60% of peak) ✓. Floors: 7/200=0.04, 465/200=2.33 ms/step ✓; 8-row estimate (32−25)×552,960×20≈77 MB → 0.39 ms marked estimate ✓. Grouped evidence: ~55 GB/s duty-cycle ceiling, grouped lost at 8 rows and tied/won only at 32 ([README.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/README.md) *What building taught* #1; [ineff-quantized-kernels-1.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/ineff-quantized-kernels-1.md) builder note; [hyp-1-fused-experts.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/implementation/hyp-1-fused-experts.md) Decision) ✓; starting point already ahead chained by 63.7/210.0 µs/layer (1.27/4.20 ms/step, [console-chain.txt](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-04-muse-round-1/console-chain.txt)) ✓; grouped bar 22.2 MB÷120.5 GB/s≈185 µs/layer ✓; 0 banked today, up to ~2.3 ms/step at 32 rows if the rate holds, negative at 8 rows on round-1 evidence ✓. **Wrong 1:** “same time on distinct bytes alone would be 177 GB/s-equivalent” — 22.15 MB in 587.6 µs is 37.7 GB/s; the usable bar is the 185 µs figure above. **Clarification 2:** per-layer 70.8 MB is the pair-model (×20 = 1,416 MB/step, the counters’ “per-pair model” column); 908 MB/step is the counter-measured saved bytes. The 465 MB excess and 2.33 ms floor are on the counter basis (908−443), as in confirmed [ineff-runs-1](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/exploration/ineff-runs-1.md); on the model basis the excess would be 973 MB. Neither correction changes the predicted saving (0 banked; floor conditional on holding ≥120.5 GB/s).

## From other agents

(none yet)
