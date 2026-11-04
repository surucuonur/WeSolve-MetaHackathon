# Inefficiency: per-pair weight re-reads at 8/32 rows — 552,960 B streamed per pair, no sharing across pairs on the same expert

- **Owner:** explorer-baseline-experts
- **Status:** refuted
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check byte arithmetic and the distinct-expert basis

## Slot

`experts`

## What happens

Each (row, expert-slot) pair gets its own 512-thread
threadgroup, which streams its expert's full gate/up/down
weights independently. The pair index and expert come from
`pair = threadgroup_position_in_grid.x`, `e = inds[pair]`
(`muse-spark/round-3/kernels/experts/attempt-1/block-0.metal:16-21`),
and every weight load uses per-pair offsets
`eW = e * 36864`, `eS = e * 9216` (`block-0.metal:34-35`) into
the helpers in `block-1.metal`. There is no sort, grouping,
tile cache or any other sharing across pairs: two pairs that
select the same expert read that expert's 552,960 B twice,
in full. At 32 rows (128 pairs over at most 60 experts,
`E = 60`, `claude/muse-agent-brief.md` *The slots*) duplicate
selections are guaranteed by pigeonhole — at least 68 of the
128 pair-reads repeat an expert another pair already read.

This restates round 2's `ineff-baseline-experts-3`
(confirmed, frozen) on the current starting point. What is
new: the sizes where it can still cost anything have shrunk.
The scalar prefill traffic it also described is gone (handed
off above 59 rows by `max_rows: 59`), and under the proposed
`min_rows: 2` the 1-row size hands off too — so this entry
prices re-reads only at 8 and 32 rows, the two sizes the
fused kernel still runs and is graded on, on the starting
point's own chained timings instead of round 1's harness.

Per-pair bytes (territory §B arithmetic): gate 184,320 B +
up 184,320 B + down 184,320 B = **552,960 B per pair**
(words `K × Ww × 4 B` plus bfloat16 scales and biases for
each of the three products).

## Cost

Measured, piece (chained µs/layer, starting point's
`muse-spark/round-3/kernels/experts/attempt-1/grade.json`;
pair bytes at 552,960 B/pair; 200 GB/s peak,
`claude/GLOSSARY.md`):

| Rows | Pairs | Pair bytes/layer | Candidate | Compiled | Effective on pair bytes |
|---|---|---|---|---|---|
| 1 | 4 | 2.21 MB | 96.9 | 86.6 | 22.8 GB/s |
| 8 | 32 | 17.69 MB | 168.9 | 238.2 | 104.8 GB/s |
| 32 | 128 | 70.78 MB | 554.7 | 883.4 | 127.6 GB/s |
| 127 / 1,016 / 2,048 | — | handed off | `same` | — | kernel does not run |

- **1 row: no cost from this entry, under either spec.** All
  4 pairs select distinct-or-not experts for one row; round
  2's counters put the whole 1-row experts excess at 7 MB
  per step (0.04 ms floor), and the 1-row loss is latency
  (round 3 `ineff-baseline-experts-2`), not bytes. With
  `min_rows: 2` the size hands off and the kernel reads
  nothing at all.
- **8 rows:** 32 pairs read 17.69 MB/layer. Distinct experts
  among the grader's captured 8-row inputs are unmeasured;
  bounded by 32 pairs and 60 experts. The byte floor on pair
  traffic is 17.69 MB ÷ 200 GB/s = 88.5 µs/layer; the
  candidate takes 168.9. How much of that traffic is
  re-reads is unknown at this size (no 8-row counter exists —
  the same gap round 2 recorded).
- **32 rows:** 128 pairs read 70.78 MB/layer. On the serving
  counter basis (GPU counters via round 2's entry: 908 MB
  read per step vs 443 MB of distinct experts), distinct
  bytes are 22.15 MB/layer, so re-reads are
  70.78 − 22.15 = **48.63 MB/layer (69% of pair traffic)**.
  The distinct count in the grader's captured layers is
  unmeasured and may differ from serving; the counter basis
  is the only measured one. At the candidate's own effective
  rate the re-read bytes occupy 48.63 MB ÷ 127.6 GB/s =
  381 µs of the 554.7 µs/layer.
- **Prefill 127/1,016/2,048: 0.** Above `max_rows: 59` the
  compiled reference runs; its own grouped reads are the
  bar's business, not this kernel's loss.
- Kernels: 1 per layer call at 8/32 rows. Fixed launch cost
  is already banked by the fusion and is not this entry.

## Kind

Overhead-bound with a bandwidth-waste component
(`claude/GLOSSARY.md`, round 2's wording for the same cause):
127.6 GB/s at 32 rows is 64% of peak bandwidth and the
kernel sustains ~8% of peak compute (territory §B), so the
wasted bytes are real but removing them banks time only at
whatever rate a grouped design actually sustains.

## Spans

`baseline-experts` (per-pair grid and offsets,
`block-0.metal:16-35`), `quantized-kernels` (the grouped
`gather_qmm_rhs` sharing this kernel lacks at decode sizes),
`g3` (the 32-request margin a wider win would feed).

## Predicted saving

Floor arithmetic first (excess bytes ÷ 200 GB/s peak — an
upper bound on any byte saving, not a promise):

- 1 row: **0** — handed off under `min_rows: 2`; ≈0 even
  without it (above).
- 8 rows: excess unmeasured. On round 1's micro pattern
  (≈25 distinct of 32 pairs, marked estimate in round 2's
  entry): (32−25) × 552,960 B = 3.87 MB/layer → **19.4
  µs/layer floor = 0.39 ms per decode step**. Estimate, no
  counter.
- 32 rows: 48.63 MB/layer ÷ 200 GB/s = **243.1 µs/layer
  floor = 4.86 ms per decode step** (counter distinct
  basis). A grouped kernel reading only distinct bytes at
  the candidate's own 127.6 GB/s would take
  22.15 MB ÷ 127.6 GB/s = 173.6 µs/layer before its
  sort/scan/barrier overhead — i.e. up to 381 µs/layer of
  the current time is re-read bytes at the current rate.
- Prefill 127/1,016/2,048: **0** at every size — the kernel
  does not run there; a grouped decode change must not
  spend the prefill hand-off.

**Prediction: 0 banked today.** Both grouped attempts that
read each expert ~once (round 1) ran into a ~55 GB/s
duty-cycle ceiling — load→barrier→compute leaves DRAM idle
— and *lost* to the per-pair kernel at 8 rows, tying or
winning only at 32 (round 2 `ineff-baseline-experts-3`,
confirmed). So the realistic claim is conditional: a
grouped or expert-cached design banks toward the 243
µs/layer floor at 32 rows only if it holds ≥127.6 GB/s
effective on distinct bytes (≤173.6 µs/layer before
overhead), and it must keep the per-pair path — or match
it — at 8 rows, where the same evidence says grouping
loses. A saving at 32 that costs 8 is not one (ADR 0011).
This entry and `ineff-baseline-experts-2` overlap on the
same µs (the re-read bytes are part of the time above the
distinct-byte floor): do not add them.

## Verdict

verifier, 2026-11-04T21:47:11Z — **refuted**: the 32-row re-read cost mixes the pair-model numerator with the counter distinct denominator and overstates the floor 2×.

- **confirmed:** per-pair streaming with no sharing — `pair = threadgroup_position_in_grid.x`, `e = inds[pair]` at [block-0.metal:16-21](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/block-0.metal:16) and `eW = e * 36864`, `eS = e * 9216` at [block-0.metal:34-35](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/block-0.metal:34); every load uses those per-pair offsets into `block-1.metal` helpers, no sort/group/cache ✓; `E = 60` ([muse-agent-brief.md:28](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md:28)), so 128 pairs guarantee ≥68 repeats (128−60) ✓.
- **confirmed:** per-pair bytes — gate/up `768×48×4 = 147,456` B words + `768×12×2×2 = 36,864` B scales/biases = 184,320 B; down `384×96×4 = 147,456` + `384×24×2×2 = 36,864` = 184,320 B; total 552,960 B/pair ✓; pair traffic 4/32/128 pairs = 2.21/17.69/70.78 MB/layer ✓.
- **confirmed:** chained times from [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/experts/attempt-1/grade.json) — 96.9 vs 86.6, 168.9 vs 238.2, 554.7 vs 883.4 µs/layer; effective on pair-model bytes 22.8/104.8/127.6 GB/s at 200 GB/s peak ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) ✓; 1-row excess 7 MB/step → 0.04 ms floor ([tables.md](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-02-gpu-counters/tables.md) 51 vs 44 MB) ✓; 8-row floor 17.69 MB ÷ 200 GB/s = 88.5 µs/layer with distinct unmeasured ✓; 8-row estimate (32−25)×552,960 = 3.87 MB → 19.4 µs/layer = 0.39 ms/step, marked estimate ✓; prefill 0 above `max_rows: 59` ✓; grouped ~55 GB/s ceiling losing at 8 rows ([ineff-baseline-experts-3.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-experts-3.md), confirmed) and 0 banked today ✓.
- **wrong:** “On the serving counter basis (908 MB vs 443 MB), re-reads are 70.78 − 22.15 = 48.63 MB/layer (69%)” and its dependents (243.1 µs/layer = 4.86 ms/step floor; 381 µs at the candidate rate). The counter table ([tables.md](/Users/onur/Desktop/Github/Hackathon-muse/results/runs/2026-11-02-gpu-counters/tables.md): 32-row experts 908 MB counter-read, 443 MB must-read model, 1,416 MB per-pair model) gives counter excess (908−443) ÷ 20 = **23.25 MB/layer**, floor **116.3 µs/layer = 2.33 ms/step**, share 465÷908 = **51%** — the confirmed round-2 figures. 70.78 MB/layer is the pair-model (1,416÷20), so 48.63 MB/layer and 69% are model-basis excess (973 MB/step) mislabelled as counter; the grouped bar 22.15 MB ÷ 127.6 GB/s = 173.6 µs/layer before overhead is unaffected. Restate on one basis (counter floor 2.33 ms/step, or model excess labelled model) and resubmit; the cause itself and the 0-banked prediction stand.

## From other agents
