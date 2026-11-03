# Inefficiency: (token, expert) pairs unsorted below 16 rows

- **Owner:** explorer-runs
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to confirm against moe_parts.txt

## Slot

`experts`

## What happens

MLX sorts pairs by expert only from 16 rows (brief). The sort experiment in `results/runs/2026-10-30-micro/moe_parts.txt` (kernel A/B, gate+up+down of one layer chained ×20) shows sorting wins at every row count: rows 8, ours sorted=False 5.08 ms vs sorted=True 3.82 ms per 20 layers; rows 16: 8.12 vs 5.65; rows 32: 15.02 vs 9.63. Below 16 rows serving pays the unsorted price; at 8 rows that price is directly measured.

## Cost

Per decode step (micro chained-layer setup, plain MLX):

- 1 row: no experiment row; 4 pairs need little sorting — cost ≈ 0.
- 8 rows: unsorted 5.08 ms vs sorted 3.82 ms per 20 layers — excess 1.26 ms.
- 32 rows: already sorted in serving — excess 0 (the 15.02 → 9.63 gap measures what sorting is worth, already collected).

## Kind

Bandwidth-bound: unsorted pairs scatter each expert's reads across DRAM instead of streaming each expert once.

## Spans

`runs`, `switch-layers`, `quantized-kernels`

## Predicted saving

Sorting the pairs at 8 rows (or an equivalent layout that streams each expert once), per decode step:

- 1 row: 0.0 ms (no data; 4 pairs, nothing to sort).
- 8 rows: 5.08 − 3.82 = 1.3 ms.
- 32 rows: 0.0 ms (already sorted).

Caveat: the micro setup's distinct/layer (up to 47.8 at R=32) exceeds serving's (38.2 at c32, `results/runs/2026-10-29-baseline/summary.md`), so the 8-row prize in serving may be smaller; the sort itself also costs a few kernels (other-kernels territory).

## Verdict

(verifier to write)

## From other agents

explorer-findings, 2026-11-03T21:55Z — premise correction from the script.
`bench/micro/moe_parts.py:155-159` builds `inds = mx.sort(...)` ONCE, then runs
BOTH `sort=False/True` arms on those same expert-ordered pairs, differing only
in the `sorted_indices` flag (:190,199,208); the header comment (:149-150) says
the experiment is "per-pair qvm vs the expert-grouped kernel (gather_qmm_rhs),
which MLX picks only when sorted and pairs/experts >= 4".
`decode-bottlenecks.md:183-195` confirms: "per-pair against grouped on equal
data ... the pairs are in expert order in both arms." So the 5.08/3.82,
8.12/5.65, 15.02/9.63 gaps measure KERNEL CHOICE, not pair order. Consequences:
(1) @32 rows the gap is NOT "already collected" — serving sorts from 16 rows
(`switch_layers.py:181`) but MLX still picks per-pair below 240 pairs (60+
rows), so serving pays the 15.02-style cost; the 9.63 grouped kernel never runs
in decode. (2) @8 rows, sorting serving's pairs alone cannot yield 3.82 — MLX
would still pick per-pair (B/E = 32/60 < 4); the prize needs a kernel change
(custom grouped kernel or MLX rule change), which puts it in
ineff-runs-1/ineff-findings-2 scope, not a separate sorting prize. At 8 rows
real routing is 1.6 pairs/expert, where the grouped win is untested
(decode-bottlenecks:195). (3) Kind `Bandwidth-bound` — same glossary point as on
ineff-runs-1. Suggest reframing or withdrawing this entry; lead to settle.

## From other agents

explorer-quantized-kernels, 2026-11-03T21:50Z: dispatch detail supporting this
entry — the sorted=True micro speedup happens *without* MLX's rhs batched
kernel. `GatherQMM::eval_gpu` takes `gather_qmm_rhs` only when
`M==1 && B>=16 && right_sorted_ && B/E>=4` (`quantized.cpp:2012 @v0.32.4`), and
`B/E = rows*4/60 <= 2.1` at `rows<=32` (needs `rows>=60`), so decode always runs
`affine_gather_qvm` sorted or not. The 15.02 -> 9.63 ms gap is therefore pure L2
locality from consecutive same-expert pairs, not a different kernel. Same note
for explorer-switch-layers: the model sort at `indices.size>=64`
(`switch_layers.py:181`) never reaches the kernel it was written for.

explorer-quantized-kernels, 2026-11-03T22:05Z — correction: my note above is
wrong about the micro (retracted). explorer-switch-layers and explorer-findings
show `moe_parts.py` slices weights to E experts (B/E>=4 in all three rows) and
runs both arms on the same expert-ordered pairs, so sorted=True *did* switch to
`gather_qmm_rhs` — the gaps measure kernel choice, not order. What stands: in
*serving* E=60 always, so B/E<=2.1 and decode never reaches rhs (agrees with
`territory-switch-layers.md` §G and `ineff-switch-layers-1`); there the sort buys
only locality inside qvm.

## From other agents

explorer-grader, 2026-11-03T21:24Z — the exact threshold code: `do_sort = inds.size >= 64`
(`bench/grader/slots.py:62`) with inds [rows,4], so sorting happens iff rows ≥ 16, via
`_gather_sort`/`_scatter_unsort` (`third_party/.../switch_layers.py:12-24`, called at
`slots.py:64-65,80-81`). Grader consequence: G2 compares *outputs* only (shape/dtype/2-ULP,
`check_candidate.py:112-130`), so a candidate that sorts at 8 rows — or streams each
expert once any other way — passes G2 as long as the numbers match; the reference's
unsorted-ness below 16 rows constrains nothing. Note the piece-timing compiled baseline at
8 rows is also the unsorted reference, so beating it there is a lower bar than the serving
prize. Suggest adding `grader` to Spans (owner decides).

## From other agents

explorer-switch-layers, 2026-11-03T21:31Z — correction to the premise (no duplicate entry posted):
- The kernel-A/B experiment slices the weights to E experts
  (`bench/micro/moe_parts.py:164-179`, `weight[:E]`), so B/E = 32/8, 64/16,
  128/30 ≥ 4 in all three rows. Its sorted=True therefore switches MLX from the
  per-pair qvm kernel to the grouped `gather_qmm_rhs` kernel (the script's own
  comment, `:149-150`). The gaps (5.08→3.82, 8.12→5.65, 15.02→9.63) measure the
  grouped kernel at equal pairs/expert, not what the sort buys in serving.
- In serving E=60 always, so sorting at 8 rows would NOT switch kernels
  (B/E = 32/60 < 4; the gate is `quantized.cpp:2012`, needs 60+ rows). It would
  only add the 6 sort kernels/layer and gain cache locality inside qvm — a much
  smaller prize than 1.3 ms, and possibly net negative. The 8-row prize is the
  grouped kernel itself, shared with `ineff-runs-1`.
- Same caveat applies at 32 rows: serving is sorted but still on qvm (128/60 < 4);
  the "already collected" part is locality (908 vs 1416 MB), while 908→443 MB
  remains for a grouped kernel. Full analysis in `territory-switch-layers.md` §G.

## From other agents

explorer-other-kernels, 2026-11-03T21:35Z — on the caveat's "the sort itself also costs a
few kernels": the gather-sort's ArgSort ops (40/step @32 rows, `tables.md:154`) use the
same single-block merge sort as routing's argpartition (`kernels/sort.h:319-364`): rows
of ≤128 elements → bn=32, one threadgroup of 32 threads per row
(`mlx/backend/metal/sort.cpp:288-314` @ v0.32.4) — the same ~4–7 µs fixed-cost regime as
the rest of the chain, so the sort glue is a handful of fixed-cost kernels per layer.
The sort-vs-unsorted call itself belongs to `switch-layers`/`quantized-kernels`; nothing
posted from this territory.
