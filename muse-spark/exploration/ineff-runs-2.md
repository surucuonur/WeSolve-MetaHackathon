# Inefficiency: routing costs ~1 ms while reading ~4 MB — pure overhead

- **Owner:** explorer-runs
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to confirm against the knock-out table

## Slot

`routing`

## What happens

The routing slot (router matmul, top-4 selection, weight normalize — ~12 kernels per layer, 240 per step per the brief) costs 1.02 ms per step at 1 row while reading only 4 MB (`results/runs/2026-11-02-gpu-counters/tables.md`, knock-out row `routing`: GB/s while running = 4). At 32 rows it costs 1.68 ms for 68 MB (40 GB/s). Cross-check: 240 kernels × 6.81 µs fixed cost per tiny kernel (`results/runs/2026-10-30-micro/step_parts.txt`) = 1.63 ms ≈ the measured 1.68 ms at 32 rows — the slot's time is launch overhead, not traffic.

## Cost

Per decode step (plain MLX, 128-token prompts):

- 1 row: 1.02 ms (knock-out, unrecorded), 4 MB read.
- 8 rows: no knock-out row; whole-step time is 13.37 ms (whole-step table). Between the r1 and r32 values.
- 32 rows: 1.68 ms of a 28.19 ms step, 68 MB read.

## Kind

Overhead-bound: 4 GB/s at 1 row against a 200 GB/s peak; the bytes are negligible and the kernels are the cost.

## Spans

`runs`, `model`, `other-kernels`

## Predicted saving

Fusing 12 kernels/layer into ~1 leaves one launch plus the traffic (4 MB ≈ 0.02 ms at peak). Saving ≈ measured cost minus ~0.1 ms floor, per decode step:

- 1 row: 1.02 − 0.1 = 0.9 ms.
- 8 rows: interpolated ≈ 1.2 ms (no direct knock-out; between r1 1.02 and r32 1.68, closer to neither — marked estimate).
- 32 rows: 1.68 − 0.3 = 1.4 ms (larger floor for the 68 MB traffic and the bigger fused kernel).

## Verdict

(verifier to write)

## From other agents

explorer-model, 2026-11-03T21:24Z — same cause from the code side; I marked my
ineff-model-1 superseded by this entry. Confirming detail from the kernel inventory
(`results/runs/2026-11-01-breakdown/tables.md:133-135`): the 12 kernels/layer are AsType,
Matmul, Sigmoid (router, `mobilemoe.py:219`), AsType, Add, Negative, ArgPartition,
GatherAxis (top-4, `mobilemoe.py:222-225`), Sum, Add, Divide, Multiply (weights,
`mobilemoe.py:226-228`). One question on bytes: the router weight is 3.7 MB/step
(`step-breakdown.md:250`), so the 68 MB @32 rows must be mostly activations plus
knock-out cache disturbance (cf. the unattributed-reads caveat, `step-breakdown.md:375-379`);
the fused kernel's traffic floor is ~4 MB weights + [rows,4] outputs either way. My saving
arithmetic agrees with yours within the floor (0.14 vs 0.1–0.3 ms).

explorer-findings, 2026-11-03T21:55Z — overlap ack + findings-side numbers. My
ineff-findings-1 covers routing+combine jointly (the brief's `routing+combine`
candidate builds both in one go); ineff-vllm-metal-1 already defers to it, so I
keep mine open and leave the merge decision to the lead. Added numbers from my
territory: (1) 8-row interpolation with arithmetic:
1.02 + (1.68 − 1.02) × 7/31 = 1.17 ms (agrees with your ~1.2); (2) per-kernel
split: 4.3 µs @1 row, 7.0 µs @32 rows; (3) the −29% compile existence proof
(decode-bottlenecks §3: 6.02 → 4.27 ms chained). On explorer-model's 68 MB
question: the findings-side must-read is 3.7 MB at BOTH 1 and 32 rows (breakdown
tables.md:419,434), so a 68 MB counter delta would be activations + knock-out
cache disturbance (cf. step-breakdown:375-379 — changes below ~60 MB are not
attributed to a piece); could you cite the exact counter row?

explorer-grader, 2026-11-03T21:24Z — the grader side of this headroom, from
`bench/grader/check_candidate.py`. G2 routing (`check_routing`, :50-94): exact shapes
(rows,4)/(rows,4) and dtypes u32/f32 (:55-62); inds < 60 (:65); expert *set* per row must
match (:71) unless the reference's 4th/5th biased scores are within 1e-6 — such rows count
as `near_tie_rows` and skip the weight check entirely (:72-80); otherwise weights matched
by expert, relative error ≤ 1e-4 (:81-87). So a one-kernel fusion is free to reorder the 4
within a row and to differ on tie rows only. Piece timing (`timed`, :141-150,209-226): 3
warmups, median of 30 × (all-20-layers eval, one GPU wait) ÷ 20 → µs/layer, and the
candidate must beat `mx.compile(ref_routing)` at all three row counts to stop the slot.
Suggest adding `grader` to Spans (owner decides).

explorer-other-kernels, 2026-11-03T21:35Z — kernel-level confirmation from my territory
(full mapping in `territory-other-kernels.md` checklist A–F); no new entry since this
covers the cause. Three details that strengthen it: (1) `ArgPartition` runs a FULL
60-element merge sort per row — `ArgPartition::eval_gpu` "direct[s] arg partition to
sort" (`mlx/backend/metal/sort.cpp:344-356` @ v0.32.4), size 60 → bn=32 threads, grid
(1, rows, 1) threadgroups (`sort.cpp:288-314`, `kernels/sort.h:319-364`) — and writes 60
uint32/row of which the slice keeps 4. (2) The weights-norm sum is `row_reduce_small`
with rows threads in ONE threadgroup — 1 thread @1 row (`reduce.cpp` row dispatch,
`kernels/reduction/reduce_row.h:199-257`). (3) Sigmoid/add/neg/gather/div/mul/astype
grids are rows*60 down to rows*4 threads (`kernels/unary.h`, `binary.h`, `copy.h`,
`indexing/gather_axis.h`) — every kernel far below occupancy, so the 4.3/7.0 µs each
is fixed cost, matching the entry's arithmetic. Saving arithmetic agreed.
