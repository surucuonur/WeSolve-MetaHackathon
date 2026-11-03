# Inefficiency: routing costs ~1 ms while reading ~4 MB — pure overhead

- **Owner:** explorer-runs
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:25:00Z
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
