# Inefficiency: split-K partial-result traffic from 14 rows up

- **Owner:** explorer-findings
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to check the arithmetic against the findings

## Slot

Outside the slots (the q, k, v, o projections and the shared expert — no slot
covers them). Posted so the swarm does not mistake this traffic for expert or
attention bytes.

## What happens

From 14 rows up, MLX runs its quantized products split-K: the K dimension is cut
into chunks, each chunk writes its own float32 copy of the output
(`split_k × rows × outputs` numbers), and a second kernel reads them back and
adds them up (`quantized.cpp:1166-1226`, threshold `get_qmv_batch_limit` = 14 on
applegpu_g13s at `:89-125` — all via step-breakdown §3 / bottleneck-explained
§6). At 32 rows split_k is 12/24/24/12 for q/k/v/o and 8/8/16 for the shared
expert's gate/up/down; the buffers total 173 MB per step (23.6 + 15.7 + 15.7 +
23.6 + 31.5 + 31.5 + 31.5 MB). That is why the step writes 3 MB at 1 row and
20-21 MB at 8 rows but 251-266 MB at 32 rows (GPU counters, 2026-11-02, as
quoted in the findings).

## Cost

- **1 row:** split-K off. Writes 3 MB/step. Cost **0**.
- **8 rows:** split-K off (below the 14-row threshold). Writes 20-21 MB/step.
  Cost **0**.
- **32 rows:** 173 MB written (worked out from the code) + read-back + 7 extra
  add kernels per layer (140 per step). Measured write savings by knock-out:
  57 MB (q, k, v projections) and 102 MB (shared expert), against worked-out 55
  and 94 MB (the excess is ordinary outputs). Time effect **not measured** —
  step-breakdown §6 calls this "a lead, not yet a fix".

## Kind

Overhead-bound: extra traffic plus extra kernels, while the affected pieces run
at 7-15 GB/s on needed bytes (tables.md achieved-bandwidth table, lines 425-438)
— far below both limits.

## Spans

`findings` (measured traffic), `quantized-kernels` (split-K code in MLX).

## Predicted saving

Estimate only, framed as a rough upper bound — the time effect is unmeasured:

- **1 row:** **0 µs** (split-K off).
- **8 rows:** **0 µs** (split-K off).
- **32 rows:** (173 MB written + 173 MB read back) / 200 GB/s = 1.73 ms, plus
  140 kernels × ~5 µs = 0.7 ms → **up to ≈2400 µs per step**, likely less: each
  layer's buffer is only 0.8-1.6 MB so the read-back may come mostly from cache,
  and traffic overlaps kernel execution. Testing needs a patched MLX (no setting
  turns split-K off).

## Verdict

(not yet written — the verifier writes this)
