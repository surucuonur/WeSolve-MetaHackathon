- **Owner:** explorer-vllm-metal
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-03T21:23:53Z
- **Next:** verifier to confirm

## Slot

outside the slots (engine CPU on a CPU-bound step).

## What happens

At 1 row vLLM's step is set by the CPU (it waits only 0.41 ms for the GPU,
0.06 ms in the least disturbed repeat), so every microsecond of engine CPU
lengthens the step (step-breakdown §2). vLLM spends 2–3 ms more CPU per step
than plain MLX (step-breakdown §4):

- 0.5 ms in the runner outside the model (batch build, metadata, sampling
  bookkeeping) — `v1/model_runner.py` `execute_model`/`_start_paged_forward`.
- ~1 ms in vllm-metal's attention-path Python: `sdpa_forward` + wrapper's own
  code 564 µs vs plain MLX's whole 233 µs attention code; `prepare_sdpa_qkv`
  250 µs (per-layer `getattr` head-count/scale/capability probes,
  `attention/impls/sdpa.py:634-667`); RoPE helper 161 µs, rebuilding the same
  `mx.array(offsets)` in each of the 20 layers
  (`attention/impls/varlen_rope_compat.py:221-228`).
- +1.1–1.5 ms encoding (`mx.async_eval` 4.10 vs 2.61 ms plain; 3.68 ms in
  vLLM's least disturbed repeat): 101 more kernel ops and 140 more view ops
  to encode per step. (Part of the gap may be paging, which counts as CPU.)

## Cost

- 1 row: ≈ 2.0–2.5 ms engine CPU above plain MLX on a 5.64 ms CPU-bound step
  (step-breakdown §2 + §4; 1-row cpu-mode figures are upper bounds because
  instrumentation lengthens the step 7.44 vs 5.64 ms bare).
- 8 rows: runner-outside-model 0.9 ms but the CPU waits 6.58 ms for the GPU —
  engine CPU is off the critical path (≈ 0 ms cost).
- 32 rows: same, 1.6 ms of runner work hidden under 19.39 ms of GPU wait
  (≈ 0 ms cost).

## Kind

Overhead-bound (glossary): CPU-side launch/record/encode work, not memory or
arithmetic. Only the 1-row step pays it.

## Spans

vllm-metal (runner + attention path). Overlaps ineff-vllm-metal-3 (the
gathers/views also cost encode time) — that entry counts GPU-chain time,
this one CPU time; do not add them.

## Predicted saving

Small Python-side fixes (cache each layer's resolved attributes, build RoPE
positions once per step ≈ 161 µs, `squeeze(0)` views cutting 140 views to
encode):

- 1 row: ≈ 0.5 ms/step (RoPE-once 0.16 ms measured-CPU + a share of the
  record/encode gap; estimate, and 1-row figures are upper bounds).
- 8 rows: ≈ 0 ms/step (GPU sets the pace; CPU work is hidden).
- 32 rows: ≈ 0 ms/step (same).

## Verdict

(verifier writes)
