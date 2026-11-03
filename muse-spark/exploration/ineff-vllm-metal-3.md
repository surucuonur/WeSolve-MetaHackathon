- **Owner:** explorer-vllm-metal
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-03T21:36Z
- **Next:** verifier to confirm

## Slot

outside the slots (attention prologue copies).

## What happens

Every layer copies q, k and v twice before the attention kernel
(`third_party/vllm-metal/vllm_metal/attention/impls/sdpa.py:722-724`):

```python
q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(kv_cache.dtype))
# ... same for k_3d, v_3d
```

`queries[0]` is a `Gather` (a copy, not a view, in MLX indexing), and
`mx.contiguous` launches a second copy from 2 rows up (at 1 row the
transposed array is already contiguous so MLX shares memory). Per step:
60 `Gather` + 60 `Contiguous` kernel operations (step-breakdown §4 table).
The exact fix is `squeeze(0)` instead of `[0]`: a view already in the
kernel's layout, so both copies stop copying (step-breakdown §5 table).

## Cost

- Kernels: 60/step at 1 row (Contiguous no-ops), 120/step at 8 and 32 rows.
- 1 row: 60 × ~5 µs ≈ 0.30 ms (arithmetic from the fixed kernel cost,
  estimate); the repeat-fix measured −0.59 ms but inside vLLM's 1-row spread
  (5.36–6.20 ms), so not claimed.
- 8 rows: −0.21 ms measured by `repeat_fixes.py` (beats the 0.03 ms spread;
  1.5% of the step) — step-breakdown §5 table.
- 32 rows: copies knock-out saves 0.24 ms (step-breakdown §3 vLLM table);
  repeat-fix measured −0.08 ms (inside the 29.63–30.20 spread).
- Bytes: negligible (activations already resident); this is pure launch
  overhead on the kernel chain — the attention kernel waits for the copies,
  which is why removing them shows (step-breakdown §5).

## Kind

Overhead-bound (glossary): on-chain launch cost, no material memory traffic.

## Spans

vllm-metal (`attention/impls/sdpa.py` only).

## Predicted saving

- 1 row: ≈ 0.3 ms/step (60 kernels × 5 µs, estimate).
- 8 rows: 0.21 ms/step (measured, `repeat_fixes.py`).
- 32 rows: 0.24 ms/step (measured, copies knock-out).

## Verdict

(verifier writes)
