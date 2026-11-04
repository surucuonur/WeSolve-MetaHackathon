# Inefficiency: redundant `gather_mm_offsets` repriced — ≤13 µs per prefill layer, now bankable (repriced)

- **Owner:** explorer-quantized-kernels
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:41:57Z
- **Next:** verifier to confirm repricing arithmetic

## Slot

`experts`

## What happens

Revisit of round 2's confirmed
[`ineff-quantized-kernels-2`](../round-2/exploration/ineff-quantized-kernels-2.md)
(same mechanism, frozen — not re-derived here): every `gather_qmm_rhs`
call recomputes `offsets = gather_mm_offsets(indices, E, M=B, ...)`
(`quantized.cpp:1795`; 60 threads in one threadgroup,
`matmul.cpp:2247-2265`), and the three products of one layer share
the same sorted `idx`, so the same 240 B result is computed 3× per
layer with no cross-call cache. What is NEW this round: the judgment
changed ([ADR 0011](../../../claude/adr/0011-decode-and-prefill.md)) —
round 2 ranked this at unbanked time-to-first-token; under two-phase
judgment it is prompt time a fast prefill branch (or the prompt-time
leg of `beats_compiled`) banks directly. No new mechanism; this entry
re-states the cost per layer at all six sizes so builders can rank it.

## Cost

Decode 1/8/32 rows: **0 kernels** — the `qvm` path builds no offsets
(round-2 territory §A). Prefill, per 20-layer chunk forward (one
chunk at 127/1,016/2,048 rows; chunking in
`bench/grader/slots.py:242`): 3 products × 20 layers = **60** offsets
launches, of which **40 are bit-identical recomputations** (20
distinct results, one per layer) — counts verified in the round-2
entry. Bytes negligible (<0.2 µs per call at 200 GB/s peak); the cost
is the fixed launch of a 60-thread kernel. Upper bound per launch
≤6.6 µs (routed-experts small-kernel scale, 140 kernels / 0.92 ms —
round-2 entry; no per-launch prefill figure exists, and pipelined
reality is smaller): 40 × ≤6.6 µs = **≤0.26 ms per chunk forward**,
i.e. **≤13.2 µs per layer** at each prefill size (2 redundant launches
per layer × ≤6.6 µs).

## Kind

Overhead-bound (`claude/GLOSSARY.md`): a 60-element counting kernel
whose runtime is its launch, repeated identically. Far below both
bandwidth and compute limits.

## Spans

`quantized-kernels` (`gather_qmm_rhs` host wrapper +
`gather_mm_offsets`) → slot `experts`. The three-call structure that
makes it redundant lives in switch-layers (`SwitchGLU.__call__`), but
the recomputation is inside this territory's host function.

## Predicted saving

0 µs per decode step at 1, 8 and 32 rows (path never runs). At
prefill: **≤13 µs per layer** at each of 127 / 1,016 / 2,048 rows
(≤0.26 ms per 20-layer chunk forward), in prompt time — bankable under
two-phase judgment toward the prefill leg. For a candidate kernel this
entry costs nothing to avoid: a fused experts kernel computes its own
grouping (or none — regrouping is free, territory §S1) and never needs
MLX's offsets at all. A `max_rows` hand-off does NOT take it (MLX
still recomputes on the reference path). Not worth a fix on its own;
recorded so the reference's prefill overhead stays fully accounted.

## Verdict

(verifier to write)

## From other agents

(none yet)
