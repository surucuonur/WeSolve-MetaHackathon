# Inefficiency: `gather_mm_offsets` recomputed identically for gate, up and down on every prefill call

- **Owner:** explorer-quantized-kernels
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T03:25:27Z
- **Next:** verifier to confirm launch counts and redundancy

## Slot

`experts`

## What happens

Every `gather_qmm_rhs` call builds its row-tile schedule from
scratch: `offsets = gather_mm_offsets(indices, E, M=B, ...)`
([quantized.cpp v0.32.4:1795](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1795)),
one extra kernel launch per call that counts the sorted
`rhs_indices` into per-expert start rows
(host launch at
[matmul.cpp v0.32.4:2247-2265](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L2247-L2265):
`offsets` is `int32[60]`, dispatched as 60 threads in one
threadgroup). The offsets depend only on the sorted index
array and `E = 60`.

Within one MoE layer the three products (gate, up, down) are
called with the *same* sorted `idx`
(`bench/grader/slots.py:67-79`, `switch_layers.py:188-194`:
one `_gather_sort`, three `gather_qmm` calls sharing `idx`),
so the same 240-byte result is computed three times per
layer, plus a 240-byte allocation each time
(`offsets.set_data(allocator::malloc(...))`, matmul.cpp
host function). Nothing caches it: the primitive has no
cross-call state, and each `GatherQMM::eval_gpu` invocation
starts fresh (`quantized.cpp:1980-2094`).

## Cost

Decode 1/8/32 rows: **0 kernels** — the `qvm` path builds no
offsets (territory §A). Prefill, per 20-layer forward (one
prefill chunk at 127/1,016/2,048 rows; chunking in
`bench/grader/slots.py:242`):

- offsets kernels launched: 3 products × 20 layers = **60**,
  of which **40 are bit-identical recomputations** (only 20
  distinct results exist, one per layer).
- bytes: 60 × 240 B written = 14.4 KB total; the kernel reads
  the `B`-element sorted index array to count it
  (508 / 4,064 / 8,192 × 4 B = 2.0 / 16.3 / 32.8 KB read per
  call, 60 calls per forward). Byte time at 200 GB/s peak
  (`claude/GLOSSARY.md:10`) is < 0.2 µs per call — negligible;
  the cost is the fixed launch of a 60-thread kernel.
- fixed cost: no per-launch µs is measured for this kernel.
  Bounding with the decode step's small-kernel scale —
  routed experts 140 kernels / 0.92 ms ≈ 6.6 µs average
  including full `qvm` products (`claude/findings/step-breakdown.md:179-185`
  via round 1 `territory-quantized-kernels.md` §A) — 40
  redundant launches cost **at most ≈0.26 ms per prefill
  forward** (40 × 6.6 µs), and less when launches pipeline
  behind the main kernels; MLX's own `// TODO: Tune 16 and 4`
  (`quantized.cpp:2011`) flags this dispatch as untuned.
  At 127 rows the main kernels are smallest, so the fixed
  part is largest *relative* there; at 2,048 rows it is
  noise against ≈929 µs of useful MMA per layer-product
  (territory §E).

Kernels × fixed cost is the whole cost; there is no
bandwidth or compute component worth modelling.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): a 60-element counting
kernel whose runtime is its launch, repeated identically.
Far below both bandwidth and compute limits.

## Spans

`quantized-kernels` (`gather_qmm_rhs` host wrapper +
`gather_mm_offsets`) → slot `experts`. The three-call
structure that makes it redundant lives in switch-layers
(`SwitchGLU.__call__`), but the recomputation is inside
this territory's host function.

## Predicted saving

0 µs per decode step at 1, 8 and 32 rows (path never runs).
At prefill: at most ≈0.26 ms per 20-layer chunk forward
(40 redundant launches × ≤6.6 µs, upper bound above;
pipelined reality smaller), in time-to-first-token only —
prefill speed is not gated (`claude/adr/0010-exact-at-every-size.md:42`).
For a candidate kernel this entry costs nothing to avoid:
a fused experts kernel computes its own grouping (or none,
territory §C invariant) and never needs MLX's offsets at
all. Recorded so the reference's prefill overhead is fully
accounted; not worth a fix on its own.

## Verdict

verifier, 2026-11-04T03:25:27Z — **confirmed**. Checked: every `gather_qmm_rhs` call recomputes `offsets = gather_mm_offsets(indices, E, M=B, ...)` (MLX v0.32.4 `quantized.cpp:1795`, fetched; host launch `matmul.cpp:2247-2265`: `int32[60]`, `dispatch_threads(60)` in one threadgroup, fresh `malloc` per call) ✓; the three products in one layer share the same sorted `idx` ([slots.py:67-79](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:67), `switch_layers.py:188-194` — one `_gather_sort`, three `gather_qmm` calls) so the same 240 B result is computed 3× per layer with no cross-call cache (`GatherQMM::eval_gpu` `quantized.cpp:1980-2094` starts fresh) ✓. Decode 0 kernels (`qvm` builds no offsets, [territory-quantized-kernels.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-quantized-kernels.md) §A) ✓. Counts per 20-layer prefill forward ([slots.py:242](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:242) chunking): 3×20=60 launches, 40 bit-identical recomputations (20 distinct) ✓. Bytes: 60×240 B=14.4 KB written; reads B×4 B = 2.0/16.3/32.8 KB per call at B=508/4,064/8,192, <0.2 µs at 200 GB/s ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) — launch-bound, negligible bytes ✓. Upper bound 0.92 ms/140≈6.6 µs average (routed experts 140 kernels/0.92 ms, `step-breakdown.md:179-185` via territory §A) ×40 ≈0.26 ms per forward, pipelined reality smaller, explicitly an upper bound with no measured per-launch figure ✓; `// TODO: Tune 16 and 4` at `quantized.cpp:2011` ✓. Predicted saving 0 µs decode; at most ≈0.26 ms per chunk forward in time-to-first-token only (prefill untimed, [0010-exact-at-every-size.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/adr/0010-exact-at-every-size.md)), free for a fused candidate that never needs MLX offsets — recorded for accounting, not worth a fix on its own, as written.

## From other agents

(none yet)
