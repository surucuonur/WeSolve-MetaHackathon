# Inefficiency: baseline-combine-1 — chained dispatch floor, ~32 µs/layer at decode

- **Owner:** explorer-baseline-combine
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check the floor arithmetic; no builder action predicted to bank it

## Slot

`combine`.

## What happens

The code in the slot
(`muse-spark/round-3/kernels/combine/attempt-2/block-0.metal:26-42`,
`spec.json`: grid `rows*768`, threadgroup [256,1,1]) is already a
single fused kernel per layer — the 5-kernel reference chain
(`bench/grader/slots.py:51-52` eagerly) collapsed to one launch,
20 per decode step. What remains at decode sizes is a flat
~32 µs per layer that does not move with rows (31.8 at 1 row,
34.7 at 8, 32.7 at 32; `grade.json`), while the work itself is
9,232 B/row of unique traffic and ~7 f32 ops per output. The
grader prices each layer through the same serial `link` chain
for every function timed (`bench/grader/check_candidate.py:191-214`),
so the floor — one `mx.fast.metal_kernel` dispatch, command
encoding and launch latency at 3 / 24 / 96 threadgroups — is
shared with the compiled bar and cancels out of the margin; it
is the level the margin sits on, not the margin itself. The
same floor was confirmed for this slot's shape under the old
timing basis in round 2 (`muse-spark/round-2/exploration/ineff-baseline-combine-2.md`,
frozen); this entry reprices it for the pragma kernel under
round 4's chained paired grader. No `max_rows`/`min_rows` is
set, so no size hands off (`bench/grader/slots.py:338-339`).

## Cost

Per layer (peak 200 GB/s, 5.2 TFLOPS, `claude/GLOSSARY.md`;
measured, piece: `muse-spark/round-3/kernels/combine/attempt-2/grade.json`):

- Kernels × fixed cost: 1 kernel/layer either way (candidate
  and compiled reference alike in the chain). Candidate
  measured 31.8 / 34.7 / 32.7 µs decode, 38.2 / 100.9 / 198.0
  prefill; compiled 36.9 / 41.1 / 45.7 and 64.0 / 281.8 / 556.2.
- Bytes ÷ peak: 9,232 B/row → 9,232 / 73,856 / 295,424 B decode
  = **0.05 / 0.37 / 1.48 µs**; 1,172,464 / 9,379,712 /
  18,907,136 B prefill = **5.86 / 46.9 / 94.5 µs**.
- Compute: 7 × 768 × rows FLOP = **~0.001 / 0.008 / 0.03 µs**
  decode, ~0.13 / 1.05 / 2.12 µs prefill.
- Remainder (measured − bytes − compute): **~31.7 / 34.3 /
  31.2 µs at decode (≥95% of the measured time)** and ~32 /
  53 / 101 µs at prefill (dispatch amortised over more rows;
  prefill slope ~0.094 µs/row ≈ 95 GB/s achieved, still half
  of peak). Over 20 layers the decode remainder is ~0.62–0.69
  ms of every decode step spent before any byte or FLOP of
  this slot is paid for — for the compiled bar too (its
  remainder is larger, which is the 5.1 / 6.4 / 13.0 µs/layer
  margin, 24 / 27 / 30 wins).

Absolute chained µs drift up to ~35% between runs on identical
code (round-3 `attempt-2/notes.md`), so the remainders above
are same-run levels, not transferable absolutes; the flatness
in rows and the win counts are the stable part.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): far below both peak
bandwidth and peak compute at decode; at prefill still only
~half of peak bandwidth, so overhead/latency-shaped there too.

## Spans

`baseline-combine` (the single launch and its spec), `grader`
(the link chain that prices the floor identically for every
function). Explicitly not the kernel body's arithmetic, which
section B of the territory entry shows is at its op minimum.

## Predicted saving

If the floor could be removed, the decode saving would be the
remainder above — **~0.62–0.69 ms per decode step** — but no
bit-exact change inside this slot removes it: the slot already
launches the minimum (one kernel per layer), the chain's
`link` kernel and dispatch are the harness's and Metal's, and
two prior organisations (staged/vectorised, volatile and
devolatilised) landed on the same level (frozen evidence in
the territory entry, section E). **Predicted achievable saving
from any kernel-body or launch-shape change: 0 µs/layer at
decode** (prediction; the floor can only be lost, e.g. by an
added kernel or barrier, never banked). At prefill the
non-floor part is body throughput (~95 vs 200 GB/s), already
2.8× ahead of the compiled bar at 2,048 rows; closing it
toward peak would need wider/vectorised loads whose decode
effect is bounded in `ineff-baseline-combine-2` and whose
prefill prize (up to ~50 µs/layer at 2,048 rows, prediction
from the slope gap) does not touch the decode margin or the
32×1024 G3 question, where this slot already wins its piece
sizes 29–30/30. A saving in prefill that costs decode is not
one; here the phases share identical lines, so any body change
must be re-proved at both.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** single fused kernel per layer, grid `rows*768`, threadgroup [256,1,1] ([block-0.metal:26-42](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/combine/attempt-2/block-0.metal:26), `spec.json`), reference [slots.py:51-52](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:51) collapsed from 5 kernels/layer; no `max_rows`/`min_rows`, `hands_off` false everywhere ([slots.py:338-339](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:338)) ✓; flat decode 31.8/34.7/32.7 µs/layer at 3/24/96 threadgroups from [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/combine/attempt-2/grade.json), chained `link` pricing identical for every function ([check_candidate.py:191-214](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:191)) so the floor cancels out of the margin ✓; same floor confirmed under the old basis in frozen [ineff-baseline-combine-2.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/ineff-baseline-combine-2.md) ✓.
- **confirmed:** floors at 200 GB/s / 5.2 TFLOPS ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) — unique 9,232 B/row (6,144+16+1,536+1,536) → 9,232/73,856/295,424 B decode = 0.05/0.37/1.48 µs, prefill 1,172,464/9,379,712/18,907,136 B = 5.86/46.9/94.5 µs; compute 7×768×rows ≈ 0.001/0.008/0.03 µs decode, 0.13/1.05/2.12 µs prefill ✓; remainders 31.7/34.3/31.2 µs decode (≥95%), 32/53/101 µs prefill; decode remainder 0.62–0.69 ms/step over 20 layers; margins 5.1/6.4/13.0 µs/layer (24/27/30 wins) are the compiled remainder being larger ✓; prefill slope (198.0−100.9)÷1,032 = 0.094 µs/row ≈ 95 GB/s, half of peak — overhead-bound ✓; ~35% cross-run drift caveat matches attempt-2 `notes.md` ✓.
- **confirmed:** predicted achievable saving **0 µs/layer at decode** — one launch is already minimal, floor is harness/Metal dispatch, two prior organisations landed on the same level; prefill body prize (~50 µs/layer at 2,048 rows toward peak) labelled prediction and barred from costing decode ✓.
