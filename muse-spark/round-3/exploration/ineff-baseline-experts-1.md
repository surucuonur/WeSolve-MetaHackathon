# Inefficiency: scalar prefill branch runs 15–28× slower than compiled — serial chain, scalar redundant loads, no cross-row reuse

- **Owner:** explorer-baseline-experts
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:46:33Z
- **Next:** done — confirmed by verifier (see Verdict)

## Slot

`experts`

## What happens

At `rows >= 60` the slot kernel takes its prefill branch
(`muse-spark/round-2/kernels/experts/attempt-7/block-0.metal:22,59-72,103-112`),
one thread per output calling `qmm_accum`
(`block-1.metal:206-230`): a single thread walks all of K
(768 for gate/up, 384 for down) in `BK=32` contiguous
order, accumulating into one float `acc`. The arithmetic
is right — G2 passes with `differ 0` at all three prefill
sizes — but the issue pattern loses three ways at once:

1. **A purely serial dependent chain per thread.** Each of
   the 884,736 iterations per pair (gate/up 384×768 each,
   down 768×384) does `acc += xv[k] * (bfloat)wq` on one
   accumulator: no second accumulator, no unrolling, so
   each iteration waits for the last — zero instruction
   overlap within a thread. Serial depth per thread (simd
   groups 0–7) is 768 (gate) + 768 (up) + 768 (down, two
   blocks) = 2,304 dependent iterations, plus silu. Gate
   and up run as fully sequential phases (384 of 512
   threads active in each, `block-0.metal:61-71`); down
   runs 2 waves (`:105-111`).
2. **Scalar redundant loads: ~7.1 MB issued per pair for
   553 KB of unique weight bytes (12.8×).** Per iteration
   each thread issues 1 W load + 1 S + 1 B device load +
   1 threadgroup `xv` load (`block-1.metal:223-226`), but
   8 lanes share each weight word (`word = wb + (lane >>
   3)`, `:217`) and all 32 lanes of a block share `S`,
   `B` and `xv[k]`. Issued per pair: W 884,736 words =
   3.54 MB vs 442,368 unique B (8×); S+B 884,736×2×2 B =
   3.54 MB vs 110,592 unique B (32×). All scalar,
   uncoalesced, per-thread.
3. **No cross-row reuse: per-pair grid at fan-ins of
   508–8,192 pairs.** Each pair streams its expert's full
   552,960 B independently, so unique-pair traffic is
   280.9 MB / 2,247 MB / 4,530 MB per layer at 127/1,016/
   2,048 rows, where MLX's grouped `gather_qmm_rhs` loads
   each appearing expert's tile once per `BM=16` row tile
   into threadgroup `Ws` and shares it across 16 rows
   (round-2 `territory-quantized-kernels.md` §C–§E,
   frozen), and issues 8×8×8 matrix instructions (~512
   MACs per simdgroup step vs this branch's 32 scalar
   MACs — a 16× ALU-issue gap per step).

This branch was written when prefill speed was free:
round-2 `ineff-baseline-experts-1` (confirmed, frozen)
states "a slow-but-exact prefill branch is free" because
prefill was untimed (`claude/adr/0010-exact-at-every-size.md:42`).
Under two-phase judgment (ADR 0011) it is the slot's
largest loss. The branch threshold itself is correct
(`rows >= 60` from dispatch source, round-2 territory
§E/§F); rows 33–59 still take the decode path.

## Cost

Measured, piece (chained median µs/layer,
`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`):

| Rows | Pairs | Reference | Compiled | Candidate | Ratio vs compiled |
|---|---|---|---|---|---|
| 127 | 508 | 963.4 | 960.8 | 15,220.0 | 15.8× |
| 1,016 | 4,064 | 3,979.3 | 4,306.0 | 119,154.4 | 27.7× |
| 2,048 | 8,192 | 7,359.8 | 11,998.9 | 240,681.1 | 20.1× |

- Candidate per pair per layer: 15,220/508 = 29.96,
  119,154/4,064 = 29.32, 240,681/8,192 = 29.38 µs —
  constant ~29.5 µs/pair across a 16× row range: the
  branch scales linearly with pairs at a fixed issue
  rate. Compiled per-pair-equivalent: 1.89 / 1.06 /
  1.46 µs (sublinear mid-range — grouped amortization;
  the candidate has none).
- Effective bandwidth on pair-model bytes at 200 GB/s
  peak (`claude/GLOSSARY.md:10`): 280.9 MB / 15,220 µs
  = 18.5, 2,247 / 119,154 = 18.9, 4,530 / 240,681 =
  18.8 GB/s — ~9.4% of peak, identical at all three
  sizes. Byte floors on the same traffic: 1,404.6 /
  11,237 / 22,650 µs; the candidate runs 10.6–10.8×
  above its own byte floor.
- Kernels: 1 per layer per call (candidate) vs the
  reference's grouped products + sort glue (count at
  prefill unmapped — decode counts 7/13 per layer do
  not apply; recorded under Unverified in the
  hypothesis this feeds). Fixed launch cost is not the
  loss: the loss is ~29.5 µs/pair of serial issue.
- At decode 1/8/32 rows this entry costs 0: the branch
  is not taken below 60 rows.

End-to-end stake (whole request, `console-vllm.txt` +
`results/runs/2026-11-04-engine-comparison/`, via
round-3 README): prompt processing 6.63× / 14.50× /
12.50× / 13.09× slower and throughput −10.0% / −37.5% /
−48.0% / −82.5% (routing + experts; routing alone adds
only 1.07–1.10× prompt time, so the remainder is this
branch).

## Kind

Overhead-bound (`claude/GLOSSARY.md`): 18.8 GB/s ≈ 9%
of peak bandwidth and a small fraction of peak compute
— time goes to issuing scalar loads and stepping a
serial dependent chain, not to bytes or arithmetic.
(All three prefill sizes show the same rate, the
signature of an issue/occupancy ceiling, not a
bandwidth slope.)

## Spans

`baseline-experts` (`block-0.metal` prefill branch,
`block-1.metal` `qmm_accum`), `quantized-kernels` (the
`gather_qmm_rhs` tiling/occupancy a fast branch must
match — round 3's reopened question), `grader` (two-phase
judgment, `max_rows` hand-off grading).

## Predicted saving

If the prefill path matched the compiled reference's
speed at equal bits (fast exact branch, or the
`max_rows` hand-off Claude checked bit-exact in
`results/runs/2026-11-04-handoff-check/`), per layer
(candidate − compiled, grade-phases file):

- 1 / 8 / 32 rows: **0 µs** — the branch is not taken;
  a prefill fix must not spend the decode wins (8 rows
  ahead 72.7, 32 rows ahead 282.5 µs/layer).
- 127 rows: 15,220.0 − 960.8 = **14,259.2 µs/layer**,
  × 20 layers = **285.2 ms per 20-layer chunk forward**.
- 1,016 rows: 119,154.4 − 4,306.0 = **114,848.4
  µs/layer** = **2,296.9 ms per chunk forward**.
- 2,048 rows: 240,681.1 − 11,998.9 = **228,682.2
  µs/layer** = **4,573.6 ms per chunk forward**.

This is a prediction: it counts only the candidate's
own measured gap and stands beside the measured value
once a fix grades. Whole-request effect (prediction):
prompt processing back to ~compiled parity from 6.6–
14.5× slower, recovering the −10% to −83% throughput
loss to whatever the decode phase then gives — a
prefill-only fix banks nothing at decode sizes and
must keep the 8/32-row wins. Fix shapes for builders
(not costed here): (a) vectorized per-pair scalar —
one thread computes a word's 8 outputs with 8
accumulators (8× overlap, 8× fewer W/S/B issues, each
output keeps contiguous `BK=32` order, exact by the
same G2-proven coincidence this branch relies on);
(b) grouped/tiled replica of `qmm_rhs` issue;
(c) `max_rows` hand-off. Reference prefill waste a
fast branch takes for free is priced in round-2
`ineff-quantized-kernels-1/2` (confirmed, frozen).

## Verdict

verifier, 2026-11-04T17:46:33Z — **confirmed**. Every claim the cost and the saving rest on holds; two corrections (a misquote and trivial rounding) change nothing.

- **confirmed:** prefill branch at `rows >= 60` (block-0:22), gate/up (block-0:59-72) and down (block-0:103-112) call `qmm_accum` (block-1:206-230): one thread per output, single float `acc`, K = 768/768/384, contiguous `BK=32` order, no unrolling.
- **confirmed:** 884,736 iterations/pair (384×768 ×2 + 768×384) ✓; serial depth 768+768+768 = 2,304 for simds 0–7 (down 2 blocks only there, block-0:105-111) ✓; gate/up sequential phases with 384 of 512 threads each (block-0:61,67 `simd < 12`) ✓.
- **confirmed:** per-iteration loads 1 W + 1 S + 1 B device + 1 `xv` threadgroup (block-1:223-226); 8 lanes share each W word (`:217`), 32 share S/B/`xv[k]`; issued W 884,736×4 B = 3.54 MB vs 442,368 unique (8×), S+B 884,736×2×2 B = 3.54 MB vs 110,592 unique (32×); total 7.08 MB vs 552,960 B unique (12.8×) ✓.
- **confirmed:** no cross-row reuse — per-pair grid (block-0:16), 508/4,064/8,192 pairs (rows×4) ✓, 552,960 B/pair ✓, unique-pair traffic 280.9/2,247.2/4,530.5 MB ✓; `rhs` loads each appearing expert's tile once per `BM=16` row tile into `Ws` shared across ≤16 rows (round-2 territory-quantized-kernels §C–§E); 8×8 frags stepped 8-wide over BK (§E) = 512 MACs vs 32 scalar = 16× issue gap ✓ (issue-rate comparison, not measured).
- **confirmed:** cost table — all six reference/compiled/candidate triples match `grade-phases-experts-attempt-7.json`; ratios 15.84/27.67/20.06 → 15.8/27.7/20.1 ✓; per-pair 29.96/29.32/29.38 ✓; compiled per-pair-equiv 1.89/1.06/1.46 ✓; effective 18.5/18.9/18.8 GB/s ≈ 9.4% of 200 (GLOSSARY.md:10-11) ✓; 10.6–10.8× above own byte floor ✓; G2 `differ 0` at all three prefill sizes (JSON) ✓.
- **confirmed:** end-to-end prompt 6.63/14.50/12.50/13.09× and throughput −10.0/−37.5/−48.0/−82.5% (round-3 README; routing-only 1.07–1.10× prompt corroborates the attribution) ✓; decode cost 0 below 60 rows (code line 22 + §E dispatch ⇔ rows ≥ 60; rows 33–59 take the decode path) ✓; reference prefill kernel count honestly recorded unverified.
- **confirmed:** saving — 8/32-row leads 72.7/282.5 ✓; 14,259.2/114,848.4/228,682.2 µs/layer ✓; ×20 = 285.2/2,296.9/4,573.6 ms ✓; `max_rows`-59 hand-off bit-exact (`results/runs/2026-11-04-handoff-check/grade-experts-a7-max59.json`, corroborated by round-3 README); round-2 `ineff-quantized-kernels-1/2` both `confirmed` ✓.
- **wrong (quote):** round-2 `ineff-baseline-experts-1` (itself `confirmed`) does not state "a slow-but-exact prefill branch is free" — it states "Prefill speed is not gated (G3 times decode only)". Substance holds; also its ADR line is :44 in the current file, not :42 (stale line inherited from round-2's own cites).
- **confirmed (rounding):** byte floors exact 1,404.5/11,236.1/22,652.3 µs vs written 1,404.6/11,237/22,650 — immaterial.

## From other agents

builder-2-c, 2026-11-04T18:05:23Z — direction (c) grouped/tiled replica
graded as `muse-spark/round-3/kernels/experts/attempt-3`: G1 fail, no
G2/timing. Cause is a Metal reserved word (`uint half` in
`block-3.metal:dequant_tile_4bit`; `half` is the half-float type),
not the grouped design (counting-sort + fused grouped MMA + select),
whose prefill predictions (900/4,100/11,500 us/layer, `faster`
everywhere) are untested, not refuted. One-line rename (`half` ->
`part`) is the next change; full verdict in the attempt's `notes.md`
and `grade.json`.

builder-2-b, 2026-11-04T18:05:14Z — direction (b) priced, and it
is not the lever. Attempt-2
(`muse-spark/round-3/kernels/experts/attempt-2/`, G1+G2 pass,
differ 0 all six) widened the prefill branch to 8 outputs per
thread with 8 accumulators (8× fewer W/S/B issues, same
per-output order): prefill timed 20,444/156,416/303,543
µs/layer vs the starting point's 15,220/119,154/240,681, and
within-run vs-compiled 18.4/23.1/18.6× vs 15.8/27.7/20.1× —
no speedup (this run's machine was slower than the basis
run's; the byte-identical decode path slowed too, so the
within-run ratio is the signal). Cause: each k step's 8 FMAs
depend on that step's loads and the k loop is unrolled
nowhere, so the serial load→FMA depth stayed 768 while active
threads fell 384→48 per group — the widening removed the
cross-thread latency hiding without shortening any chain. A
scalar-branch speedup must unroll or pipeline across k, not
across outputs; the no-cross-row-reuse cause stands as the
structural fix. Full arithmetic in the attempt's `notes.md`
(§Miss analysis).

builder-2-d, 2026-11-04T18:09:35Z — direction (c) passes G2 and prices
the grouped design. Attempt-4
(`muse-spark/round-3/kernels/experts/attempt-4/`, attempt-3 + `half` ->
`part` rename only): G1 pass, G2 pass with `differ == 0` at all six
sizes — the grouped prefill path (counting-sort + fused grouped MMA +
select) is bit-exact, refuting the frag-mapping suspect. Prefill timed
2,684/9,311/17,229 µs/layer vs compiled 1,006/4,073/7,417 (`slower`
x3, 2.67/2.29/2.32x) — the scalar branch is gone (cross-run 5.7x/12.8x/
14.0x faster than the starting point's 15,220/119,154/240,681) but the
"compiled rate minus banked waste" prediction missed ~3x. Candidate
causes, unprofiled: (1) `x` gathered 12x per tile (`g`-outer/`k`-inner
nest, `block-2.metal:54-78`: 288 KB vs 24 KB gathers per tile group);
(2) scalar-fed MMA (~34 TG loads + 2 barriers per `mma_step`) at 64
threads/group behind ~19 KB TG; (3) partial-tile padding at small
sizes. Decode verdicts match the starting point (slower/faster/faster)
but overhead missed (+30/+44/+69 vs +3/+3/+4 predicted; ~24 µs/layer
at 1 row unexplained). Full arithmetic in the attempt's `notes.md`.
