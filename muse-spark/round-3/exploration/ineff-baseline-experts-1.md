# Inefficiency: scalar prefill branch runs 15–28× slower than compiled — serial chain, scalar redundant loads, no cross-row reuse

- **Owner:** explorer-baseline-experts
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:39:32Z
- **Next:** verifier to confirm issue-amplification arithmetic and per-pair linearity

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

(verifier to write)

## From other agents

(none yet)
