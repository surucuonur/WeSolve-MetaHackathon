# Hypothesis 2: experts prefill at compiled speed

- **Owner:** lead
- **Status:** boundary
- **Version:** 1
- **Updated:** 2026-11-04T17:53:22Z
- **Next:** boundary-2 draws the boundary

## Inefficiency

[ineff-baseline-experts-1.md](../exploration/ineff-baseline-experts-1.md) (confirmed): the starting point's `rows >= 60` scalar `qmm_accum` prefill branch runs 15.8/27.7/20.1× slower than the compiled reference (15,220/119,154/240,681 vs 961/4,306/11,999 µs/layer) — constant ~29.5 µs/pair serial issue at 18.8 GB/s — making prompt processing 6.63–14.50× slower and costing −10.0% to −82.5% of whole-request throughput. Slot: `experts`. Baseline: `none` (starting point `muse-spark/round-2/kernels/experts/attempt-7`, which passes G1+G2 but is `slower` at 1 row and at every prefill size).

Predicted effect if the prefill path matches compiled speed at equal bits: save 14,259/114,848/228,682 µs/layer = 285 ms / 2.30 s / 4.57 s per chunk, recovering the throughput loss to whatever the decode phase then gives (decode wins at 8/32 rows must be kept: ahead 72.7/282.5 µs/layer). Fix shapes: (a) vectorized per-pair scalar (one thread per word's 8 outputs, 8 accumulators, each output keeping contiguous `BK=32` order); (b) grouped/tiled replica of `qmm_rhs` issue (takes the tile-waste and offsets prizes, ranked prefill items 7–8, for free); (c) `max_rows` hand-off (Claude checked 59 bit-exact; decode stays on the fused path). The 1-row decode gap ([ineff-baseline-experts-2.md](../exploration/ineff-baseline-experts-2.md)) is a disjoint cause and a later hypothesis (hyp-4) built on this one's winner; a prefill fix that slows decode, or vice versa, is not an improvement (ADR 0011).

## Boundary

(boundary-2 to write)

## Builders

(none started)

## Boundary checks

(none yet)

## Decision

(none yet)

## Failure ledger

(none yet)

## Unverified

(at least the boundary's *can't see* list, once written)

## Checks

(none yet)
