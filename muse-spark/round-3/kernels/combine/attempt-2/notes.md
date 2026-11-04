# Attempt 2 — builder-1-b, direction (b): FP-contract-off pragma barrier

First attempt in this direction (round 3, `hyp-1-combine-devolatilize`,
slot `combine`, baseline `none`). Builds on the starting point
`muse-spark/round-2/kernels/combine/attempt-1`, which passes G1+G2 and
`beats_compiled` but fails G3 (+1.051 ms/step in vLLM at 32 requests).

## What changed

Only the barrier mechanism. Grid (`rows*768`), threadgroup [256,1,1],
index/base lines, inputs/outputs and the final
`bfloat(acc) + shared[sbase]` line are verbatim from the starting
point. The seven `volatile float` locals (`p0..p3`, `s01`, `s012`,
`acc`) become seven plain `float` locals with the identical 7-op
sequence, and a new header block (`block-1.metal`, wired via
`"header_block": 1` — the only `spec.json` delta) disables FP
contraction for the kernel:

```metal
#pragma STDC FP_CONTRACT OFF
#pragma clang fp contract(off)
```

MLX emits `header` before the main function, so the pragmas precede the
kernel definition, which is where FP-option pragmas take effect. No
`volatile` anywhere in code, no `__attribute__((noinline))` helpers,
no extra functions, calls, or memory traffic of any kind.

Why this is contraction-proof: with contract(off) the backend may not
fuse any fmul+fadd into an fma — this is the compiler's own
contraction switch, so there is no transform left that can keep a
product unrounded into its consumer add. Each of the 7 statements is
then one rounded f32 op on register values: the same 7 materialisation
points, same fold order `((p0+p1)+p2)+p3`, same two bfloat16 roundings
as the starting point. Reassociation needs no pragma: it is off without
fast-math flags, and the grader sets none. The proof is G2 itself: any
surviving contraction shows the starting point's old signature (decode
0/2/2, prefill 6/50/130 differ).

## Inefficiency removed

`ineff-baseline-combine-1` (confirmed): the seven volatiles' 7 stores +
7 loads per output (56 B/thread of thread-private traffic, 43,008
B/row/layer) stall the decode step (+7.1/+25.4/+72.1 µs/layer vLLM)
and the 32×128 prompt (+1,501.7 ms twin). The pragmas keep every value
in registers — ~10 live floats, no spill pressure — so the barrier
costs zero bytes and zero instructions. Same fix covers both phases:
both run these identical lines.

## Prediction (µs per layer, chained paired basis)

Starting point (re-grade `grade-phases-combine-attempt-1.json`, via
`ineff-g3-1`): decode reference/candidate 52.8/45.7 @1, 71.2/47.9 @8,
87.0/64.6 @32; compiled 53.4/56.4/69.1. Prefill candidate
62.8/226.5/273.0 vs compiled 66.1/271.6/520.4.

Arithmetic:

- Slot traffic unchanged: 9,232 B/row ÷ 200 GB/s = 0.05 / 0.37 / 1.48
  µs decode, 5.9 / 46.9 / 94.5 µs prefill. Floor dominates decode.
- Spill floor removed: 43,008 B/row ÷ 200 GB/s = 0.22 / 1.7 / 6.9 µs
  decode, 27 / 218 / 440 µs prefill — an upper bound the quiet grader
  chain mostly does not price (the inefficiency entry), so decode is
  predicted at the starting point ±5% paired noise, prefill
  same-or-slightly-better.
- Predicted candidate: **45 / 47 / 63** decode, **61 / 220 / 265**
  prefill. Margins to compiled (≈8/9/6 decode, ≈5/50/255 prefill)
  reproduce the starting point's 28–30/30 wins: `faster` at all six
  sizes, both phases `faster`, `beats_compiled` true.
- G1 pass, G2 differ 0 / max ULP 0.0 at all six sizes.

The piece prize is keeping `beats_compiled`; the real prize (vLLM
≈0.5 ms/step at 8 requests, ≈1.4 ms/step at 32; order-1-s prompt
recovery) is invisible to the piece timing by construction and needs
the ready check.

## max_rows

None. The kernel runs identical lines at all sizes, combine has no
MLX kernel switch underneath, and the boundary's grader note says no
hand-off is needed for this hypothesis's direction.

## What this attempt leaves unmeasured

For the hypothesis entry's **Unverified** list: the in-step stall
recovery (only the ready check / G3 at 32 requests shows it);
exactness at row counts other than the six graded ones (boundary tool
`check_sizes.py` can probe 25 sizes incl. 59/60/61); the 33–126 band;
sizes above 2,048; prefill beyond the first seeded-random chunk;
whether the pragma pair is honoured by any toolchain other than the
graded one (irrelevant to the verdict, which runs on this one).

## Verdict

G1 pass, G2 pass at all six sizes, `beats_compiled` true. Baseline is
`none`, so there is no `beats_baseline` bar. Single grade run,
2026-11-04T18:05:21Z, verdict in `grade.json`.

- Correctness (differ / max ULP, summed over 20 layers): decode 1/8/32
  rows 0 / 0.0 each; prefill 127/1,016/2,048 rows 0 / 0.0 each.
  First difference: none. The pragma pair disables contraction: the
  plain-`float` stepwise fold is bit-exact everywhere the volatiles
  were, with no `volatile`, no helper calls, no memory traffic.
- Timing (µs per layer, reference / compiled / candidate, wins,
  verdict): rows 1: 59.1/36.9/**31.8** (24, faster); rows 8:
  45.1/41.1/**34.7** (27, faster); rows 32: 62.9/45.7/**32.7** (30,
  faster); prefill 127: 85.1/64.0/**38.2** (30, faster); prefill 1,016:
  419.6/281.8/**100.9** (30, faster); prefill 2,048: 828.9/556.2/
  **198.0** (29, faster). Same-run margins to compiled: −5.1/−6.4/−13.0
  decode, −25.8/−180.9/−358.2 prefill. `phases_vs_compiled`: decode
  faster, prefill faster; `beats_compiled` true.

### Why the timing missed the prediction (>20%)

Predicted decode 45/47/63 and prefill 61/220/265 from the re-grade's
absolute medians; measured 31.8/34.7/32.7 and 38.2/100.9/198.0 —
−25 to −55%. The miss is the anchor, not the kernel: chained absolute
µs drift run to run with machine state. In this same run the
*unchanged* compiled reference timed 36.9/41.1/45.7 decode vs
53.4/56.4/69.1 in the re-grade and 55.5/56.8/72.8 in builder-1-a's
run minutes earlier — a ~35% swing on identical code — while the
eager reference swung the other way at rows 1 (59.1 here vs 52.8
re-grade). Absolute chained medians therefore do not transfer across
runs; the valid evidence is the same-run paired verdict, which is
`faster` at all six sizes (24–30/30 wins) with both phases `faster`.
Corroborating cross-check on the same-run basis: builder-1-a's
noinline attempt (different run, hotter machine) shows the same shape
— `faster` everywhere with large prefill margins (−23.2/−173.7/−402.0
vs mine −25.8/−180.9/−358.2) — so both devolatilizations agree the
piece timing sees part of the spill prize at prefill sizes. Lesson:
predict combine piece times as same-run margins to compiled, never as
cross-run absolutes. (Noted in `ineff-baseline-combine-1.md`.)

Next change if hardening is wanted: none for this slot on the piece
basis — exact everywhere and `beats_compiled` with margin. The open
question is the vLLM step/prompt recovery, which only the ready check
can show; if it under-delivers, the next probe is whether the pragma
leaves any backend transform the volatiles also blocked (e.g. check
the compiled binary for fma), not another barrier shape.
