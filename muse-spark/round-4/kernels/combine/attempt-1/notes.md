# Attempt 1 — builder-3-a, direction (a): byte-identical copy

Round 4, `hyp-3-combine-setcopy`, slot `combine`, baseline `none`.
Builds on the starting point `muse-spark/round-3/kernels/combine/attempt-2`
(pragma FP_CONTRACT OFF devolatilized kernel, no `max_rows`; G2 exact,
`faster` at all six sizes, `beats_compiled` true in both phases,
`muse-spark/round-3/kernels/combine/attempt-2/grade.json`).

## What changed

No change at all since the starting point. `block-0.metal`,
`block-1.metal` and `spec.json` are copied byte-identically into this
folder and verified with `diff` (both blocks and `spec.json` identical,
including the `#pragma STDC FP_CONTRACT OFF` + `#pragma clang fp
contract(off)` header block at `block-1.metal:23-24` emitted via
`"header_block": 1`). No kernel line, no `spec.json` key is added,
removed or reordered: grid `rows*768`, threadgroup [256,1,1], template
`{}`, inputs `y, weights, shared`, output `out` [`rows`,768] bfloat16
all verbatim. This is the only direction the boundary allows for
`hyp-3-combine-setcopy` (*Boundary, What must stay identical — the copy
may change nothing*).

## Inefficiency addressed

None inside the slot. The starting point sits on the chained dispatch
floor (~32 µs/layer at decode, `muse-spark/round-4/exploration/ineff-
baseline-combine-1.md`, confirmed): bytes+compute are ≤1.5 µs of it at
decode, and exploration confirms no bit-exact change widens its decode
margin (`ineff-baseline-combine-2`, both confirmed with achievable
saving 0 / below paired noise). What the round needs from this slot is
a round-4 attempt the ready check can time as part of the set
`muse-all`: the pair's measured whole-request gains exist only with
combine replaced, and round-3's attempt-2 is another round's folder.
This attempt exists so the ready check can time this slot in this
round's set; any margin claim for the set comes from the ready check,
not from this grade.

## Prediction (µs per layer, chained paired basis)

Prediction is the starting point's own grade, byte-identical code on
the same grader. Starting point (`muse-spark/round-3/kernels/combine/
attempt-2/grade.json`, reference / compiled / candidate, wins,
verdict): rows 1: 59.1/36.9/31.8 (24, faster); rows 8: 45.1/41.1/34.7
(27, faster); rows 32: 62.9/45.7/32.7 (30, faster); prefill 127:
85.1/64.0/38.2 (30, faster); prefill 1,016: 419.6/281.8/100.9 (30,
faster); prefill 2,048: 828.9/556.2/198.0 (29, faster);
`phases_vs_compiled` decode faster, prefill faster; `beats_compiled`
true. G1 pass, G2 differ 0 / max ULP 0.0 at all six sizes.

Arithmetic for the level: slot traffic is `y` 4×768 bf16 (6,144 B) +
`weights` 4×f32 (16 B) + `shared` 768 bf16 (1,536 B) + `out` 768 bf16
(1,536 B) = 9,232 B/row. At 200 GB/s peak that is 0.05 / 0.37 / 1.48
µs at decode 1/8/32 rows and 5.9 / 46.9 / 94.5 µs at prefill
127/1,016/2,048 rows; compute 7×768×rows FLOP at 5.2 TFLOPS adds
~0.001 / 0.008 / 0.03 µs decode and ~0.13 / 1.05 / 2.12 µs prefill.
Measured (31.8 / 34.7 / 32.7 decode, 38.2 / 100.9 / 198.0 prefill) sits
far above bytes+compute at decode — the remainder is the shared
chained dispatch floor, identical for candidate and compiled bar and
cancelling out of the margin.

Predicted candidate for this attempt: **31.8 / 34.7 / 32.7** decode,
**38.2 / 100.9 / 198.0** prefill, same-run margins to compiled
−5.1 / −6.4 / −13.0 decode and −25.8 / −180.9 / −358.2 prefill,
`faster` at all six sizes, both phases `faster`, `beats_compiled`
true, G1 pass, G2 differ 0 / max ULP 0.0 at all six sizes. Absolute
chained µs drift ~35% run to run on identical code (round-3
attempt-2 notes), so a >20% absolute miss against this anchor is
expected drift, not a kernel effect; the evidence is the same-run
paired verdict.

## max_rows

None. Neither `max_rows` nor `min_rows` appears in `spec.json`, so
`hands_off(rows)` is false at every row count
(`bench/grader/slots.py:338-339`): every decode step (1–32 rows) and
every prefill chunk (127/1,016/2,048 graded, and ungraded 33–126,
59/60/61 and above-2,048 counts alike) runs these identical lines.
Combine has no MLX kernel switch underneath, so there is no 60-row
side to place.

## What this attempt leaves unmeasured

For the hypothesis entry's **Unverified** list: row counts other than
the six graded ones, all kernel-run here (no hand-off) — decode 2–7
and 9–31, the 33–126 serving band, both sides of 60, sizes above
2,048, and prefill beyond the first seeded-random chunk; byte-identity
itself (G2 also passes for any other exact kernel — only the `diff`
above proves the copy); toolchain dependence of the `FP_CONTRACT OFF`
pragmas (exactness holds only where the graded toolchain honours
them); threadgroup memory, still unchecked by `slots.build`;
unwritten threadgroup reads and thread-timing dependence, which one
equality run per layer cannot expose; whole-request noise at 1
request; and the three-slot interaction, which only the ready check's
`muse-all` measures. The piece timing cannot distinguish the copy
from its starting point beyond paired noise — identical code graded
twice differs only by machine state — so this grade proves exactness
and the compiled bar, nothing about the set.

## Verdict

G1 pass, G2 pass at all six sizes, `beats_compiled` true. Baseline is
`none`, so there is no `beats_baseline` bar. Single grade run,
2026-11-04T22:08:02Z, verdict in `grade.json`.

- Correctness (differ / max ULP, summed over 20 layers): decode 1/8/32
  rows 0 / 0.0 each; prefill 127/1,016/2,048 rows 0 / 0.0 each.
  First difference: none. The byte-identical copy preserves the
  pragma barrier exactly, so the plain-`float` stepwise fold is
  bit-exact everywhere the starting point was.
- Timing (µs per layer, reference / compiled / candidate, wins,
  verdict): rows 1: 43.8/47.1/**39.3** (25, faster); rows 8:
  73.7/60.5/**45.5** (25, faster); rows 32: 89.5/69.6/**43.2** (30,
  faster); prefill 127: 91.0/66.9/**43.9** (29, faster); prefill 1,016:
  425.8/284.1/**95.1** (30, faster); prefill 2,048: 924.1/567.2/
  **178.8** (29, faster). Same-run margins to compiled: −7.8/−15.0/
  −26.4 decode, −23.0/−189.0/−388.4 prefill. `phases_vs_compiled`:
  decode faster, prefill faster; `beats_compiled` true.

### Why the absolute timing missed the prediction (>20% at decode)

Predicted 31.8 / 34.7 / 32.7 decode and 38.2 / 100.9 / 198.0 prefill
from the starting point's grade; measured 39.3 / 45.5 / 43.2 decode
(+24% / +31% / +32%) and 43.9 / 95.1 / 178.8 prefill (+15% / −6% /
−10%). The decode miss is anchor drift, not a kernel effect: the copy
is byte-identical (`diff` clean), so no line changed. In this same
run the *unchanged* compiled reference timed 47.1 / 60.5 / 69.6 decode
vs 36.9 / 41.1 / 45.7 in the starting point's run — a +28% / +47% /
+52% swing on identical code — and the eager reference swung the
other way at rows 1 (43.8 here vs 59.1 starting point). Absolute
chained medians therefore do not transfer across runs; the valid
evidence is the same-run paired verdict, which is `faster` at all six
sizes (25–30/30 wins) with both phases `faster`, and same-run margins
that are *larger* than the starting point's (−7.8/−15.0/−26.4 vs
−5.1/−6.4/−13.0 decode; −23.0/−189.0/−388.4 vs −25.8/−180.9/−358.2
prefill). The copy reproduces the starting point's compiled bar
exactly as the boundary requires; only the ready check can say
whether carrying combine in the set helps `muse-all`.

Next change if hardening is wanted: none inside this slot — the
boundary forbids any kernel or spec edit and the copy is exact with
`beats_compiled` true in both phases. The open question is the
three-slot set, which only the ready check measures.
