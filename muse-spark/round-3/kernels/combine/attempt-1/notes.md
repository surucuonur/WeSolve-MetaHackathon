# Attempt 1 — builder-1-a, direction (a): noinline device-function barriers

First attempt in this direction (round 3, `hyp-1-combine-devolatilize`,
slot `combine`, baseline `none`). Builds on the starting point
`muse-spark/round-2/kernels/combine/attempt-1` (G1+G2 pass,
`beats_compiled` true, fails G3 on volatile-stall cost).

## What changed

Only the barrier mechanism. Indexing, grid (`rows*768`), threadgroup
[256,1,1], inputs/outputs, the fold order, both bfloat16 roundings, and
the final line `bfloat(acc) + shared[sbase]` are verbatim from the
starting point. The seven `volatile float` barriers:

```metal
volatile float p0 = float(y[ybase]) * weights[wbase];
...
volatile float s01 = p0 + p1;
volatile float s012 = s01 + p2;
volatile float acc = s012 + p3;
```

become seven calls through two helpers living in `header_block`
(`block-1.metal`, wired via `"header_block": 1` in `spec.json`):

```metal
__attribute__((noinline)) float combine_mul(float a, float b) { return a * b; }
__attribute__((noinline)) float combine_add(float a, float b) { return a + b; }
...
float p0 = combine_mul(float(y[ybase]), weights[wbase]);
...
float s01 = combine_add(p0, p1);
float s012 = combine_add(s01, p2);
float acc = combine_add(s012, p3);
```

Each mul/add executes in its own non-inlined call, so its result rounds
once to f32 at the return and no expression tree spans the call
boundary for the compiler to FP-contract. Values cross in registers:
no thread-private memory traffic. Same 7 f32 materialisations, same
left fold `((p0+p1)+p2)+p3`, same two bfloat16 roundings — the only
change the Boundary permits.

## Inefficiency removed

`ineff-baseline-combine-1` (confirmed): the seven volatiles cost
+0.142/+0.508/+1.441 ms per decode step in vLLM (+7.1/+25.4/+72.1
µs/layer) plus the +1,501.7 ms prompt twin on 32×128, while the chained
piece timing prices them near zero. This attempt keeps the exact
per-op rounding the volatiles bought and removes the memory traffic
that caused the context-dependent stall.

## Prediction (µs per layer, chained piece timing)

The piece timing cannot see the prize (Boundary *can't see*,
load-bearing): the starting point already beats compiled at all six
sizes there. Basis: the starting point's re-grade medians
(`results/runs/2026-11-04-muse-round-2/grade-phases-combine-attempt-1.json`):

- decode 1 / 8 / 32 rows: 45.7 / 47.9 / 64.6 (compiled 53.4 / 56.4 / 69.1)
- prefill 127 / 1,016 / 2,048 rows: 62.8 / 226.5 / 273.0 (compiled 66.1 / 271.6 / 520.4)

Arithmetic: replacing 7 volatile spill pairs per output with 7 register
calls changes only sub-floor terms. Call overhead ≈ 7 calls × ~768
outputs/row × a few cycles, under 0.5 µs/layer against the ~45–65 µs
chained floor at decode; at prefill the kernel is traffic-shaped and
the barrier term is likewise negligible. Same-kernel-twice noise is up
to 5% (brief). So predicted candidate ≈ starting point medians ±5%:
**≈45.7 / 47.9 / 64.6 decode, ≈62.8 / 226.5 / 273.0 prefill**,
each size `faster` or `same` vs compiled, `beats_compiled` true, G2
differ 0 at all six sizes. The real prize — recovery to ≈
−0.51/−0.71/−0.39 ms/step (net vs compiled −0.2/−0.4/−0.1 ms/step at
1/8/32 requests) plus ~1 s prompt recovery — shows only in the ready
check / G3 at 32 requests, not here.

No `max_rows`: identical lines run at all sizes, and combine has no
60-row kernel switch underneath (Boundary), so no hand-off is needed.

## Unverified (for the hypothesis entry)

- The prize itself: vLLM step/prompt recovery at 32 requests — only the
  ready check / G3 measures it.
- Row counts other than the six graded ones (incl. the 33–126 band),
  sizes above 2,048, prefill beyond the first chunk — the Boundary's
  *can't see* list.
- That `noinline` survives every Metal compiler version/flag path the
  grader and vLLM use (G2 at six sizes is the check run; other sizes
  rely on `check_sizes.py` if the boundary agent runs it).

## Verdict

G1 pass, G2 pass at all six sizes, `beats_compiled` true. Baseline is
`none`, so no `beats_baseline` bar. Single grader run
(`check_candidate.py ... combine --out .../grade.json`, waited for the
GPU behind the experts attempt-1 run, then completed in one go);
verdict kept in `grade.json`.

- Correctness (differ / max ULP, summed over 20 layers): decode 1/8/32
  rows 0/0.0, 0/0.0, 0/0.0; prefill 127/1,016/2,048 rows 0/0.0, 0/0.0,
  0/0.0. First difference: none. The noinline barriers keep every
  rounding the volatiles bought.
- Chained timing, µs per layer (reference / compiled / candidate,
  wins, verdict): rows 1: 59.5 / 55.5 / **46.8** (28, faster); rows 8:
  69.6 / 56.8 / **42.0** (28, faster); rows 32: 83.3 / 72.8 / **49.2**
  (30, faster); prefill 127: 93.5 / 67.4 / **44.2** (29, faster);
  prefill 1,016: 437.2 / 278.3 / **104.6** (30, faster); prefill 2,048:
  861.8 / 590.0 / **188.0** (30, faster).
- `phases_vs_compiled`: decode faster, prefill faster;
  `beats_compiled` true.
- Against the starting point's re-grade medians (45.7/47.9/64.6
  decode, 62.8/226.5/273.0 prefill): +2.4% / −12% / −24% decode,
  −30% / −54% / −31% prefill.

### Why the timing beat the prediction (>20% at four sizes)

Predicted ≈ starting-point medians ±5% (volatiles "priced near zero"
in the chained timing); measured −24% at 32 rows and −30/−54/−31% at
the three prefill sizes. The miss is in the prediction's extension of
the near-zero claim from decode to prefill, and it favours the
attempt. Per the verifier's arithmetic
(`ineff-baseline-combine-1.md` Verdict), the seven volatiles are 7
stores + 7 loads = 56 B per output, i.e. spill traffic of 43 KB / 344
KB / 1.375 MB per layer call at decode 1/8/32 rows but 5.5 / 43.7 /
88 MB at prefill 127/1,016/2,048 rows, with bandwidth floors 0.26 /
2.1 / 8.4 µs decode vs 33 / 265 / 535 µs prefill. At decode the floor
is indeed ~0 against the ~45 µs chained floor (1-row matched at
+2.4%), but at prefill the spill floor is material, and this run
recovers part of it: −18.6 / −121.9 / −85.0 µs, each inside its floor
(caches absorb the rest). Causal, not machine state: between the two
runs the reference and compiled controls moved slightly *against* the
candidate (e.g. at 1,016 rows reference 425.2→437.2, compiled
271.6→278.3) while the candidate moved −54%. Lesson: the piece timing
sees part of the devolatilization prize at prefill sizes; the "~0 in
the grader" half of the inefficiency title holds at decode only (note
appended to `ineff-baseline-combine-1.md`). The vLLM step/prompt
recovery itself remains for the ready check to measure.

Next change if hardening is wanted: none for this direction — the
kernel is exact everywhere, beats compiled in both phases, and the
remaining prize is in serving (G3), not in the kernel body.
