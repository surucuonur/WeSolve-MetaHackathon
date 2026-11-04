# Inefficiency: baseline-combine-1 — FMA contraction breaks bit-exactness

- **Owner:** explorer-baseline-combine
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T03:17:47Z
- **Next:** verifier to confirm; builder fix is contraction suppression, not reordering

## Slot

`combine`.

## What happens

The starting point (`muse-spark/kernels/combine/attempt-1/block-0.metal:20-25`)
writes the 4-term f32 sum as one expression,
`float(y0)·w0 + float(y1)·w1 + float(y2)·w2 + float(y3)·w3`, then
`bfloat(acc) + shared`. Compiled as a single Metal kernel, the compiler
FP-contracts the mul+add pairs into FMAs, keeping each product unrounded
into its add. The reference can never do this: `ref_combine`
(`bench/grader/slots.py:51-52`) runs eagerly as separate kernels —
`binary_g` multiply (`Multiply` is plain `x * y`, product rounded once
to f32), then `col_reduce_small` (`Sum` is plain `a + b`, init 0) folding
((p₀+p₁)+p₂)+p₃ with a rounding after every add, then `copy_v` bf16
round, then `binary_vv` bf16 add (headers under
`.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/`;
full mapping in `territory-baseline-combine.md` checklist B–D). The
candidate's `acc` therefore differs from the reference fold by ~1 f32
ULP on some elements; the bf16 rounding absorbs it except when `acc`
sits at a rounding midpoint, and the following `+ shared` can magnify
the reported gap when it cancels toward zero (ULP floored at
`|r| = 0.001`, `bench/grader/check_candidate.py:84-99`).

This is contraction, not association: the candidate's written order is
already the reference's left fold, and round 1's CPU emulation showed
all 120 pure association trees give ≤1.0 ULP at 8 rows with zero matches
to the grader fingerprint while the FMA-chain evaluation reproduces it
exactly (`muse-spark/implementation/hyp-3-fused-combine.md`, *The
2.0-ULP touch*). Attempt-2's explicit `acc += …` loop has the identical
fingerprint, so neither loop form nor staging avoids contraction — the
fix must suppress it (the `precise` accumulator round 1's boundary
named there) while keeping the ((p₀+p₁)+p₂)+p₃ fold and both bf16
rounding points.

Under round 2's bit-exact G2 (`claude/adr/0010-exact-at-every-size.md`)
the kernel fails at five of six sizes
(`results/runs/2026-11-04-muse-round-1/grade-exact-combine.json`, summed
over 20 layers): decode 1 row 0 differ (pass), 8 rows **2** (max
2.0 ULP), 32 rows **2** (max 1.0 ULP); prefill 127 rows **6** (1.0),
1,016 rows **50** (16.0), 2,048 rows **130** (32.0). The count grows at
a roughly constant 3–16 ppm of rows·768·20 elements — the same lottery
sampled more times, not a prefill kernel switch (combine's kernels are
row-count-invariant; the 60-row experts switch cannot enter through
captured `y`). In the model, these differences changed 7 of 32 vLLM
sequences from token 0 (`claude/findings/muse-round-1.md` §2), which is
why the new G2 admits none.

## Cost

This inefficiency costs correctness, not time — FMA contraction is the
faster evaluation. Its cost is the slot's entire bankable saving: while
G2 fails, the grader never times the attempt
(`check_candidate.py:308-337`) and the kernel cannot be used at any
size. The forfeited saving against the compiled reference (old
independent piece timing, the only same-run basis available; Claude
re-grade, `claude/findings/muse-round-1.md` §1):

| Rows | Compiled ref | Candidate | Gap ×20 layers |
|---|---|---|---|
| 1 | 23.8 µs/layer | 18.6 µs/layer | 5.2 × 20 = **0.104 ms/step** |
| 8 | 26.1 µs/layer | 17.0 µs/layer | 9.1 × 20 = **0.182 ms/step** |
| 32 | 32.0 µs/layer | 18.9 µs/layer | 13.1 × 20 = **0.262 ms/step** |

End to end, combine attempt-1 alone saved 0.51 / 0.71 / 0.39 ms in vLLM
against the compiled reference's own 0.29 / 0.26 / 0.27 ms
(`muse-round-1.md` §3) — real headroom that stays unclaimable until
every element matches.

## Kind

Overhead-bound slot (see `ineff-baseline-combine-2`); this entry is the
correctness gate on that saving, in the terms of `claude/GLOSSARY.md`
an exactness blocker rather than a bandwidth/compute loss.

## Spans

`baseline-combine` (candidate arithmetic), `grader` (bit-exact G2 at
six sizes). Explicitly not `baseline-experts` or `baseline-routing`:
the combine part is graded on captured `y` and reference `weights`.

## Predicted saving

Removing it — suppress contraction, keep the fold and both roundings,
kernel otherwise unchanged — is predicted to cost <0.5 µs/layer (the
kernel is dispatch-bound at ~0.05–1.5 µs of traffic and ~0.03 µs of
compute per layer; losing FMA fusion cannot surface above the floor;
both round-1 organizations landed within 1 µs of each other). Net
predicted saving against the compiled reference, if the fixed kernel
passes G2 and is re-timed: **≈0.10 / 0.18 / 0.26 ms per decode step at
1 / 8 / 32 rows** on the independent piece basis above (5.2 / 9.1 /
13.1 µs/layer × 20). Caveat: round 2 ranks by the chained timing, which
has never run on this slot (it requires G2 first); the chained figure
must replace this prediction before any saving is banked, and G3
decides end to end.

## Verdict
