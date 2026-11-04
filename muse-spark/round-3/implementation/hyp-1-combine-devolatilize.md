# Hypothesis 1: combine without volatiles

- **Owner:** lead
- **Status:** building
- **Version:** 3
- **Updated:** 2026-11-04T17:58:28Z
- **Next:** boundary-1 checks attempt-1 (G2 pass); builder-1-b still building attempt-2

## Inefficiency

[ineff-baseline-combine-1.md](../exploration/ineff-baseline-combine-1.md) (confirmed): the starting point's seven `volatile float` rounding barriers (`p0..p3`, `s01`, `s012`, `acc`) cost +0.142/+0.508/+1.441 ms per decode step in vLLM (+7.1/+25.4/+72.1 µs/layer, ±0.1 ms control) while the grader prices them near zero. Measurement side: [ineff-g3-1.md](../exploration/ineff-g3-1.md) (confirmed). The same barriers drive the +1,501.7 ms prompt twin on 32×128 (ranked prefill item 6), so one fix covers both phases. Slot: `combine`. Baseline: `none` (starting point `muse-spark/round-2/kernels/combine/attempt-1`, which passes G1+G2 and `beats_compiled` but fails G3).

Predicted effect if the volatiles go with exact per-op rounding kept: return to ≈ −0.51/−0.71/−0.39 ms/step (the volatiles-free twin's medians), i.e. recover ≈0.5 ms/step at 8 requests and ≈1.4 ms/step at 32 (≈0 claimed at 1 row, within noise), net vs compiled −0.2/−0.4/−0.1 ms/step (verifier-corrected), plus prompt recovery of order 1 s on combine-including 32×128 prompts. Suggested direction: `__attribute__((noinline))` device-function barriers for the mul and the adds (values in registers, no memory traffic; helpers in `header_block`).

## Boundary

*written by boundary-1, 2026-11-04T17:54:29Z — delta to round 2's boundary. Only what changed is written here; every round-2 line not listed below still holds verbatim.*

Round 2's boundary is [hyp-3-combine-exact.md](../../round-2/implementation/hyp-3-combine-exact.md) *Boundary* (`muse-spark/round-2/implementation/hyp-3-combine-exact.md:18-64`, itself a delta to round 1's `muse-spark/implementation/hyp-3-fused-combine.md:19-62`: inputs/outputs table, before/after in `MoE.__call__`, identical-math, grader, GPU limits). I checked each of its lines against the starting point `muse-spark/round-2/kernels/combine/attempt-1` (`spec.json`; `block-0.metal:25-41`) and against the current two-phase G2 (`claude/muse-agent-brief.md:182-194`; `claude/adr/0011-decode-and-prefill.md`; `bench/grader/check_candidate.py:88-99,139-150,308-378`; `bench/grader/slots.py:51-52,304-337`). Still holding, not rewritten below: input/output names, shapes, dtypes and layouts at all six sizes; the slot reference `(y * weights[..., None]).sum(axis=-2).astype(bfloat16) + shared` (`bench/grader/slots.py:51-52`, byte-unchanged); neighbour contracts (positional `weights[r,j]` ↔ `y[r,j,:]` pairing, shared expert outside the slots, residual-add consumer — `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232,258` and `bench/grader/slots.py:97-119` unchanged); the exact 5-step evaluation sequence, the left fold `((p₀+p₁)+p₂)+p₃`, both bfloat16 rounding points, no-FMA/no-reassociation rule, and no `rows`-dependent kernel switch for combine; the tie rule belonging to routing only with strict positional pairing here; G1/G2 mechanics (six sizes, `differ == 0` every element in all 20 layers, shape/dtype faults fail G2 with G1 true, NaN-vs-NaN equal, `routing+combine` parts checked in isolation on captured reference inputs); GPU limits (1,024 threads/threadgroup, 32 KB threadgroup memory). Baseline is `none` — no `beats_baseline` bar this round either (`muse-spark/round-3/exploration/territory-grader.md` §1; round-3 README *Starting point*). Only what changed follows.

### Inputs and outputs (unchanged)

No change. The starting point's launch surface is byte-identical to round 1's (`spec.json` diff empty; `muse-spark/round-3/exploration/territory-baseline-combine.md` checklist A): grid `rows*768`, threadgroup [256,1,1], inputs `y, weights, shared`, output `out` [rows,768] bfloat16. Round-2 table (`hyp-3-combine-exact.md:28-33`) holds verbatim at decode 1/8/32 and prefill 127/1,016/2,048 rows.

### Before and after (unchanged)

No change. Same `MoE.__call__` neighbours with the same expectations; checklist delta per `.agents/skills/muse-round/tracking.md` *The checklist*: producer `SwitchGLU` — holds; producer routing-weights chain — holds; producer shared-expert MLP — holds; consumer residual add — holds, still expects `out` bit-identical to the reference. The fix changes kernel-internal barrier mechanics only; no neighbour-visible contract moves.

### What must stay identical — still bit for bit; the starting point now passes it (changed)

- **The arithmetic contract is unchanged.** The reference chain, the 5-step sequence (each product rounded once to float32; fold `((p₀+p₁)+p₂)+p₃` with a rounding after every add; one bfloat16 round; single-rounding bfloat16 `+ shared`), the FMA-contraction ban, and the no-60-row-switch statement all hold verbatim — `ref_combine` is unchanged, so the required evaluation is identical to round 2's.
- **Starting point against this boundary (changed).** Round 2's starting point failed this boundary on contraction (0/2/2 differ decode, 6/50/130 prefill). The round-3 starting point `muse-spark/round-2/kernels/combine/attempt-1` **passes** it: G1 pass, G2 pass with differ 0 / max ULP 0.0 at all six sizes (`muse-spark/round-2/implementation/hyp-3-combine-exact.md:79-83`; re-graded `results/runs/2026-11-04-muse-round-2/grade-phases-combine-attempt-1.json`). Its seven `volatile float` barriers (`p0..p3`, `s01`, `s012`, `acc`, `block-0.metal:33-39`) materialise exactly the required roundings — each written once and read once in program order (`hyp-3-combine-exact.md:136-137`). What the fix may change is therefore **only the barrier mechanism** (e.g. the hypothesised `__attribute__((noinline))` device-function barriers, expressible via `header_block`, `bench/grader/slots.py:313-322`; thread/layout decomposition stays free as in round 2). It may not change any rounding: same 7 float32 materialisations, same fold order, same two bfloat16 roundings. The failure it must avoid is G2 (exactness), and the failure it must cure is G3: the starting point beats compiled at all six piece sizes yet costs +1.051 ms/step in vLLM at 32 requests (`muse-spark/round-3/exploration/ineff-g3-1.md:15`; mechanism in `ineff-baseline-combine-1.md:15-36`).
- **`weights` still arbitrary finite float32** (round-2 line holds verbatim): no non-negativity or sum-to-2.5 assumption, no re-sorting/re-normalising, positional pairing only.

### What the grader checks, and what it can't see (changed)

- **G1/G2 (unchanged).** Same six sizes, same `differ == 0` rule, same per-part isolation for `routing+combine` (`bench/grader/check_candidate.py:308-341`). New citation, same semantics: size labels now carry the phase prefix (`rows N` vs `prefill rows N`, `bench/grader/check_candidate.py:51-52,313-320`).
- **Piece timing (changed to two-phase paired).** Timing still runs only if G2 passes, but now at all six sizes with paired per-size verdicts (`bench/grader/check_candidate.py:345-378`; `claude/muse-agent-brief.md:206-224`; `claude/adr/0011-decode-and-prefill.md`; `muse-spark/round-3/exploration/territory-grader.md` §1): per size, candidate vs `mx.compile(ref_combine)` over 30 chained repeats, `faster` with ≥22 wins, `slower` with ≥22 losses, else `same`; per phase, `faster` iff every size is, `slower` iff any size is, else `not slower`; `beats_compiled` iff one phase is `faster` and neither is `slower`. Round 2's "candidate median < compiled median at all three decode sizes" bar is retired. Prefill speed is now timed — a fix slower at prefill sizes fails `beats_compiled` instead of being invisible. The starting point beats compiled at all six sizes under both rules (decode 45.7/47.9/64.6 vs 53.4/56.4/69.1, prefill 62.8/226.5/273.0 vs 66.1/271.6/520.4 µs/layer; `ineff-g3-1.md:19,46`), so the fix must keep all six `same`-or-better with at least one phase all-`faster`. The slot stops at the first attempt passing G1+G2 and the two-phase bar, or after 5 attempts (`claude/muse-agent-brief.md:217`; round 2 cited :201, moved).
- **`max_rows` hand-off (new).** With `"max_rows"` in the spec, inputs with `rows > max_rows` run `mx.compile(ref_combine)` instead of the kernels, graded bit-exact and timed at the prefill sizes like the rest (`bench/grader/slots.py:304-337`; `claude/muse-agent-brief.md:174`; `muse-spark/round-3/exploration/territory-grader.md` §2). Combine has no kernel switch underneath, so any `max_rows` ≥ 32 keeps decode on the kernels and hands every prefill size to compiled (grader note for this slot, `territory-grader.md:86`). This hypothesis's direction runs identical lines at all sizes, so no hand-off is needed — but if a builder adds one, the boundary check verifies where the kernels stop running in real serving (33–126-row chunks), not just the six graded sizes.
- **G3 and the ready check (new in-round).** G3 is now vLLM-only two-phase judgment — decode step, prompt time, whole-request time over 1×128/8×128/32×128/32×1024, 5 repeats, pass iff decode or prompt is `faster` and none of the three is `slower`, against both base and compiled (`bench/grader/time_candidates.py:163-201`; `claude/muse-agent-brief.md:195-204`; `claude/adr/0011-decode-and-prefill.md`). The lead's ready check (`bench/grader/check_ready.py`, `claude/adr/0013-measure-what-you-claim.md`) runs that same G3 inside the round with every candidate in one call, plus same-greedy-tokens; only its passing verdict makes an attempt `ready`, and Claude's post-round G3 stays the authority (`muse-spark/round-3/exploration/territory-grader.md` §§4–5). For this hypothesis the discriminating check is the ready check timing the fix combine-alone at 32 requests (`ineff-baseline-combine-1.md:123-126`); end-to-end greedy tokens (24 tokens at 1 and 32 requests) are part of it.
- **What it can't see** (checked by hand later — *check an attempt*; seeds this entry's **Unverified** list): row counts other than the six graded ones and, new, the `max_rows` band 33–126 where kernels run in real serving but no piece verdict exists (`territory-grader.md`, *can't see* 1–2); sizes above 2,048 and prefill beyond the first seeded-random chunk (items 3–4); hand-off exactness above `max_rows` at untested sizes (item 5); whole-request noise at 1 request (item 6); threadgroup memory still unchecked by `slots.build` (item 8, now `bench/grader/slots.py:341-342`); link-kernel caveats carried over (item 9); plain-MLX behaviour no longer gated (item 10). **Load-bearing for this hypothesis:** the chained piece timing cannot see context-dependent stall costs by construction — it prices the starting point's seven volatiles near zero (beats compiled everywhere) while the step pays +72.1 µs/layer at 32 rows (`ineff-baseline-combine-1.md:72-80`; note in `territory-grader.md:108`). A devolatilized fix therefore proves nothing about its prize in the piece timing; only the ready check / G3 at 32 requests shows the recovery. New since round 2: `bench/grader/check_sizes.py` (25 sizes incl. 59/60/61) closes between-sizes exactness where the boundary agent runs it — boundary tooling, not a gate (`territory-grader.md` §3). Boundary checks will probe exactly these.

### The GPU limits (unchanged)

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory (`claude/muse-agent-brief.md:100`); `grid` counts threads (`claude/muse-agent-brief.md:178`); Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` in Metal (brief:101). Starting point for scale: 256 threads/threadgroup, no threadgroup memory (`spec.json`; `block-0.metal:25-41`), valid at every row count including 2,048 rows. Any replacement must stay inside both limits at all six sizes. Enforcement gap stands: `slots.build` checks only the 1,024-thread product (`bench/grader/slots.py:341-342`); Metal rejects oversize kernels as opaque G1 exceptions.

## Builders

### builder-1-a — direction (a): noinline device-function barriers

Helpers `combine_mul` / `combine_add` (`__attribute__((noinline))`, one op each, in `header_block`) replace the seven `volatile float` barriers; calls in the body keep the exact fold `((p0+p1)+p2)+p3`, both bfloat16 roundings, grid and indexing. Values pass in registers, no thread-private memory traffic.

- **attempt-1** (`muse-spark/round-3/kernels/combine/attempt-1`): G1 pass; G2 pass, differ 0 / max ULP 0.0 at all six sizes, first difference none. Predicted ≈ starting-point medians ±5% (45.7/47.9/64.6 decode, 62.8/226.5/273.0 prefill, `results/runs/2026-11-04-muse-round-2/grade-phases-combine-attempt-1.json`); measured (`grade.json`, µs/layer, reference / compiled / candidate, wins, verdict) — rows 1: 59.5/55.5/**46.8** (28, faster); rows 8: 69.6/56.8/**42.0** (28, faster); rows 32: 83.3/72.8/**49.2** (30, faster); prefill 127: 93.5/67.4/**44.2** (29, faster); prefill 1016: 437.2/278.3/**104.6** (30, faster); prefill 2048: 861.8/590.0/**188.0** (30, faster). `phases_vs_compiled` decode faster, prefill faster; `beats_compiled` true; no baseline. Taught: the piece timing sees part of the prize at prefill sizes — the volatiles' 56 B/output spill (5.5/43.7/88 MB per layer call, floors 33/265/535 µs) is material there, and the attempt recovers −18.6/−121.9/−85.0 µs of it; "~0 in the grader" holds at decode only (see `notes.md` and the note in `ineff-baseline-combine-1.md`). The vLLM step/prompt recovery is for the ready check.

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
