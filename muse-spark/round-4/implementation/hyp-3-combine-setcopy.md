# Hypothesis 3: combine carried into the set, byte-identical

- **Owner:** lead
- **Status:** hardening
- **Version:** 1
- **Updated:** 2026-11-04T22:05:00Z
- **Next:** boundary-3 writes the Boundary section

**Inefficiency:** none inside the slot is left to remove: the starting point sits on the chained floor (~32 µs/layer at decode, bytes+compute ≤1.5 µs of it) and exploration confirms no bit-exact change widens its decode margin (`../exploration/ineff-baseline-combine-1.md`, `-2`, both confirmed with achievable saving 0 / below paired noise). What the round needs from this slot is a round-4 attempt the ready check can time as part of the set: the pair's measured whole-request gains exist only with combine replaced, and round 3's attempt-2 is another round's folder.

**Slot:** `combine`. **Baseline:** none. **Starting point:** `muse-spark/round-3/kernels/combine/attempt-2` (pragma FP_CONTRACT OFF devolatilized kernel, no `max_rows`; G2 exact, `faster` at all six sizes, `beats_compiled` true in both phases).

**Direction:** (a) byte-identical copy of the starting point (both blocks and `spec.json`), graded once in this round — the round-3 attempt-6 noise-test precedent. No kernel change is attempted; any margin claim for the set comes from the ready check, not from this grade.

## Boundary

*written by boundary-3, 2026-11-04T22:05:14Z — delta to round 3's boundary. Only what changed is written here; every round-3 line not listed below still holds verbatim.*

Round 3's boundary is [hyp-1-combine-devolatilize.md](../../round-3/implementation/hyp-1-combine-devolatilize.md) *Boundary* (`muse-spark/round-3/implementation/hyp-1-combine-devolatilize.md:15-45`, itself a delta to round 2's `muse-spark/round-2/implementation/hyp-3-combine-exact.md:18-64` and round 1's `muse-spark/implementation/hyp-3-fused-combine.md:19-62`: inputs/outputs table, before/after in `MoE.__call__`, identical-math, grader, GPU limits). I checked each of its lines against the starting point `muse-spark/round-3/kernels/combine/attempt-2` (`spec.json`; `block-0.metal:26-42`; `block-1.metal:23-24`) and against the current G2 (`claude/muse-agent-brief.md:173-176,204-218`; `claude/adr/0015-sets-and-hand-off-below.md`; `bench/grader/slots.py:51-52,330-350`; `bench/grader/check_candidate.py:244-271,320-400`). Still holding, not rewritten below: input/output names, shapes, dtypes and layouts at all six sizes; the slot reference `(y * weights[..., None]).sum(axis=-2).astype(bfloat16) + shared` (`bench/grader/slots.py:51-52`, byte-unchanged); neighbour contracts (positional `weights[r,j]` ↔ `y[r,j,:]` pairing, shared expert outside the slots, residual-add consumer — `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232,258` and `bench/grader/slots.py:97-119` unchanged); the exact 5-step evaluation sequence, the left fold `((p₀+p₁)+p₂)+p₃`, both bfloat16 rounding points, no-FMA/no-reassociation rule, and no `rows`-dependent kernel switch for combine; the tie rule belonging to routing only with strict positional pairing here; G1/G2 mechanics (six sizes, `differ == 0` every element in all 20 layers, shape/dtype faults fail G2 with G1 true, NaN-vs-NaN equal, `routing+combine` parts checked in isolation on captured reference inputs); GPU limits (1,024 threads/threadgroup, 32 KB threadgroup memory). Baseline is `none` — no `beats_baseline` bar this round either (this entry, *Slot*/*Starting point*). Only what changed follows.

### Inputs and outputs (unchanged)

No change. Rechecked against the starting point: `spec.json` lists inputs `y, weights, shared`, output `out` [`rows`,768] bfloat16, grid `rows*768`, threadgroup [256,1,1], `header_block: 1` (`muse-spark/round-3/kernels/combine/attempt-2/spec.json`; `bench/grader/slots.py:31`); shapes `y` [`rows`,4,768] bfloat16, `weights` [`rows`,4] float32, `shared` [`rows`,768] bfloat16 (`block-0.metal:11-12`) at decode 1/8/32 and prefill 127/1,016/2,048 rows. Round-2 table (`hyp-3-combine-exact.md:28-33`) holds verbatim.

### Before and after (unchanged)

No change. Same `MoE.__call__` neighbours with the same expectations; checklist delta per `.agents/skills/muse-round/tracking.md` *The checklist*: producer `SwitchGLU` — holds; producer routing-weights chain — holds; producer shared-expert MLP — holds; consumer residual add — holds, still expects `out` bit-identical to the reference. The copy changes no neighbour-visible contract.

### What must stay identical — still bit for bit; the copy may change nothing (changed)

- **The arithmetic contract is unchanged.** The reference chain, the 5-step sequence (each product rounded once to float32; fold `((p₀+p₁)+p₂)+p₃` with a rounding after every add; one bfloat16 round; single-rounding bfloat16 `+ shared`), the FMA-contraction ban, and the no-60-row-switch statement all hold verbatim — `ref_combine` is unchanged, so the required evaluation is identical to round 3's.
- **Starting point against this boundary (changed).** Round 3's starting point was the volatile `muse-spark/round-2/kernels/combine/attempt-1`; round 4's starting point is round 3's fix itself, `muse-spark/round-3/kernels/combine/attempt-2`, which already passes this boundary: G1 pass, G2 pass with differ 0 / max ULP 0.0 at all six sizes, `faster` at all six sizes, `beats_compiled` true in both phases (`muse-spark/round-3/kernels/combine/attempt-2/grade.json`). Round 3 allowed changing the barrier mechanism; round 4's direction (a) allows **no kernel change at all**: both Metal blocks and `spec.json` are copied byte-identically into this round's attempt folder, graded once (this entry, *Direction*). Thread/layout decomposition is no longer free — any edit, however exact it stays, is outside this hypothesis.
- **The FP_CONTRACT OFF header block is part of the copy (changed).** Exactness now rests on `block-1.metal:23-24` (`#pragma STDC FP_CONTRACT OFF` + `#pragma clang fp contract(off)`, emitted before the kernel function via `"header_block": 1` in `spec.json`) forbidding mul+add fusion into FMAs, so the seven plain `float` locals (`p0..p3`, `s01`, `s012`, `acc`, `block-0.metal:34-40`) each round once to float32 in registers. Dropping, renaming or reordering the header block, or copying `block-0.metal` without it, re-enables contraction and breaks G2 with the round-2 signature (decode 0/2/2, prefill 6/50/130 differ); the pragma pair must survive the copy verbatim. Reassociation needs no barrier: the compiler does not reassociate without fast-math flags and the grader passes none.
- **No hand-off in this slot: the kernel runs at all sizes (changed).** The starting point's `spec.json` carries neither `max_rows` nor `min_rows`, and the copy must keep both absent. `hands_off(rows)` is therefore false at every row count (`bench/grader/slots.py:338-339`; `claude/muse-agent-brief.md:173-174`): every decode step (1–32 rows) and every prefill chunk (127/1,016/2,048 graded, and ungraded 33–126, 59/60/61 and above-2,048 counts alike) runs these identical lines. Combine has no MLX kernel switch underneath, so there is no 60-row side to place and no hand-off band to verify; adding either bound would change the claim (rows handed to the compiled reference can never be better than the bar there) and is outside direction (a). With 1 request (1 row per step) the kernels still run every step — this slot never falls back to MLX's own code.
- **`weights` still arbitrary finite float32** (round-3 line holds verbatim): no non-negativity or sum-to-2.5 assumption, no re-sorting/re-normalising, positional pairing only.

### What the grader checks, and what it can't see (changed)

- **G1/G2 (unchanged).** Same six sizes, same `differ == 0` rule, same per-part isolation for `routing+combine` (`bench/grader/check_candidate.py:320-351`); size labels carry the phase prefix (`rows N` vs `prefill rows N`).
- **Piece timing (hand-off reporting vacuous here).** Timing still runs only if G2 passes, at all six sizes with paired per-size verdicts (`bench/grader/check_candidate.py:352-400`; `claude/muse-agent-brief.md:213-218`): `faster` with ≥22 wins, `slower` with ≥22 losses, else `same`; per phase, `faster` iff every timed size is, `slower` iff any is, else `not slower`; `beats_compiled` iff one phase is `faster` and neither is `slower`. Round 4's `handed off` verdict (`bench/grader/slots.py:338-345`; `bench/grader/check_candidate.py:372-394`; `claude/muse-agent-brief.md:216`; `claude/adr/0015-sets-and-hand-off-below.md`) never triggers for this slot: with no bounds, all six sizes are timed exactly as in round 3, and the starting point's `faster` ×6 / `beats_compiled` true is the bar the copy must reproduce (up to run drift — chained absolutes move ~35% on identical code, so only same-run paired verdicts are evidence). No baseline exists, so no `beats_baseline` bar applies. The slot stops at the first attempt passing G1+G2 and `beats_compiled`, or after 5 attempts (`claude/muse-agent-brief.md:222`).
- **G3 and the ready check (set rule new).** G3 mechanics are unchanged (vLLM two-phase judgment — decode step, prompt time, whole-request time over 1×128/8×128/32×128/32×1024, 5 repeats, pass iff decode or prompt is `faster` and none of the three is `slower`, against both base and compiled, plus same greedy tokens; `bench/grader/time_candidates.py`; `claude/muse-agent-brief.md:195-206`). What round 4 changes is the reading: when `muse-all` passes the ready check, its candidates are `ready` together as a **set** even where one fails alone, and G3 judges the set as one (`claude/muse-agent-brief.md:206`; `claude/adr/0015-sets-and-hand-off-below.md:11`); the set's verdict is `ready-verdict.json`'s `variants."muse-all".pass`, not the file's top-level `pass`. Round 3's combine-alone ready failure (decode `not slower` on a 1×128 range overlap, medians −0.53/−0.59/−0.44/−0.26 ms in the candidate's favour) therefore no longer blocks this slot alone: this hypothesis exists to give the ready check a round-4 combine attempt to time inside the set (this entry, *Inefficiency*), and any whole-request claim for the copy comes from that set timing, never from its piece grade.
- **What it can't see** (checked by hand later — *check an attempt*; seeds this entry's **Unverified** list): row counts other than the six graded ones, all kernel-run here (no hand-off) — decode 2–7 and 9–31, the 33–126 serving band, both sides of 60, sizes above 2,048, and prefill beyond the first seeded-random chunk; byte-identity itself (G2 also passes for any other exact kernel — only a `diff` of both blocks and `spec.json` against the starting point, header pragmas included, proves the copy); toolchain dependence of the `FP_CONTRACT OFF` pragmas (exactness holds only where the graded toolchain honours them — re-grade after any MLX/macOS/Xcode change); threadgroup memory still unchecked by `slots.build` (item carried from round 3); unwritten threadgroup reads and thread-timing dependence, which one equality run per layer cannot expose; whole-request noise at 1 request; and the three-slot interaction, which only the ready check's `muse-all` measures. **Load-bearing for this hypothesis:** the piece timing cannot distinguish the copy from its starting point beyond paired noise — identical code graded twice differs only by machine state — so the copy's grade proves exactness and the compiled bar, nothing about the set; only the ready check shows whether carrying combine in helps the set. `bench/grader/check_sizes.py` (25 sizes incl. 59/60/61) still closes between-sizes exactness where the boundary agent runs it — boundary tooling, not a gate. Boundary checks will probe exactly these.

### The GPU limits (unchanged)

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory (`claude/muse-agent-brief.md:100`); `grid` counts threads (`claude/muse-agent-brief.md:180`); Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` in Metal (`claude/muse-agent-brief.md:101`). Starting point for scale: 256 threads/threadgroup, no threadgroup memory (`spec.json`; `block-0.metal:26-42`), valid at every row count including 2,048 rows. The byte-identical copy inherits both footprints at all six sizes and beyond. Enforcement gap stands: `slots.build` checks only the 1,024-thread product (`bench/grader/slots.py:348-350`); Metal rejects oversize kernels as opaque G1 exceptions.

## Builders

### builder-3-a — direction (a): byte-identical copy

**Direction:** copy `muse-spark/round-3/kernels/combine/attempt-2`
byte-identically into this round (both blocks and `spec.json`,
including the `FP_CONTRACT OFF` header at `block-1.metal:23-24` via
`"header_block": 1`), no kernel or spec change at all, graded once so
the ready check can time this slot in this round's set. Baseline
`none`, no `max_rows`/`min_rows` — the kernel runs at all sizes.

- **attempt-1** (`muse-spark/round-4/kernels/combine/attempt-1/`):
  `block-0.metal`, `block-1.metal` and `spec.json` verified
  byte-identical to the starting point with `diff` (all three
  identical). Predicted 31.8 / 34.7 / 32.7 decode and 38.2 / 100.9 /
  198.0 prefill µs/layer (the starting point's own grade,
  `muse-spark/round-3/kernels/combine/attempt-2/grade.json`:
  traffic 9,232 B/row = 0.05 / 0.37 / 1.48 µs decode and 5.9 / 46.9 /
  94.5 µs prefill at 200 GB/s, remainder the shared chained dispatch
  floor), `faster` at all six sizes, `beats_compiled` true.
  **Measured: G1 pass, G2 pass at all six sizes (differ 0 / max ULP
  0.0 each, first difference none); candidate 39.3 / 45.5 / 43.2 vs
  compiled 47.1 / 60.5 / 69.6 decode and 43.9 / 95.1 / 178.8 vs
  66.9 / 284.1 / 567.2 prefill µs/layer, wins 25 / 25 / 30 / 29 / 30 /
  29, `faster` at all six sizes, `phases_vs_compiled` decode faster,
  prefill faster, `beats_compiled` true** (`grade.json`, reference
  43.8 / 73.7 / 89.5 decode, 91.0 / 425.8 / 924.1 prefill in the same
  run). Baseline `none`, so no `beats_baseline`. The decode absolutes
  missed the anchor by +24% / +31% / +32% while prefill stayed within
  +15% / −6% / −10%: absolute drift across runs on identical code
  (same-run compiled bar moved +28% / +47% / +52% decode vs the
  starting point's run), judged by same-run verdicts — margins this
  run −7.8 / −15.0 / −26.4 decode and −23.0 / −189.0 / −388.4 prefill
  exceed the starting point's −5.1 / −6.4 / −13.0 and −25.8 / −180.9 /
  −358.2. Taught: the copy reproduces the starting point's
  exactness and compiled bar exactly; it removes no inefficiency
  inside the slot (the slot sits on its chained floor), and any set
  effect is measurable only by the ready check's `muse-all`, not by
  this piece grade.

## Boundary checks

### attempt-1 — pass (boundary-3, 2026-11-04T22:09:25Z)

G1+G2 pass (`muse-spark/round-4/kernels/combine/attempt-1/grade.json`: `g1` true, `g2` true, differ 0 / max ULP 0.0 at all six sizes, `beats_compiled` true, decode `faster`, prefill `faster`). Byte-identity with the starting point `muse-spark/round-3/kernels/combine/attempt-2` verified by `diff` (no output) and `cmp` on all three files, SHA-256 `block-0.metal` `ed65c9cec5b5dcbc10ef1912b692e92fff1c41bd5bf74a8bf962e7ebda8c9e83`, `block-1.metal` `d2766b2a838fe14494bda328c49edb226e4ca459b9cd53df4d26051639e83eae`, `spec.json` `313946bf026ac2422b44aa624cd2545a333701ca45105a35b57356b27859d31d` — identical in both folders, including the `FP_CONTRACT OFF` header pair at `block-1.metal:23-24` emitted via `"header_block": 1` (`spec.json`). G2 alone cannot prove this (any other exact kernel also passes); only the `diff` does.

- **Row counts other than the graded ones.** The kernel is strictly per-output with no `rows`-dependent branch, loop bound or fixed-size buffer: `idx = thread_position_in_grid.x`, `row = idx / 768u`, `d = idx - row * 768u` (`block-0.metal:26-28`), grid `rows*768` (`spec.json`), then the fixed 5-step sequence `p0..p3`, `((p0+p1)+p2)+p3`, one `bfloat` round and `+ shared` (`block-0.metal:34-42`). Decode 2–7 and 9–31, the 33–126 serving band, both sides of 60 (59/60/61) and sizes above 2,048 therefore execute the identical lines the graded 1/8/32 and 127/1,016/2,048 ran; combine has no MLX kernel switch at 60 rows, so there is no second arithmetic regime to miss. Exactness at ungraded counts rests on this rows-independent construction, not on G2; speed there is unmeasured by the piece timing — noted, not a defect.
- **No hand-off: kernel runs at all sizes.** `spec.json` carries neither `min_rows` nor `max_rows` at top level or inside `kernels[0]` (keys are only `atomic_outputs, grid, header_block, init_value, inputs, name, outputs, source_block, template, threadgroup`), so `hands_off(rows)` is false at every count (`bench/grader/slots.py:338-339`). Every decode step 1–32 and every prefill chunk runs these lines; with 1 request (1 row per step) the kernel still runs every step — this slot never falls back to MLX's own code, and there is no 60-row side to place.
- **Threadgroup memory: none, far from the limit.** Neither block declares threadgroup storage (no `threadgroup` qualifier, array or allocation in `block-0.metal` or `block-1.metal`); use is 0 B against the 32 KB ceiling. Threadgroup is [256,1,1] = 256 threads (`spec.json`) against the 1,024-thread limit. `slots.build` checks only the thread product (`bench/grader/slots.py:348-350`); the 0 B footprint was counted by hand here.
- **Unwritten threadgroup reads: none.** With no threadgroup memory there is nothing to read unwritten. All reads are device buffers at deterministic in-bounds offsets for any `rows` ≥ 1: `y[row*3072 + d + {0,768,1536,2304}]`, `weights[row*4 + {0..3}]`, `shared[row*768 + d]` (`block-0.metal:30-42`).
- **No thread-timing dependence.** No barrier, atomic or cross-thread reduction anywhere: `"atomic_outputs": false` (`spec.json`), no `threadgroup_barrier`, `atomic_*` or `simd_*` call in either block. Each thread writes the unique `out[row*768 + d]` from private registers (`p0..p3`, `s01`, `s012`, `acc`, `block-0.metal:34-40`); the fold order is fixed by the source sequence and contraction is disabled by the header pragmas, so the result cannot vary with scheduling.
- **No captured-input assumptions.** The kernel hard-codes only the slot constants G1 enforces (hidden 768, 4 experts, stride 3,072 = 4×768; `y` [`rows`,4,768] bfloat16, `weights` [`rows`,4] float32, `shared` [`rows`,768] bfloat16). It branches on no input value, expert id, magnitude or row count, never sorts or re-normalises, and keeps strict positional pairing `weights[r,j]` ↔ `y[r,j,:]`; `weights` stays arbitrary finite float32 exactly as the Boundary requires, and NaN/Inf propagate through the same rounded ops as the reference.

## Decision

**Best (only) attempt: `muse-spark/round-4/kernels/combine/attempt-1`** — the byte-identical copy of the starting point, graded in this round: G1 pass, G2 differ 0 at all six sizes, `beats_compiled` true with both phases `faster` (decode 39.3 / 45.5 / 43.2 vs compiled 47.1 / 60.5 / 69.6 µs per layer, 25 / 25 / 30 wins; prefill 43.9 / 95.1 / 178.8 vs 66.9 / 284.1 / 567.2, 29 / 30 / 29 wins). No hardening change exists to name: exploration confirms the slot is at its chained floor and no bit-exact change widens its margin, so a second builder direction would be a second byte-identical grade. The attempt goes to the ready check as the combine member of the set.

**Predicted change in whole requests (PREDICTION):** this slot's own contribution, from its same-run margins × 20 layers, is about −0.16 / −0.30 / −0.53 ms per decode step at 1 / 8 / 32 requests against the compiled reference; the set's whole-request change is measured only by the ready check and is not a sum of slots.

## Failure ledger

## Failure ledger

No graded attempt lost: attempt-1 passes G1, G2 and `beats_compiled`. Nothing to harden against.

## Unverified

- Whether the pragma pair leaves any FMA in the compiled binary: G2 shows no contraction effect (differ 0 where the contracted kernel differed), but no AIR disassembly was available (no Metal toolchain in the grader environment); exactness rests on G2, not on inspecting the binary.
- Row counts other than the six graded ones (any decode count 1–32, other prefill chunk sizes): correctness rests on the per-element, rows-independent construction, not on G2. Sizes above 2,048, prefill beyond the first seeded chunk, other lengths/distributions and mixed batches untested.
- The set's whole-request behaviour (three-slot interaction at 8 requests; 1-request whole-request noise) is measured only by the ready check. Unwritten threadgroup reads: none by construction (no threadgroup memory), checked by hand in the boundary check. NaN-vs-NaN counts as equal in G2.

## Checks

*(checker-3 writes here before candidate.)*
