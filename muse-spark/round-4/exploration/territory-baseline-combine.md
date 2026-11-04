# Territory: baseline-combine

- **Owner:** explorer-baseline-combine
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T21:45:18Z
- **Next:** none — checklist filled (A–E), 2 inefficiencies posted open, note left in territory-g3, no open notes on mine

## What it does

Each of the 20 layers returns
`out = (y * weights[..., None]).sum(axis=-2).astype(bf16) + shared`
(`bench/grader/slots.py:51-52`; in the model
`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:230-232`). The code now
in the `combine` slot is round 3's
`muse-spark/round-3/kernels/combine/attempt-2/`: one fused
thread-per-output kernel whose seven `volatile` rounding barriers are
seven plain `float` locals, with FP contraction disabled by a header
pragma pair instead of by memory traffic. It passes G1+G2 bit-exact
and `beats_compiled` at all six sizes, but failed G3 alone on
5-repeat range overlap, so it is a starting point, not a baseline
(`muse-spark/round-4/exploration/README.md` *Starting point*;
`claude/findings/muse-round-3.md` §§1–2). Round 3's
`territory-baseline-combine.md`, `ineff-baseline-combine-1.md` and
`implementation/hyp-1-combine-devolatilize.md` are frozen and cited;
this entry writes only what is new for round 4: whether any
bit-exact change widens the decode margin, what the pragma leaves in
the compiled binary, and the chained-floor costs of this kernel.

## Checklist

### A. Launch spec: `spec.json` (attempt-2)

- **Callers:** `slots.build` turns the spec into one
  `mx.fast.metal_kernel` call (`bench/grader/slots.py:304-326`); the
  grader calls it per layer as `fn(y=…, weights=…, shared=…)`
  (`bench/grader/check_candidate.py:152-154,323-347`), and in serving
  it replaces the combine line via `slots.make_moe_call`
  (`bench/grader/slots.py:97-119`, combine at `:116`). Shapes: decode
  1 / 8 / 32 rows, prefill 127 / 1,016 / 2,048 rows.
- **Callees:** one kernel launch per layer, 20 per step.
  `spec.json` (single line): name `fused_combine`, `source_block` 0,
  `header_block` 1, inputs `y, weights, shared`, output `out`
  [rows,768] bfloat16, grid `rows*768`, threadgroup [256,1,1],
  `template` {}, `init_value` null, `atomic_outputs` false. **No
  `max_rows`, no `min_rows`:** `hands_off(rows)` is therefore false
  at every row count (`bench/grader/slots.py:330-343`), so the
  round-4 `handed off` verdict never applies to this slot and every
  graded size runs this kernel. Adding a hand-off now would only
  surrender a win (see *Decode margin* below).
- **Data:** `y` bf16 [rows,4,768] row-contiguous, `weights` f32
  [rows,4], `shared` bf16 [rows,768] → `out` bf16 [rows,768];
  9,232 B/row unique traffic (6,144 + 16 + 1,536 + 1,536). Frozen
  accounting in round 3's territory checklist A still holds; the
  kernel adds no threadgroup memory, no atomics, no second kernel.
- **Neighbours:** before: experts' `y`, routing's `weights`
  (positional pairing `weights[r,j] ↔ y[r,j,:]`), shared expert's
  `shared` (outside the slots); after: residual add
  `h + feed_forward(…)` (`mobilemoe.py:258`). A replacement must
  keep shapes, dtypes, the f32 fold and both bf16 rounding points.
- **Cost:** measured, piece: `grade.json` — candidate 31.8 / 34.7 /
  32.7 µs/layer decode and 38.2 / 100.9 / 198.0 prefill vs compiled
  36.9 / 41.1 / 45.7 and 64.0 / 281.8 / 556.2 (same-run paired
  verdicts `faster` at all six, 24 / 27 / 30 / 30 / 30 / 29 wins).
  Full floor arithmetic is in section D and
  `ineff-baseline-combine-1`.
- **Invariants:** G2 `differ == 0` at all six sizes (grader compares
  against the eager reference, `check_candidate.py:152-163,351`;
  `mx.compile` appears only in timing). Any change must keep: exact
  bf16→f32 casts; each product rounded once to f32; left fold
  `((p0+p1)+p2)+p3` with a rounding after every add; one bf16 round;
  single-rounding bf16 `+ shared`. Combine has no 60-row kernel
  switch underneath (that switch is the experts slot's).

### B. Kernel body: `block-0.metal` (attempt-2)

- **Callers / Callees:** as in A; the body is the whole kernel
  (`block-0.metal:26-42`). One thread per output element:
  `idx = thread_position_in_grid.x`, `row = idx / 768`,
  `d = idx - row*768` (`:26-28`), then `wbase`, `ybase`, `sbase`
  (`:30-32`).
- **Data / arithmetic:** four products
  `p_j = float(y[ybase + j*768]) * weights[wbase + j]` (`:34-37`),
  fold `s01 = p0+p1`, `s012 = s01+p2`, `acc = s012+p3` (`:38-40`),
  final line verbatim from every predecessor:
  `out[sbase] = bfloat(acc) + shared[sbase]` (`:42`). Seven f32
  materialisations, no `volatile`, no helper calls, no threadgroup
  memory in this attempt (contrast attempt-1's `noinline` helpers,
  `muse-spark/round-3/kernels/combine/attempt-1/block-1.metal`).
  Loads: 4 bf16 `y` values strided 768 elements (coalesced within
  each expert plane — consecutive threads read consecutive `d`),
  the same 4 `weights` re-read by all 768 threads of a row
  (L1-broadcast, 16 B unique), 1 bf16 `shared`, 1 bf16 store.
  Grid is always whole threadgroups (768 = 3×256, and 256 is a
  multiple of SIMD width 32), so no partial threadgroup and no
  bounds check is needed at any row count (round-3 boundary check,
  `hyp-1-combine-devolatilize.md` *Boundary checks* attempt-2).
- **Neighbours:** identical to A. Indexing is row-count
  independent straight-line code, so the six G2 passes carry the
  rounding argument to untested sizes by construction; only the
  six sizes are certified (boundary *can't see* list).
- **Cost:** see D. Body terms that are not unique traffic: one
  `idx/768` integer divide per thread and 768× redundant weight
  loads per row — bounded in `ineff-baseline-combine-2`.
- **Invariants:** the written fold is already the reference's
  association (round-2 territory checklist C: `col_reduce_small`
  folds `((p0+p1)+p2)+p3`). Reassociating to any other tree,
  including the balanced `((p0+p1)+(p2+p3))`, changes f32 rounding
  on ppm of elements and fails G2 (round-1 CPU emulation, cited in
  round-2 `ineff-baseline-combine-1`); contracting any mul into its
  add does the same (section C). The arithmetic is therefore
  frozen; only data movement, mapping and launch shape may move.

### C. Header: `block-1.metal` — what the pragma leaves in the binary

- **What it is:** two lines, `#pragma STDC FP_CONTRACT OFF` and
  `#pragma clang fp contract(off)` (`block-1.metal:23-24`), wired
  by `"header_block": 1` — the only `spec.json` delta from the
  volatile starting point (`attempt-2/notes.md` *What changed*).
  `slots.build` passes the header text to `mx.fast.metal_kernel`
  (`slots.py:317-325`), which emits it before the generated kernel
  function, where FP-option pragmas take effect
  (`block-1.metal:6-13`; MLX help for `metal_kernel`: header is
  included before the main function).
- **Compile flags:** `slots.build` passes no `compile_options`, so
  MLX's default `math_mode` Safe applies
  (`.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/common/metal_kernel.h:7-15`).
  Without fast-math the compiler does not reassociate
  (`block-1.metal:19-22`); contraction is the only transform that
  could keep a product unrounded into its add, and
  `contract(off)` forbids exactly that fusion.
- **FMA or not — the evidence that exists:** no FMA *effect*
  survives. `grade.json` shows `differ 0 / max_ulp 0.0` at all six
  sizes (summed over 20 layers), while the contracted evaluation's
  fingerprint is known from the volatiles-free twin: 0 / 2 / 2
  differ decode, 6 / 50 / 130 prefill (round-3 territory checklist
  B; round-2 `ineff-baseline-combine-1`). Any surviving
  mul→add fusion in the four products or the three adds would
  reproduce that signature at ppm rate; it is absent at
  31.5M prefill elements, so the graded binary evaluates 4
  separate `fmul` + 3 separate `fadd` roundings (plus the bf16
  cast and bf16 add, neither of which has a mul to contract).
  The final line cannot contract in any case: `acc` is already a
  materialised local and `shared` is a load, not a product.
- **What remains unverified (not an inefficiency):** no `.air` /
  `.metallib` disassembly was produced. `xcrun metal` cannot run
  on this machine (`cannot execute tool 'metal' due to missing
  Metal Toolchain; use: xcodebuild -downloadComponent
  MetalToolchain` — checked 2026-11-04), MLX compiles in-process
  through the Metal framework so there is no kernel file on disk
  to inspect, and running MLX to force a compile is a GPU job an
  explorer must not start (tracking.md *Rules*). Which of the two
  spellings is load-bearing (the C99 `STDC` form is expected to be
  inert in Metal's C++-based language; the `clang fp` form is the
  operative one) is likewise unprobed: separating them needs two
  graded builder attempts, not explorer work. Functionally the
  pair as graded is proven; at instruction level the answer
  above is inferred from G2, labelled as such. Round 3's notes
  already named this probe (`attempt-2/notes.md` *Next change*:
  check the compiled binary for fma) — it is still open, and it
  changes no kernel decision until a toolchain or a re-grade
  exists, because G2 re-proves the effect on every grade.

### D. Cost table: `grade.json` and the chained floor

Measured, piece (`attempt-2/grade.json`, chained paired timing,
`check_candidate.py:191-251` — 20 layers linked by the `link`
kernel, candidate vs compiled in alternating repeats, `faster`
needs ≥22/30 wins). Floors at 200 GB/s peak and 5.2 TFLOPS peak
(`claude/GLOSSARY.md`); per-row traffic 9,232 B, ~7 f32 ops per
output:

| Size | Candidate | Compiled | Margin | Wins | BW floor | Compute floor |
|---|---|---|---|---|---|---|
| decode 1 | 31.8 | 36.9 | 5.1 µs/layer (0.10 ms/step) | 24 | 0.05 µs | ~0.001 µs |
| decode 8 | 34.7 | 41.1 | 6.4 (0.13) | 27 | 0.37 | ~0.008 |
| decode 32 | 32.7 | 45.7 | 13.0 (0.26) | 30 | 1.48 | ~0.03 |
| prefill 127 | 38.2 | 64.0 | 25.8 | 30 | 5.86 | ~0.13 |
| prefill 1,016 | 100.9 | 281.8 | 180.9 | 30 | 46.9 | ~1.05 |
| prefill 2,048 | 198.0 | 556.2 | 358.2 | 29 | 94.5 | ~2.12 |

- **Decode is the chained floor.** Traffic + compute are
  ≤1.5 µs of a ~32–35 µs/layer measurement at every decode size;
  the rest is one dispatch + `link` + launch latency per layer,
  identical for candidate and compiled bar (both are timed
  through the same chain, `check_candidate.py:197-214,372-384`).
  Candidate time is flat in rows (31.8 → 32.7 for 32× the work),
  the signature of a fixed floor, already confirmed for this
  slot's shape in round 2 (`ineff-baseline-combine-2`, frozen).
  Absolute chained µs drift up to ~35% between runs on identical
  code (round-3 lesson, `attempt-2/notes.md` *Why the timing
  missed*), so only the same-run margins and win counts above
  are evidence; cross-run absolutes (e.g. attempt-1's 46.8 /
  42.0 / 49.2 in its hotter run) are machine state.
- **Prefill is body-shaped but not bandwidth-bound.** Candidate
  slope ≈0.094 µs/row/layer between 1,016 and 2,048 rows against
  the 0.046 µs/row bandwidth slope: ~95 GB/s achieved, about
  half of peak. Compiled slope ≈0.256 µs/row (~34 GB/s). The
  candidate's scalar, latency-shaped body (section B) therefore
  still beats the compiled reference's generic body by 2.8× at
  2,048 rows, while neither is close to peak bandwidth in the
  glossary's sense. Kind: overhead-bound throughout
  (`claude/GLOSSARY.md`), least so at large prefill.
- **Where the margin comes from.** The margin grows with rows
  (5.1 → 13.0 decode, 25.8 → 358.2 prefill) because it is a
  per-element body gap riding on a common fixed floor: at 1 row
  the floor compresses a ~2.8× body advantage into 5.1 µs and
  24 wins — the thinnest decode verdict in the slot (threshold
  22, `check_candidate.py:59-63,244-250`). That thinness, not
  any single size's median, is what G3's range rule then meets
  at 32×128 / 32×1024 (`claude/findings/muse-round-3.md` §1:
  combine fails alone vs `compiled-combine` there on overlap,
  no phase slower anywhere).

### E. Whether any bit-exact change widens the decode margin

No candidate survives the arithmetic. The kernel is already at
the minima on all three axes a bit-exact change may touch:

- **Arithmetic: frozen.** 4 muls + 3 adds + bf16 cast + bf16 add
  is the reference's own op count; every reassociation or
  contraction variant fails G2 (sections B–C). There is no
  seventh-op saving to take.
- **Bytes: minimal.** Every `y` element feeds exactly one output,
  `shared` and `out` are touched once, `weights` unique traffic
  is 16 B/row. Routing weights are strictly positive
  (sigmoid outputs normalised ×2.5, `slots.py:44-48`), so no
  zero-weight sparsity exists to skip reads for — and a
  data-dependent skip would still need the weight read first.
- **Launches: minimal.** One kernel per layer; zero is not a
  kernel. The remaining decode time is the shared dispatch
  floor (section D), which no body edit removes — round 1's
  staged/vectorised organisation and round 2's staged variant
  landed on the same floor (round-2 territory checklist E,
  frozen; re-bounded for this body in
  `ineff-baseline-combine-2`).

The three mapping changes a builder could still try, each
bit-exact by construction, bound as follows (predictions,
arithmetic in the inefficiency entry): removing the per-thread
`idx/768` divide with a 2-D grid (`grid ["768","rows"]`,
`thread_position_in_grid.y` as the row — expressible, since
`dim()` accepts literal and `rows` terms, `slots.py:283-293`);
staging the 4 weights per threadgroup instead of 768× broadcast
loads; and widening to one thread per 2–4 outputs (fewer threads,
weights loaded once per group of outputs). Each attacks
<1 µs/layer of body time under a ~32 µs floor — inside the
~5% same-kernel paired noise the brief warns of
(`claude/muse-agent-brief.md` *The piece timing*) and below
what 22/30 wins can certify. Threadgroup-size retuning
(128/512/768) changes only how the same 768·rows threads are
packed into the same single launch, so it cannot move the
dispatch term either. A `min_rows`/`max_rows` hand-off is
strictly narrowing here: `min_rows: 2` would hand the 1-row win
(5.1 µs, the set's 1-request combine contribution) to the
compiled reference, and `max_rows: 32` would hand away
25.8–358.2 µs/layer of prefill margin that G3's prompt time
currently banks. **Conclusion for builders: do not spend combine
attempts on widening decode margin; the slot's remaining G3
problem is 5-repeat spread at 32×1024, which only set
composition (the `g3` territory's question) or repeat count —
neither a kernel property — can move.** This matches round 3's
close (`hyp-1-combine-devolatilize.md` *Decision*: no further
kernel change worth making on 7 plain ops), now with the
round-4 hand-off surface checked as well.

## Relations

- **Fed by:** `baseline-experts` (`y`), `baseline-routing`
  (`weights`; the combine part is graded on reference weights, so
  routing exactness cannot save or fail this slot), shared expert
  (outside the slots, `shared`), `grader` (six-size G2, chained
  timing, `handed off` rules that never fire here), `g3` (the
  32×1024 set-margin question this slot's thin 1-row / 32-row
  verdicts feed).
- **Feeds:** residual add / next layer; the ready check's set
  (`muse-all`), where this kernel's 1-request speed is part of
  what lets a `min_rows` experts attempt pass at 1 request.

## Inefficiencies

- `ineff-baseline-combine-1` — the chained dispatch floor:
  ~32 µs/layer at decode against ≤1.5 µs of traffic + compute;
  already minimal at one launch, so no bit-exact change banks
  it (open; repriced for this kernel).
- `ineff-baseline-combine-2` — per-thread divide + redundant
  weight-broadcast loads: the only body terms left above the
  floor, bounded under 1 µs/layer (open; posted so the bound is
  verified, not chased).

## Files read

- `muse-spark/round-3/kernels/combine/attempt-2/{spec.json,block-0.metal,block-1.metal,notes.md,grade.json}`
  — the code in the slot: launch surface, 7-op body, pragma
  header, builder account and the six-size verdict/margins.
- `muse-spark/round-3/kernels/combine/attempt-1/{block-0.metal,block-1.metal,spec.json,grade.json,notes.md}`
  — the `noinline` twin: same body, call barriers, same verdict
  shape in a hotter run (cross-run drift control).
- `muse-spark/round-2/kernels/combine/attempt-1/block-0.metal`
  — the volatile predecessor (7 barriers the pragma replaces).
- `bench/grader/slots.py:44-52,97-119,283-343`,
  `bench/grader/check_candidate.py:59-63,152-163,191-251,253-271,352-400`
  — reference, MoE split, size expressions, G2, link chain,
  paired verdicts, hand-off handling.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232,251-258`
  — the combine line and its residual consumer.
- `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/common/metal_kernel.h:7-15`
  — `math_mode` Safe default (no `compile_options` are passed).
- `claude/findings/muse-round-3.md` §§1–2,4,
  `claude/adr/0015-sets-and-hand-off-below.md`,
  `claude/muse-agent-brief.md` (slots, piece timing, `min_rows`),
  `claude/GLOSSARY.md` (kind terms, peak figures).
- Frozen, cited not repeated:
  `muse-spark/round-3/exploration/territory-baseline-combine.md`
  (volatile accounting, install path, op balance),
  `muse-spark/round-3/exploration/ineff-baseline-combine-1.md`
  (twin swing +1.441 ms/step, confirmed),
  `muse-spark/round-3/implementation/hyp-1-combine-devolatilize.md`
  (boundary, both attempts, ready-check failure, Decision),
  `muse-spark/round-2/exploration/territory-baseline-combine.md`
  checklists B–E and `ineff-baseline-combine-1/2.md` (reference
  kernel chain, association proof, floor, micro-path bound).

## From other agents

(none yet)
