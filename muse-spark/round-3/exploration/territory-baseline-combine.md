# Territory: baseline-combine

- **Owner:** explorer-baseline-combine
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T17:46:30Z
- **Next:** none — checklist filled (A–F), 1 inefficiency posted, notes left in g3/grader entries, no open notes on mine

## What it does

Same slot as round 2 built on: each of the 20 layers returns
`out = (y * weights[..., None]).sum(axis=-2).astype(bf16) + shared`
(`bench/grader/slots.py:51-52`). The code now in the slot is the round-2
starting point `muse-spark/round-2/kernels/combine/attempt-1/` (one fused
kernel/layer, exact at all six sizes, `beats_compiled` true) — it is no
baseline because G3 times it +1.05 ms/step slower than the unchanged model
with 32 requests. Round 2's
`muse-spark/round-2/exploration/territory-baseline-combine.md` (frozen)
mapped the reference chain, the sum association, the G2 gap and the dispatch
floor; this entry writes only what is new: the starting point's kernel and
spec as code in the slot, its two twins (round-1 attempt-1 without volatiles,
round-2 attempt-2 staged with volatiles), the G3 install path, and the
cross-run attribution of the +1 ms to the volatile barriers.

## Checklist

### A. Starting point: `fused_combine` (`muse-spark/round-2/kernels/combine/attempt-1/`)

- **Callers:** unchanged from round 2's mapping (gradi `slots.build` +
  per-layer `fn(y, weights, shared)`; in the model the combine line of
  `MoE.__call__` via `slots.make_moe_call`, `slots.py:97-119`). Shapes:
  decode 1 / 8 / 32 rows, prefill 127 / 1,016 / 2,048 rows.
- **Callees:** one `mx.fast.metal_kernel` launch per layer, 20 per step.
  `spec.json`: grid `rows*768`, threadgroup [256,1,1], inputs
  `y, weights, shared`, one bf16 output `out` [rows,768]. **New:** this
  spec is byte-identical to round-1 attempt-1's (`diff` of the two
  `spec.json` files is empty) — grid, threadgroup, inputs, outputs and
  launch surface did not change between rounds.
- **Data:** unchanged from round 2's mapping: 9,232 B/row touched
  (6,144 y + 16 weights + 1,536 shared + 1,536 out). **New:** each thread
  additionally performs 7 volatile stores + 7 volatile loads of f32
  (`block-0.metal:33-39`: `p0..p3`, `s01`, `s012`, `acc`), i.e. 14
  thread-private memory accesses × 4 B = 56 B/thread = rows×43,008
  B/layer: 43 KB / 344 KB / 1.375 MB at decode 1 / 8 / 32 rows, 5.5 /
  43.7 / 88 MB at prefill 127 / 1,016 / 2,048 rows. By `volatile`
  semantics every one is a real memory access (no register forwarding,
  no elision); thread address space has no SRAM backing, so they are
  stack traffic. How much the L1/SLC absorbs is the unverified
  micro-lever in `ineff-baseline-combine-1`; the access count itself is
  exact from the code.
- **Neighbours:** unchanged (experts' `y`, routing's `weights`,
  shared expert's `shared` before; residual add after). **New:** in the
  step the kernel runs with a full layer of large kernels around it
  (~22 MB/layer experts streaming at 32 rows, shared-expert MLP,
  attention); in the grader it runs with only the tiny `link` kernel
  between layers (`check_candidate.py:186-209`). Same kernel, different
  cache context — the discrimination `ineff-baseline-combine-1` uses.
- **Cost:** piece (Claude re-grade,
  `results/runs/2026-11-04-muse-round-2/grade-phases-combine-attempt-1.json`):
  decode 45.7 / 47.9 / 64.6 vs compiled 53.4 / 56.4 / 69.1 µs/layer,
  prefill 62.8 / 226.5 / 273.0 vs 66.1 / 271.6 / 520.4 — faster at all
  six sizes. In-step G3 (vLLM): −0.372 / −0.197 / **+1.051** ms/step at
  1 / 8 / 32 requests (`claude/findings/muse-round-2.md` §3). The
  measurement side is `ineff-g3-1` (swing +75.0 µs/layer vs eager at
  32 rows); the kernel-side mechanism is `ineff-baseline-combine-1`.
- **Invariants:** bit-exact G2 at all six sizes (differ 0 everywhere,
  grade-phases file above). Any change must keep the per-op rounding
  the volatiles currently enforce: each product rounded once to f32,
  fold `((p0+p1)+p2)+p3` with a rounding after every add, one bf16
  round, single-rounding bf16 `+ shared` (round-2 hyp-3 boundary,
  `muse-spark/round-2/implementation/hyp-3-combine-exact.md:41-50`).

### B. Twin without volatiles: round-1 `muse-spark/kernels/combine/attempt-1/`

- **Callers / Callees / Data:** identical to A except the arithmetic:
  `block-0.metal:12-25` has the same three index lines (12-14), the
  same three base lines (16-18) and the same final line (25) as round-2
  attempt-1's lines 25-27, 29-31 and 41; only lines 20-23 (one
  contracted f32 expression) differ from round-2's lines 33-39 (seven
  volatile stepwise ops). Spec byte-identical (checklist A). **This is
  the only code delta between the rounds for this slot.**
- **Neighbours:** identical (same install path, same slot).
- **Cost:** G3 vLLM in round 1's run: −0.514 / −0.705 / −0.390 ms/step
  at 1 / 8 / 32 requests (`claude/findings/muse-round-1.md` §3) — a
  saving at every size, vs −0.372 / −0.197 / +1.051 now. Engine swing
  (round-2 minus round-1): **+0.142 / +0.508 / +1.441 ms/step**
  (+7.1 / +25.4 / +72.1 µs/layer). Cross-run drift is bounded by the
  control both runs timed: round-1's three kernels together measured
  +0.980 / +0.163 / −1.756 then and +0.923 / +0.127 / −1.849 now
  (muse-round-1.md §3 vs muse-round-2.md §3) — within 0.06–0.09 ms at
  every size, an order of magnitude below the 1.44 ms swing. Plain MLX
  agrees (swing +1.574 ms at 32: −0.406 then, +1.168 now).
- **Invariants:** fails bit-exact G2 (0 / 2 / 2 differ decode, 6 / 50 /
  130 prefill) — the twin is a measurement control, not a fix.

### C. Twin with staging: round-2 `muse-spark/round-2/kernels/combine/attempt-2/`

- **Callees:** grid `rows*192`, threadgroup 192 (one threadgroup/row),
  3,088 B threadgroup memory (`wsh` + `shsh`), one barrier, vectorized
  u64 loads (`block-0.metal:20-63`, `spec.json`).
- **Data:** same 9,232 B/row of slot traffic, plus 24 volatile
  round-trips per thread (6 per output × 4 outputs, lines 66-83: `p0..p3`
  then two fold steps per `k`) = 48 accesses × 4 B = 192 B/thread.
  At 32 rows: 6,144 threads × 192 B = 1.18 MB/layer — near attempt-1's
  1.375 MB/layer. Same volatile dose in total spill bytes, different
  shape (fewer threads, more barriers each).
- **Cost:** chained 45.2 / 37.6 / 33.8 vs compiled 47.3 / 55.6 / 43.0
  µs/layer in round 2's own run (builder-3-b note on round-2
  `ineff-baseline-combine-1`); **never G3-timed**, so its in-step
  penalty is unmeasured. Dose prediction: similar in-step loss to
  attempt-1 at 32 rows (similar total spill bytes); timing it would
  test dose-vs-shape but is not needed for the fix.
- **Invariants:** bit-exact at all six sizes (same run).

### D. G3 install path (exonerated)

- `time_candidates.py:65-66` installs the candidate as
  `lambda y, weights, shared: fn(...)["out"]` inside
  `slots.make_moe_call` (`slots.py:97-119`); `slots.build`
  (`slots.py:304-359`) calls `mx.fast.metal_kernel` lazily — no
  `mx.eval`, no per-layer sync, no command-buffer split beyond what the
  graph already has. The compiled reference is installed the same way
  (`mx.compile(slots.REFERENCE[p])`, `time_candidates.py:105,119`),
  also lazy. Install mechanics therefore cannot explain a
  candidate-vs-compiled in-step gap: both replace the same line with
  one lazy node. The +1.234 ms/step candidate-vs-compiled G3 gap at 32
  requests (muse-round-2.md §3: +1.051 vs −0.183) must come from the
  kernel's GPU execution in context, not from Python-side handling.

### E. Op counts (no hidden kernels)

- `muse-plain.jsonl` `kind=check` records (graph ops via
  `mx.export_to_dot`, `repeat_fixes.py:137-144`): at 32 rows
  muse-combine has exactly 906 kernels vs base 986 (−80 = 4/layer:
  Add −20, AsType −40, Sum −20, Multiply −20, CustomKernel +20) and
  vLLM `console-vllm.txt` shows 1007 vs 1047 for compiled-combine
  (−40 = the 2 extra fused kernels/layer compiled keeps). No extra
  graph-level op (copy, cast, gather) appears for the candidate at any
  size. Backend-inserted contiguity copies would be invisible here, but
  they would hit the round-1 twin equally (identical inputs and launch
  surface) — the 1.44 ms twin swing rules them out as the main cause,
  and the twin's engine≈grader match bounds them near zero (detail in
  `ineff-baseline-combine-1`).

### F. What the grader chain prices (pointer)

- The chained timing (`check_candidate.py:192-221`) links layers
  through `link` on the smallest input (`weights`, [rows,4]) and
  divides one `mx.eval` by 20. It prices serial dispatch latency
  equally for every function and runs the kernel against a quiet cache
  (only link traffic between layers). It therefore cannot show a cost
  that lives in the kernel's interaction with a busy step — cache
  pressure from neighbouring layers' megabytes, or any stall the
  link-chain's steady state absorbs. Note left in round-3
  `territory-grader.md`; the consequence for this slot is in
  `ineff-baseline-combine-1`.

## Relations

- **Fed by:** `baseline-experts` (`y`), `baseline-routing` (`weights`),
  shared expert (outside the slots), `grader` (G2 + chained timing),
  `g3` (`ineff-g3-1` owns the measurement side of the gap; this
  territory owns the kernel-side mechanism).
- **Feeds:** residual add / next layer; the ready check (round 3 must
  reproduce the 32-request combine-alone step or re-learn the gap).

## Inefficiencies

- `ineff-baseline-combine-1` — the seven volatile rounding barriers
  cost ~1.4 ms/step in the engine at 32 rows while the grader prices
  them near zero; exactness without volatiles recovers it.

## Files read

- `muse-spark/round-2/kernels/combine/attempt-1/{spec.json,block-0.metal,notes.md}`
  — the code in the slot: launch surface, volatile lines 33-39, verdict.
- `muse-spark/kernels/combine/attempt-1/{spec.json,block-0.metal}` —
  the volatiles-free twin; spec diff empty, body identical but for lines
  20-23 vs 33-39.
- `muse-spark/round-2/kernels/combine/attempt-2/{spec.json,block-0.metal}`
  — staged volatile twin: 24 round-trips/thread, 1.18 MB/layer at 32 rows.
- `bench/grader/slots.py:51-52,97-119,304-359`,
  `bench/grader/check_candidate.py:186-221`,
  `bench/grader/time_candidates.py:65-66,100-136` — reference, install,
  link chain; install exonerated.
- `results/runs/2026-11-04-muse-round-2/grade-phases-combine-attempt-1.json`,
  `muse-plain.jsonl` (check records), `console-vllm.txt` (repeats,
  op counts) — piece times, exact −80-kernel balance, tight G3 ranges.
- `claude/findings/muse-round-1.md` §3, `claude/findings/muse-round-2.md`
  §§1-4 — twin G3 numbers both rounds plus the reproducing control.
- `muse-spark/round-2/implementation/hyp-3-combine-exact.md:41-129` —
  boundary (per-op rounding contract) and both builders' barrier notes.
- Round-2 frozen `territory-baseline-combine.md`,
  `ineff-baseline-combine-1.md`, `ineff-baseline-combine-2.md` —
  reference chain, association proof, dispatch floor; cited, not repeated.
- Round-3 `exploration/README.md` (starting point, item 4),
  `ineff-g3-1.md` (measurement side, prefill twin), `claude/GLOSSARY.md`.

## From other agents

(none yet)
