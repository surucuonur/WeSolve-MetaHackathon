# Territory: baseline-combine

- **Owner:** explorer-baseline-combine
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T03:18:22Z
- **Next:** none — checklist filled, 2 inefficiencies posted, grader note folded in

## What it does

In every decode step and every prefill chunk, each of the 20 layers runs
`MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:230-232`): the
routed experts produce `y` [rows,4,768] bf16 in `inds` order, routing produces
`weights` [rows,4] f32, the shared expert produces `shared` [rows,768] bf16,
and combine returns `out = (y * weights[..., None]).sum(axis=-2).astype(bf16)
+ shared` [rows,768] bf16 (`bench/grader/slots.py:51-52`). This territory is
the starting point now occupying the `combine` slot —
`muse-spark/kernels/combine/attempt-1` (one fused kernel/layer) — the
reference behind it, and the MLX kernels the reference launches. Round 1
mapped the same chain as territory `other-kernels` checklist G and
`ineff-model-2` (confirmed); this entry writes only what is new under round
2's rules (bit-exact G2 at decode + prefill sizes, chained piece timing;
`claude/adr/0010-exact-at-every-size.md`), citing round 1 by path.

## Checklist

### A. Starting point: `fused_combine` (`muse-spark/kernels/combine/attempt-1/`)

- **Callers:** the grader builds it from `spec.json` via `slots.build`
  (`bench/grader/slots.py:303-345`) and calls it per layer as
  `fn(y=…, weights=…, shared=…)` (`bench/grader/check_candidate.py:139-141`);
  in the model it replaces the combine line of `MoE.__call__`
  (`mobilemoe.py:231-232`, split form `slots.py:116`). Shapes: decode
  1 / 8 / 32 rows and prefill 127 / 1,016 / 2,048 rows (brief: rows is tokens
  in the step; prefill rows from `slots.capture_prefill`, `slots.py:214-254`).
- **Callees:** one `mx.fast.metal_kernel` launch per layer, 20 per step.
  `spec.json:1`: grid `rows*768` threads, threadgroup [256,1,1], inputs
  `y, weights, shared`, one bf16 output `out` [rows,768], no header block,
  no template, `atomic_outputs: false`. Body: `block-0.metal:12-25`.
- **Data:** `y` bf16 [rows,4,768] row-contiguous (6,144 B/row), `weights`
  f32 [rows,4] (16 B/row), `shared` bf16 [rows,768] (1,536 B/row) → `out`
  bf16 [rows,768] (1,536 B/row); 9,232 B/row touched in total. One thread
  per output element: `row = idx/768`, `d = idx - row*768`
  (`block-0.metal:12-14`); thread reads `y[row,0..3,d]` (strides of 768
  elements), `weights[row,0..3]`, `shared[row,d]`, computes
  `acc = Σ float(y_j)·weights_j` in f32 (`:20-23`), writes
  `bfloat(acc) + shared` (`:25`). No threadgroup memory, no barriers, no
  atomics; threads fully independent. Grid is always whole threadgroups
  (768 = 3×256), and the indexing is exact for every row count (prefill
  2,048 rows = 1,572,864 threads = 6,144 threadgroups), so its G2 failure
  is arithmetic only (round 1 boundary check, `muse-spark/implementation/hyp-3-fused-combine.md` *Boundary checks*).
- **Neighbours:** before: experts' `y` and routing's `weights` (positional
  pairing `weights[r,j] ↔ y[r,j,:]`, never re-sorted), shared expert's
  `shared` (outside the slots); after: residual add `h + feed_forward(…)`
  in `DecoderLayer.__call__` (`mobilemoe.py:258`), then the next layer.
  A replacement must keep shapes, dtypes, the f32 accumulation, and both
  bf16 rounding points (the astype, then the bf16 `+ shared`).
- **Cost:** 1 kernel/layer. Bytes ÷ peak (200 GB/s, brief): 9,232 /
  73,856 / 295,424 B per layer at 1 / 8 / 32 rows = **0.05 / 0.37 /
  1.48 µs**; compute ≈7 f32 ops per output element ≈0.03 µs/layer even at
  32 rows against 5.2 TFLOPS peak — both negligible. Measured (old
  independent piece timing): candidate **18.6 / 17.0 / 18.9 µs/layer**
  against compiled reference 23.8 / 26.1 / 32.0 and reference
  23.1 / 22.5 / 39.5 (Claude re-grade, `claude/findings/muse-round-1.md`
  §1, `results/runs/2026-11-04-muse-round-1/grade-combine.json`); round
  1's own run: 17.2 / 17.0 / 18.5 against compiled 22.9 / 26.2 / 28.0
  (`attempt-1/notes.md` *Verdict*). Flat in rows: time is the dispatch
  floor, not work — see `ineff-baseline-combine-2`. No chained timing
  exists yet for this slot: the round-2 grader times only after G2 passes
  (`check_candidate.py:308-337`), and this kernel fails G2.
- **Invariants:** round 2 G2 is bit-exact at all six sizes (brief *How each
  attempt is judged*; ADR 0010). Round 1's 2-ULP rule is gone. The kernel
  keeps both rounding points but contracts its mul+add chain to FMAs,
  which the reference never does — 2 elements differ at decode 8/32 rows,
  6 / 50 / 130 at prefill (`grade-exact-combine.json`); see
  `ineff-baseline-combine-1` and checklist C–D below.

### B. Reference: `ref_combine` (`bench/grader/slots.py:51-52`)

- **Callers:** `MoE.__call__` split form (`slots.py:97-119`, combine at
  `:116`); the grader's correctness check calls it directly per layer
  (`check_candidate.py:140`) — **the G2 reference is the eager MLX graph,
  not the compiled one**; `mx.compile(ref_combine)` appears only in timing
  (`check_candidate.py:322`). In a `routing+combine` candidate the combine
  part still receives the captured reference weights (brief; `slots.py`
  slot table `:29-40`).
- **Callees:** eagerly, 5 kernels/layer, 100/step (`results/runs/2026-11-01-breakdown/tables.md:140`:
  AsType 40, Multiply 20, Sum 20, Add 20 per step), mapped in round 1's
  `muse-spark/exploration/territory-other-kernels.md` checklist G:
  1. `y` bf16→f32: `copy_v` (`…/metal/kernels/copy.h`, `cast_to<U>` per
     element; exact — bf16 ⊂ f32), grid rows·4·768 threads;
  2. broadcast multiply by `weights[...,None]`: `binary_g`
     (`…/metal/kernels/binary.h`), `Multiply` op `x * y`
     (`…/metal/kernels/binary_ops.h`), f32 product rounded once to f32,
     grid rows·4·768;
  3. sum over axis −2 (size 4, stride 768): `col_reduce_small`
     (`…/metal/kernels/reduction/reduce_col.h`), threadgroup (32,4,1) =
     128 threads (strided-small dispatch; round 1 territory-other-kernels
     G, citing MLX v0.32.4 `reduce.cpp` — installed version confirmed
     0.32.4); `Sum` op `a + b`, init 0 (`…/reduction/ops.h`);
  4. f32→bf16: `copy_v` again — the single astype rounding (RNE);
  5. `+ shared`: `binary_vv` (`binary.h`), `Add` op `x + y` in bf16,
     grid rows·768.
  (Header paths under `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/`.)
- **Data:** intermediates are f32 [rows,4,768] after steps 1–2 (12,288
  B/row written + read), f32 [rows,768] after the sum, bf16 after step 4.
  Kernel-touch bytes ≈1.3 / 11 / 43 MB/step at 1 / 8 / 32 rows, but the
  intermediates stay in GPU cache between dependent kernels; unique DRAM
  traffic is the 9,232 B/row above (round 1 territory-other-kernels G).
- **Neighbours:** identical to A — that identity is the boundary
  (`muse-spark/implementation/hyp-3-fused-combine.md` *Boundary*).
- **Cost:** knock-out 0.61 ms/step at 1 row, 0.33 ms at 32 rows
  (`tables.md:335`; per layer 30.5 / 16.5 µs — the knock-out and the piece
  harness disagree on standalone cost, round 1 `ineff-model-2`); piece
  timing as in A. `mx.compile` fuses the casts (Claude:
  `claude/findings/step-breakdown.md`, via round 1 ineff-model-2) and
  saves 0.29 / 0.26 / 0.27 ms/step end to end alone
  (`claude/findings/muse-round-1.md` §3), yet its piece timing
  (23.8–26.1–32.0 µs/layer) still trails the fused candidate at every
  row count.
- **Invariants:** the arithmetic contract a replacement must reproduce
  bit for bit: exact bf16→f32 casts; each product `p_j` rounded once to
  f32 by the multiply kernel; the 4-term sum folded by separate f32 adds
  (checklist C); one bf16 rounding of the sum; one bf16 add of `shared`.
  Because each step is its own kernel, **no mul+add contraction can occur
  in the reference** — that asymmetry is the whole G2 gap (checklist D).

### C. The reference sum's association (`col_reduce_small`, size-4 axis)

Per output element (fixed row, fixed column `d`), `reduce_col.h` runs:
`totals[i]` (4 column lanes per thread) start at `Sum::init = 0`; the loop
over the reduction (`r = lid.y; r < rows·4; r += lsize.y`, `lsize.y = 4`)
gives each `lid.y` lane exactly one product, so lane `j` holds
`0 + p_j = p_j` exactly; then, if `lsize.y > 1`, the threadgroup fold on
`lid.y == 0` computes `t = p_0; t = p_1 + t; t = p_2 + t; t = p_3 + t`,
each `op(a,b) = a + b` rounded once to f32. IEEE addition commutes, so the
reference association is exactly the left fold **((p₀+p₁)+p₂)+p₃** with a
rounding after every add. That association does **not** change with rows:
the reduced axis is always size 4; more rows only add threadgroups /
non-column reductions. Round 1's boundary emulation already proved the
point from the candidate side: all 120 pure association trees of the four
products give ≤1.0 ULP at 8 rows with zero matches to the grader
fingerprint, while the FMA-chain evaluation reproduces it exactly
(`muse-spark/implementation/hyp-3-fused-combine.md`, *The 2.0-ULP touch*).

### D. The candidate's arithmetic (`block-0.metal:20-25`)

The source writes the same left fold, `p₀ + p₁ + p₂ + p₃` as one f32
expression, then `bfloat(acc) + shared[sbase]`. Compiled as a single
Metal kernel (MLX `mx.fast.metal_kernel`, `slots.py:309-316`; no compiler
flags are expressible in `spec.json`), the mul+adds are FP-contracted
into FMAs: `fma(y_j, w_j, partial)` keeps the product unrounded, so `acc`
can differ from the reference fold by ~1 f32 ULP. The bf16 rounding of
`acc` absorbs that almost always; it flips only when `acc` lies within
the delta of a bf16 rounding midpoint, yielding a 1-ULP-different sum,
which the bf16 `+ shared` then preserves — or magnifies in ULP terms
when `shared ≈ −sum` cancels toward zero (the grader floors the ULP at
`|r| = 0.001`, `check_candidate.py:84-99`, so near-zero results report
large ULP gaps from tiny absolute differences). Attempt-2's explicit
`acc += product` loop shows the identical fingerprint (0.0 / 2.0 / 1.0),
so loop form and staging are exonerated too: only suppressing
contraction (e.g. the `precise` accumulator round 1's boundary named)
while keeping the ((p₀+p₁)+p₂)+p₃ fold can close it. Counts under the
new grader (`results/runs/2026-11-04-muse-round-1/grade-exact-combine.json`,
summed over 20 layers; totals = rows·768·20 elements):

| Size | Differ | Max ULP | Elements | Rate |
|---|---|---|---|---|
| decode 1 row | 0 | 0.0 | 15,360 | 0 |
| decode 8 rows | 2 | 2.0 | 122,880 | 16 ppm |
| decode 32 rows | 2 | 1.0 | 491,520 | 4 ppm |
| prefill 127 rows | 6 | 1.0 | 1,950,720 | 3 ppm |
| prefill 1,016 rows | 50 | 16.0 | 15,605,760 | 3 ppm |
| prefill 2,048 rows | 130 | 32.0 | 31,457,280 | 4 ppm |

The count grows with the element count at a roughly constant few-ppm
rate — more samples of the same rounding-boundary lottery, **not** a
kernel switch at prefill sizes (combine's MLX kernels are identical at
every row count; the 60-row `gather_qvm → gather_qmm_rhs` switch belongs
to the experts slot and cannot enter here, since the grader feeds
captured `y`). The larger max ULP at prefill is the tail of the same
lottery plus near-zero cancellation after `+ shared`. It fails G2 at
five of six sizes; in the model the round-1 kernel changed 7 of 32
vLLM sequences from token 0 (`claude/findings/muse-round-1.md` §2).
Folded in from explorer-grader's note below: the fix must hold at
2,048 rows, not only at decode sizes (the per-element sum is
row-count-independent, so a correct fix scales by construction, but
only the six graded sizes certify it), and a fix that is slower at
prefill sizes would be invisible to `beats_compiled`, which times
decode sizes only (`check_candidate.py:311`).

### E. Micro-path of the candidate body (bounded, not posted)

Each thread does one `idx/768` integer divide, 4 scalar bf16 loads
strided 768 elements apart (coalesced within each expert plane), the
same 4 weights re-loaded by all 768 threads of a row (cache-broadcast),
and scalar shared/out accesses. Round 1's attempt-2 (vectorized u64
loads, staged weights/shared, one barrier, 192-thread threadgroup/row)
measured 17.9 / 16.1 / 18.1 µs/layer (`muse-spark/kernels/combine/attempt-2/notes.md`)
— the same floor and the same 8→32-row slope (+1.5–2 µs for 4× work,
against +1.1 µs predicted from traffic alone). Vectorizing or staging
therefore buys nothing measurable; the slope is memory latency at
growing grid size, and the divide/loads sit under the dispatch floor.
No entry: expected saving <1 µs/layer, inside run-to-run spread.

## Relations

- **Fed by:** `baseline-experts` (produces `y`; its prefill rounding
  changes `y`'s values but never combine's arithmetic — the grader feeds
  captured `y` either way), `baseline-routing` (produces `weights`; the
  combine part is graded on reference weights, so routing's exactness
  cannot fail or save this slot), the shared expert (outside the slots,
  produces `shared`), and `grader` (the six-size bit-exact G2 and the
  chained timing define what "identical" now means).
- **Feeds:** the residual add and the next layer in `MoE.__call__`; its
  out is the previous call's output the chained timing links through
  (`check_candidate.py:178-201`).

## Inefficiencies

- `ineff-baseline-combine-1` — FMA contraction vs the reference's
  separate mul/add kernels: fails bit-exact G2 at 5 of 6 sizes and
  forfeits the slot's whole saving until fixed.
- `ineff-baseline-combine-2` — the fused kernel still spends ~17
  µs/layer on the dispatch floor: overhead-bound, flat in rows, already
  banked against the compiled reference; no kernel-body change in
  round 1 moved it.

## Files read

- `muse-spark/kernels/combine/attempt-1/{block-0.metal,spec.json,notes.md}`
  — the starting point: grid, arithmetic, round-1 verdict and the
  ~17 µs floor lesson.
- `muse-spark/kernels/combine/attempt-2/{block-0.metal,notes.md}` —
  competing organization; identical timing floor and G2 fingerprint,
  exonerating staging/vectorization and the loop form.
- `bench/grader/slots.py:51-52,97-119,214-254,303-345` — `ref_combine`,
  the MoE split, prefill capture, and how a spec becomes kernels.
- `bench/grader/check_candidate.py:84-99,139-150,172-228,277-337` —
  bit-exact differencing, `check_combine`, the link/chained timing, and
  that timing runs only when G2 passes.
- MLX headers (installed 0.32.4) under
  `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/`:
  `copy.h` (`copy_v` casts), `binary.h` (`binary_g`/`binary_vv`),
  `binary_ops.h` (`Multiply`/`Add` are plain `x*y` / `x+y`),
  `reduction/reduce_col.h` (`col_reduce_small` fold), `reduction/ops.h`
  (`Sum` init 0, `a+b`).
- `claude/findings/muse-round-1.md` §1–§3, §6 and
  `results/runs/2026-11-04-muse-round-1/grade-combine.json` +
  `grade-exact-combine.json` — re-grade timings, exact-G2 differ counts,
  G3 alone-vs-compiled figures.
- `muse-spark/implementation/hyp-3-fused-combine.md` — round 1 boundary,
  both attempts, and the CPU-emulation proof that FMA contraction, not
  association, causes the fingerprint.
- `muse-spark/exploration/territory-other-kernels.md` checklist G and
  `muse-spark/exploration/ineff-model-2.md` — round 1's kernel inventory
  and knock-out costs for the 5-kernel reference chain.
- `results/runs/2026-11-01-breakdown/tables.md:140,335`,
  `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`,
  `claude/adr/0010-exact-at-every-size.md`, `claude/muse-agent-brief.md`
  (slots, judging), `claude/GLOSSARY.md` (kind terms).

## From other agents

explorer-grader, 2026-11-04T03:17:52Z — the grader side of your exactness gap, from `bench/grader/` (full map in my `territory-grader.md` §4–5). (1) The gate is `differ == 0` at **six** sizes (decode 1/8/32 + prefill 127/1,016/2,048): `differences` counts bitwise-unequal elements and `passes` requires zero ([check_candidate.py:90-99, 172-175](../../../bench/grader/check_candidate.py)); `max_ulp` is now diagnostic only. Your starting point's 2 differing elements at decode 8/32 rows and 130 at prefill 2,048 — all inside round 1's 2-ULP allowance — therefore already fail G2; an association/FMA fix must hold at 2,048 rows, not only at decode sizes (the per-element sum is row-count-independent, so a correct fix should scale, but the grader will only certify it at the six sizes). (2) In a `routing+combine` candidate the combine part is graded on the captured reference `y/weights/shared`, never the candidate routing part's output ([check_candidate.py:277-304](../../../bench/grader/check_candidate.py)) — a combine error cannot be cancelled by a compensating routing error, and vice versa. (3) Timing is chained and decode-only ([check_candidate.py:311](../../../bench/grader/check_candidate.py)); prefill slowness is invisible to `beats_compiled`.

explorer-g3, 2026-11-04T03:18:59Z — your attempt end-to-end, from `results/runs/2026-11-04-muse-round-1/` (my `territory-g3.md` pieces 5/6): yours is the only attempt that over-delivers in-step at 1 request (−0.514 ms vLLM vs a piece promise of −0.104 ms/step, `console-vllm.txt` vs `grade-combine.json`) and the slot's only clean G3 pass against both baselines, at 8 requests. The caution is compositional: adding routing+combine to the experts attempt costs +0.98/+2.40 ms (vLLM, 8/32 requests) where the two alone save 1.10/0.88 ms (`ineff-g3-1`) — a per-slot win may not survive being installed together, which no single-slot timing will show.
