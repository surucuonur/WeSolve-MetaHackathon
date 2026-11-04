# Hypothesis 1: experts with `min_rows` — hand 1 row to the compiled reference

- **Owner:** lead
- **Status:** dropped
- **Version:** 10
- **Updated:** 2026-11-04T21:49:10Z
- **Next:** builders grading

**Inefficiency:** the experts starting point loses `beats_compiled` on the 1-row decode size alone (96.9 vs 86.6 µs/layer) while winning at 8/32 rows by 69.3 / 328.7 µs/layer (round 3 grade; ranked item 1 in `../exploration/README.md`). Round 3's kernel fixes for 1 row bought nothing (vectorised scalar) or 2× worse (unfused), both refuted. Round 4's `min_rows` (ADR 0015) hands 1-row inputs to the compiled reference instead.

**Slot:** `experts`. **Baseline:** none. **Starting point:** `muse-spark/round-3/kernels/experts/attempt-1` (byte-identical round-2 attempt-7 kernels, `max_rows: 59`; G1 pass, G2 differ 0 at all six sizes, prefill `same` ×3 by hand-off).

**Directions:** (a) pure spec change — copy the starting point, add `min_rows: 2`, keep `max_rows: 59`; (b) same hand-off with `max_rows: 32`, handing the 33–59 band to the compiled reference too. Both keep every kernel byte and accumulation order; they differ only in which ungraded band runs the candidate.

## Boundary

(written by boundary-1; delta only — every line carries its source)

**Cited boundary, checked line by line.** Round 3's boundary for this slot is
[`muse-spark/round-3/implementation/hyp-2-experts-prefill.md`](../../round-3/implementation/hyp-2-experts-prefill.md)
*Boundary* (itself a delta on round 2's
[`muse-spark/round-2/implementation/hyp-1-experts-exact.md`](../../round-2/implementation/hyp-1-experts-exact.md)).
I checked each line against the starting point
`muse-spark/round-3/kernels/experts/attempt-1` (`block-0.metal`,
`block-1.metal`, `spec.json`) and against the current G2
(`claude/muse-agent-brief.md:173-175,185-193,211-218`;
`claude/adr/0015-sets-and-hand-off-below.md:12-13,17-23,37`;
`bench/grader/slots.py:330-345`; `bench/grader/check_candidate.py:253-271,361-400`).
Still holding, not rewritten below: input/output names, dtypes, layouts and
weight shapes/bytes at all six sizes; `y[r,j]` positional in `inds` order;
neighbour contracts; the four bfloat16 rounding points in kind; the two
reference arithmetics and the kernel switch at 60 rows; launch-spec surface;
GPU limits. Baseline is `none` — no slot has one this round, so the bar is
the compiled reference everywhere (this entry, *Slot*/*Starting point*;
`../exploration/README.md` *Starting point*). Only what changed follows:
round 4 adds a hand-off below a row count (`min_rows`), and handed-off sizes
are no longer timed.

### Inputs and outputs — unchanged

Shapes, dtypes and layouts are round 3's verbatim (`x` [rows,768] bfloat16,
`inds` [rows,4] uint32, the nine weight/scale/bias inputs, `y` [rows,4,768]
bfloat16 at 1/8/32 decode rows and 127/1,016/2,048 prefill rows;
`claude/muse-agent-brief.md:74-82`; `bench/grader/slots.py:33-36`). Rechecked
against the starting point: its `spec.json` input list is the slot's full 11
inputs and its output `y` is `["rows",4,768]` bfloat16; it reads `rows` from
`x_shape[0]` (`block-0.metal:22`), the same value the grader's dispatch reads
from the first slot input's `shape[0]` (`bench/grader/slots.py:342`) and
`build`'s `hands_off(rows)` receives. `min_rows`/`max_rows` add no input,
output, shape, dtype or layout.

### Before and after — unchanged

Neighbour code and contracts are round 3's verbatim (post-attention layernorm
→ `x`; router → `inds`, `weights`; `experts(x, inds)` → `y`; combine pairs
`y[r,j]` with `weights[r,j]` positionally;
`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`;
`bench/grader/slots.py:97-119`). The hand-off lives entirely inside
`slots.build`'s dispatch: on a handed-off call the compiled slot reference
receives the same named inputs and returns the same `y`
(`bench/grader/slots.py:336-345`), so nothing crosses a neighbour boundary.

### What must stay identical — changed: the hand-off is now two-sided

- **Bit-exactness unchanged.** G2 still requires `differ == 0` in all 20
  layers at all six sizes, handed-off sizes included
  (`claude/muse-agent-brief.md:175,185-191`;
  `bench/grader/check_candidate.py:166-182,349-351`); `max_ulp` stays
  diagnostic only. The two reference arithmetics are unchanged:
  `gather_qvm` below 60 rows (float32 dequant, strided-lane accumulation +
  `simd_sum`), `gather_qmm_rhs` at/above (bfloat16-rounded dequant tiles,
  `BK=32` MMA-order accumulation). Tie rule and pair order unchanged: this
  slot inherits routing's ids as given and keeps `y[r,j]` positional,
  because combine sums positionally (`bench/grader/slots.py:51-52,116`).
- **New: rows below `min_rows` run the compiled reference, not the kernels.**
  With `"min_rows"` in `spec.json` (int ≥ 1, bools rejected, at most
  `max_rows` when both are given), `slots.build` sends inputs with
  `rows < min_rows` to the slot's reference under `mx.compile`, exactly as
  `rows > max_rows` already did (`bench/grader/slots.py:330-339`;
  `claude/muse-agent-brief.md:174`;
  `claude/adr/0015-sets-and-hand-off-below.md:12,18-19`). G2 checks `run()`,
  so handed-off sizes exercise the compiled reference through the attempt's
  own dispatch; the compiled reference was bit-exact with the reference at
  all six sizes on 2026-11-04 (`bench/micro/compiled_exact.py`,
  `results/runs/2026-11-04-handoff-check/`).
- **`min_rows` must be exactly 2: only the 1-row decode size hands off.**
  `rows` is a row count ≥ 1, so `rows < 2` is exactly `rows == 1`. A decode
  step has 1–32 rows (`claude/muse-agent-brief.md:26`), hence among the
  graded decode sizes 1 row runs the compiled reference and 8/32 rows run
  the kernels; in serving, every 1-row decode step hands off and every
  2–32-row decode step runs the kernels. `min_rows: 1` would hand off
  nothing (no call has 0 rows) and leave the 1-row loss this hypothesis
  exists to remove; `min_rows ≥ 3` would also hand off decode counts
  2…`min_rows−1`, which neither direction in this entry proposes (both keep
  `min_rows: 2`, *Directions* (a)/(b)). The boundary therefore requires
  `min_rows == 2`.
- **With 1 request, this slot runs MLX's own code for the whole request.**
  With 1 request every decode step has 1 row
  (`claude/muse-agent-brief.md:174`;
  `claude/adr/0015-sets-and-hand-off-below.md:37`), so with `min_rows: 2`
  every `MoE.__call__` experts call in that workload hands off, in all 20
  layers and every step; the 1-request prefill (127 graded rows, and any
  chunk above `max_rows`) hands off too. The experts slot then contributes
  no speed change at 1 request — alone it is no faster than its compiled
  bar there, and it can pass G3 / the ready check only in a set whose
  other kernels are faster with 1 request
  (`claude/adr/0015-sets-and-hand-off-below.md:37`;
  `claude/muse-agent-brief.md:174,206`). This is a whole-request (G3)
  fact: the piece timing cannot show it beyond the 1-row `handed off`
  verdict below.
- **The top hand-off still sits below MLX's switch: `max_rows: 59`.**
  MLX changes its own experts kernel at 60 rows — `gather_qvm` below,
  `gather_qmm_rhs` at/above (`bench/grader/slots.py:219-220`;
  `claude/muse-agent-brief.md:189`). With direction (a), `min_rows: 2` +
  `max_rows: 59`, the kernels run only on **2–59 rows**: 59 is the last
  `gather_qvm` row count, and the first handed-off count above, 60, is
  exactly MLX's first `gather_qmm_rhs` count, taken by the compiled
  reference. The hand-off therefore sits on the `qvm` side of the switch:
  no kernel-run row ever needs `qmm_rhs` arithmetic, and the starting
  point's in-kernel dispatch `x_shape[0] >= 60` (`block-0.metal:22`) and
  its scalar `qmm_accum` prefill branch are unreachable wherever the
  kernels run — dead code kept only as fallback. Direction (b),
  `max_rows: 32`, is also inside this boundary: the kernels then run
  2–32 rows, and the 33–59 band hands off to the compiled reference on
  the same `qvm` side of the switch. The valid window is therefore
  **32 ≤ `max_rows` ≤ 59** with `min_rows: 2`: `max_rows ≥ 32` keeps every
  decode step from 2 to 32 rows on the fused path (1 row is the sole,
  intended exception), and `max_rows ≤ 59` keeps the kernels off the
  `qmm_rhs` side entirely. Above `max_rows` and below `min_rows` the
  attempt runs the compiled bar itself, so it can never be better there —
  its claim is "faster at 8/32-row decode, unchanged everywhere else"
  (`claude/adr/0011-decode-and-prefill.md:35,48`, extended below by
  `claude/adr/0015-sets-and-hand-off-below.md:13`).
- **Kernel arithmetic for the rows that still run is unchanged.** Every
  kernel-run count (2–59 in (a), 2–32 in (b)) is below 60, so only the
  `gather_qvm` arithmetic must stay bit-exact there: float32 dequant,
  strided-lane accumulation + `simd_sum`, the four bfloat16 rounding
  points, positional `y`. Both directions keep every kernel byte and
  accumulation order of the starting point (*Directions* (a)/(b)); a
  kernel change is outside this hypothesis.

### What the grader checks, and what it can't see — changed: handed-off sizes aren't timed

Checks (`bench/grader/check_candidate.py`): G1/G2 mechanics unchanged — G1
is true only after the whole six-size correctness loop completes without
exception (`:349`); per-layer shape/dtype faults fail G2 with G1 true
(`:171-172,330-333`). G2 still checks every size bit for bit, handed-off
sizes included (`claude/muse-agent-brief.md:175`). **Timing change (round
4):** a size the spec hands off is not timed against the bar — it takes the
compiled reference's times and its verdict is `handed off`
(`claude/muse-agent-brief.md:216`;
`claude/adr/0015-sets-and-hand-off-below.md:13,21-23`;
`bench/grader/check_candidate.py:376-394`). For this hypothesis that means:
1-row decode and all three prefill sizes (127/1,016/2,048 > 59) report
`handed off` — round 3's prefill `same` ×3 wording is superseded, and the
starting point's 1-row `slower` verdict disappears as `handed off`, not as
a win. Decode's phase verdict is then judged on 8 and 32 rows alone, both
of which must be `faster` (a phase is `faster` when every size the kernels
run is, with at least one such size); prefill's phase verdict is
`handed off`; `beats_compiled` is true iff one phase is `faster` and
neither is `slower` (`claude/muse-agent-brief.md:217-218`;
`bench/grader/check_candidate.py:253-271`). No `--baseline` bar exists this
round (no baselines; `../exploration/README.md` *Starting point*).
What it still can't see (checked by hand later — *check an attempt*; this
is the start of this entry's **Unverified** list):
- **The exact hand-off bounds.** At the six graded sizes, any
  `min_rows` in 2–8 hands off only the 1-row size (8 rows runs the kernels
  for all of them), and any `max_rows` in 32–59 runs 8/32 and hands off
  all prefill — the grader cannot tell `min_rows: 2` (right) from 3–8
  (which silently hand off serving decode counts 2–7 too), nor
  `max_rows: 59` (right side of 60) from 32 (also valid here) or from a
  value above 59 (wrong side: kernels would meet `gather_qmm_rhs` rows).
  Only reading `spec.json` (`min_rows == 2`, `32 ≤ max_rows ≤ 59`, ints,
  not bools) distinguishes them; `check_sizes.py`-style probing at
  2/59/60/61 rows is boundary tooling, not a gate.
- **Ungraded row counts.** Kernel speed and (for direction (a)) kernel
  correctness at decode counts 2–7 and 9–31 and at 33–59 rows are
  unmeasured by the piece timing and untested by G2; correctness there
  rests on the per-pair, rows-independent construction round 3 checked.
  Sizes above 2,048, prefill beyond the first chunk of seeded 128-token
  prompts, other lengths/distributions and mixed batches are untested.
- **The 1-request whole-request consequence.** The piece timing shows only
  `handed off` at 1 row; that this slot then runs MLX's own code for the
  whole 1-request workload, and can pass only inside a faster set, is
  visible only to the ready check / G3 (`check an attempt` cannot see it
  either — it is recorded here so the *Decision* does not claim a 1-request
  gain). The three-slot interaction at 8 requests is likewise untested by
  the piece timing (`../exploration/README.md` item 3).
- Unwritten threadgroup reads, thread-timing dependence, the 32 KB
  threadgroup limit below; NaN-vs-NaN counts as equal
  (`bench/grader/check_candidate.py:106`); 1-request whole-request noise
  in G3.

### The GPU limits — unchanged, plus the pipeline ceiling

At most 1,024 threads per threadgroup (count enforced,
`bench/grader/slots.py:349-350`; `claude/muse-agent-brief.md:100`) and
32 KB of threadgroup memory (not enforced by `build` — verify by hand from
the Metal source). Starting point for scale, unchanged by the hand-off:
512 threads/group (`spec.json`) and 3,840 B (`tg_x[768] + tg_gate/up/h[384]`
bfloat16, `block-0.metal:24-27`), valid at every row count the kernels can
still see (2–59). Carried invariant from round 3 (itself from round 2,
attempt-6 G1 fail → attempt-7 pass): the binding ceiling for this
512-thread fused kernel is the compiled pipeline's
`maxTotalThreadsPerThreadgroup`, not the 1,024 count — headroom is exactly
two distinct 32-float thread arrays (`p`, shared `q`,
`block-0.metal:42-43`); a third drops the compiled maximum below 512 and
fails G1 loudly. A pure `spec.json` hand-off change touches none of this.

## Builders

### builder-1-a — direction (a): pure spec change (`min_rows: 2`, `max_rows: 59`)

- **Attempt:** `muse-spark/round-4/kernels/experts/attempt-1` —
  `block-0.metal` / `block-1.metal` byte-identical to the starting
  point `muse-spark/round-3/kernels/experts/attempt-1` (MD5
  `a1e95fc49844f714afd1a1e377967dae` / `631e710f5dd343427667a2270a166e6d`,
  `diff` clean); `spec.json` = starting point's plus `"min_rows": 2`,
  `"max_rows": 59` kept. Kernels run 2–59 rows; 1 row and all prefill
  sizes hand off. Full detail in the attempt's `notes.md`.
- **Predicted µs/layer** (from the starting point's grade run;
  handed-off = that run's compiled, kernel-run = its candidate):
  1 row 86.6 `handed off`; 8 rows 168.9 `faster` (saving
  238.2 − 168.9 = 69.3); 32 rows 554.7 `faster` (883.4 − 554.7 =
  328.7); prefill 127 / 1,016 / 2,048: 1,131.1 / 4,133.5 / 7,557.8
  `handed off`. G1 pass, G2 differ 0, `beats_compiled` true.
- **Measured** (`grade.json` in the attempt folder; candidate vs
  reference vs compiled): **G1 pass; G2 pass, differ 0 /
  max_ulp 0.0 at all six sizes, no first difference.** 1 row
  73.9 / 76.2 / 73.9 (`handed off`); 8 rows 157.6 / 233.0 / 232.1
  (`faster`, 30 wins); 32 rows 571.1 / 842.5 / 832.6 (`faster`,
  30 wins); prefill 127: 1,026.6 / 1,016.0 / 1,026.6 (`handed off`);
  1,016: 4,013.4 / 4,070.0 / 4,013.4 (`handed off`); 2,048:
  7,412.5 / 7,452.5 / 7,412.5 (`handed off`).
  `phases_vs_compiled`: decode `faster`, prefill `handed off`.
  **`beats_compiled` true.** Baseline `none` — no `beats_baseline`.
  Vs the starting point's run: the 1-row `slower` (96.9 vs 86.6,
  5 wins) is now `handed off`; 8/32-row margins this run
  74.5 / 261.5 µs/layer (232.1 − 157.6, 832.6 − 571.1).
- **Miss:** none over 20% — measured vs predicted −14.7% (1 row),
  −6.7% (8), +3.0% (32), −9.2% / −2.9% / −1.9% (prefill). Level
  shift is run noise (compiled bar moved 86.6 → 73.9 at 1 row
  between runs); handed-off sizes equal compiled exactly by
  construction; kernel sizes run byte-identical kernels.
- **Taught:** the spec change alone flips `beats_compiled`
  false → true with no kernel change (the hand-off's own timing
  cost is not isolated by any measurement; the two runs differ by
  a level shift), on the already-measured 8/32-row wins. Unproven remains whole-request: with 1 request
  this slot runs MLX's own code everywhere, so G3 passes only
  inside a faster set (see **Unverified**).

### builder-1-b — direction (b): same 1-row hand-off with narrower kernel window (`min_rows: 2`, `max_rows: 32`)

- **Attempt:** `muse-spark/round-4/kernels/experts/attempt-2` —
  `block-0.metal` / `block-1.metal` byte-identical to the starting
  point `muse-spark/round-3/kernels/experts/attempt-1` (MD5
  `a1e95fc49844f714afd1a1e377967dae` / `631e710f5dd343427667a2270a166e6d`,
  `diff` clean); `spec.json` = starting point's plus `"min_rows": 2`
  with `"max_rows"` narrowed 59 → 32. Kernels run 2–32 rows; 1 row,
  the ungraded 33–59 band and all prefill sizes hand off. Full
  detail in the attempt's `notes.md`.
- **Predicted µs/layer** (from the starting point's grade run;
  handed-off = that run's compiled, kernel-run = its candidate):
  1 row 86.6 `handed off`; 8 rows 168.9 `faster` (saving
  238.2 − 168.9 = 69.3); 32 rows 554.7 `faster` (883.4 − 554.7 =
  328.7); prefill 127 / 1,016 / 2,048: 1,131.1 / 4,133.5 / 7,557.8
  `handed off`. G1 pass, G2 differ 0, `beats_compiled` true.
- **Measured** (`grade.json` in the attempt folder; candidate vs
  reference vs compiled): **G1 pass; G2 pass, differ 0 /
  max_ulp 0.0 at all six sizes, no first difference.** 1 row
  76.1 / 79.2 / 76.1 (`handed off`); 8 rows 170.5 / 237.2 / 236.8
  (`faster`, 30 wins); 32 rows 538.1 / 809.1 / 823.2 (`faster`,
  30 wins); prefill 127: 991.6 / 1,043.8 / 991.6 (`handed off`);
  1,016: 3,998.9 / 4,090.0 / 3,998.9 (`handed off`); 2,048:
  7,441.5 / 7,440.4 / 7,441.5 (`handed off`).
  `phases_vs_compiled`: decode `faster`, prefill `handed off`.
  **`beats_compiled` true.** Baseline `none` — no `beats_baseline`.
  Margins this run 66.3 / 285.1 µs/layer (236.8 − 170.5,
  823.2 − 538.1); graded verdicts match direction (a)'s
  (attempt-1) in kind, as predicted — the 33–59 difference is
  ungraded.
- **Miss:** none over 20% — measured vs predicted −12.1% (1 row),
  +0.9% (8), −3.0% (32), −12.3% / −3.3% / −1.5% (prefill). Level
  is run noise (compiled bar 76.1 / 236.8 / 823.2 this run vs
  86.6 / 238.2 / 883.4 in the starting point's run); handed-off
  sizes equal compiled exactly by construction; kernel sizes run
  byte-identical kernels.
- **Taught:** narrowing the window to 2–32 rows matches direction
  (a)'s verdict at every graded size (any timing difference is
  inside the separate runs' noise, not measured as zero) and flips `beats_compiled` false → true exactly
  as direction (a) does; the 33–59 band's hand-off cost stays
  unmeasured by the piece timing. Unproven remains whole-request:
  with 1 request this slot runs MLX's own code everywhere, so G3
  passes only inside a faster set (see **Unverified**).

## Boundary checks

### attempt-1 — pass (boundary-1, 2026-11-04T21:54:05Z)

G1+G2 pass (`grade.json`: differ 0 / max_ulp 0.0 at all six sizes; `beats_compiled` true, decode `faster`, prefill `handed off`). Kernels verified byte-identical to the starting point `muse-spark/round-3/kernels/experts/attempt-1` (`diff` clean; MD5 `block-0.metal` `a1e95fc49844f714afd1a1e377967dae`, `block-1.metal` `631e710f5dd343427667a2270a166e6d`); `spec.json` differs by one added key `"min_rows": 2` (`spec.json:50`), `"max_rows": 59` kept (`spec.json:46`), both ints (not bools), `min_rows <= max_rows`, accepted by `bench/grader/slots.py:330-339`.

- **Row counts other than the graded ones, each side of 60.** `hands_off(rows)` is `(rows > 59) or (rows < 2)` (`bench/grader/slots.py:338-339`), dispatch on `inputs[slot_in[0]].shape[0]` (`:342`). Kernel-visible rows are exactly 2–59: every decode count 2–32 runs the kernels (grid `rows*2048`, `spec.json:5-9`; 512 threads/group, `:39-43`), and any count 60+ — prefill chunks of 60, 61, any size above 2,048 — hands off to the compiled reference (`bench/grader/slots.py:343-345`). 59 is the last `gather_qvm` count and the kernel's own dispatch `prefill = x_shape[0] >= 60` (`block-0.metal:22`) is false for every kernel-visible count, so the scalar `qmm_accum` prefill branch (`block-1.metal:206-230`; `block-0.metal:59-71,103-112`) is unreachable dead fallback; 60 is the first handed-off count and exactly MLX's first `gather_qmm_rhs` count. A prefill chunk of 33–59 rows runs the kernels on the same `qvm` arithmetic the reference uses below 60. Kernel work is per-pair with no cross-row interaction (`r = pair >> 2`, `e = inds[pair]`, `block-0.metal:20-21`), so the 8/32-row G2 verdict extends by construction to decode 2–7, 9–31 and to 33–59; speed at those ungraded kernel-run counts (2–7, 9–31, 33–59) is unmeasured by the piece timing — noted, not a defect.
- **Hand-offs (`min_rows: 2`, `max_rows: 59`).** `rows < 2` is exactly `rows == 1` (`rows` ≥ 1): 1-row decode runs the compiled reference, so with 1 request every decode step (1 row per step, all 20 layers, every step) runs MLX's own code for the whole request, and the 1-request prefill above 59 rows hands off too — this slot contributes no speed change at 1 request and can pass G3 only inside a faster set (whole-request fact; the piece timing shows it only as `handed off` at 1 row, `grade.json` 73.9 = compiled 73.9). Decode up to 32 stays on the kernel except the intended 1-row exception: 2–32 all run the fused path, `max_rows: 59` keeps the top hand-off below MLX's 60-row switch (`gather_qvm` below, `gather_qmm_rhs` at/above), so no kernel-run row ever needs `qmm` arithmetic. Handed-off sizes equal compiled exactly by construction (1,026.6 / 4,013.4 / 7,412.5 = compiled at 127 / 1,016 / 2,048 rows).
- **Unwritten threadgroup reads: none.** `tg_x[768]` is fully covered by the strided loop (`block-0.metal:29-31`, 512 threads cover 0–767) before the barrier at `:32`; `tg_gate`/`tg_up[384]` are fully covered in both branches by the 12 active simdgroups writing distinct `b*32+lane` (decode `:51-58`, prefill `:61-71`) before the barrier at `:73`; `tg_h[384]` is fully covered by the strided silu loop (`:77-85`) before the barrier at `:86`. Every read follows its phase's barrier; idle lanes/simdgroups write nothing that is read unwritten.
- **Threadgroup memory far from the limit.** 3,840 B (`tg_x[768] + tg_gate[384] + tg_up[384] + tg_h[384]` bfloat16, `block-0.metal:24-27`) vs the 32 KB ceiling (~11.7%); 512 threads/group vs the 1,024 count, and the inherited pipeline ceiling is unchanged — exactly two distinct 32-float thread arrays `p`, `q` (`block-0.metal:42-43`), so the round-2 attempt-6 G1 failure mode (a third array dropping the compiled maximum below 512) does not recur.
- **No thread-timing dependence.** All cross-thread sharing passes through unconditional `threadgroup_barrier`s (`block-0.metal:32,73,86` — outside every divergent branch, so all threads converge); `y` slices are disjoint per (pair, simdgroup, lane) (`:100-101,108-109`); no atomics (`"atomic_outputs": false`, `spec.json:4`); `simd_sum` reductions are fixed trees (`block-1.metal:120-127,187-194`). The shared kernel-scope `q[32]` is thread-private storage, and each helper re-zeroes both accumulators on entry (`block-1.metal:85-88,152-155`), so the gate/up → down reuse is timing-independent.
- **No captured-input assumptions.** The attempt adds no input-dependent logic beyond `rows` read from `x_shape[0]` (`block-0.metal:22`), the same value the grader's dispatch reads (`bench/grader/slots.py:342`) and `hands_off` receives. `inds` flows through untouched and `y[r,j]` stays positional in `inds` order; expert ids are used only as offsets (`eW`/`eS`, `block-0.metal:34-35`), with no assumption that ids are distinct, sorted or drawn from the captured distribution, and no branching on `x` values, expert ids, or any fixed row count (1/8/32 appear nowhere in the kernels).

### attempt-2 — pass (boundary-1, 2026-11-04T21:55:06Z)

G1+G2 pass (`grade.json`: differ 0 / max_ulp 0.0 at all six sizes; `beats_compiled` true, decode `faster`, prefill `handed off`). Kernels verified byte-identical to the starting point `muse-spark/round-3/kernels/experts/attempt-1` (`diff` clean; MD5 `block-0.metal` `a1e95fc49844f714afd1a1e377967dae`, `block-1.metal` `631e710f5dd343427667a2270a166e6d`); `spec.json` is the starting point's plus `"min_rows": 2` (`spec.json:50`) with `"max_rows"` narrowed 59 → 32 (`spec.json:46`), both ints (not bools), `min_rows <= max_rows`, accepted by `bench/grader/slots.py:330-339`.

- **Row counts other than the graded ones, each side of 60.** `hands_off(rows)` is `(rows > 32) or (rows < 2)` (`bench/grader/slots.py:339`), dispatch on `inputs[slot_in[0]].shape[0]` (`:342`). Kernel-visible rows are exactly 2–32: every decode count 2–32 runs the kernels (grid `rows*2048`, `spec.json:5-9`; 512 threads/group, `:39-43`), and any count 33+ — prefill chunks of 33, 59, 60, 61, any size above 2,048 — hands off to the compiled reference (`bench/grader/slots.py:343-345`). In particular the ungraded 33–59 band, which attempt-1 keeps on the kernels, hands off here too; that is the only behavioural difference between the two directions and no graded size can see it. Both handed-off neighbours of the 60-row switch are taken by the compiled reference: 59 is the last `gather_qvm` count and 60 is exactly MLX's first `gather_qmm_rhs` count (`gather_qvm` below 60, `gather_qmm_rhs` at/above). The kernel's own dispatch `prefill = x_shape[0] >= 60` (`block-0.metal:22`) is false for every kernel-visible count, so the scalar `qmm_accum` prefill branch (`block-1.metal:206-230`; `block-0.metal:59-71,103-112`) is unreachable dead fallback, and no kernel-run row ever needs `qmm_rhs` arithmetic. Kernel work is per-pair with no cross-row interaction (`r = pair >> 2`, `e = inds[pair]`, `block-0.metal:20-21`), so the 8/32-row G2 verdict extends by construction to decode 2–7 and 9–31; correctness at 33–59 is the compiled reference's. Speed at those ungraded kernel-run counts (2–7, 9–31) and the 33–59 hand-off cost are unmeasured by the piece timing — noted, not a defect.
- **Hand-offs (`min_rows: 2`, `max_rows: 32`).** `rows < 2` is exactly `rows == 1` (`rows` ≥ 1): 1-row decode runs the compiled reference, so with 1 request every decode step (1 row per step, all 20 layers, every step) runs MLX's own code for the whole request, and the 1-request prefill above 32 rows hands off too — this slot contributes no speed change at 1 request and can pass G3 only inside a faster set (whole-request fact; the piece timing shows it only as `handed off` at 1 row, `grade.json` 76.1 = compiled 76.1). Decode up to 32 stays on the kernel except the intended 1-row exception: 2–32 all run the fused path, `max_rows: 32` is the bottom of the boundary's valid window 32 ≤ `max_rows` ≤ 59 and keeps the top hand-off below MLX's 60-row switch, with the first handed-off count above, 33, still on the `qvm` (`gather_qvm`) side taken by the compiled reference. Handed-off sizes equal compiled exactly by construction (76.1 / 991.6 / 3,998.9 / 7,441.5 = compiled at 1 / 127 / 1,016 / 2,048 rows).
- **Unwritten threadgroup reads: none.** `tg_x[768]` is fully covered by the strided loop (`block-0.metal:29-31`, 512 threads cover 0–767) before the barrier at `:32`; `tg_gate`/`tg_up[384]` are fully covered in both branches by the 12 active simdgroups writing distinct `b*32+lane` (decode `:51-58`, prefill `:61-71`) before the barrier at `:73`; `tg_h[384]` is fully covered by the strided silu loop (`:77-85`) before the barrier at `:86`. Every read follows its phase's barrier; idle lanes/simdgroups write nothing that is read unwritten.
- **Threadgroup memory far from the limit.** 3,840 B (`tg_x[768] + tg_gate[384] + tg_up[384] + tg_h[384]` bfloat16, `block-0.metal:24-27`) vs the 32 KB ceiling (~11.7%); 512 threads/group vs the 1,024 count, and the inherited pipeline ceiling is unchanged — exactly two distinct 32-float thread arrays `p`, `q` (`block-0.metal:42-43`), so the round-2 attempt-6 G1 failure mode (a third array dropping the compiled maximum below 512) does not recur.
- **No thread-timing dependence.** All cross-thread sharing passes through unconditional `threadgroup_barrier`s (`block-0.metal:32,73,86` — outside every divergent branch, so all threads converge); `y` slices are disjoint per (pair, simdgroup, lane) (`:100-101,108-109`); no atomics (`"atomic_outputs": false`, `spec.json:4`); `simd_sum` reductions are fixed trees (`block-1.metal:120-127,187-194`). The shared kernel-scope `q[32]` is thread-private storage, and each helper re-zeroes both accumulators on entry (`block-1.metal:85-88,152-155`), so the gate/up → down reuse is timing-independent.
- **No captured-input assumptions.** The attempt adds no input-dependent logic beyond `rows` read from `x_shape[0]` (`block-0.metal:22`), the same value the grader's dispatch reads (`bench/grader/slots.py:342`) and `hands_off` receives. `inds` flows through untouched and `y[r,j]` stays positional in `inds` order; expert ids are used only as offsets (`eW`/`eS`, `block-0.metal:34-35`), with no assumption that ids are distinct, sorted or drawn from the captured distribution, and no branching on `x` values, expert ids, or any fixed row count (1/8/32 appear nowhere in the kernels).

## Decision

**Best attempt: `muse-spark/round-4/kernels/experts/attempt-1`** (`min_rows: 2`, `max_rows: 59`, kernels byte-identical to the starting point). It passes G1, G2 (differ 0 at all six sizes) and its boundary check (`pass`), with `beats_compiled` true: decode `faster` at 8 and 32 rows (157.6 / 571.1 vs compiled 232.1 / 832.6 µs per layer, 30 wins each), 1-row and all prefill sizes `handed off`. Attempt-2 (`max_rows: 32`) also passes everything and is faster at 32 rows (538.1) but slower at 8 (170.5 vs 157.6); attempt-1 is picked because the 8-request workload is where the untested three-slot interaction (+2.13 to +2.86 ms/step, `ineff-g3-2`) bites, and because it keeps the starting point's full 2–59 kernel window. No hardening change is named: the kernels are unchanged by design and every kernel-level 1-row fix is refuted; the remaining cause (the 8/32-row chain headroom, `ineff-baseline-experts-2`) is a different hypothesis if the set fails.

**Predicted change in whole requests (PREDICTION):** this slot alone adds nothing with 1 request (1 row handed off — MLX's own code runs) and, additively from its same-run piece margins × 20 layers, about −1.49 ms per decode step at 8 requests and −5.23 ms at 32 requests against the compiled reference (`ineff-g3-2.md` arithmetic); prompts unchanged (handed off). The ready check replaces this prediction with a measurement; the three-slot interaction may take some of it back at 8 requests.

**Ready check (measured, whole request):** `../checks/ready-1/ready-verdict.json` FAILS this attempt alone and in the three-slot set (`muse-all` fail vs base and vs compiled-all; every phase `not slower` — range overlaps, same tokens everywhere). Decode medians vs base +0.06 / −1.57 / −4.02 / −5.09 ms at 1×128 / 8×128 / 32×128 / 32×1024: alone it adds nothing at 1 request exactly as predicted (1 row handed off), and the set it could hide in is slower at 8×128 (+0.32 ms/step, the three-slot interaction). Against the PREDICTION above (−1.49 at 8, −5.23 at 32 vs compiled): the 8-request decode gain measured −1.57 vs base, the 32-request gains −4.02 / −5.09 — the kernel's speed is real; the verdict is spread plus the set's 8-request loss. **Dropped:** no ready check remains (two used), and no kernel change addresses 5-repeat spread. Slot keeps the reference.

## Failure ledger

## Failure ledger

No graded attempt lost: attempts 1 and 2 both pass G1, G2 and `beats_compiled`. Nothing to harden against.

## Unverified

- The exact hand-off bounds: the grader cannot tell `min_rows: 2` from 3–8 (which would hand off serving decode counts 2–7) nor grade the bounds themselves; only reading `spec.json` (`min_rows: 2`, `max_rows: 59`, plain ints) distinguishes them. Boundary check verified both by reading the spec.
- Kernel speed at ungraded decode counts 2–7 and 9–31 and (attempt-1 only) at 33–59 rows is unmeasured; correctness there rests on the per-pair, rows-independent construction round 3 checked, not on G2.
- Sizes above 2,048 rows, prefill beyond the first seeded chunk, other prompt lengths/distributions and mixed batches are untested.
- The 1-request consequence: with 1 request this slot runs MLX's own code for the whole workload (1-row hand-off), so it adds nothing there and the set passes at 1×128 only on routing+combine. Only the ready check / G3 measures this.
- The three-slot interaction at 8 requests (+2.13 to +2.86 ms/step in rounds 1–2) is untested by the piece timing; only the ready check's `muse-all` measures it with these kernels.
- Unwritten threadgroup reads and thread-timing dependence: checked by hand in the boundary check (pass), not by the grader. Threadgroup memory 3,840 B against the 32 KB limit, counted by hand. NaN-vs-NaN counts as equal in G2. Whole-request noise in the ready check / G3 at 1 request (5-repeat ranges) is itself unmeasured until the ready check runs.

## Checks

- **checker-1, 2026-11-04T22:02:20Z — check: fail** (defects below; do not close as `candidate` until fixed):
  1. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:330-332` — **Unverified** is only the placeholder “(starts from the boundary's can't-see list.)” and covers none of the boundary's can't-see list at `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:164-192`: exact hand-off bounds indistinguishable at graded sizes (`min_rows` 2–8, `max_rows` 32–59; only reading `spec.json` distinguishes them), ungraded row counts (kernel speed/correctness at decode 2–7 and 9–31 and, for direction (a), 33–59; sizes above 2,048, prefill beyond the first chunk, other lengths/distributions and mixed batches), the 1-request whole-request consequence (visible only to the ready check / G3; this slot can pass only inside a faster set), the three-slot interaction at 8 requests, and unwritten threadgroup reads / thread-timing dependence / the 32 KB limit, NaN-vs-NaN equality and 1-request G3 noise. Mentions scattered in builder/boundary notes do not substitute for the **Unverified** list.
  2. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:245-247` — a cost is called zero without a measurement: “at zero kernel cost”. Byte-identity proves no kernel change, not zero timing cost; attempt-1 and the starting point were graded in separate runs with an acknowledged level shift (compiled bar 86.6 → 73.9 µs/layer at 1 row, `:242-243`), so no measurement isolates the spec/hand-off cost as zero.
  3. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:287-288` — a cost is called zero without a measurement: narrowing to 2–32 rows “costs nothing at any graded size”. Attempts 1 and 2 were graded in separate runs (compiled 232.1 / 832.6 vs 236.8 / 823.2 µs/layer; candidate 157.6 vs 170.5 µs/layer at 8 rows), so only the verdict kind matching (`:278-280`) is measured, not a zero difference. (The 33–59 hand-off cost is correctly left unmeasured at `:289-290`.)
- **Verified (no defect):** every measured piece number was found in its cited `grade.json` at the named size/key and the arithmetic redone — attempt-1 (`muse-spark/round-4/kernels/experts/attempt-1/grade.json`): G1/G2 pass, differ 0 / max_ulp 0.0 at all six sizes; candidate/reference/compiled 73.9/76.2/73.9 handed off, 157.6/233.0/232.1 faster (30 wins), 571.1/842.5/832.6 faster (30 wins), prefill 1026.6/1016.0/1026.6, 4013.4/4070.0/4013.4, 7412.5/7452.5/7412.5 handed off; margins 232.1−157.6=74.5 and 832.6−571.1=261.5 µs/layer; prediction misses −14.7/−6.7/+3.0/−9.2/−2.9/−1.9% (none over 20%). Attempt-2 (`muse-spark/round-4/kernels/experts/attempt-2/grade.json`): 76.1/79.2/76.1 handed off, 170.5/237.2/236.8 and 538.1/809.1/823.2 faster (30 wins each), prefill 991.6/1043.8/991.6, 3998.9/4090.0/3998.9, 7441.5/7440.4/7441.5 handed off; margins 236.8−170.5=66.3 and 823.2−538.1=285.1 µs/layer; misses −12.1/+0.9/−3.0/−12.3/−3.3/−1.5%. Boundary-check equalities (handed-off candidate = compiled) match both grade files. Predictions are labelled with arithmetic counting only this slot's own measured effect (builder predictions 238.2−168.9=69.3, 883.4−554.7=328.7 µs/layer; Decision `:322` is labelled PREDICTION: same-run margins × 20 layers = 74.5×20=1.49 ms and 261.5×20=5.23 ms per decode step); no combination gain is summed from slots (the +2.13 to +2.86 ms/step three-slot term at `:320` is quoted as untested historical context, not added to this slot's gain). Failure ledger (`:324-328`) recounts correctly: neither grade file lost — both attempts pass G1, G2 and `beats_compiled` with no `slower` size or phase, so there is nothing to harden against.
- **checker-1, 2026-11-04T22:03:48Z — re-check: check: fail** (Unverified placeholder fixed; defects below remain; do not close as `candidate` until fixed):
  1. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:245-246` — a cost is still called zero without a measurement: “at zero kernel cost”. Unchanged since the first check. Byte-identity proves no kernel change, not zero timing cost; attempt-1 and the starting point were graded in separate runs with an acknowledged level shift (compiled bar 86.6 → 73.9 µs/layer at 1 row, `:242-243`), so no measurement isolates the spec/hand-off cost as zero.
  2. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:287-288` — a cost is still called zero without a measurement: narrowing to 2–32 rows “costs nothing at any graded size”. Unchanged since the first check. Attempts 1 and 2 were graded in separate runs (compiled 232.1 / 832.6 vs 236.8 / 823.2 µs/layer; candidate 157.6 vs 170.5 µs/layer at 8 rows), so only the verdict-kind matching (`:278-280`) is measured, not a zero difference. (The 33–59 hand-off cost is correctly left unmeasured at `:289-290`.)
  3. `muse-spark/round-4/implementation/hyp-1-experts-minrows.md:330-337` — **Unverified** now covers the boundary's can't-see list (`:164-192`) except one sub-item: 1-request whole-request noise in G3 (`:191-192`) is named nowhere in `:330-337`. `:335` says only the ready check / G3 measures the 1-request consequence; it does not list G3 noise itself as unmeasured.
- **Re-verified (no defect):** Unverified fix verified — `:332` exact hand-off bounds (`min_rows: 2` vs 3–8 handing off 2–7, bounds graded only by reading `spec.json`; specs re-read: attempt-1 `min_rows: 2`/`max_rows: 59`, attempt-2 `min_rows: 2`/`max_rows: 32`, plain ints), `:333` kernel speed/correctness at ungraded 2–7, 9–31 and (attempt-1) 33–59 resting on construction not G2, `:334` sizes above 2,048 / prefill beyond the first seeded chunk / other lengths, distributions and mixed batches, `:335` 1-request whole-request consequence (slot runs MLX code, passes only inside a faster set, ready check / G3 only), `:336` three-slot interaction at 8 requests (`muse-all` only), `:337` unwritten threadgroup reads / thread-timing dependence checked by hand, 3,840 B vs 32 KB counted by hand, NaN-vs-NaN equality. Every measured piece number re-found in its cited `grade.json` at the named size/key and arithmetic redone — starting point (`muse-spark/round-3/kernels/experts/attempt-1/grade.json`): 96.9 vs 86.6 `slower` (5 wins), 168.9 vs 238.2 and 554.7 vs 883.4 `faster`; predictions 238.2−168.9=69.3, 883.4−554.7=328.7 µs/layer, handed-off prefill = compiled 1,131.1 / 4,133.5 / 7,557.8. Attempt-1 (`muse-spark/round-4/kernels/experts/attempt-1/grade.json`): G1/G2 pass, differ 0 / max_ulp 0.0 at all six sizes; 73.9/76.2/73.9 `handed off`, 157.6/233.0/232.1 `faster` (30 wins), 571.1/842.5/832.6 `faster` (30 wins), prefill 1,026.6/1,016.0/1,026.6, 4,013.4/4,070.0/4,013.4, 7,412.5/7,452.5/7,412.5 `handed off`; `phases_vs_compiled` decode `faster`, prefill `handed off`, `beats_compiled` true; margins 232.1−157.6=74.5, 832.6−571.1=261.5 µs/layer; misses −14.7/−6.7/+3.0/−9.2/−2.9/−1.9% (none over 20%). Attempt-2 (`muse-spark/round-4/kernels/experts/attempt-2/grade.json`): 76.1/79.2/76.1 `handed off`, 170.5/237.2/236.8 and 538.1/809.1/823.2 `faster` (30 wins each), prefill 991.6/1,043.8/991.6, 3,998.9/4,090.0/3,998.9, 7,441.5/7,440.4/7,441.5 `handed off`; margins 236.8−170.5=66.3, 823.2−538.1=285.1 µs/layer; misses −12.1/+0.9/−3.0/−12.3/−3.3/−1.5%. No measured whole-request claim cites a grader file, timing run or arithmetic (Decision `:322` is labelled PREDICTION, same-run margins × 20 layers = 74.5×20=1.49 ms and 261.5×20=5.23 ms per decode step, counting only this slot's own effect, ready check to replace it); no combination gain is summed from slots (the +2.13 to +2.86 ms/step three-slot term at `:320`/`:336` is untested historical context, not added). Failure ledger (`:324-328`) recount verified from both `grade.json` files: neither attempt lost (both G1 pass, G2 pass, `beats_compiled` true, no `slower` size or phase), so no approach-change entry is owed.
- **checker-1, 2026-11-04T22:04:34Z — re-check: check: pass** — all three defects from the 2026-11-04T22:03:48Z re-check are fixed and the full check (steps 1–6) was re-run. Fixes verified: (1) builder-1-a *Taught* (`:245-250`) no longer claims zero cost — it now states the hand-off's own timing cost is not isolated by any measurement and the two runs differ by a level shift; (2) builder-1-b *Taught* (`:288-294`) no longer claims narrowing costs nothing — it now states any timing difference is inside the separate runs' noise, not measured as zero (the 33–59 hand-off cost remains correctly unmeasured); (3) **Unverified** (`:339`) now lists the missing sub-item — whole-request noise in the ready check / G3 at 1 request (5-repeat ranges) is itself unmeasured until the ready check runs. Remaining zero/free scan: no builder or Decision text calls a cost free, neutral or zero without a measurement (the only other “unchanged” is the boundary's handed-off claim at `:131`, where handed-off sizes run the compiled bar itself and equal compiled exactly in both `grade.json` files). Full re-check, no defect: every measured piece number re-found in its cited `grade.json` at the named size/key with arithmetic redone — starting point (`muse-spark/round-3/kernels/experts/attempt-1/grade.json`): 96.9 vs 86.6 `slower` (5 wins), 168.9 vs 238.2 and 554.7 vs 883.4 `faster`, predictions 238.2−168.9=69.3 and 883.4−554.7=328.7 µs/layer, handed-off prefill = compiled 1,131.1 / 4,133.5 / 7,557.8; attempt-1 (`muse-spark/round-4/kernels/experts/attempt-1/grade.json`): G1/G2 pass, differ 0 / max_ulp 0.0 at all six sizes, 73.9/76.2/73.9 `handed off`, 157.6/233.0/232.1 `faster` (30 wins), 571.1/842.5/832.6 `faster` (30 wins), prefill 1,026.6/1,016.0/1,026.6, 4,013.4/4,070.0/4,013.4, 7,412.5/7,452.5/7,412.5 `handed off`, margins 232.1−157.6=74.5 and 832.6−571.1=261.5 µs/layer, misses −14.7/−6.7/+3.0/−9.2/−2.9/−1.9% (none over 20%); attempt-2 (`muse-spark/round-4/kernels/experts/attempt-2/grade.json`): 76.1/79.2/76.1 `handed off`, 170.5/237.2/236.8 and 538.1/809.1/823.2 `faster` (30 wins each), prefill 991.6/1,043.8/991.6, 3,998.9/4,090.0/3,998.9, 7,441.5/7,440.4/7,441.5 `handed off`, margins 236.8−170.5=66.3 and 823.2−538.1=285.1 µs/layer, misses −12.1/+0.9/−3.0/−12.3/−3.3/−1.5%; `spec.json` re-read (attempt-1 `min_rows: 2`/`max_rows: 59`, attempt-2 `min_rows: 2`/`max_rows: 32`, plain ints). No measured whole-request claim exists (no `ready-verdict.json` in round 4; Decision `:324` is labelled PREDICTION, same-run margins × 20 layers = 74.5×20=1.49 ms and 261.5×20=5.23 ms per decode step, counting only this slot's own effect, ready check to replace it), so no `check_ready.py --audit` was owed; no combination gain is summed from slots (the +2.13 to +2.86 ms/step three-slot term is untested historical context, not added). **Unverified** (`:334-339`) now covers the full boundary can't-see list (`:166-192`): exact hand-off bounds, ungraded row counts and sizes, the 1-request whole-request consequence, the three-slot interaction, unwritten threadgroup reads / thread-timing dependence / 32 KB limit, NaN-vs-NaN equality and 1-request G3 noise. Failure ledger (`:326-330`) recount verified from both `grade.json` files: neither attempt lost (both G1 pass, G2 pass, `beats_compiled` true, no `slower` size or phase), so no approach-change entry is owed. Eligible to close as `candidate`.
