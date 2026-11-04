# Hypothesis 1: experts with `min_rows` — hand 1 row to the compiled reference

- **Owner:** lead
- **Status:** building
- **Version:** 2
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

*(one section per builder, filled as attempts return.)*

## Boundary checks

*(boundary-1 appends here for each attempt that passed G2.)*

## Decision

*(lead writes at close.)*

## Failure ledger

*(lead keeps: each graded attempt that lost, the size or phase, its named cause, its approach.)*

## Unverified

*(starts from the boundary's can't-see list.)*

## Checks

*(checker-1 writes here before candidate.)*
