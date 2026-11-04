# Hypothesis 3: routing prefill at compiled speed

- **Owner:** lead
- **Status:** hardening
- **Version:** 7
- **Updated:** 2026-11-04T18:24:53Z
- **Next:** builder-3-f re-grades a byte-identical copy of attempt-1 (one run only — tests the identical-code-noise call at 2048 rows)

## Inefficiency

[ineff-baseline-routing-1.md](../exploration/ineff-baseline-routing-1.md) (confirmed, traffic: per-row `w_router` re-read, excess 116.1/935.4/1,886.5 µs/layer at 127/1,016/2,048 rows) plus [ineff-baseline-routing-2.md](../exploration/ineff-baseline-routing-2.md) (confirmed, shape: 24 barriers, 28 KB residency cap, 23% single-partition duty; residual 24.1/142.7/336.8 µs/layer). Together they partition the starting point's prefill gap exactly (measured − compiled = 139.8/1,077.7/2,222.6 µs/layer; 2.6/5.9/8.7× compiled), which makes prompts 7–10% slower and costs −3.5% whole-request throughput at 32×1,024. Slot: `routing`. Baseline: `none` (starting point `muse-spark/round-2/kernels/routing/attempt-4`, which passes G1+G2 and wins every decode size but is `slower` at every prefill size).

Predicted effect if the prefill path matches compiled speed at equal bits: save 2.80/21.6/44.5 ms per chunk (traffic 2.32/18.7/37.7 + shape 0.48/2.85/6.74), keeping the decode wins (15.3/22.2/23.9 µs/layer). Fix shapes: (a) `max_rows` hand-off (decode stays on the staged scalar path, prefill on the compiled reference — banks the prize at `same`); (b) GEMM-shaped kernel sharing `w_router` across rows (removes traffic and shape together, but round-1 tiled sharing was slower everywhere, so the design needs proof on the chained timing); (c) narrower staged footprint restoring residency (fewer barriers, fuller lanes in the single-partition regime). Every direction must keep the exact `rows`-branched accumulation orders and add no fixed cost at decode sizes.

## Boundary

*written by boundary-3, 2026-11-04T18:09:20Z — delta to round 2's boundary. Only what changed is written here; every round-2 line not listed below still holds verbatim.*

Round 2's boundary is [hyp-2-routing-exact.md](../../round-2/implementation/hyp-2-routing-exact.md) *Boundary* (`muse-spark/round-2/implementation/hyp-2-routing-exact.md:22-85`, itself a delta to round 1's `muse-spark/implementation/hyp-2-fused-routing.md:19-62`). I checked each of its lines against the starting point `muse-spark/round-2/kernels/routing/attempt-4` (`spec.json`; `block-0.metal`; `block-1.metal`) and against the current two-phase G2 (`claude/muse-agent-brief.md:26,100-101,173,184-194,195-202,209-213,217`; `claude/adr/0011-decode-and-prefill.md`; `bench/grader/check_candidate.py:51-58,115-144,180-183,224-264,308-378`; `bench/grader/slots.py:19,44-52,304-342`). Still holding, not rewritten below: input/output names, shapes, dtypes and layouts at all six sizes; the producer (`post_attention_layernorm(h)` → flat `[rows,768]`, `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232` byte-unchanged) and both consumers (`SwitchGLU` sort from `indices.size >= 64`, `third_party/mlx-lm/mlx_lm/models/switch_layers.py:181` unchanged; combine summing the 4 pairs in emitted order, `mobilemoe.py:231`, `bench/grader/slots.py:51-52` unchanged); the retired round-1 freedoms (1e-4 tolerance, unordered set equality, 1e-6 near-tie exemption, "order is free" — the brief's routing table still says order is free at `claude/muse-agent-brief.md:49`, but G2's ordered rule at brief:190 governs, as round 2 ruled); the bit-for-bit ordered G2 rule; `ref_routing` byte-unchanged (`bench/grader/slots.py:44-48`); the three accumulation regimes (GEMV at `rows == 1`, split-K for 1 < rows ≤ 768, regular steel GEMM above) and the scalar≡MMA identity the starting point proved; MLX's verbatim sigmoid; the bias rule, unbiased gather and left-fold 4-term sum with `+1e-20`, divide, `×2.5`; the strictly-greater tie rule and earliest-index-wins order; G1/G2 mechanics; the GPU limits. Baseline is `none` — no slot has one this round, so the bar is the compiled reference everywhere (`claude/adr/0011-decode-and-prefill.md:46`). Only what changed follows.

### Inputs and outputs (unchanged)

No change. The starting point's launch surface matches round 2's table exactly: inputs `x`, `w_router`, `bias`; outputs `inds` `[rows,4]` uint32, `weights` `[rows,4]` float32; grid `rows*256`, threadgroup 256, no `max_rows` (`spec.json`). `D, E, K = 768, 60, 4` (`bench/grader/slots.py:19`); shape/dtype enforcement moved to `bench/grader/check_candidate.py:115-122` (was :107) and `inds < 60` to :125-126 (was :117) — same semantics. Six graded sizes unchanged: decode 1/8/32, prefill 127/1,016/2,048 (`bench/grader/check_candidate.py:51-52,308-312`).

### Before and after (unchanged)

No change. Checklist delta per `.agents/skills/muse-round/tracking.md` *The checklist*: producer — holds (`mobilemoe.py:217-232`, `bench/grader/slots.py:97-119` unchanged; `x` still bf16, cast inside the reference); `SwitchGLU` consumer — holds (sort threshold `switch_layers.py:181` unchanged; order still load-bearing through G2's ordered comparison at `bench/grader/check_candidate.py:127-137`); combine consumer — holds (positional sum, bf16 cast outside the slot). The starting point emits reference order from its thread-0 scan (`block-0.metal:139-174`); any fix shape keeps that.

### What must stay identical — the exactness gates are closed; the fix must preserve them (changed)

- **The arithmetic contract is unchanged.** Ordered bitwise equality at all six sizes (`differ = (inds != ref_inds).any(axis=1) | (w != ref_w).any(axis=1)`, `rows_differ == 0` in all 20 layers, `bench/grader/check_candidate.py:128,180-183,344`; `max_rel_err` still diagnostic-only, :130-136); the three `rows`-branched accumulation regimes with their exact orders (round-2 table :64-68 still binds — a candidate branches on `rows`, never on the six graded values); the verbatim `1/(1+precise::exp(abs(s)))` sigmoid with the `(s<0)?y:1-y` branch; bias-add selection with unbiased gather; the `(((s0+s1)+s2)+s3)+1e-20`, divide, `×2.5` fold in emitted order; the strictly-greater tie rule with earliest index winning exact ties. `ref_routing` is byte-unchanged, so the required evaluation is identical to round 2's.
- **Starting point against this boundary (changed).** Round 2's starting point (round-1 attempt-1) failed G2 at every size on the product order and the sigmoid. The round-3 starting point `muse-spark/round-2/kernels/routing/attempt-4` **passes** it: G1 pass, G2 pass with 0 rows differing at all six sizes, chaining 47.5/69.1/72.9 µs/layer vs compiled 60.7/84.9/89.5 at decode (`hyp-2-routing-exact.md:175-179`) — and `slower` at every prefill size (2.6/5.9/8.7× compiled; this entry, *Inefficiency*). Its regime branches (`rows == 1` GEMV shuffle tree, `rows <= 768` split-K partitions from the `rows`-only formula, single accumulator above, `block-0.metal:35-74`), verbatim sigmoid (:134-135), strictly-greater top-4 (:142-159) and weights fold (:160-173) are therefore the decode behaviour to keep: decode rows stay on this winning staged scalar path, and every decode step up to 32 rows must stay at or below any `max_rows` the attempt names (see below).
- **New: rows above `max_rows` run the compiled reference, not the kernels.** With `"max_rows"` in `spec.json` (int ≥ 1, bools rejected), `slots.build` sends inputs with `rows > max_rows` to the slot's reference under `mx.compile` (`bench/grader/slots.py:327-337`; `claude/muse-agent-brief.md:173`; `claude/adr/0011-decode-and-prefill.md:33-35`). G2 checks `run()`, so sizes above `max_rows` exercise the compiled reference through the attempt's own dispatch; the compiled reference was bit-exact with the reference at all six sizes on 2026-11-04 (`bench/micro/compiled_exact.py`, `results/runs/2026-11-04-handoff-check/`). This replaces the ADR 0010 line that required the kernels to run in every call (`claude/adr/0011-decode-and-prefill.md:17`).
- **Decode stays on the candidate: `max_rows >= 32`.** A decode step has 1–32 rows (`claude/muse-agent-brief.md:26`); every decode step up to 32 rows stays at or below the hand-off, running the staged scalar path that already wins there. Below the hand-off the kernels face only the GEMV (`rows == 1`) and split-K (1 < rows ≤ 768) roundings.
- **No MLX switch at 60 rows for this slot — the window is 32 ≤ `max_rows` ≤ 126.** The 60-row kernel change (`gather_qmm_rhs` vs `gather_qvm`) is the *experts* slot's (`bench/grader/slots.py:219-220`; `claude/muse-agent-brief.md:187`); routing's reference regimes change at `rows = 1` and `rows = 768` only. So any `max_rows ≥ 32` keeps decode on the kernels and hands every graded prefill size (127/1,016/2,048) to compiled (grader note for this slot, `muse-spark/round-3/exploration/territory-grader.md:86`); `max_rows ≤ 126` hands all three graded prefill sizes off while `max_rows ≤ 768` additionally dead-ends the regular-GEMM branch wherever the kernels run. Kernel-visible counts 33–126 take the split-K path (partition counts from the same `rows`-only formula: 16×3 at 33–64, 8×6 at 65–128), already exercised in kind by the graded 8/32/127-row groupings. Above `max_rows` the attempt times `same` as the compiled bar by construction, never better — its claim is "faster decode, unchanged prefill" (`claude/adr/0011-decode-and-prefill.md:35,48`).
- **Consequence for the three fix shapes** (this entry, *Inefficiency*): (a) `max_rows` hand-off keeps the staged scalar decode path verbatim and deletes or dead-ends nothing it needs — the G2-exactness at kernel-visible counts is inherited, and the boundary check verifies the hand-off value, not the arithmetic; (b) GEMM-shaped sharing must reproduce the *same per-row regime orders* while sharing `w_router` across rows — sharing bytes is free, reordering any row's accumulation is a G2 failure; (c) a narrower staged footprint keeps the same orders with less staging. Every direction adds no fixed cost at decode sizes and keeps the tie rule and pair order bit-exact.

### What the grader checks, and what it can't see (changed)

- **G1/G2 (unchanged).** Same six sizes, same ordered `rows_differ == 0` rule in all 20 layers, same per-part isolation (`bench/grader/check_candidate.py:308-344`); G1 is true only after the whole six-size correctness loop completes without exception (:342); shape/dtype faults fail G2 with G1 true (:324-326). Size labels now carry the phase prefix (`rows N` vs `prefill rows N`, :51-52,313-320).
- **Piece timing (changed to two-phase paired).** Timing still runs only if G2 passes, but now at all six sizes with paired per-size verdicts (`bench/grader/check_candidate.py:345-378`; `claude/muse-agent-brief.md:209-213`; `claude/adr/0011-decode-and-prefill.md:21-32`; `muse-spark/round-3/exploration/territory-grader.md` §1): per size, candidate vs `mx.compile(ref_routing)` over 30 chained repeats in rotating order, `faster` with ≥22 wins, `slower` with ≥22 losses, else `same`; per phase, `faster` iff every size is, `slower` iff any size is, else `not slower`; `beats_compiled` iff one phase is `faster` and neither is `slower`. Round 2's "candidate median beats compiled median at all three decode sizes" bar is retired. **Prefill speed is now timed** — round 2's can't-see "prefill speed" item is retired, and this hypothesis exists because the starting point is `slower` at every prefill size. A `max_rows` fix banks prefill `same` × 3 and wins the slot on decode `faster` × 3; a kernel fix must additionally be `same`-or-better at every prefill size. The slot stops at the first attempt passing G1+G2 and the two-phase bar, or after 5 attempts (`claude/muse-agent-brief.md:217`; round 2 cited :201, moved).
- **G3 and the ready check (new in-round).** G3 is now vLLM-only two-phase judgment — decode step, prompt time, whole-request time over 1×128/8×128/32×128/32×1024, 5 repeats, pass iff decode or prompt is `faster` and none of the three is `slower`, against both base and compiled (`bench/grader/time_candidates.py:163-201`; `claude/muse-agent-brief.md:195-202`; `claude/adr/0011-decode-and-prefill.md:36-42`). The lead's ready check (`bench/grader/check_ready.py`, `claude/adr/0013-measure-what-you-claim.md`) runs that same G3 inside the round with every candidate in one call, plus same-greedy-tokens; only its passing verdict makes an attempt `ready`, and Claude's post-round G3 stays the authority (`territory-grader.md` §§4–5). For this hypothesis the discriminating check is prompt time on the 8×128/32×128/32×1024 workloads, where the starting point fails G3.
- **What it can't see** (checked by hand later — *check an attempt*; seeds this entry's **Unverified** list): row counts other than the six graded — ungraded decode 2–7 and 9–31 on the graded split-K path but either side of nothing (the GEMV/split-K boundary is at `rows = 1`); the `max_rows` band 33–126 where kernels run in real serving (vLLM chunked prefills) but no piece verdict exists, straddling the partition-count transitions at 33/65/129 (`territory-grader.md`, *can't see* 1–2); prefill chunks of other sizes on both sides of the split-K/regular boundary at `rows = 768` (`territory-baseline-routing.md` §3.1 regime table) and of the transitions at 129/257 — a kernel hard-coding the six sizes passes G2 and breaks this boundary; sizes above 2,048 (item 3); prefill beyond the first seeded-random chunk (item 4); hand-off exactness above `max_rows` at untested sizes (item 5); whole-request noise at 1 request (item 6); other prompts, later decode steps, layers' drift, unseen exact 4th/5th ties (no graded row is one), score magnitudes saturating the sigmoid differently; threadgroup memory still unchecked by `slots.build` (item 8, now `bench/grader/slots.py:341-342`); unwritten threadgroup reads and thread-timing dependence, which one equality run per layer cannot expose. New since round 2: `bench/grader/check_sizes.py` (25 sizes incl. 33/65/129/257/768/769) closes between-sizes exactness where the boundary agent runs it — boundary tooling, not a gate (`territory-grader.md` §3; the 59/60/61 probes are the experts slot's switch, not routing's). Boundary checks will probe exactly these.

### The GPU limits (unchanged)

At most 1,024 threads per threadgroup and 32 KB of threadgroup memory (`claude/muse-agent-brief.md:100`); `grid` counts threads (`claude/muse-agent-brief.md:178`); Apple M1 Pro, 16 cores, SIMD width 32, `bfloat` in Metal (brief:101). Starting point for scale: 256 threads/threadgroup, `(816 + 60 + 2,048 + 4,096) × 4 B = 28,080 B` — 85.7% of the limit, the closest of any routing attempt, leaving 4,688 B (`block-0.metal:15-25`); flat 69→73 µs from 8 to 32 rows confirms the footprint does not bind residency at decode sizes, but any fix that stages more (wider groups, cross-row tiles) must re-derive this sum before it can be trusted. Enforcement gap stands: `slots.build` checks only the 1,024-thread product (`bench/grader/slots.py:341-342`); Metal rejects oversize kernels as opaque G1 exceptions.

## Builders

### builder-3-a — direction (a): the `max_rows` hand-off

Copy the starting point's kernels verbatim (staged scalar decode
path + G2-exact split-K/regular branches as dead code above the
hand-off) and hand every row count above decode to the compiled
reference with `max_rows: 32` (bottom of the boundary's 32–126
window: decode rows ≤ 32 stay on the candidate; routing has no
60-row MLX switch, and the staged path's traffic floor alone
exceeds compiled above decode sizes, so no measured win exists to
preserve in 33–126 — the can't-see band empties by construction).

- **attempt-1** (`muse-spark/round-3/kernels/routing/attempt-1/`):
  predicted decode 49.8 / 113.5 / 121.3 (`faster` × 3) and prefill
  = compiled (`same` × 3). Measured (`grade.json`): G1 pass; G2
  pass all six, differ 0. Decode 1/8/32 rows: 103.6 / 129.9 /
  126.5 vs compiled 139.4 / 157.2 / 149.7 — `faster` (25/28/25).
  Prefill 127/1,016/2,048 rows: 91.3 / 229.3 / 299.7 vs compiled
  95.1 / 228.4 / 291.7 — `same` (18) / `same` (18) / `slower`
  (8). `phases_vs_compiled` decode `faster`, prefill `slower`;
  `beats_compiled` false. Taught: the 2,048-row `slower` is
  paired noise on identical code — above `max_rows` the candidate
  IS the compiled reference (separate `mx.compile` instance,
  `slots.py:331` vs `check_candidate.py:361`), medians within
  2.7%, and only the last-timed size hit the 22-loss threshold
  (see notes.md; prize genuinely banked). The 1-row absolute miss
  (+108%) is machine-state drift — compiled moved +114% in the
  same run, relative wins hold. Next: re-grade a byte-identical
  copy under a new attempt number to confirm the noise event does
  not reproduce; no kernel edit can move identical-code timing.

### builder-3-b — direction (b): GEMM-shaped prefill kernel sharing `w_router`

Two kernels behind a size branch. Kernel 0 is the starting point
verbatim plus `if (rows > 32) return;` (outputs renamed `inds0` /
`weights0`), so decode rows stay on the winning staged path bit for
bit. Kernel 1 is new: grid `rows*60`, threadgroup 960 — one threadgroup
spans 16 rows with one thread per (row, expert), staging each
`w_router` K-chunk once per 16 rows with attempt-4's exact coalesced
`float4` mapping (24,576 B threadgroup); per-row accumulation orders
copied op-for-op (split-K partition partials in registers, folded in
partition order; single serial sum above 768 rows), verbatim sigmoid,
strictly-greater top-4 and left-fold normalise; at rows ≤ 32 it only
copies kernel 0's outputs. Predicted 53 / 117 / 124 / 50 / 148 / 267
µs/layer (decode = attempt-4 + 3 µs copy launch; prefill = 16x-cut
traffic floors + barrier/wave allowances — thin at 2,048 rows, where
the chained timing decides).

- **attempt-2** (`muse-spark/round-3/kernels/routing/attempt-2/`):
  G1 fail, no G2, no timing (`grade.json`: `Thread group size (960)
  is greater than the maximum allowed threads per threadgroup (640)`).
  Taught: the brief's 1,024-thread cap is the device max — Metal
  derives each pipeline's actual cap from the kernel's resource use
  (here 640, likely register-bound by the `part[16]` partials; the
  GEMM direction itself is unrefuted, the kernel never ran). Next: a
  copied new attempt with 10 rows per threadgroup (600 threads, grid
  stays `rows*60`, 60 still divides the group size so no bounds
  checks); re-derived prefill floors ~13 / ~102 / ~205 µs (see
  notes.md).

### builder-3-c — direction (c): narrower staged footprint (drop `parts` where partitions = 1)

Two kernels split at the regime boundary `rows = 768` (threadgroup
arrays are statically sized, so one kernel cannot shed `parts`).
Kernel-A is the starting point verbatim plus a uniform
`if (rows > 768) return;` (outputs renamed `inds_a` / `weights_a`).
Kernel-B copies kernel-A through at rows ≤ 768 and runs the
single-partition path above: each expert's full 768-product chain in
one thread-private accumulator, no `parts`, 19,888 B (60.7% of 32 KB)
instead of 28,080 B. Orders, staging, barriers (24), sigmoid, top-4
and fold verbatim. Ruled out while writing: single-kernel register
partials (final ordered partition-sum needs a full-size gather
scratch ≈ `parts` itself); 8-chunks/group (8,192 `wsh` floats alone
exceed 32 KB) and 6-chunks/group (36,272 B with `parts`) — both
infeasible; sharding one expert's K across threads (reorders its
serial chain, fails G2).

- **attempt-3** (`muse-spark/round-3/kernels/routing/attempt-3/`):
  predicted 50 / 72 / 76 / 230 / 1266 / 2446 (decode = attempt-4 +
  ~2-3 µs copy launch; prefill-big = anchor + empty launch − ~10%
  of the shape residual), decode `faster` × 3, prefill `slower` × 3,
  `beats_compiled` false. Measured (`grade.json`): G1 pass; G2 pass
  all six, differ 0. Decode: 66.1 / 96.1 / 109.5 vs compiled 64.7 /
  95.4 / 103.9 — `same` (14/15/9). Prefill: 266.0 / 1098.3 / 2097.3
  vs compiled 96.9 / 235.3 / 299.7 — `slower` (0/2/0).
  `phases_vs_compiled` decode `not slower`, prefill `slower`;
  `beats_compiled` false. Taught: the decode miss (+32/+33/+44%) is
  machine-state drift meeting the small real wiring cost — the
  compiled bar swung ~2× between adjacent runs (attempt-1 saw
  139.4/157.2/149.7 minutes earlier), compressing attempt-4's margin
  to coin-flip races where ~2-3 µs of copy launch flips `faster` to
  `same` (see notes.md). Prefill-big is the encouraging half:
  −15%/−17% vs anchor against a +6%/+3% bar headwind, consistent
  with removing most of the 355/610 µs shape residual — but
  cross-run and drift-confounded, needs a same-run baseline to
  confirm. Next: re-grade a byte-identical copy WITH `--baseline`
  round-2 attempt-4 (lead must bless the flag; verdict-neutral, adds
  the paired column) to isolate copy cost and residency saving.

### builder-3-d — direction (b) fix: 10 rows per threadgroup (600 threads)

Copy of attempt-2 with exactly attempt-2's notes' prescribed fix:
kernel 1 goes from 16 to 10 rows per threadgroup (600 threads, grid
stays `rows*60`, 60 still divides 600 so groups stay row-aligned with
no bounds checks); `xst`/`sc` to 10×64, `Rg` cap and row stride 16→10,
`wsh[4096]` and the `w`-staging bound unchanged (K-chunk geometry);
threadgroup 24,576 → 21,504 B; `block-0`/`block-1` byte-identical
(`diff` empty). Predicted 53 / 117 / 124 / 55 / 220 / 400 µs/layer
(decode = attempt-4 + 3 µs copy launch, byte-identical code; prefill =
10x-cut traffic floors ~13 / ~102 / ~205 + barrier/wave allowances —
tied at 1,016, `slower` at 2,048 under the 0.5-µs/barrier guess, where
the chained timing decides).

- **attempt-4** (`muse-spark/round-3/kernels/routing/attempt-4/`):
  G1 fail, no G2, no timing (`grade.json`: `Thread group size (600)
  is greater than  the maximum allowed threads per threadgroup (576)`).
  Taught: the pipeline cap moves with the kernel — attempt-2's kernel
  reported 640, this near-identical recompile reports 576, so "design
  under 640" was wrong advice; both caps are whole-SIMD multiples
  (20×32, 18×32), pointing at register allocation shifting ~11% under
  recompilation (inferred — threadgroup memory shrank, so it cannot
  explain a lower cap; the GEMM direction itself is unrefuted, the
  kernel never ran). Next: a copied new attempt with 8 rows/group
  (480 threads, inside round-1 attempt-2's proven-passing band); if
  the cap drops below 480 again, attack per-thread registers next
  (e.g. shrink `part[16]`), not just the group size (see notes.md).

### builder-3-e — direction (b) fix: 8 rows per threadgroup (480 threads)

Copy of attempt-4 with exactly attempt-4's notes' prescribed fix:
kernel 1 goes from 10 to 8 rows per threadgroup (480 threads, grid
stays `rows*60`, 60 still divides 480 so groups stay row-aligned with
no bounds checks); `xst`/`sc` to 8x64, `Rg` cap and row stride 10->8,
`wsh[4096]` and the `w`-staging bound unchanged (K-chunk geometry);
threadgroup 21,504 -> 20,480 B; `block-0`/`block-1` byte-identical
(`diff` empty). Predicted 53 / 117 / 124 / 57 / 256 / 485 us/layer
(decode = attempt-4 + 3 us copy launch, byte-identical code; prefill =
8x-cut traffic floors ~16 / ~125 / ~252 + barrier/wave allowances).

- **attempt-5** (`muse-spark/round-3/kernels/routing/attempt-5/`):
  G1 pass; G2 pass all six, differ 0 — the first direction-(b)
  kernel to compile AND prove exact. Measured (`grade.json`):
  decode 1/8/32 rows: 72.6 / 123.3 / 111.9 vs compiled 76.2 /
  120.4 / 107.7 — `faster` (24) / `same` (10) / `same` (16);
  original 78.0 / 118.6 / 127.1. Prefill 127/1,016/2,048 rows:
  147.3 / 561.6 / 1,052.7 vs compiled 102.1 / 222.6 / 300.8 —
  `slower` (3/0/0); original 109.8 / 244.9 / 325.7.
  `phases_vs_compiled` decode `not slower`, prefill `slower`;
  `beats_compiled` false. Taught: the sharing works (0.65x/0.43x/
  0.42x the staged-path anchor at prefill, removing over half the
  gap) but per-group overhead runs ~4x the model (~3.2 us/group vs
  ~0.8: barriers ~2 us each or ~4x occupancy serialization from
  `part[16]` register pressure, inferred) — 1.44x/2.52x/3.50x
  slower than compiled, which shares `w_router` across all rows
  while this re-stages it every 8. Decode verdicts are drift
  compression (`same` on coin flips), not code cost. Next: a copied
  new attempt staging each `w` chunk once per group while streaming
  2-4 sequential 8-row blocks of `x` through it (traffic/row and
  barriers/row fall 2-4x, no more threads, G2-safe); then shrink
  live `part[16]` if occupancy still binds (see notes.md).

## Boundary checks

### attempt-1 — pass (boundary-3, 2026-11-04T18:14:51Z)

G1+G2 pass (`grade.json`: 0 rows differing at all six sizes). Kernels byte-identical to the starting point (`diff` of `block-0.metal` and `block-1.metal` against `muse-spark/round-2/kernels/routing/attempt-4/` empty); `spec.json` is the starting point's plus `"max_rows": 32` (JSON int, not bool — `slots.build` would raise otherwise, `bench/grader/slots.py:327-331`). Per the boundary, exactness at kernel-visible counts is inherited and this check verifies the hand-off, not the arithmetic. Checked against `muse-spark/round-3/kernels/routing/attempt-1/`:

- **Every decode step stays at or below `max_rows`.** `run()` hands off iff `rows > max_rows` (`bench/grader/slots.py:335`); a decode step has 1–32 rows (brief:26), all ≤ 32, so every decode step runs the staged scalar path. `max_rows = 32` sits at the bottom of the boundary's 32–126 window.
- **No MLX 60-row switch for this slot.** Routing's reference regimes change at `rows = 1` and `rows = 768` only; the 60-row change is the experts slot's. No side-of-switch analysis needed; all graded prefill sizes (127/1,016/2,048) hand to compiled.
- **Ungraded decode counts 2–7, 9–31:** same path in-kind as graded 8/32, not merely the same branch. The `rows`-only formula (`block-0.metal:60-70`) gives `tm = 1`, `ratio = 24`, `partitions = 32`, `base_iters = 1` for every `rows` in 2–32 — the graded 8 and 32 run this exact configuration. `rows == 1` GEMV is itself graded. No ungraded decode count reaches a new configuration.
- **The 33–126 band is empty by construction.** Every count above 32 hands to `mx.compile(ref_routing)` through the attempt's own dispatch, in graded and chunked-prefill serving alike; no piece verdict is missing anywhere the kernels run. The split-K-for-33–768 and single-accumulator branches are dead code at every reachable kernel count, kept verbatim.
- **Threadgroup memory:** unchanged 28,080 B of 32 KB (85.7%, `block-0.metal:15,22-25`), threadgroup 256 ≤ 1,024. Close to the limit but inherited from the G2-passing starting point, and no fix direction stages more — flag stands as a note, not a defect.
- **No unwritten threadgroup reads:** `xpad` pad slots (`c*17+16`) are never read (both readers index chunk offsets ≤ 15: `:28-29,42-45,111-114`); all 60 `scores` written on both branches (`:52-54,121-127`) before the sigmoid barrier (`:129`); every `parts[p*64+e]` read at `:123-124` was zeroed (`:30-32`) and every partition's chunks accumulated exactly once (`:95-119`); every `wsh` index read at `:114` is written by the staging loop (`:81-93`, `kv*4+j` covers `k` 0–15).
- **No thread-timing dependence:** barriers separate every staging/compute/consume phase (`:33,94,119,129,137`); the thread-0 top-4 scan and weights fold run after the final barrier (`:139-174`).
- **No new input assumptions:** dispatch, grids, shapes, dtypes and the strictly-greater tie rule with emitted-order fold are untouched; above `max_rows` the attempt runs the compiled reference whose six-size exactness was verified 2026-11-04.
- **The 2,048-row `slower` (8/30 wins) is not a boundary defect.** Above `max_rows` the candidate IS the compiled reference (separate `mx.compile` instance); medians within 2.7% on identical code, inside the brief's 5% same-kernel band. Verdict-level noise, already in the failure ledger.

Still unverified (inherited can't-see items, for **Unverified**): hand-off exactness above `max_rows` at untested sizes (compiled-vs-reference beyond the six graded); sizes above 2,048; prefill beyond the first seeded-random chunk; other prompts, later decode steps, layers' drift, unseen exact 4th/5th ties; score magnitudes saturating the sigmoid differently; whole-request noise at 1 request. `check_sizes.py` not run (boundary tooling, not a gate; nothing kernel-visible left ungraded below the hand-off).

## Decision

(none yet)

## Failure ledger

- attempt-1 (max_rows 32): G1+G2 pass, decode `faster` x3, prefill `same`/`same`/`slower` (2,048 rows: 8/30 wins, exactly the 22-loss threshold; candidate medians within 2.7% of compiled on literally identical code — paired noise, not a slowdown). Cause (suspected): noise on identical code; the builder's proposed next change is a re-grade of a byte-identical copy under a new number. Approach: hand-off (no kernel change can move identical-code timing).
- attempt-2 (GEMM-shaped sharing, 16 rows/group): G1 fail — kernel 1 asks 960 threads/group (16 rows × 60 experts), pipeline max is 640; no G2, no timing. Cause: compile error (threadgroup size), not a speed cause; fix is 10 rows/group (600 threads) in a copied attempt.
- attempt-3 (drop `parts` where partitions = 1, two-kernel split at rows = 768): G1+G2 pass, decode `same` × 3 (66.1/96.1/109.5 vs 64.7/95.4/103.9; 14/15/9 wins — the ~2–3 µs copy launch flips `faster` to `same`), prefill `slower` × 3 (266.0/1,098.3/2,097.3 vs 96.9/235.3/299.7 — the traffic re-read dominates; shape residual removal consistent with −15%/−17% but drift-confounded). Cause: per-row re-read untouched plus copy-launch wiring cost at decode. Approach: narrower staged footprint. (Declined: the proposed `--baseline` re-grade — this round has no baselines, so no `--baseline` runs.)
- attempt-4 (GEMM-shaped sharing, 10 rows/group): G1 fail — 600 threads/group against a pipeline max of 576 (the cap moved with the kernel: attempt-2's recompile reported 640); no G2, no timing. Cause: compile error (register allocation shifting the pipeline cap ~11% under recompilation, inferred). Fix: 8 rows/group (480 threads) in a copied attempt — inside round-1 attempt-2's proven-passing band.
- attempt-5 (GEMM-shaped sharing, 8 rows/group): G1+G2 pass — first direction-(b) kernel to run, and exact. Decode `faster`/`same`/`same` (72.6/123.3/111.9 vs 76.2/120.4/107.7; 24/10/16 wins), prefill `slower` × 3 (147.3/561.6/1,052.7 vs 102.1/222.6/300.8; 3/0/0 wins — 1.44/2.52/3.50×). Cause: per-group overhead ~4× the model (~3.2 vs ~0.8 µs/group — barriers or occupancy serialization, inferred); sharing itself works (0.65×/0.43×/0.42× the staged-path anchor). Approach: GEMM-shaped sharing. Line ends: two gaps away (prefill 1.4–3.5× plus decode `same`→`faster` at 8/32), while attempt-1 is one noise-test away — hardening continues on attempt-1.

## Unverified

(at least the boundary's *can't see* list, once written)

## Checks

(none yet)
