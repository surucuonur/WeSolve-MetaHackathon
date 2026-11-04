# Territory: baseline-routing — the routing slot's starting point (attempt-4) under two-phase judgment

- **Owner:** explorer-baseline-routing
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T17:39:44Z
- **Next:** none — checklist filled for all pieces, ineff-baseline-routing-1..2 posted, grader note folded

## What it does

Same slot contract as round 2
(`muse-spark/round-2/exploration/territory-baseline-routing.md:9-21`, frozen):
each of the 20 layers casts `x` to float32, multiplies by `w_router.T`,
sigmoid, adds `bias` for selection only, takes top-4 experts per row, and
normalises the unbiased scores (`+1e-20`, divide, ×2.5). Outputs `inds`
[rows,4] uint32 (feeds experts) and `weights` [rows,4] float32 (feeds
combine, in emitted order). G2 is still ordered bitwise equality at all six
sizes; the judgment is new ([ADR 0011](../../../claude/adr/0011-decode-and-prefill.md)):
an attempt must beat the compiled reference in both phases, decode and
prefill.

The code in the slot is now round 2's
`muse-spark/round-2/kernels/routing/attempt-4/` (frozen): one fused kernel
per layer, one 256-thread threadgroup per row, with a `rows`-branched
product (GEMV shuffle replica at `rows == 1`, staged scalar split-K while
`rows <= 768`, single staged accumulator above), MLX's verbatim sigmoid,
the strictly-greater insertion top-4, and the `((w0+w1)+w2)+w3` normalise.
Round 2's entries `ineff-baseline-routing-1..4` (frozen, all confirmed)
describe the round-1 starting point; entries 1–3 are removed by this code
(exact product in all three regimes, verbatim sigmoid, 256 threads/row),
entry 4 (per-row `w_router` re-read) is untouched and is now the prefill
story. This entry maps attempt-4 itself and finds why it wins at decode
sizes but scales to 2.6/5.9/8.7× compiled at prefill sizes.

## The headline finding (details in ineff-baseline-routing-1 and -2)

Chained µs/layer, reference / compiled / candidate
(`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`,
frozen): decode 68.4/65.1/**49.8** at 1 row, 169.2/135.7/**113.5** at 8,
159.3/145.2/**121.3** at 32 — faster everywhere; prefill 102.3/87.3/**227.1**
at 127 rows, 231.3/221.1/**1298.8** at 1,016, 310.0/290.4/**2513.0** at
2,048 — slower everywhere, widening with rows. `beats_compiled` false
(decode `faster`, prefill `slower`).

The shape of the loss is traffic plus occupancy, both linear in rows on the
candidate and near-constant on the compiled reference:

- Every row's threadgroup streams its own full copy of `w_router`
  (60×768×4 B = 184 KB) from device memory: rows×184 KB per layer, i.e.
  0.19/1.5/5.9 MB at decode 1/8/32 rows but 23.6/189/381 MB at prefill
  127/1,016/2,048 rows. Traffic floors at 200 GB/s: 0.9/7.4/29.7 µs
  decode, 118/944/1903 µs prefill — the prefill floors alone are 1.35×,
  4.3× and 6.6× the compiled reference's whole measured time
  (87.3/221.1/290.4). The reference reads `w_router` once per layer and
  reuses each tile across rows. This is round-2 `ineff-baseline-routing-4`
  (confirmed) with prefill sizes now judged: at decode it was an upper
  bound of 0/0.13/0.57 ms/step nobody needed; at prefill it is ~72–75% of
  the measured time.
- The remainder (measured − traffic floor: 109/355/610 µs/layer at
  127/1,016/2,048) is the staged scalar path's per-row shape cost: 24
  barriers per row, a 28,080 B threadgroup footprint (85.7% of 32 KB)
  capping residency as row counts grow into the hundreds, and — only in
  the single-partition regular-GEMM regime above 768 rows — 60 of 256
  threads active through all 12 scalar groups (23% duty). At decode sizes
  the same structure wins: 8–32 resident threadgroups overlap staging
  loads and barriers with other threadgroups' compute (marginal
  ≈0.16 µs/row, attempt-4 `notes.md:96-97`), and one kernel replaces the
  reference's ~12.

## Checklist

### 1. The starting-point kernel — `muse-spark/round-2/kernels/routing/attempt-4/block-0.metal` (174 lines)

One threadgroup of 256 threads per row at every size (`spec.json`: grid
`rows*256`, threadgroup 256). Threadgroup memory: `xpad[816]` +
`scores[60]` + `parts[2048]` + `wsh[4096]` floats = 28,080 B of 32 KB
(block-0:15-25). Five phases per row, separated by barriers (lines
33/94/119/129/137):

- **(a) `x` staging + `parts` zeroing (lines 27-33).** Cooperative:
  `x[row]` (768 bfloat16 → float32) into padded `xpad[(d>>4)*17+(d&15)]`
  (one pad float per BK=16 chunk kills the attempt-3 bank conflicts);
  `parts[2048]` zeroed. 1 barrier. Cost per row: 1.5 KB device read +
  ~11 float4-worth of threadgroup writes per thread.
- **(b) GEMV branch, `rows == 1` only (lines 35-55).** 8 simdgroups × 32
  lanes; simdgroup `sg` takes experts `sg, sg+8, …`; each lane serially
  sums 24 products (`d = i*128 + lane*4 + 0..3`, `i = 0..5`), then the
  shuffle tree 16,8,4,2,1; lane 0 writes `scores[e]`. Reads `xpad` (padded
  layout) and `w_router` uncoalesced straight from device — unchanged from
  attempt-3 apart from the padded index. Full 256-thread duty. This branch
  is why 1 row wins: 49.8 vs 65.1 µs/layer.
- **(c) Staged scalar path, `rows > 1` (lines 56-128).** The `rows`-branched
  partition geometry (lines 57-75): 32 partitions at 2–32 rows (base 1,
  last absorbs chunks 31–47), 16/8/4/2 across 33–768 rows, 1 partition
  above 768 (audited at ungraded counts in
  `muse-spark/round-2/implementation/hyp-2-routing-exact.md:204-206`,
  frozen). Then 12 groups of 4 K-chunks (lines 76-120): stage
  `gn*240` float4 loads of `w_router` coalesced into transposed
  `wsh[slot][k][expert]` (stride 64), barrier, scalar FMAs from `wsh`/`xpad`
  resuming `parts[p*64+e]` in increasing `d`, barrier — 24 barriers per
  row on this path. Scalar duty per group is `dparts*60` of 256 threads,
  where `dparts` is the partitions the group's 4 chunks touch: at 8/32
  rows groups 0–7 run 240/256 (94%), groups 8–11 run 60/256 (last
  partition); at 127 rows (8×6) each group touches ≤2 partitions (≤47%
  duty); at 1,016/2,048 rows (1 partition) every group runs 60/256 (23%
  duty). Final sequential partial sum per expert (lines 121-127, 60/256
  threads). Per-row device traffic: full 184 KB `w_router` re-read — the
  prefill bottleneck.
- **(d) Verbatim sigmoid (lines 132-136).** `1/(1+precise::exp(abs(s)))`,
  `(s<0)?y:1-y`, cooperative over 60 experts. Removes round-2
  `ineff-baseline-routing-2` (confirmed) at zero time cost.
- **(e) Serial thread-0 top-4 + weights (lines 139-174).**
  Strictly-greater insertion over 60 biased scores, earliest index wins
  ties; `s0+s1+s2+s3+1e-20f` left-to-right, divide, ×2.5. Unchanged since
  round 1; 255 threads idle through it, but 60 short iterations are lost
  in the fixed cost at every size.

- **Callers:** the grader builds it via `slots.build` into one
  `mx.fast.metal_kernel` (`bench/grader/slots.py:303-345`) and calls it
  once per layer per size in G2 and in the chained two-phase piece timing.
  In the model it would replace the routing part of `make_moe_call`
  (`slots.py:97-119`). Shapes: rows = 1/8/32 (decode), 127/1,016/2,048
  (prefill); grid `rows*256` threads, one threadgroup per row at every
  size.
- **Callees:** none — one Metal kernel. 1 kernel/layer (20/step) where the
  reference launches ~12/layer (brief:47).
- **Data:** in: `x` [rows,768] bfloat16, `w_router` [60,768] float32,
  `bias` [60] float32; out: `inds` [rows,4] uint32, `weights` [rows,4]
  float32 (`spec.json`; brief:30-36). All row-contiguous (brief:24).
  Threadgroup: 28,080 B, row-count independent.
- **Neighbours:** before: post-attention layernorm output reshaped to
  [rows,768] (`mobilemoe.py:258`, `slots.py:107`); after: experts gather by
  `inds`, combine sums `(y * weights[...,None])` in emitted order
  (`mobilemoe.py:230-231`). A replacement must keep shapes, dtypes, the
  unbiased-score rule and the reference's exact per-row order (round-2
  hyp-2 boundary, frozen).
- **Cost:** chained µs/layer, reference / compiled / candidate
  (`grade-phases-routing-attempt-4.json`): 1 row 68.4/65.1/**49.8**; 8
  rows 169.2/135.7/**113.5**; 32 rows 159.3/145.2/**121.3**; 127 rows
  102.3/87.3/**227.1**; 1,016 rows 231.3/221.1/**1298.8**; 2,048 rows
  310.0/290.4/**2513.0**. End to end (G3, vLLM, round-3 README): decode
  step −0.47/−0.43/−0.54 ms at 1/8/32 requests (passed every rule of its
  time); whole requests +10.2/+1.8/+0.3/−3.5% throughput with prompts
  1.07–1.10× slower. Traffic per layer: rows×184 KB (`w_router`) +
  rows×1.5 KB (`x`): 0.9/7.4/29.7 µs floors at decode, 118/944/1903 µs at
  prefill (200 GB/s peak).
- **Invariants:** G1+G2 pass at all six sizes (`rows_differ` 0,
  `rows_other_experts` 0, `max_rel_err` 0.0). Bit-exactness rests on: the
  exact bf16→f32 cast of `x`; the `rows`-branched regime matching MLX's
  dispatch (GEMV only at `rows == 1`; split-K while `rows <= 768`;
  regular above) with the split-K partition formula as a function of
  `rows` (not the six graded values); scalar≡MMA rounding at every
  grouping (round-2 refuted claim, proven by attempts 2–4); the verbatim
  sigmoid; the strictly-greater insertion tie rule (earliest index wins,
  matching the reference's stable sort); the
  `((w0+w1)+w2)+w3 + 1e-20`, divide, ×2.5 association. A builder changing
  the feeding (staging, barriers, duty) without touching accumulation
  order keeps G2, as attempt-4 itself proved against attempt-3.

### 2. `spec.json` and `block-1.metal`

- **Callers/callees/data/neighbours:** `spec.json` declares the single
  kernel `fused_routing_exact` (inputs `x`, `w_router`, `bias`; outputs
  `inds`, `weights`; grid `rows*256`, threadgroup 256; no `max_rows`, so
  the kernel runs at prefill sizes too). `slots.build` enforces only the
  1,024-thread cap (`slots.py:326-341`); the 32 KB threadgroup limit is
  unchecked by the grader (round-2 hyp-2 boundary, *can't see* list).
  `block-1.metal` is 3 lines: `#include <metal_simdgroup>` for the GEMV
  shuffle tree.
- **Cost:** no direct cost; the spec's one row-count-independent grid is
  what forces the single-partition regime at 1,016/2,048 rows onto the
  same 256-thread, 28 KB threadgroup the decode sizes use. A `max_rows`
  hand-off (supported by `slots.build`, `slots.py:306-310`) would keep
  prompt-sized inputs on the compiled reference — the round-3 README's
  item 3 names it as one of this slot's two outs.
- **Invariants:** `atomic_outputs` false; grid counts threads, not
  threadgroups (brief:98).

### 3. The measurement tables

- `results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`
  (frozen): the six chained timings quoted above, `beats_compiled` false
  (decode `faster`, prefill `slower`), G2 pass everywhere. This is the
  piece evidence for both inefficiency entries.
- Round-3 exploration README (v1): G3 decode-step effects (−0.47/−0.43/−0.54
  ms), whole-request throughput (+10.2/+1.8/+0.3/−3.5%), prompt multiples
  (1.07–1.10×), and *What is left* items 3 (this territory's prefill gap)
  and 7 (the decode side of the re-read, already ranked upper-bound-only).

## Relations

- **Fed by:** the model territory (layernorm output `x`; router weight and
  `expert_bias`, `mobilemoe.py:187-199,258`); the grader territory
  (captured inputs at 1/8/32 rows; prefill inputs recorded live,
  `slots.py:214-254`).
- **Feeds:** `baseline-experts` (`inds` chooses the (row, expert) pairs and
  their order); `baseline-combine` (`weights` multiply the experts'
  outputs in emitted order — the reason order is part of G2); `g3` (decode
  wins −0.47/−0.43/−0.54 ms/step banked; prefill loss 7–10% prompt time,
  −3.5% whole-request at 1,024-token prompts).
- **Shared structure:** the prefill loss has the same shape as experts'
  (per-row work that the reference amortises across rows), but the cause
  is this slot's own re-read plus its staged-path shape, not experts'
  scalar `qmm_accum`. The `max_rows` hand-off is shared grader machinery
  (see the `grader` territory); if it banks this slot, the decode win must
  survive composition (round-3 README item 5).

## Inefficiencies

- `ineff-baseline-routing-1` — `w_router` re-read once per row at prefill
  sizes: 23.6/189/381 MB per layer against the traffic the compiled
  reference pays; predicted saving 0/0.13/0.57 ms per decode step and
  2.32/18.71/37.73 ms per prefill chunk (upper bound).
- `ineff-baseline-routing-2` — the staged path's per-row shape cost at
  prefill: 24 barriers/row, 28 KB residency cap, 23% scalar duty in the
  single-partition regime; predicted saving ~0 at decode and 0.48/2.85/6.74
  ms per prefill chunk (upper bound). The two entries partition the prefill
  gap exactly (measured − compiled = 139.8/1077.7/2222.6 µs/layer).

## Files read

- `muse-spark/round-2/kernels/routing/attempt-4/{block-0.metal,spec.json,block-1.metal,notes.md}`
  — the code in the slot: staged scalar path, regime branches, spec grid,
  the 28 KB / 24-barrier / flat-decode account.
- `results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`
  — the six chained timings and the decode-faster/prefill-slower verdict.
- `muse-spark/round-2/exploration/territory-baseline-routing.md` (frozen)
  — the slot contract, the three MLX accumulation regimes, the op-by-op
  exactness account this code resolves.
- `muse-spark/round-2/exploration/ineff-baseline-routing-1.md` (frozen,
  confirmed; scalar≡MMA since refuted by building) — the exactness gate
  attempt-4 removes.
- `muse-spark/round-2/exploration/ineff-baseline-routing-2.md` (frozen,
  confirmed) — the sigmoid gate attempt-4 removes at zero cost.
- `muse-spark/round-2/exploration/ineff-baseline-routing-3.md` (frozen,
  confirmed) — the 1-row occupancy deficit attempt-4 removes with 256
  threads/row (GEMV branch) while the serial thread-0 scan stays.
- `muse-spark/round-2/exploration/ineff-baseline-routing-4.md` (frozen,
  confirmed) — the per-row re-read with decode-only upper bounds and the
  round-1 attempt-2 counter-evidence; the starting point for the new
  prefill entries.
- `muse-spark/round-2/implementation/hyp-2-routing-exact.md` (frozen) —
  the four builders' verdicts (what each attempt taught), the attempt-4
  boundary check (ungraded row counts, unwritten reads, 28,080 B
  derivation, timing independence), and the decision.
- `muse-spark/round-3/exploration/README.md` (v1) — the two-phase judgment
  (no baselines), this territory's opening line, G3 effects, *What is
  left* items 3 and 7.
- `bench/grader/slots.py:303-345` — `slots.build`: one-kernel build, the
  1,024-thread cap, the `max_rows` hand-off the spec does not use.
- `claude/muse-agent-brief.md:24-47` — slot table (shapes, dtypes, the
  reference, ~12 kernels/layer).
- `claude/GLOSSARY.md` — overhead/bandwidth/compute-bound, occupancy,
  prefill, peak bandwidth (200 GB/s).

## From other agents

- explorer-grader, held in `territory-grader.md` (*Inefficiencies*) and folded
  here 2026-11-04T17:39:44Z: no kernel switch underneath routing/combine, so
  any `max_rows` ≥ 32 keeps G2/timing identical at decode sizes and hands
  every prefill size to compiled; the only choice is where kernels stop
  running in real serving. **Folded:** agreed — attempt-4's `rows` branch
  selects accumulation orders inside one kernel, not different kernels, so
  a hand-off at any value in 32..126 is G2-identical at decode sizes by
  construction (the compiled reference is the G2 target). This is the
  round-3 README item-3 hand-off out for this slot: it banks both
  inefficiency entries' prefill savings at once (the decode win must still
  survive composition, README item 5).
- explorer-g3, 2026-11-04T17:41:19Z — measurement-side companion to your ineff-baseline-routing-1/-2: ineff-g3-3 shows routing prefill piece excess composes to prompt time within 15% (+3.3/+19.1/+77.7/+662.5 ms at 1x128/8x128/32x128/32x1024); -3.5% whole-request throughput at 32x1024 is the +663ms prompt excess, not decode. Also: experts 1-row piece sign flips between runs (grade vs grade-phases), treat 1-row piece margins as noise.
