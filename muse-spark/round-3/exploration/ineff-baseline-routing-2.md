# Inefficiency: the staged path's per-row shape cost — 24 barriers, 28 KB residency cap, 23% duty above 768 rows

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:41:11Z
- **Next:** done — confirmed by verifier (see Verdict); builders may use the residual bounds

## Slot

`routing`

## What happens

Past the traffic in `ineff-baseline-routing-1`, each row pays a shape cost
for how the staged scalar path is built
(`muse-spark/round-2/kernels/routing/attempt-4/block-0.metal`, frozen):

1. **24 barriers per row.** The 12 staged groups each pay 2
   `threadgroup_barrier`s (stage-then-compute, compute-then-overwrite,
   block-0:94,119), plus 3 outside the loop (lines 33,129,137). Every
   barrier stalls the whole 256-thread threadgroup.
2. **28,080 B threadgroup footprint** (`xpad[816]` + `scores[60]` +
   `parts[2048]` + `wsh[4096]` floats, block-0:15-25) — 85.7% of the
   32 KB limit. Only ~1 such threadgroup fits a core's allocation at a
   time, so as row counts grow into the hundreds the waves of threadgroups
   serialise and barrier/staging latency that overlapped at decode sizes
   is exposed. Attempt-4's own notes measured the decode side of this:
   flat 69→73 µs from 8 to 32 rows because staging and barriers overlapped
   other resident threadgroups' compute (`notes.md:88-103`) — that overlap
   is what the residency cap takes away as rows scale.
3. **23% scalar duty in the single-partition regime.** Each group's scalar
   phase runs `dparts*60` of 256 threads (block-0:99-101), where `dparts`
   is the partitions the group's 4 chunks touch. From the partition
   formula (block-0:57-75; audited in round-2 hyp-2, frozen): at 8/32 rows
   (32 partitions) groups 0–7 run 240/256 threads (94%) and only groups
   8–11 drop to 60/256; at 127 rows (8 partitions × 6 chunks) each group
   touches ≤2 partitions (≤47% duty); at 1,016/2,048 rows (1 partition,
   the regular-GEMM regime) **all 12 groups run 60/256 threads (23%)**,
   and the final sequential partial sum runs 60/256 at every size
   (block-0:121-127). The per-row FMA count is fixed (60×768 = 46,080),
   but above 768 rows it issues through one quarter of the lanes in
   serial dependent chains.

None of this is the serial thread-0 scan (block-0:139-174): 60 short
iterations, unchanged since round 1 and lost in the fixed cost at every
size — round-2 `ineff-baseline-routing-3`'s (confirmed) occupancy deficit
is removed at decode sizes by the 256-thread GEMV branch; this entry is
the occupancy deficit that remains at prefill sizes.

## Cost

Measured chained µs/layer minus the traffic floor from
`ineff-baseline-routing-1` (measured/compiled from
`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`,
frozen):

| rows | measured | traffic floor | residual | compiled total | compiled overhead (≈) |
|---|---|---|---|---|---|
| 1 | 49.8 | 0.9 | 48.9 | 65.1 | ~64 |
| 8 | 113.5 | 7.4 | 106.1 | 135.7 | ~128 |
| 32 | 121.3 | 29.7 | 91.6 | 145.2 | ~115 |
| 127 | 227.1 | 118.0 | 109.1 | 87.3 | ~85 |
| 1,016 | 1298.8 | 944.1 | 354.7 | 221.1 | ~212 |
| 2,048 | 2513.0 | 1903.2 | 609.8 | 290.4 | ~273 |

Compiled overhead ≈ total minus its own small traffic (≈1/8/30 µs decode,
≈2/9/17 µs prefill). At decode the candidate's residual is *below* the
compiled reference's overhead at every size — the staged path wins there.
At prefill the residual exceeds compiled's overhead by 24/143/337
µs/layer and grows with rows (0.86/0.35/0.30 µs/row marginal — falling per
row as fixed launch cost amortises, but the barrier/residency/duty slice
scales with the threadgroup count). Kernels × fixed cost: 1 kernel/layer
here against the reference's ~12, so the residual is not launch count —
it is per-threadgroup shape (barriers × residency serialization × idle
lanes), i.e. the part of the 2.6/5.9/8.7× prefill multiples that bytes do
not explain.

## Kind

Overhead-bound in `claude/GLOSSARY.md` terms (time goes to barriers, idle
thread slots and residency serialization, not to bytes or FLOPs — 46,080
FLOP/row is ~0.01 µs at peak compute), sitting on top of the
bandwidth-bound re-read in `ineff-baseline-routing-1`.

## Spans

`baseline-routing`. The residency argument touches the `grader` territory
only in that the 32 KB limit is unchecked by `slots.build`
(`bench/grader/slots.py:326-341`); the duty argument is entirely inside
this slot's kernel.

## Predicted saving

Upper bound if the per-row shape cost were cut to the compiled reference's
overhead level (residual minus compiled overhead), per layer and per
prefill chunk (× 20 layers); decode saves ~0 because the candidate's
residual already beats compiled's overhead there:

| rows | per layer | per decode step / prefill chunk |
|---|---|---|
| 1 | ~0 µs | ~0 ms |
| 8 | ~0 µs | ~0 ms |
| 32 | ~0 µs | ~0 ms |
| 127 | 24.1 µs | 0.48 ms |
| 1,016 | 142.7 µs | 2.85 ms |
| 2,048 | 336.8 µs | 6.74 ms |

Arithmetic, e.g. at 2,048 rows: residual 2513.0 − 1903.2 = 609.8 µs/layer;
compiled overhead ≈ 290.4 − 16.6 ≈ 273 µs/layer (3.3 MB traffic ≈ 16.6 µs
floor); saving ≈ 609.8 − 273 ≈ 337 µs/layer × 20 = 6.74 ms per prefill
chunk. What could remove it: fewer, wider staged groups (attempt-4's notes
name staging 8 chunks per group to halve the 24 barriers, at ~2× the `wsh`
footprint — but only 4,688 B of headroom remain under 32 KB, so that trade
needs re-derivation); a smaller footprint to restore residency (e.g. narrow
the expert stride, or drop `parts` where partitions = 1); and fuller lanes
in the single-partition regime (shard experts' K within the group instead
of one thread per expert). Each must keep the exact accumulation orders
(feeding changes without order changes keep G2, as attempt-4 proved). This
entry and `ineff-baseline-routing-1` partition the prefill gap exactly:
116.1+24.1 = 139.8, 935.4+142.7 = 1077.7, 1886.5+336.8 = 2222.6 µs/layer
= measured − compiled; the two savings do not add beyond that, and a
GEMM-shaped kernel sharing weights across rows would remove both together.

## Verdict

verifier, 2026-11-04T17:41:11Z — **confirmed**. Every claim the cost and the saving rest on holds; four corrections below (two wording, one unsourced-but-immaterial, one mechanism without a source) change no conclusion.

- **confirmed:** 24 in-loop barriers (block-0:94,119 × 12 groups) plus 3 outside (lines 33,129,137); attempt-4 notes:27 agrees ("24 barriers on the scalar path").
- **confirmed:** 28,080 B footprint = 85.7% of 32 KB (block-0:15 comment; 28080/32768 = 85.7%); headroom 4,688 B ✓.
- **confirmed:** scalar duty `dparts*60` of 256 threads (block-0:99-101); partition formula recomputed from block-0:57-75: 32 partitions at 8/32 rows with groups 0–7 at 240/256 and 8–11 at 60/256, 8 partitions at 127 rows (≤2 touched per group, ≤46.9% duty), 1 partition above 768 rows (all 12 groups at 60/256 = 23.4%); final partial sum 60/256 at every size (block-0:121-127); thread-0 serial scan, 60 iterations (block-0:139-174); 60×768 = 46,080 FMAs/row.
- **confirmed:** attempt-4 notes overlap account (flat 69.1→72.9 µs, staging/barriers overlapped across threadgroups, notes:80-103); staging-8-halves-barriers at ~2× `wsh` (notes:107); round-2 `ineff-baseline-routing-3` is `confirmed`; `slots.build` enforces only the 1,024-thread cap so the 32 KB limit is unchecked (slots.py:326-341, corroborated by round-2 hyp-2 boundary).
- **confirmed:** cost table — residuals 48.9/106.1/91.6/109.1/354.7/609.8 recomputed from the grade-phases JSON ✓; prefill compiled overhead ≈85/212/273 ✓; excess over compiled overhead 24.1/142.7/336.8 ✓; prefill multiples 2.60/5.88/8.65 → 2.6/5.9/8.7 ✓; ~12 reference kernels (round-1 attempt-2 notes: 12 kernels/layer).
- **confirmed:** saving table and the 2,048-row worked example (609.8 − 273.8 ≈ 337 × 20 = 6.74 ms) ✓; decode ~0 because the residual already beats compiled overhead there.
- **no source:** "only ~1 such threadgroup fits a core's allocation at a time, so waves serialise" — a mechanism inference, consistent with the notes' overlap account and the scaling, but residency itself is unmeasured. The residual cost does not rest on it.
- **wrong (immaterial):** decode compiled traffic "≈1/8/30" — the ≈8/30 equal the candidate's own floors and contradict entry 1's sourced read-once model (≈1/1/1 at decode). Under either model the candidate residual stays below compiled overhead at decode, so the ~0 decode saving stands.
- **wrong (label):** "0.86/0.35/0.30 µs/row marginal" — these are per-row averages (residual ÷ rows); true marginals are ≈0.28/0.25. The falling pattern holds either way.
- **wrong (wording):** "partition the prefill gap exactly" — sums exceed measured−compiled by 0.4/0.4/0.7 µs (rounding of ≈ compiled overhead; see entry-1 verdict). Holds to rounding; non-additivity stands.

## From other agents

builder-3-c, 2026-11-04T18:18:49Z — direction-(c) evidence from
`muse-spark/round-3/kernels/routing/attempt-3/` (G1+G2 pass, 0 rows
differing all six sizes; decode `same` ×3 at 66.1/96.1/109.5 vs
compiled 64.7/95.4/103.9; prefill `slower` ×3 at 266.0/1098.3/2097.3
vs 96.9/235.3/299.7; `grade.json` in the attempt folder):

- The two-kernel split at `rows = 768` works and keeps G2: the
  single-partition path with register-held expert chains (no `parts`,
  19,888 B) is bit-exact at 1,016/2,048 rows, confirming feeding
  changes without order changes keep G2 once more.
- Prefill-big is consistent with most of the shape residual removed:
  −15%/−17% vs the grade-phases anchor against a +6%/+3% slower
  compiled bar. But cross-run and drift-confounded (the bar swung ~2×
  between adjacent runs that hour), so the residency mechanism stays
  unmeasured — a same-run `--baseline` re-grade is the proposed next
  step (see the attempt's notes.md).
- Three of this entry's suggested fixes are infeasible as stated:
  (1) 8-chunks/group needs 8,192 `wsh` floats = 32,768 B before any
  other array — over the cap alone; 6-chunks/group needs 36,272 B
  with `parts` (34,224 B even at stride 60) — also over, so
  attempt-4's notes' "stage 8, halve the barriers" does not fit;
  (2) register-held partials in ONE kernel cannot shed `parts`,
  because the final ordered partition-sum needs all partitions × 60
  partials simultaneously visible (a gather scratch of up to 1,920
  floats ≈ `parts` itself) — only the partitions = 1 regime (fixed
  thread→expert ownership, no gather) can drop it, which is what the
  split exploits; (3) sharding one expert's K across threads would
  reorder its serial accumulation chain and fail bit-exact G2.
- The split's decode cost is real (~2–3 µs copy/empty launch per
  layer, as predicted from (reference − attempt-4)/11 kernels) and
  flips `faster` to `same` when machine-state drift compresses
  attempt-4's margin to coin-flip races (14/15/9 wins here).
- builder-3-e, 2026-11-04T18:24:35Z — data point for the shape cost: direction-(b) attempt-5 (`muse-spark/round-3/kernels/routing/attempt-5/grade.json`, G1+G2 pass) shares w across 8 rows/group (25 barriers/group, 20,480 B footprint) and measures ~3.2 us/group overhead at prefill-big vs ~0.8 modeled — either ~2-us barriers (4x the 0.5 guess) or ~4x occupancy serialization from part[16] register pressure (inferred). Barrier/occupancy cost, not traffic, now dominates the GEMM-shaped kernel; details in the attempt's `notes.md`.
