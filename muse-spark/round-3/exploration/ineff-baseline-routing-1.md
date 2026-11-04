# Inefficiency: `w_router` re-read once per row costs 116/935/1887 µs per layer at prefill sizes

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:41:11Z
- **Next:** done — confirmed by verifier (see Verdict); builders may use the excess bounds

## Slot

`routing`

## What happens

Every row's threadgroup stages its own full copy of the router weight from
device memory. In the GEMV branch (`rows == 1`) each lane reads `w_router`
straight from device (`muse-spark/round-2/kernels/routing/attempt-4/block-0.metal:41-45`,
frozen); on the staged scalar path (`rows > 1`) all 256 threads cooperatively
load all 48 K-chunks × 60 experts as `float4`s into `wsh` across the 12 groups
(block-0:81-93) — coalesced, but still a private full copy per row. Nothing is
shared across rows: per layer the kernel reads rows × 60×768×4 B = rows ×
184,320 B of `w_router`, plus rows × 1,536 B of `x` (bfloat16).

The reference's GEMM/GEMV reads `w_router` once per layer and reuses each tile
across all rows in the tile (round-2 `territory-baseline-routing.md`
section 3.1, frozen). This entry is the prefill side of round-2
`ineff-baseline-routing-4` (frozen, confirmed), which ranked the same re-read
0/0.13/0.57 ms/step at decode sizes and 0 at prefill because prefill was
untimed (ADR 0010:42). Under two-phase judgment (ADR 0011) the prefill side
is the slot's largest measured cost.

## Cost

Traffic floors at the 200 GB/s peak (brief:9), candidate per layer:

| rows | `w_router` | `x` | total | floor | measured | compiled |
|---|---|---|---|---|---|---|
| 1 | 0.18 MB | 0.00 MB | 0.19 MB | 0.9 µs | 49.8 µs | 65.1 µs |
| 8 | 1.47 MB | 0.01 MB | 1.49 MB | 7.4 µs | 113.5 µs | 135.7 µs |
| 32 | 5.90 MB | 0.05 MB | 5.95 MB | 29.7 µs | 121.3 µs | 145.2 µs |
| 127 | 23.41 MB | 0.20 MB | 23.60 MB | 118.0 µs | 227.1 µs | 87.3 µs |
| 1,016 | 187.30 MB | 1.56 MB | 188.83 MB | 944.1 µs | 1298.8 µs | 221.1 µs |
| 2,048 | 377.49 MB | 3.15 MB | 380.63 MB | 1903.2 µs | 2513.0 µs | 290.4 µs |

Measured/compiled: chained µs/layer from
`results/runs/2026-11-04-muse-round-2/grade-phases-routing-attempt-4.json`
(frozen). The compiled reference's traffic per layer is ~184 KB once plus
rows×1.5 KB of `x` (≈0.4/1.7/3.3 MB at prefill sizes, i.e. ≈2/9/17 µs
floors) — the re-read excess is (rows−1)×184,320 B per layer:

- decode: 0 / 6.5 / 28.6 µs/layer at 1/8/32 rows (as round-2 entry 4);
- prefill: 116.1 / 935.4 / 1886.5 µs/layer at 127/1,016/2,048 rows.

At prefill the candidate's traffic floor alone exceeds the compiled
reference's whole measured time by 1.35×/4.3×/6.6×, and accounts for
52%/73%/76% of the candidate's own measured time. End to end this is the
7–10% slower prompt processing and the −3.5% whole-request throughput at
1,024-token prompts (round-3 README, G3).

## Kind

Bandwidth-bound at prefill sizes (time scales with bytes: measured
227/1299/2513 µs against floors 118/944/1903 µs, and the 1016→2048
doubling adds 1214 µs against a 959 µs floor delta); overhead-bound at
decode sizes, where the same re-read is ≤30 µs of 50–121 µs measured
(`claude/GLOSSARY.md:39-42`). In the terms of round-2 entry 4's verdict,
the slot was overhead-bound with a bandwidth-waste component at 32 rows;
at prefill sizes the waste component is the cost.

## Spans

`baseline-routing`. The traffic is inside this slot's kernel; the
comparison point (read-once tile reuse) is MLX's GEMM dispatch in the
reference, and the `max_rows` hand-off alternative is `grader` territory
machinery.

## Predicted saving

Upper bound if each weight byte were read once per layer at peak bandwidth
(the excess above), per layer and per step/chunk (× 20 layers):

| rows | per layer | per decode step / prefill chunk |
|---|---|---|
| 1 | 0 µs | 0 ms |
| 8 | 6.5 µs | 0.13 ms |
| 32 | 28.6 µs | 0.57 ms |
| 127 | 116.1 µs | 2.32 ms |
| 1,016 | 935.4 µs | 18.71 ms |
| 2,048 | 1886.5 µs | 37.73 ms |

Arithmetic: (rows−1) × 184,320 B ÷ 200 GB/s; e.g. at 2,048 rows
2047 × 184,320 B = 377.3 MB ÷ 200 GB/s = 1886.5 µs/layer × 20 = 37.73 ms
per prefill chunk. A saving in one phase that costs the other is not one:
at decode the candidate already beats compiled by 15.3/22.2/23.9 µs/layer,
so the decode bounds only deepen an existing win — provided the sharing
design does not add fixed cost. Confidence is low without a design, on
measured counter-evidence: round-1 attempt-2 tiled `w_router` across rows
to remove this re-read and was slower at every decode size (36.3/34.9/49.0
vs 28.5/23.6/41.8 µs/layer, `muse-spark/kernels/routing/attempt-2/notes.md`,
frozen) — barriers and strided reads ate the saving. A builder claiming
this saving needs cross-row weight sharing that keeps the exact
accumulation orders (the `rows`-branched regimes are per-row orders over
shared data, so sharing is compatible with G2 in principle) without
attempt-2's barrier structure — e.g. one threadgroup spanning several rows
in a GEMM shape — and must prove it on the chained two-phase timing, not
the traffic arithmetic. This entry and `ineff-baseline-routing-2` partition
the prefill gap exactly: traffic excess + shape residual = measured −
compiled (139.8/1077.7/2222.6 µs/layer); the two savings do not add beyond
that.

## Verdict

verifier, 2026-11-04T17:41:11Z — **confirmed**. Every claim the cost and the saving rest on holds; three citation/wording corrections below change nothing in the arithmetic.

- **confirmed:** GEMV branch reads `w_router` from device per row (block-0.metal:35-55; one threadgroup per row via line 16); staged path cooperatively loads 48 K-chunks × 60 experts as `float4` into `wsh` across 12 groups (block-0:76-93, 256-stride loop line 81, `float4` line 87).
- **confirmed:** per-layer traffic rows×184,320 B + rows×1,536 B (60×768×4 = 184,320; 768×2 = 1,536); peak 200 GB/s (brief:9); reference read-once tile reuse (round-2 territory §3.1, corroborated by round-2 entry-4's verdict).
- **confirmed:** round-2 `ineff-baseline-routing-4` is `confirmed` and ranked 0/0.13/0.57 ms/step; this entry's 6.5 µs/layer at 8 rows is the correctly rounded 6.4512 (round 2 wrote 6.4).
- **confirmed:** cost table — all six measured/compiled pairs match `grade-phases-routing-attempt-4.json` timing rows; floors 0.9/7.4/29.7/118.0/944.1/1903.2 recomputed ✓; excess 0/6.5/28.6 and 116.1/935.4/1886.5 recomputed ✓; ratios 1.35×/4.3×/6.6× and shares 52/73/76% recomputed ✓; 1016→2048 deltas 1214.2 vs 959.1 ✓.
- **confirmed:** G3 7–10% slower prompts and −3.5% throughput at 1,024-token prompts (round-3 README routing section); decode win 15.3/22.2/23.9 µs/layer ✓; saving table and the 2,048-row worked example (2047×184,320 = 377.3 MB → 1886.5 µs → 37.73 ms) ✓.
- **confirmed:** round-1 attempt-2 counter-evidence — tiled sharing direction (notes:9-11), 36.3/34.9/49.0 vs attempt-1's 28.5/23.6/41.8, slower at every row count (notes table + line 41).
- **wrong (cite only):** "prefill was untimed (ADR 0010:42)" — the claim holds but the line is ADR 0010:44 ("Not covered: prefill speed"); line 42 is the grader-run-takes-longer bullet.
- **wrong (cite only):** "in the terms of `claude/GLOSSARY.md:39-42`" — lines 39-42 define overhead-bound only; bandwidth-bound is at :33-34. The kind reasoning itself checks.
- **wrong (wording):** "partition the prefill gap exactly" — 116.1+24.1 = 140.2 vs 139.8, 935.4+142.7 = 1078.1 vs 1077.7, 1886.5+336.8 = 2223.3 vs 2222.6 measured−compiled; the ≤0.7 µs gap is rounding of the ≈ compiled overhead. Holds to rounding; the non-additivity conclusion stands.

## From other agents

builder-3-a, 2026-11-04T18:14Z — the hand-off banks this entry's
prize at compiled *time* but the paired verdict can still read
`slower` on identical code: `muse-spark/round-3/kernels/routing/attempt-1`
(`max_rows` 32, kernels byte-identical to round-2 attempt-4) graded
G1+G2 pass with prefill medians 4.0% under / 0.4% over / 2.7% over
compiled — all inside the brief's 5% same-kernel-twice band — yet
2,048 rows went 8/30 wins (`slower`, exactly the 22-loss
threshold) while 127/1,016 went 18/30 (`same`). Above `max_rows`
the candidate is a second `mx.compile` of the same reference
(`bench/grader/slots.py:331` vs `bench/grader/check_candidate.py:361`),
so no kernel defect can explain it; it is a verdict-level noise
event over 30 correlated repeats (detail in the attempt's
`notes.md`). Cross-run caution for this entry's tables: the same
run's 1-row decode absolutes drifted ~2× vs the basis run
(candidate 49.8 → 103.6, compiled 65.1 → 139.4) with the relative
win intact — compare within-run, not across runs.
- builder-3-b, 2026-11-04T18:16Z — direction-(b) data point for the
  sharing design this entry asks for: a 960-thread threadgroup (16 rows
  x 60 experts, one thread per (row, expert), 24,576 B threadgroup,
  per-row orders copied op-for-op) fails G1 — Metal derives this
  pipeline's max threads per threadgroup as 640, not the brief's device
  max of 1,024 (`muse-spark/round-3/kernels/routing/attempt-2/grade.json`).
  Likely register-bound (`part[16]` partials per thread). Sharing across
  rows must fit ≤ 640 threads/group (e.g. 10 rows x 60 = 600), which
  re-derives the traffic floors to ~13 / ~102 / ~205 µs at
  127/1,016/2,048 rows — still a 10x cut of the re-read, thin at 2,048.
  The GEMM direction is unrefuted (the kernel never ran); details in the
  attempt's `notes.md`.
- builder-3-d, 2026-11-04T18:21Z — follow-up: the 10-rows/group fix
  (600 threads, 21,504 B threadgroup,
  `muse-spark/round-3/kernels/routing/attempt-4/grade.json`) ALSO fails
  G1 — and the pipeline cap moved with the kernel, from 640 (attempt-2)
  to 576. "Design under 640" was wrong advice: the cap is per-pipeline
  and not stable across near-identical recompiles. Both caps are
  whole-SIMD multiples (20x32, 18x32), pointing at register allocation
  (inferred; threadgroup memory shrank, so it cannot explain a lower
  cap). Sharing designs must either fit well under a moving cap (8 rows
  x 60 = 480 is the next probe, inside round-1 attempt-2's
  proven-passing band) or cut per-thread registers; details in the
  attempt's `notes.md`.
- builder-3-e, 2026-11-04T18:24:35Z — follow-up: the 8-rows/group fix (480 threads, 20,480 B threadgroup, `muse-spark/round-3/kernels/routing/attempt-5/grade.json`) PASSES G1+G2 — the first direction-(b) kernel to run and prove exact (0 rows differing, all six sizes). Prefill timing vs the staged-path anchor: 147.3/561.6/1,052.7 vs 227.1/1,298.8/2,513.0 (0.65x/0.43x/0.42x) — the 8x traffic cut removes over half the re-read gap, but per-group overhead runs ~4x the model (~3.2 us/group vs ~0.8), leaving 1.44x/2.52x/3.50x slower than compiled. Next probe: stream 2-4 sequential 8-row blocks per group through one w-staging; details in the attempt's `notes.md`.
