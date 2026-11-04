# attempt-5 (round 2): hardening — parallel block stores (builder-1-e)

Hardening attempt for `hyp-1-experts-exact` (slot `experts`, baseline
`none`). Builds on `muse-spark/round-2/kernels/experts/attempt-4`
(interleaved gate/up, fused single kernel, 512 threads/pair, G2 pass
all six, 94.0 / 177.0 / 591.3 µs/layer chained vs compiled
73.5 / 239.0 / 796.3 — loses only at 1 row).

## The one main change

Parallelize the block stores, decode branch only. Nothing else
changes: same `qvm_block` / `qvm_gate_up_pair` / `qmm_accum` bodies
(`block-1.metal` byte-identical to attempt-4, verified by diff),
same interleaved gate/up K loop, same silu, same prefill branch,
same grids and `spec.json` (byte-identical).

- **Before (attempt-4):** after each block's `simd_sum`, lane 0 alone
  stored the block: `for (j = 0..31) tg_gate[b*32+j] = p[j]` and
  likewise `tg_up`, and for down `y[pair*768 + b*32 + j] = p[j]`.
  That is 48 blocks per pair (12 gate + 12 up + 24 down) × 32 serial
  stores, all on the 16 lanes-0 of the pair's threadgroup (12 active
  in gate/up, up to 16 per down wave), while the other 31 lanes of
  each simdgroup idle at the following barrier / next wave.
- **Now:** after each `simd_sum` every lane holds the full reduced
  32-vector (`p[j] = simd_sum(p[j])` for all `j` in `block-1.metal`,
  so each lane's `p[0..31]` is the whole block output). Lane `j`
  stores element `j` — one store per lane, 32-way parallel:
  `tg_gate[b*32 + lane] = (bfloat)p[lane]`,
  `tg_up[b*32 + lane] = (bfloat)pu[lane]`, and for down
  `y[pair*768 + b*32 + lane] = (bfloat)p[lane]` (still inside the
  `b < 24` guard, so idle simdgroups store nothing). The
  `(bfloat)` rounding at the store is unchanged, so the bfloat16
  rounding points are identical.
- **Prefill (`rows >= 60`) untouched:** it already stores one output
  per lane (`tg_gate[b*32 + lane]`, `y[... + lane]`), so there is no
  serial store to parallelize there.

Correctness rests on `simd_sum` being a broadcast reduction: if any
lane held only its own partial, the parallel store would write
partials and G2 would fail loudly with large `differ` at decode
sizes. That is the intended check.

## Which inefficiency it removes, and how

- `ineff-baseline-experts-2` (chained 1-row latency): attempt-4's
  miss analysis left the still-sequential tails — chained fixed
  cost, `tg_x` load, silu, barriers and the down chain — plus these
  serial stores on the critical path. On the per-pair critical path
  the serial stores are: gate/up phase, 64 threadgroup stores by
  each active simdgroup's lane 0 (32 gate + 32 up, 12 simdgroups
  concurrent) before the silu barrier; down, 32 device stores per
  wave × 2 waves. That is ~128 serial store issues (plus their loop
  control) that every other lane waits on. Parallel stores cut that
  to 4 parallel store steps (gate 1, up 1, down 1 per wave),
  removing ~124 serial issues from the path without touching any
  accumulation order, so decode bits are attempt-4's by construction
  if the broadcast premise holds.
- `ineff-baseline-experts-1` (prefill G2 gate): already removed by
  attempt-4's branch; kept by reusing its scalar `qmm_accum`
  verbatim.
- `ineff-baseline-experts-3` (per-pair re-reads): deliberately
  kept, as in attempts 1–4 — weight bytes per pair stay 552,960 B;
  the store bytes are tiny beside them (`y` 1,536 B/pair = 0.28%).

## Limits (by hand; the grader does not check the 32 KB cap)

512 threads/group ≤ 1,024 and threadgroup memory unchanged at
3,840 B (`tg_x[768] + tg_gate[384] + tg_up[384] + tg_h[384]`
bfloat16) of 32,768 B, rows-independent (`spec.json` identical to
attempt-4). No new state: `p[lane]` reads the existing per-thread
accumulator. Valid at every `rows ≥ 1` on both sides of 60; decode
stores cover every `(pair, block, lane)` exactly once, as before
(gate/up: 12 simdgroups × 32 lanes; down: 24 blocks × 32 lanes over
2 waves).

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Attempt-4 (other run): 94.0 / 177.0 / 591.3 vs compiled
73.5 / 239.0 / 796.3. Baseline `none`, so no
`wins_vs_baseline` / `beats_baseline`.

- **1 row: 91.** Latency-bound. ~124 serial store issues leave the
  critical path; at ~20 exposed cycles each (single-lane store
  pipeline plus 32-iteration loop overhead, GPU ≈ 1.3 GHz) that is
  124 × 20 / 1.3e9 ≈ 1.9 µs, plus the down device stores' extra
  latency on lane 0, ≈ 3 µs total → 94.0 − 3 ≈ **91 µs**. Still
  above compiled 73.5 — this trims a tail, it does not close the
  remaining down-chain / fixed-cost gap.
- **8 rows: 175.** 32 groups per call partly hide one group's store
  tail behind other groups' K loops, so roughly half the 1-row
  saving survives: 177.0 − 2 ≈ **175 µs**, keeping the win over
  compiled 239.0.
- **32 rows: 589.** Streaming-bound (attempt-4: 119.7 GB/s on pair
  bytes). Store bytes are 0.28% of pair traffic and group-level
  parallelism already hides the store latency, so the change is
  ~neutral: 591.3 − 2 ≈ **589 µs**, keeping the win over compiled
  796.3. Risk: none structural — no register or threadgroup change;
  the only risk is the broadcast premise, which G2 tests directly.

G2 prediction: pass at all six sizes with `differ == 0` — the
parallel store writes the same `(bfloat)p[j]` values attempt-4's
lane 0 wrote, just from lane `j` instead of lane 0, and the prefill
path is byte-identical to attempt-4's.

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0
everywhere** (decode 1/8/32 and prefill 127/1,016/2,048) — the
broadcast premise holds: every lane's `p[0..31]` after `simd_sum`
is the full reduced block, so lane `j` storing `p[j]` reproduces
attempt-4's serial lane-0 stores bit for bit, for gate, up and
down alike. Chained timing µs/layer (this run; candidate vs
reference vs compiled): **1 row 92.1 / 95.7 / 75.9 (loses to
compiled, beats reference)**; 8 rows 162.0 / 232.4 / 234.6 (wins);
32 rows 534.8 / 802.3 / 798.2 (wins). `beats_compiled` false —
only the 1-row size misses, as in attempt-4. Baseline `none` — no
`wins_vs_baseline`/`beats_baseline`.

Against attempt-4 (different run; compiled shifted 73.5 → 75.9 at
1 row and reference 81.2 → 95.7 between runs, so treat cross-run
deltas as indicative): 92.1 vs 94.0 at 1 row (**−2.0%**), 162.0
vs 177.0 at 8 rows (**−8.5%**), 534.8 vs 591.3 at 32 rows
(**−9.6%**). Effective bandwidth on pair bytes: 24.0 / 109.2 /
132.3 GB/s (attempt-4: 23.5 / 99.9 / 119.7). The 32-row rate is
back at attempt-1's 132.1 GB/s level while keeping the
interleave's 1/8-row wins — attempt-5 dominates attempt-4 at
every size and matches the best previous 32-row time (attempt-1:
535.7).

## Miss analysis (prediction 91 / 175 / 589, measured 92.1 / 162.0 / 534.8)

Misses are +1.2% / −7.4% / −9.2% — all inside the 20% bar, so no
formal miss, but the size pattern was inverted versus the
prediction (smallest gain at 1 row, largest at 32 rows) and that
needs explaining:

1. **At 1 row the serial stores were a small tail, as the
   arithmetic said, but the fixed remainder is larger than the
   store tail.** Removing ~124 serial issues bought only ~2 µs of
   the predicted ~3; the remaining ~16 µs gap to compiled (92.1 vs
   75.9) is the still-sequential down chain (12 strided steps × 2
   waves behind the silu barrier), the `tg_x` load, silu, barriers
   and the chained launch/link cost — none of which this change
   touches.
2. **At 8/32 rows the serial device stores cost more than pure
   latency.** Lane 0's 32 serial stores to `y` per down block are
   32 separate single-lane transactions to consecutive addresses;
   32 lanes storing in parallel coalesce into full transactions.
   Paid per block (24 per pair) on every streaming step, that
   inefficiency — plus the serial threadgroup stores delaying the
   silu barrier — taxed attempt-4 by ~9% at 32 rows, which the
   latency-only prediction (store bytes = 0.28% of pair traffic)
   missed. Parallel stores removed it, which is also why the
   interleave's 32-row register tax (attempt-4 notes) no longer
   shows: 534.8 ≈ attempt-1's non-interleaved 535.7.

Lesson for `ineff-baseline-experts-2`: a serial lane-0 store
after a broadcast reduction is never free — at latency sizes it
is a barrier tail, at streaming sizes the uncoalesced device
stores tax every block. Any future block kernel should store
from all 32 lanes by default.

## Next change

Make the down product's two waves overlap like gate/up: one
simdgroup computes two down blocks in one interleaved K loop
(separate accumulators, per-block `qvm_block` order and the new
parallel store kept), attacking the still-sequential 12-step
down chain that is now the largest named remainder at 1 row.
The row-conditional interleave attempt-4 proposed is moot —
attempt-5 already restored attempt-1's 32-row rate (534.8 vs
535.7) with the interleave kept.
