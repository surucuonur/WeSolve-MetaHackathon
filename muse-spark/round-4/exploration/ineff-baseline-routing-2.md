# Inefficiency: the staged scalar path's shape cost at decode sizes — 27 barriers, a 23%-duty tail third, serial tails

- **Owner:** explorer-baseline-routing
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check the duty/barrier counts and the residual table; builders take only the exact-preserving items listed

## Slot

`routing`

## What happens

Past the bytes in `ineff-baseline-routing-1`, the staged scalar path that runs at every decode size above 1 row pays a per-row shape cost in `muse-spark/round-3/kernels/routing/attempt-6/block-0.metal` (the code in the slot). Round 3's `ineff-baseline-routing-2` (frozen, confirmed) priced this shape at *prefill* sizes and put the decode saving at ~0 because the candidate's residual already beat the compiled reference's overhead. That verdict stands; what is new in round 4 is the question — not "reach the compiled reference's overhead" but "cut the candidate's own residual", since every µs/layer cut at 32 rows widens the set's 32×1024 margin 1:1 (territory entry, headline). The items, all inside one 256-thread threadgroup per row:

1. **27 barriers per row.** The 12 staged groups pay 2 `threadgroup_barrier`s each (stage-then-compute block-0:94, compute-then-overwrite block-0:119), plus 3 outside the loop (block-0:33,129,137). Every barrier stalls the whole threadgroup; the `rows == 1` GEMV branch pays only the 3 outside ones.
2. **The last third of K-chunks runs at 23% thread duty.** At every decode size 2–32 the partition formula (block-0:57-75) gives 32 partitions, `base_iters = 1`, the last partition absorbing chunks 31–47. Groups at `gbase` 0–28 therefore touch 4 partitions and run 240/256 threads (94%); the four groups at `gbase` 32–44 touch only partition 31 and run 60/256 threads (23.4%, block-0:99-101) — 16 of 48 chunks (33% of each row's 46,080 FMAs) issue through a quarter of the lanes, behind the same 2 barriers per group as the full groups. The final sequential partial sum also runs 60/256 (block-0:121-127). Recomputed from the formula this round; the graded 8/32 and every ungraded 2–7/9–31 share this geometry.
3. **28,080 B threadgroup footprint** (`xpad[816]` + `scores[60]` + `parts[2048]` + `wsh[4096]` floats, block-0:15,22-25) — 85.7% of the 32 KB limit, 4,688 B of headroom. At decode sizes the starting point's own account says this does not yet bind (flat 69→73 µs from 8 to 32 rows as staging/barriers overlap across resident threadgroups, `muse-spark/round-2/kernels/routing/attempt-4/notes.md`); it is a constraint on fixes, not a measured decode cost — wider staging is already proven infeasible (6 chunks/group needs 36,272 B with `parts`, 8 chunks/group needs 32,768 B of `wsh` alone; hyp-3 builder-3-c, frozen), and the round-3 verifier flagged the residency mechanism itself as unmeasured.
4. **Dead work at `rows == 1`.** Lines 27–32 stage `xpad` and zero `parts[2048]` (8 floats per thread) before the branch; the GEMV branch (block-0:35-55) then reads neither `parts` nor `wsh`. The zeroing is pure waste on the 1-row path, in front of a barrier.
5. **Serial tails.** The sigmoid runs 60/256 threads (block-0:132-136); the top-4 + weights fold is thread 0 alone (block-0:139-174): 60 strictly-greater insertion iterations, each adding a `bias[e]` it loads from device itself, then 4 divisions — 255 threads idle after the last barrier, on every row at every size. (Unchanged since round 1; round 2's entry 3 priced it as lost in the fixed cost, which was the right call at prefill scale and is re-listed here only because the round-4 prize is single-digit µs/layer.)

Items 1–2 cannot be fixed by rebalancing accumulation: the partition grouping *is* the reference's split-K rounding order (changing partition boundaries or sharding one expert's K across threads reorders a serial chain and fails G2 — hyp-3 boundary and builder-3-c, frozen). What remains legal is feeding: let the 196 idle threads in tail groups pre-stage or pre-load instead of waiting at barriers, and delete dead work — changes that keep every row's accumulation order and so keep G2, as attempt-4 itself proved against attempt-3.

## Cost

Measured chained µs/layer (candidate) minus the traffic floor from `ineff-baseline-routing-1` (200 GB/s), both runs of the identical bytes; compiled overhead ≈ compiled total minus its own ~0.9 µs read-once floor:

| rows | measured (re-grade / round-3 grade) | traffic floor | candidate residual | compiled overhead (re-grade) |
|---|---|---|---|---|
| 1 | 45.1 / 51.7 | 0.9 µs | 44.2 / 50.8 µs | ~61.9 µs |
| 8 | 144.2 / 90.9 | 7.4 µs | 136.8 / 83.5 µs | ~159.1 µs |
| 32 | 151.3 / 132.2 | 29.7 µs | 121.6 / 102.5 µs | ~170.1 µs |
| 127 / 1,016 / 2,048 | `handed off` | — | 0 (kernels do not run) | — |

Kernels × fixed cost: 1 kernel/layer here against the reference's ~12 (brief:49), so the residual is not launch count — it is the chained per-call floor (shared with the compiled reference, whose overhead is larger at every decode size) plus the per-threadgroup shape above. The residual therefore *overstates* what this entry can claim: the shape-attributable slice inside it (barrier exposure, tail-group latency, serial tails, dead zeroing) has never been decomposed by measurement, and the one barrier-cost datum in the tree is inferred and confounded (round 3's sharing kernel: ~3.2 µs/group overhead against ~0.8 modelled, "barriers ~2 µs each or ~4× occupancy serialisation, inferred", hyp-3 attempt-5). At decode sizes the overlap across 8–32 resident threadgroups hides much of the barrier/staging latency the prefill pricing exposed (round-3 entry 2's residual grew with rows; the decode residual does not scale per row — 136.8 µs at 8 rows vs 121.6 at 32 in the same run, i.e. rows past the first are nearly free).

## Kind

Overhead-bound (`claude/GLOSSARY.md:39-42`): time goes to barriers, idle thread slots and the chained fixed floor, not to bytes (entry 1) or FLOPs (46,080 FMA/row ≈ 0.01 µs at peak compute). At prefill sizes: not applicable — handed off.

## Spans

`baseline-routing`; the prize lands in `g3` (set margin at 32×1024). This entry owns the per-threadgroup shape; `ineff-baseline-routing-1` owns the bytes. The two do not add beyond the candidate's measured total, and neither adds beyond the residual-plus-floor decomposition of it.

## Predicted saving

Upper bound: the shape-attributable slice of the candidate residual above — unmeasured, so no single figure is claimed. Bounded sub-items, per layer (×20 for per-step), prediction only, counting only this slot's own effect, prefill 0 at every size (nothing here runs above `max_rows: 32`, so the other phase cannot be cost):

- **Dead `parts` zeroing at 1 row** (item 4): removing 2,048 threadgroup writes in front of a barrier is worth at most their issue/mem time — sub-µs/layer; certain in direction, negligible in size. 0 at 8/32 rows (the zeroing is used there).
- **Tail-group feeding overlap** (items 1–2): if the four 23%-duty groups' exposed staging/barrier latency were fully hidden by idle-thread pre-staging, the saving is at most those groups' barrier-to-barrier time beyond their 16-FMA-per-thread scalar chain — with the tree's only barrier datum (~2 µs/barrier, inferred, prefill-confounded) as an outer envelope, ≤ ~8 barriers' worth ≈ single-digit µs/layer at 8/32 rows, 0.02 ms/step per µs/layer at 32 rows. Realised value is likely well under the envelope because cross-threadgroup overlap already hides part of it at decode sizes.
- **Serial tails** (item 5): parallelising the sigmoid is already 60-wide; the thread-0 scan could be widened only by a parallel top-4 that reproduces the strictly-greater, earliest-index-wins tie order bit-exactly (a tree reduction with index-ordered tie-breaking does, in principle) — worth at most the scan's serial latency, unmeasured, plausibly ~1 µs/layer at every decode size.

Total honest expectation: single-digit µs/layer at 32 rows — material only combined with `ineff-baseline-routing-1` and the combine slot's margin against the +7.5–12.5 µs/layer (routing+combine) that 32×1024 non-overlap needs, and each item must be proven on the same-run chained timing (round 3's lesson: a ~2–3 µs copy launch flipped decode `faster` to `same`, attempt-3). A builder who cannot decompose the residual first should treat this entry as a list of legal micro-directions, not a banked saving.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** shape items in [block-0.metal](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/routing/attempt-6/block-0.metal:27) — barriers at :33, :94, :119 per staged group, :129, :137 → 12×2+3 = **27/row**, GEMV branch pays only the 3 outside ✓; partition formula at :57-75 gives `tm = 1, ratio = 24, partitions = 32, base_iters = 1` at decode 2–32, last partition chunks 31–47; groups `gbase` 0–28 run `tid < 4×60` = 240/256 (94%), `gbase` 32–44 run 60/256 (23.4%) at :99-101, final sum 60/256 at :121-127; 16 of 48 chunks = 33% of 46,080 FMAs/row through a quarter of lanes ✓; footprint (816+60+2048+4096)×4 = 28,080 B = 85.7% of 32 KB, 4,688 B headroom (:15,22-25) ✓; dead `parts` zeroing at :27-32 unused by GEMV :35-55 (which reads `xpad`/`w_router` only) ✓; sigmoid 60/256 at :132-136, thread-0 top-4 at :139-174 ✓; reordering partitions fails G2, feeding-only fixes legal ([hyp-3-routing-prefill.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/implementation/hyp-3-routing-prefill.md) boundary) ✓.
- **confirmed:** residuals (measured − traffic floor from entry 1) — re-grade 45.1−0.9 = 44.2, 144.2−7.4 = 136.8, 151.3−29.7 = 121.6 µs; round-3 grade 50.8/83.5/102.5 µs; compiled overhead ~61.9/159.1/170.1 µs (compiled total − ~0.9 µs read-once floor, `~` as written) ✓; 1 kernel/layer vs ~12 reference, 46,080 FMA/row ≈ 0.01 µs at 5.2 TFLOPS — overhead-bound ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)) ✓; shape slice never decomposed, sole barrier datum ~3.2 vs ~0.8 µs/group inferred/confounded from attempt-5, cross-threadgroup overlap hiding decode latency — all carried as caveats, residual not claimed as saving ✓.
- **confirmed:** saving bounds labelled prediction only — dead zeroing sub-µs at 1 row, tail feeding single-digit µs/layer envelope at 8/32 (0.02 ms/step per µs/layer), serial tail ~1 µs/layer unmeasured; total single-digit at 32 rows against the +7.5–12.5 µs/layer set need, prefill 0 by `max_rows: 32`, same-run chained proof required ✓.
