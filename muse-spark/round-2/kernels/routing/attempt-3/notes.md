# Round-2 attempt 3 — builder-2-c, hardening of hyp-2-routing-exact

Builds on the best full-G2 attempt, `muse-spark/round-2/kernels/routing/attempt-1`
(G1 pass, G2 pass at all six sizes; chained 54.9/91.4/114.8 µs/layer vs
compiled 56.2/91.2/98.9 — wins only at 1 row). Baseline: none, so no
`beats_baseline`; the bar is the compiled reference.

## The one main change

Replace attempt-1's simdgroup-MMA threadgroup staging with scalar
partition accumulation, nothing else:

- Removed: `Ash`/`Bsh`/`Csh` staging, `simdgroup_float8x8` loads/MMA/store,
  and the 48 chunks × 2 threadgroup barriers per row (plus the 7/8 wasted
  rows of each 8×8 tile). The `<metal_simdgroup_matrix>` include goes too.
- Added: a `parts[60×32]` threadgroup array. Each (expert, partition)
  partial is one thread's serial scalar sum `part += xsh[d] * w_router[e*768+d]`
  over its K range in increasing `d`; one barrier; then partials are summed
  sequentially in partition order (`acc = 0; acc += parts[p]`), exactly
  attempt-1's partial order with the MMA inner accumulation replaced by
  the scalar sum attempt-2 proved bit-identical at decode.

Kept verbatim from attempt-1: the regime branch on `rows` (GEMV only at
`rows == 1`; split-K while `rows <= 768`; regular above), the split-K
partition formula (`next_pow2(48/((rows+31)/32*2))` clamped to [2,32],
base `48/partitions` BK=16 chunks, last partition to chunk 48 — so 32
partitions at 8/32 rows, 8 partitions of 96 at 127 rows, 2 at 257–768),
the single partition covering all 48 chunks above 768 rows, the GEMV
lane order and shuffle tree 16,8,4,2,1, MLX's verbatim sigmoid
(`1/(1+precise::exp(abs(s)))`, `(s<0)?y:1-y`), the strictly-greater
insertion top-4 (earliest index wins ties), and
`((w0+w1)+w2)+w3 + 1e-20`, divide, ×2.5 on unbiased scores. Same grid
(`rows*256`), threadgroup (256) and one-kernel layout. Threadgroup memory
(768+60+1920)×4 B = 10,992 B of 32 KB.

## Inefficiency removed

`ineff-baseline-routing-1` stays removed (same orders as attempt-1), and
attempt-1's staging cost — the reason it loses at 32 rows (its notes:
48×2 barriers/row, 7/8 tile waste, per-row `w_router` re-read) — is
attacked directly: the scalar path has 2 barriers per row total instead
of 96+, and no tile waste. The per-row `w_router` re-read
(`ineff-baseline-routing-4`) and serial thread-0 scan (part of
`ineff-baseline-routing-3` kept for the tie rule) are untouched.

## Predicted µs per layer at 1 / 8 / 32 rows

Anchor: attempt-1 measured 54.9/91.4/114.8, compiled 56.2/91.2/98.9
(same grader run, hyp entry). Traffic per row is 184 KB of `w_router`
(0.9 µs at 200 GB/s), so traffic floors are 0.9/7.4/29.5 µs; the rest is
chained-call latency and per-row staging/serial work.

- 1 row: the GEMV branch is unchanged, so ≈ attempt-1's 54.9 → **55**.
- 8 rows: attempt-1's per-row increment over its 1-row time is
  (91.4−54.9)/7 ≈ 5.2 µs/row on the MMA path. Removing 96 barriers and
  the tile staging per row, the scalar critical path per expert is the
  last partition's 272 serial products, hidden across 8 threadgroups ×
  8 warps; estimate the increment falls to ≈1.4 µs/row → 54.9+7×1.4 ≈ **65**.
- 32 rows: same increment → 54.9+31×1.1 ≈ **89** (per-row increment
  falls further as 32 threadgroups fill the GPU; traffic floor 29.5 µs
  is inside this, not added to it).

Prediction: **55 / 65 / 89 µs** — beats compiled (56.2/91.2/98.9) at all
three if the staging model holds. G2 prediction: pass at all six sizes
if attempt-2's scalar≡MMA result extends to the 8-partition (127-row)
and single-accumulator (>768) groupings, which are the only scalar
groupings attempt-2 did not already prove.

## Verdict (2026-11-04T03:51:53Z, one grader run)

G1 pass, **G2 pass at all six sizes** (rows_differ 0, rows_other_experts
0, max_rel_err 0.0 at 1/8/32 and 127/1016/2048). Chained µs/layer
(ref / compiled / candidate): 1 row 68.7 / 58.0 / **46.9** (beats
compiled); 8 rows 106.2 / 84.5 / 300.0; 32 rows 106.7 / 90.9 / 297.2.
`beats_compiled` false. Baseline none, so no `beats_baseline`.

The exactness half of the bet won everywhere: attempt-2's scalar≡MMA
result extends to the two groupings it never tested — 8 scalar
partitions of 96 at 127 rows and the single serial accumulator over 48
chunks above 768 rows both reproduce the steel regimes bit for bit on
all graded inputs. Scalar partition accumulation inside attempt-1's
regime logic is therefore a full-G2 design, and the fastest at 1 row
(46.9 vs attempt-1's 54.9, same GEMV branch — the win there is the
removed MMA staging code around it, or run-to-run drift; the branch
itself is unchanged).

## Why the 8/32-row timings missed the prediction by >20%

Predicted 55/65/89, measured 46.9/300.0/297.2: the 1-row prediction
holds (−15%), the scalar rows are 4.6×/3.3× over. Analysis (no second
GPU run; this is code-level, not an isolated measurement):

- The scalar path is latency-bound, not bandwidth-bound. The candidate
  is flat 300→297 µs while rows quadruple (traffic 1.5→5.9 MB), where
  attempt-1 rose 91.4→114.8. Per row the path issues 60×768 = 46,080
  scalar `w_router` loads as one dependent FMA chain per
  (expert, partition) thread; the threads holding partition 31 chain
  272 dependent device loads, and at 8 rows only 64 warps (8
  threadgroups × 8) are resident across 16 GPU cores — too few to hide
  that chain. Attempt-1's MMA consumed 16 K values per hardware op
  after one cooperative stage per chunk; the scalar version pays a
  full load-to-FMA latency per product.
- The loads are also uncoalesced and bank-conflicted. A warp is one
  expert's 32 partitions, so lanes read `w_router` 16 floats (64 B)
  apart — 16 cache lines per load step — and read `xsh[p*16+k]` with
  all even partitions on the same threadgroup bank (`(p*16+k) mod 32`
  is `k` or `k+16`), a 16-way bank conflict on every product. The
  prediction assumed staging removal was pure profit; in fact
  attempt-1's per-chunk staging was also its coalescing mechanism.

Next change (one): keep the scalar sums and partition order, but stage
each partition's `w_router` block into threadgroup memory with
coalesced vector loads (as attempt-1 staged chunks) and sum from
threadgroup with a swizzled/padded `xsh` layout, so the scalar chain
reads conflict-free local memory instead of uncoalesced device memory.
